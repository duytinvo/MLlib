# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import torch
import torch.nn as nn
from mlmodels.modules.embeddings import Emb_layer
from mlmodels.modules.encoders import Word_Encoder
from mlmodels.modules.attention import Word_alignment, Col_awareness


class Word_Decoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs, drop_rate=0.5, num_labels=None, enc_att=False, sch_att='',
                 col_dim=-1, tab_dim=-1):
        super(Word_Decoder, self).__init__()
        self.embedding = Emb_layer(word_HPs[0])
        self.decoder = Word_Encoder(word_HPs[1])
        self.num_labels = num_labels
        self.enc_att = enc_att
        self.sch_att = sch_att
        hidden_dim = word_HPs[1][2]
        self.fn_dim = hidden_dim
        self.finaldrop_layer = nn.Dropout(drop_rate)
        if self.enc_att:
            self.enc_attention = Word_alignment(hidden_dim, hidden_dim)
            self.fn_dim += hidden_dim
        if self.sch_att == "en_hidden":
            assert col_dim > 0
            self.col_attention = Col_awareness(col_dim, hidden_dim)
            self.fn_dim += col_dim

            assert tab_dim > 0
            self.tab_attention = Col_awareness(tab_dim, hidden_dim)
            self.fn_dim += tab_dim
        elif self.sch_att == "de_hidden":
            assert col_dim > 0
            self.col_attention = Word_alignment(col_dim, hidden_dim)
            self.fn_dim += col_dim

            assert tab_dim > 0
            self.tab_attention = Word_alignment(tab_dim, hidden_dim)
            self.fn_dim += tab_dim
        else:
            pass

        if num_labels > 2:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()

    def forward(self, word_inputs, word_lengths, init_hidden=None,
                enc_out=None, enc_mask=None, enc_hn=None,
                colemb=None, tabemb=None, colmask=None, tabmask=None):
        rnn_out, hidden_out = self.get_all_hiddens(word_inputs, word_lengths, init_hidden, enc_out, enc_mask, enc_hn,
                                                    colemb, tabemb, colmask, tabmask)
        de_score = self.scoring(rnn_out)
        return de_score, rnn_out, hidden_out

    def get_all_hiddens(self, word_inputs, word_lengths, init_hidden, enc_out, enc_mask, enc_hn,
                        colemb, tabemb, colmask=None, tabmask=None):
        emb_inputs = self.embedding(word_inputs)
        rnn_out, hidden_out = self.decoder(emb_inputs, word_lengths, init_hidden)
        enc_context = None
        if self.enc_att:
            # enc_context: [batch, seq_length2, hidden_dim]
            enc_context = self.enc_attention(enc_out, rnn_out, enc_mask)
            # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
        if self.sch_att == "en_hidden":
            # enc_hn: [batch, hidden_dim]
            # colemb: [batch, num_col, col_features]
            # col_context: [batch, 1, col_features]
            col_context = self.col_attention(colemb, enc_hn, colmask)
            # col_context: [batch, seq_length2, col_features]
            col_context = col_context.expand(-1, rnn_out.size(1), -1)

            # tab_context: [batch, 1, tab_features]
            tab_context = self.tab_attention(tabemb, enc_hn, tabmask)
            # tab_context: [batch, seq_length2, tab_features]
            tab_context = tab_context.expand(-1, rnn_out.size(1), -1)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        elif self.sch_att == "de_hidden":
            # col_context: [batch, seq_length2, col_dim]
            col_context = self.col_attention(colemb, rnn_out, colmask)

            # tab_context: [batch, seq_length2, tab_dim]
            tab_context = self.tab_attention(tabemb, rnn_out, tabmask)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        else:
            col_context = None
            tab_context = None

        if enc_context is not None:
            rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
        if col_context is not None:
            rnn_out = torch.cat((rnn_out, col_context), dim=-1)
        if tab_context is not None:
            rnn_out = torch.cat((rnn_out, tab_context), dim=-1)
        return rnn_out, hidden_out

    def scoring(self, rnn_out):
        de_score = self.hidden2tag_layer(rnn_out)
        de_score = self.finaldrop_layer(de_score)
        return de_score

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            # label_score = [B, C]; label_tensor = [B, ]
            de_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1,))
        else:
            # label_score = [B, *]; label_tensor = [B, *]
            de_loss = self.lossF(label_score, label_tensor.float().view(-1, 1))
        return de_loss

    def inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = torch.softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred

    def logsm_inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = torch.nn.functional.log_softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred


class Word_Decoder_v2(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs):
        super(Word_Decoder_v2, self).__init__()
        print("\t\t- Add sql embedding module")
        self.embedding = Emb_layer(word_HPs[0])
        print("\t\t- Add decoder module")
        self.decoder = Word_Encoder(word_HPs[1])

    def forward(self, word_inputs, word_lengths, init_hidden=None):
        rnn_out, hidden_out = self.get_all_hiddens(word_inputs, word_lengths, init_hidden)
        return rnn_out, hidden_out

    def get_all_hiddens(self, word_inputs, word_lengths, init_hidden):
        emb_inputs = self.embedding(word_inputs)
        rnn_out, hidden_out = self.decoder(emb_inputs, word_lengths, init_hidden)
        return rnn_out, hidden_out