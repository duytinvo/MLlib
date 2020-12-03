# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import torch
import torch.nn as nn
from mlmodels.modules.cnns import CNN_layer
from mlmodels.modules.rnns import RNN_layer


class Word_Encoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs):
        super(Word_Encoder, self).__init__()
        self.nn_bidirect = word_HPs[4]
        if word_HPs[0] == "cnn":
            self.word_nn_embs = CNN_layer(word_HPs)
        else:
            self.word_nn_embs = RNN_layer(word_HPs)

    def forward(self, emb_inputs, word_lengths, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, word_lengths, init_hidden)

    def get_all_hiddens(self, emb_inputs, word_lengths, init_hidden=None):
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # hidden_out = (h_n,c_n) --> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
        rnn_out, hidden_out = self.word_nn_embs(emb_inputs, word_lengths, init_hidden)
        return rnn_out, hidden_out

    def get_last_hiddens(self, hidden_out):
        # TODO: it may be wrong for CNN
        # hidden_out = (h_n,c_n) --> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
        if type(hidden_out) == tuple:
            if self.nn_bidirect:
                # extract the last feature vector from h_n
                h_n = torch.cat((hidden_out[0][-2, :, :], hidden_out[0][-1, :, :]), -1)
            else:
                h_n = hidden_out[0][-1, :, :]
        else:
            if self.nn_bidirect:
                h_n = torch.cat((hidden_out[-2, :, :], hidden_out[-1, :, :]), -1)
            else:
                h_n = hidden_out[-1, :, :]
        # h_n = self.word_nn_embs.get_last_hiddens(emb_inputs, word_lengths, init_hidden)
        return h_n