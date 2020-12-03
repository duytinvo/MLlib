# -*- coding: utf-8 -*-
"""
Created on 2020-03-03
@author: duytinvo
"""
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from mlmodels.modules.embeddings import PositionalEncoding, Emb_layer


class Encoder_base(nn.Module):
    def __init__(self, HPs):
        """
        :param ninp: embedding size
        :param nhead: number of heads
        :param nhid: the dimension of the feedforward network model
        :param nlayers: number of layers
        :param dropout: dropout rate
        """
        super(Encoder_base, self).__init__()
        nn_mode, ninp, nhid, nlayers, nhead, dropout, activation, norm, his_mask = HPs
        # nn_mode, nlayers, ninp, nhead, nhid, dropout, activation, norm, his_mask = HPs
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, norm)

    def forward(self, src_emb, src_mask, src_key_padding_mask=None):
        output = self.transformer_encoder(src_emb, src_mask, src_key_padding_mask)
        return output


class Encoder_layer(nn.Module):
    def __init__(self, word_HPs, enc_HPs):
        super(Encoder_layer, self).__init__()
        self.ninp = enc_HPs[1]
        self.his_mask = enc_HPs[-1]
        # self.wordemb_layer = Emb_layer(word_HPs)
        self.posemb_layer = PositionalEncoding(word_HPs)
        self.transformer_encoder = Encoder_base(enc_HPs)
        self.src_mask = None

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_key_padding_mask=None):
        if self.his_mask:
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                device = src.device
                mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        # wordemb = self.wordemb_layer(src)
        wordposemb = self.posemb_layer(src).transpose(0, 1) * math.sqrt(self.ninp)
        output = self.transformer_encoder(wordposemb, self.src_mask, src_key_padding_mask)
        return output.transpose(0, 1)


class Decoder_base(nn.Module):
    def __init__(self, HPs):
        """
        :param ninp: embedding size
        :param nhead: number of heads
        :param nhid: the dimension of the feedforward network model
        :param nlayers: number of layers
        :param dropout: dropout rate
        """
        super(Decoder_base, self).__init__()
        nn_mode, ninp, nhid, nlayers, nhead, dropout, activation, norm, his_mask = HPs
        # nn_mode, nlayers, ninp, nhead, nhid, dropout, activation, norm, his_mask = HPs
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout, activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers, norm)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = self.transformer_decoder(tgt, memory,
                                          tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output


class Decoder_layer(nn.Module):
    def __init__(self, word_HPs, dec_HPs):
        super(Decoder_layer, self).__init__()
        self.ninp = dec_HPs[1]
        self.his_mask = dec_HPs[-1]
        # self.wordemb_layer = Emb_layer(word_HPs)
        self.posemb_layer = PositionalEncoding(word_HPs)
        self.transformer_decoder = Decoder_base(dec_HPs)
        self.tgt_mask = None

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.his_mask:
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(1):
                device = tgt.device
                mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
                self.tgt_mask = mask
        else:
            self.tgt_mask = None
        # wordemb = self.wordemb_layer(tgt)
        wordposemb = self.posemb_layer(tgt).transpose(0, 1) * math.sqrt(self.ninp)
        memory = memory.transpose(0, 1)
        output = self.transformer_decoder(wordposemb, memory,
                                          self.tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output.transpose(0, 1)


if __name__ == '__main__':
    word_HPs = [30000, 512, None, 0.0, False, True]  # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad]
    wordemb_layer = Emb_layer(word_HPs)

    # pos_HPs = [512, 0.1, 5000]  # [d_model, dropout, max_len]
    max_len = 5000
    posemb_layer = PositionalEncoding(word_HPs + [max_len])

    x = torch.randint(1000, (32, 50)).to(dtype=torch.long)
    wordemb = wordemb_layer(x)
    wordposemb = posemb_layer(x)

    # nn_mode, ninp, nhid, nlayers, nhead, dropout, activation, norm, his_mask = HPs
    enc_HPs = ["self_attention", 512, 2048, 6, 8, 0.1, 'relu',
               None, False]

    transenc_layer = Encoder_layer(word_HPs + [max_len], enc_HPs)

    enc_out = transenc_layer(x)

    y = torch.randint(1000, (32, 100)).to(dtype=torch.long)

    # nn_mode, ninp, nhid, nlayers, nhead, dropout, activation, norm, his_mask = HPs
    dec_HPs = ["self_attention", 512, 2048, 6, 8, 0.1, 'relu',
               None, True]

    transdec_layer = Decoder_layer(word_HPs + [max_len], dec_HPs)

    dec_out = transdec_layer(y, enc_out)
