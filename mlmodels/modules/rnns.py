# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN_layer(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer to extract:
        - all hidden features
        - last hidden features
        - all attentional hidden features
        - last attentional hidden features
    """
    def __init__(self, HPs):
        super(RNN_layer, self).__init__()
        [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        nn_rnn_dim = nn_out_dim // 2 if nn_bidirect else nn_out_dim
        self.nn_bidirect = nn_bidirect
        if nn_mode == "rnn":
            if nn_layers == 1:
                self.hidden_layer = nn.RNN(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers,
                                           batch_first=True, bidirectional=nn_bidirect)
            else:
                self.hidden_layer = nn.RNN(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                           batch_first=True, bidirectional=nn_bidirect)

        elif nn_mode == "gru":
            if nn_layers == 1:
                self.hidden_layer = nn.GRU(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers,
                                           batch_first=True, bidirectional=nn_bidirect)
            else:
                self.hidden_layer = nn.GRU(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                           batch_first=True, bidirectional=nn_bidirect)
        else:
            if nn_layers == 1:
                self.hidden_layer = nn.LSTM(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers,
                                            batch_first=True, bidirectional=nn_bidirect)
            else:
                self.hidden_layer = nn.LSTM(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                            batch_first=True, bidirectional=nn_bidirect)
            # # Set the bias of forget gate to 1.0
            # for names in self.hidden_layer._all_weights:
            #     for name in filter(lambda n: "bias" in n, names):
            #         bias = getattr(self.hidden_layer, name)
            #         n = bias.size(0)
            #         start, end = n // 4, n // 2
            #         bias.data[start:end].fill_(1.)

    def forward(self, emb_inputs, input_lengths, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, input_lengths, init_hidden)

    def get_last_hiddens(self, emb_inputs, input_lengths, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, hidden_dim)
        """
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # hc_n = (h_n,c_n)
        # h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
        rnn_out, hc_n = self.get_all_hiddens(emb_inputs, input_lengths, init_hidden=init_hidden)
        # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
        if type(hc_n) == tuple:
            if self.nn_bidirect:
                h_n = torch.cat((hc_n[0][-2, :, :], hc_n[0][-1, :, :]), -1)
            else:
                h_n = hc_n[0][-1, :, :]
        else:
            if self.nn_bidirect:
                h_n = torch.cat((hc_n[-2, :, :], hc_n[-1, :, :]), -1)
            else:
                h_n = hc_n[-1, :, :]
        return h_n

    def get_all_hiddens(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, seq_length, hidden_dim)
        """
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # hc_n = (h_n,c_n);
        # h_n = tensor(num_layers*num_directions, batch_size, rnn_dim)
        if input_lengths is not None:
            total_length = emb_inputs.size(1)
            # pack_input = pack_padded_sequence(emb_inputs, input_lengths.cpu().numpy(), True)
            pack_input = pack_padded_sequence(emb_inputs, input_lengths.cpu(), True)
            self.hidden_layer.flatten_parameters()
            rnn_out, hc_n = self.hidden_layer(pack_input, init_hidden)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=total_length)
        else:
            self.hidden_layer.flatten_parameters()
            rnn_out, hc_n = self.hidden_layer(emb_inputs, init_hidden)
        return rnn_out, hc_n