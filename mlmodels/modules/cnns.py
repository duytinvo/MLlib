# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import torch
import torch.nn as nn


class CNN_layer(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an CNN layer to extract:
        - all hidden features
        - last hidden features
    """
    def __init__(self, HPs):
        super(CNN_layer, self).__init__()
        [_, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, kernel_size] = HPs
        nn_cnn_dim = nn_out_dim // 2 if nn_bidirect else nn_out_dim
        # mlmodels.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation, groups, bias, padding_mode)
        # L_out = [L_in + 2 × padding − dilation × (kernel_size−1) − 1] / stride + 1; dilation=1, stride=1
        # L_out = L_in + 2 × padding −  kernel_size + 1
        # L_out = L_in ==> 2 × padding −  kernel_size + 1 = 0
        # kernel_size = 3 ==> padding = 1; kernel_size = 5 ==> padding = 2
        padding = int((kernel_size-1)/2)
        self.nn_bidirect = nn_bidirect
        self.nn_layers = nn_layers
        self.fw_layers = nn.ModuleList()
        self.fw_layers.append(nn.Conv1d(nn_inp_dim, nn_cnn_dim, int(kernel_size), padding=padding))
        for i in range(nn_layers-1):
            self.fw_layers.append(nn.Conv1d(nn_cnn_dim, nn_cnn_dim, int(kernel_size), padding=padding))
        if nn_bidirect:
            self.bw_layers = nn.ModuleList()
            self.bw_layers.append(nn.Conv1d(nn_inp_dim, nn_cnn_dim, int(kernel_size), padding=padding))
            for i in range(nn_layers - 1):
                self.bw_layers.append(nn.Conv1d(nn_cnn_dim, nn_cnn_dim, int(kernel_size), padding=padding))

    def forward(self, emb_inputs, input_lengths=None, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, input_lengths, init_hidden)

    def get_all_hiddens(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
            input:
                emb_inputs: tensor(batch_size, seq_length, emb_dim)
            output:
                tensor(batch_size, seq_length, hidden_dim)
                hn: final feature vector of a sequence
                hc: max-pooling feature vector
        """
        # emb_inputs: (batch_size, seq_length, emb_dim) --> (batch_size, emb_dim, seq_length)
        emb_inputs = emb_inputs.transpose(1, -1)
        # w_0 --> w_n
        inp_fwcnn = emb_inputs
        hn_fw = []
        hc_fw = []
        out_fwconv = torch.empty_like(inp_fwcnn)
        for i in range(self.nn_layers):
            # out_fwconv: (batch_size, hidden_dim, seq_length)
            out_fwconv = self.fw_layers[i](inp_fwcnn)
            # hn_bw, hc_bw: (batch_size, hidden_dim)
            hn_fw.append(out_fwconv[:, :, -1])
            # equivalent with max pooling
            hc_fw.append(out_fwconv.max(-1)[0])
            inp_fwcnn = out_fwconv
        # hn: (num_layers, batch_size, hidden_dim)
        h_n = torch.stack(hn_fw)
        h_c = torch.stack(hc_fw)
        # cnn_out: (batch_size, hidden_dim, seq_length)
        cnn_out = out_fwconv

        if self.nn_bidirect:
            # reverse seq_length w_n --> w_0
            inp_bwconv = emb_inputs.flip(-1)
            hn_bw = []
            hc_bw = []
            out_bwconv = torch.empty_like(inp_bwconv)
            for i in range(self.nn_layers):
                # out_bwconv: (batch_size, hidden_dim, seq_length)
                out_bwconv = self.bw_layers[i](inp_bwconv)

                # hn_bw, hc_bw: (batch_size, hidden_dim)
                hn_bw.append(out_bwconv[:, :, -1])
                # equivalent with max pooling
                hc_bw.append(out_bwconv.max(-1)[0])
                inp_bwconv = out_bwconv
            # hn_bw: (num_layers, batch_size, hidden_dim)
            hn_bw = torch.stack(hn_bw)
            hc_bw = torch.stack(hc_bw)
            # h_n: (num_layers * num_directions, batch_size, hidden_dim)
            h_n = torch.cat([h_n, hn_bw], dim=0)
            h_c = torch.cat([h_c, hc_bw], dim=0)
            # cnn_out: (batch_size, num_directions*hidden_dim, seq_length)
            # reverse back w_0 --> w_n before concatenating
            out_bwconv = out_bwconv.flip(-1)
            cnn_out = torch.cat([cnn_out, out_bwconv], dim=1)

        # cnn_out: (batch_size, seq_length, num_directions*hidden_dim)
        cnn_out.transpose_(1, -1)
        return cnn_out, (h_n, h_c)

    def get_last_hiddens(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
            input:
                emb_inputs: tensor(batch_size, seq_length, emb_dim)
            output:
                tensor(batch_size, seq_length, hidden_dim)
        """
        # emb_inputs: (batch_size, seq_length, emb_dim) --> (batch_size, emb_dim, seq_length)
        emb_inputs.transpose_(1, -1)
        inp_fwcnn = emb_inputs
        hn_fw = []
        out_fwconv = torch.empty_like(inp_fwcnn)
        for i in range(self.nn_layers):
            # out_fwconv: (batch_size, hidden_dim, seq_length)
            out_fwconv = self.fw_layers[i](inp_fwcnn)
            # hn_bw, hc_bw: (batch_size, hidden_dim)
            hn_fw.append(out_fwconv[:, :, -1])
            inp_fwcnn = out_fwconv
        # hn: (batch_size, hidden_dim)
        h_n = hn_fw[-1]
        # cnn_out: (batch_size, hidden_dim, seq_length)
        cnn_out = out_fwconv

        if self.nn_bidirect:
            # reverse seq_length w_n --> w_0
            inp_bwconv = emb_inputs.flip(-1)
            hn_bw = []
            out_bwconv = torch.empty_like(inp_bwconv)
            for i in range(self.nn_layers):
                # out_bwconv: (batch_size, hidden_dim, seq_length)
                out_bwconv = self.bw_layers[i](inp_bwconv)

                # hn_bw, hc_bw: (batch_size, hidden_dim)
                hn_bw.append(out_bwconv[:, :, -1])
                inp_bwconv = out_bwconv
            # hn_bw: (num_layers, batch_size, hidden_dim)
            hn_bw = torch.stack(hn_bw)
            # h_n: (batch_size, hidden_dim * num_directions)
            h_n = torch.cat([h_n, hn_bw[-1]], dim=-1)
            # cnn_out: (batch_size, num_directions*hidden_dim, seq_length)
            # reverse back w_0 --> w_n before concatenating
            out_bwconv = out_bwconv.flip(-1)
            cnn_out = torch.cat([cnn_out, out_bwconv], dim=1)

        # cnn_out: (batch_size, seq_length, num_directions*hidden_dim)
        cnn_out.transpose_(1, -1)
        return h_n