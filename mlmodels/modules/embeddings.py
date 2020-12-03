# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import math
import torch
import numpy as np
import torch.nn as nn


class Emb_layer(nn.Module):
    """
    This module take (characters or words) indices as inputs and outputs (characters or words) embedding
    """

    def __init__(self, HPs):
        super(Emb_layer, self).__init__()
        [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        self.zero_padding = zero_padding
        self.embeddings = nn.Embedding(size, dim, padding_idx=0)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))
        if not requires_grad:
            print("Fixed pre-trained embeddings")
            self.embeddings.weight.requires_grad = requires_grad
        self.drop = nn.Dropout(drop_rate)

    def forward(self, inputs, auxiliary_embs=None):
        return self.get_embs(inputs, auxiliary_embs)

    def get_embs(self, inputs, auxiliary_embs=None):
        """
        embs.shape([0, 1]) == auxiliary_embs.shape([0, 1])
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0, 1, 2, 3])
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        if auxiliary_embs is not None:
            assert embs_drop.shape[:-1] == auxiliary_embs.shape[:-1]
            embs_drop = torch.cat((embs_drop, auxiliary_embs), -1)
        return embs_drop

    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index, :] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs

    def set_zeros(self, idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)


class PositionalEncoding(nn.Module):
    def __init__(self, HPs):
        super(PositionalEncoding, self).__init__()
        [size, dim, pre_embs, drop_rate, zero_padding, requires_grad, max_len] = HPs
        HPs[3] = 0.0

        self.wordemb_layer = Emb_layer(HPs[:-1])
        self.dropout = nn.Dropout(p=drop_rate)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, inputs, auxiliary_embs=None):
        """
        :param inputs: word tensors in (batch, length)
        :return: wordpos embs in (batch, length, size)
        """
        x = self.wordemb_layer(inputs, auxiliary_embs)
        x = x + self.pe[:, x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    word_HPs = [30000, 512, None, 0.0, False, True]  # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad]
    wordemb_layer = Emb_layer(word_HPs)

    # pos_HPs = [512, 0.1, 5000]  # [d_model, dropout, max_len]
    max_len = 5000
    posemb_layer = PositionalEncoding(word_HPs + [max_len])

    x = torch.randint(1000, (50, 32)).to(dtype=torch.long)
    wordemb = wordemb_layer(x)
    wordposemb = posemb_layer(x)
