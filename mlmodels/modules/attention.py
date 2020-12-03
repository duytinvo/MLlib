# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import random
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class GlobalAttention(nn.Module):
    r"""
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.

    All models compute the output as
    :math:`c = \sum_{j=1}^{\text{SeqLength}} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`\text{score}(H_j,q) = H_j^T q`
       * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       attn_type (str): type of attention to use, options [dot,general,mlp]
       attn_func (str): attention function to use, options [softmax,sparsemax]

    """

    def __init__(self, sdim, tdim, adim=None, attn_type="general"):
        super(GlobalAttention, self).__init__()
        self.adim = adim
        self.sdim = sdim
        self.tdim = tdim
        assert attn_type in ["dot", "general", "mlp"], \
            ("Please select a valid attention type (got {:s}).".format(attn_type))
        self.attn_type = attn_type

        if self.attn_type == "general":
            self.linear_in = nn.Linear(tdim, sdim, bias=False)
        elif self.attn_type == "mlp":
            if adim is None:
                adim = sdim
            self.linear_query = nn.Linear(tdim, adim, bias=True)
            self.linear_context = nn.Linear(sdim, adim, bias=False)
            self.v = nn.Linear(adim, 1, bias=False)
        # # mlp wants it with bias
        # out_bias = self.attn_type == "mlp"
        # self.linear_out = mlmodels.Linear(sdim + tdim, adim, bias=out_bias)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim)``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.contiguous().view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
            else:
                assert src_dim == tgt_dim, "source dim ({}) and target dim ({}) are not equal".format(src_dim, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            adim = self.adim
            sdim = self.sdim
            tdim = self.tdim
            wq = self.linear_query(h_t.view(-1, tdim))
            wq = wq.view(tgt_batch, tgt_len, 1, adim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, adim)

            uh = self.linear_context(h_s.contiguous().view(-1, sdim))
            uh = uh.view(src_batch, 1, src_len, adim)
            uh = uh.expand(src_batch, tgt_len, src_len, adim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)
            # (batch, t_len, s_len)
            return self.v(wquh.view(-1, adim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, memory_bank, source, memory_mask=None):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``
          memory_mask (LongTensor): the source context length mask ``(batch, src_len)``

        Returns:
          (FloatTensor, FloatTensor):
          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
        """
        # one step input
        if source.dim() == 2:
            # (batch, 1, dim)`
            source = source.unsqueeze(1)

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()

        # compute attention scores, as in Luong et al.
        # (batch, t_len, s_len)
        align = self.score(source, memory_bank)

        if memory_mask is not None and align.size(0) == memory_mask.size(0):
            # (batch, 1, s_len)
            memory_mask = memory_mask.unsqueeze(1)  # Make it broadcastable.
            # align[~memory_mask] = -100
            align.masked_fill_(~memory_mask, -float('inf'))

        # if memory_lengths is not None:
        #     mask = sequence_mask(memory_lengths, max_len=align.size(-1))
        #     mask = mask.unsqueeze(1)  # Make it broadcastable.
        #     align.masked_fill_(~mask, -float('inf'))

        align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        # (batch, target_l, source_l)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        # (batch, t_len, dim)
        c = torch.bmm(align_vectors, memory_bank)
        # # concatenate
        # concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        # attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        # if self.attn_type in ["general", "dot"]:
        #     attn_h = torch.tanh(attn_h)

        return c, align_vectors


class Word_alignment(nn.Module):
    def __init__(self, in_features, out_features):
        super(Word_alignment, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.xavier_uniform_(self.weight)

    def forward(self, input1, input2, input_mask=None):
        """

        :param input1: [batch, seq_length1, in_features]
        :param input2: [batch, seq_length2, out_features]
        :param input_mask: mask of input1
        :return:
        """
        out1 = F.linear(input1, self.weight)
        # out1: [batch, seq_length1, out_features]
        # input2: [batch, seq_length2, out_features]
        out2 = torch.matmul(out1, input2.transpose(1, -1))
        # print(input1.shape, input2.shape)
        # print(out2.shape, input_mask.shape)
        if input_mask is not None and out2.size(0) == input_mask.size(0):
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, seq_length2]
        # input1: [batch, seq_length1, in_features]
        satt = torch.matmul(F.softmax(out2, dim=1).transpose(1, -1), input1)
        # satt: [batch, seq_length2, in_features]
        return satt


class Col_awareness(nn.Module):
    def __init__(self, col_features, enc_features):
        super(Col_awareness, self).__init__()
        self.enc_features = enc_features
        self.col_features = col_features
        self.weight = Parameter(torch.Tensor(enc_features, col_features))
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.xavier_uniform_(self.weight)

    def forward(self, input1, input2, input_mask=None):
        """
        :param input1: [batch, seq_length1, col_features]
        :param input2: [batch, enc_features]
        :param input_mask: mask of input1
        :return:
        """
        # input1: [batch, seq_length1, col_features]
        # weight: [enc_features, col_features]
        # out1: [batch, seq_length1, enc_features]
        out1 = F.linear(input1, self.weight)
        # out1: [batch, seq_length1, enc_features]
        # input2: [batch, enc_features] -- > input2.unsqueeze(-1): [batch, enc_features, 1]
        out2 = torch.matmul(out1, input2.unsqueeze(-1))
        if input_mask is not None and out2.size(0) == input_mask.size(0):
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, 1] --> out2.transpose(1, -1): [batch, 1, seq_length1]
        # input1: [batch, seq_length1, col_features]
        satt = torch.matmul(F.softmax(out2.transpose(1, -1), dim=2), input1)
        # satt: [batch, 1, col_features]
        return satt

