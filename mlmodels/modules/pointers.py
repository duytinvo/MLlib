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


class Pointer_net(nn.Module):
    def __init__(self, in_features, out_features):
        super(Pointer_net, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.lossF = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.xavier_uniform_(self.weight)

    def forward(self, input1, input2, input_mask=None):
        pointer_score = self.scoring(input1, input2, input_mask)
        return pointer_score

    def scoring(self, input1, input2, input_mask=None):
        """
        :param input1: [batch, seq_length1, in_features]
        :param input2: [batch, seq_length2, out_features]
        :param input_mask: mask of input1
        """
        out1 = F.linear(input1, self.weight)
        # out1: [batch, seq_length1, out_features]
        # input2: [batch, seq_length2, out_features]
        out2 = torch.matmul(out1, input2.transpose(1, -1))
        # TODO: use mask tensor to filter out padding in out2[:,seq_length1,:]
        if input_mask is not None and out2.size(0) == input_mask.size(0):
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, seq_length2]
        # pointers: [batch, seq_length1, seq_length2]
        # pointers = F.softmax(out2, dim=1)
        # out2: [batch, seq_length2, seq_length1]
        return out2.transpose(1, -1)

    def inference(self, label_score, k=1):
        # label_score: [batch, seq_length2, seq_length1]
        # pointers: [batch, seq_length2, seq_length1]
        pointers = F.softmax(label_score, dim=-1)
        label_prob, label_pred = pointers.data.topk(k, dim=-1)
        return label_prob, label_pred

    def logsm_inference(self, label_score, k=1):
        # label_score: [batch, seq_length2, seq_length1]
        label_prob = F.log_softmax(label_score, dim=-1)
        label_prob, label_pred = label_prob.data.topk(k, dim=-1)
        return label_prob, label_pred

    def NLL_loss(self, label_score, label_tensor):
        # label_score = [B, C]; label_tensor = [B, ]
        batch_loss = self.lossF(label_score, label_tensor)
        return batch_loss


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`

    Args:
       hidden_size (int): size of input representation
       num_labels (int): size of output vocabulary
    """

    def __init__(self, hidden_size, num_labels, p_factor=1):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
        self.p_factor = p_factor
        if p_factor <= 2:
            self.linear_copy = nn.Linear(hidden_size, 1)
        else:
            self.linear_copy = nn.Linear(hidden_size, p_factor)

    def forward(self, hidden, attn, src_map=None):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        # hidden = (tgt_len * batch, hidden)
        # attn = (tgt_len * batch, src_len)
        # src_map = (src_len * batch, cvocab)
        # Original probabilities.
        # logits = (tgt_len * batch, tvocab)
        logits = self.linear(hidden)
        # logits[:, self.pad_idx] = -float('inf')
        # prob = (tgt_len * batch, tvocab)
        prob = torch.softmax(logits, 1)
        if self.pp_factor <= 2:
            # Probability of copying p(z=1) batch.
            # p_copy = (tgt_len * batch, 1)
            p_copy = torch.sigmoid(self.linear_copy(hidden))
            # Probability of not copying: p_{word}(w) * (1 - p(z))
            # out_prob = (tgt_len * batch, tvocab)
            out_prob = torch.mul(prob, 1 - p_copy)
            # mul_attn = (tgt_len * batch, src_len)
            copy_prob = torch.mul(attn, p_copy)
            # copy_prob = (batch, tgt_len, src_len) x (batch, src_len, cvocab) --> (batch, tgt_len, cvocab)
            # copy_prob --> (tgt_len, batch, cvocab)
            if src_map is not None:
                batch_by_tlen_, slen = attn.size()
                slen_, batch, cvocab = src_map.size()
                copy_prob = torch.bmm(copy_prob.view(-1, batch, slen).transpose(0, 1),
                                      src_map.transpose(0, 1)).transpose(0, 1)
                # copy_prob --> (tgt_len * batch, cvocab)
                copy_prob = copy_prob.contiguous().view(-1, cvocab)
            # --> (tgt_len * batch, tvocab + cvocab)
            return torch.cat([out_prob, copy_prob], 1)
        else:
            # Probability of copying p(z=1) batch.
            # p_copy = (tgt_len * batch, 1)
            p_dis = torch.softmax(self.linear_copy(hidden), 1)
            # Probability of not copying: p_{word}(w) * (1 - p(z))
            # out_prob = (tgt_len * batch, tvocab)
            out_prob = torch.mul(prob, 1 - p_copy)
            # mul_attn = (tgt_len * batch, src_len)
            copy_prob = torch.mul(attn, p_copy)
            # copy_prob = (batch, tgt_len, src_len) x (batch, src_len, cvocab) --> (batch, tgt_len, cvocab)
            # copy_prob --> (tgt_len, batch, cvocab)
            if src_map is not None:
                batch_by_tlen_, slen = attn.size()
                slen_, batch, cvocab = src_map.size()
                copy_prob = torch.bmm(copy_prob.view(-1, batch, slen).transpose(0, 1),
                                      src_map.transpose(0, 1)).transpose(0, 1)
                # copy_prob --> (tgt_len * batch, cvocab)
                copy_prob = copy_prob.contiguous().view(-1, cvocab)
            # --> (tgt_len * batch, tvocab + cvocab)
            return torch.cat([out_prob, copy_prob], 1)

    def NLL(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss