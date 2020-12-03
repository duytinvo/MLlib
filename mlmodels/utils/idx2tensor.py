# -*- coding: utf-8 -*-
"""
Created on 2020-03-10
@author: duytinvo
"""
import torch
import random
import numpy as np


class seqPAD:
    @staticmethod
    def flatten(labels):
        """
        Binary flatten labels
        """
        flabels = []
        for i, sublist in enumerate(labels):
            nitems = []
            for item in sublist:
                if isinstance(item, int):
                    nitems.append(item)
                else:
                    nitems.extend(item)
            flabels.append(nitems)
        return flabels

    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with

        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    @staticmethod
    def pad_sequences(sequences, pad_tok, nlevels=1, wthres=-1, cthres=-1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with
            nlevels: "depth" of padding, for the case where we have word ids

        Returns:
            a list of list where each sublist has same length

        """
        if nlevels == 1:
            max_length = max(map(lambda x: len(x), sequences))
            max_length = wthres if wthres > 0 else max_length
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)

        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = cthres if cthres > 0 else max_length_word
            word_padded, word_length = [], []
            for seq in sequences:
                # pad the word-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                word_padded += [sp]
                word_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x: len(x), sequences))
            max_length_sentence = wthres if wthres > 0 else max_length_sentence
            sequence_padded, sequence_length = seqPAD._pad_sequences(word_padded, [pad_tok] * max_length_word,
                                                       max_length_sentence)
            # set sequence length to 1 by inserting padding
            word_length, _ = seqPAD._pad_sequences(word_length, 1, max_length_sentence)
        elif nlevels == 3:
            max_length_token = max([max([max(map(lambda x: len(x), wd)) for wd in seq]) for seq in sequences])
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            ts_pad = []
            for tb in sequences:
                ts_pad_ids, _ = seqPAD.pad_sequences(tb, pad_tok=pad_tok, wthres=max_length_word,
                                                           cthres=max_length_token, nlevels=2)
                ts_pad.append(ts_pad_ids)
            sequence_padded, sequence_length = seqPAD.pad_sequences(ts_pad,
                                                                    pad_tok=[[pad_tok]*max_length_token]*max_length_word)
        return sequence_padded, sequence_length


class Data2tensor:
    @staticmethod
    def idx2tensor(indexes, dtype=torch.long, device=torch.device("cpu")):
        vec = torch.tensor(indexes, dtype=dtype, device=device)
        return vec

    @staticmethod
    def sort_tensors(word_ids, seq_lens, char_ids=None, wd_lens=None, dtype=torch.long,
                     device=torch.device("cpu")):
        word_tensor = Data2tensor.idx2tensor(word_ids, dtype, device)
        seq_len_tensor = Data2tensor.idx2tensor(seq_lens, dtype, device)
        seq_len_tensor, seqord_tensor = seq_len_tensor.sort(0, descending=True)
        word_tensor = word_tensor[seqord_tensor]
        _, seqord_recover_tensor = seqord_tensor.sort(0, descending=False)

        if char_ids is not None:
            char_tensor = Data2tensor.idx2tensor(char_ids, dtype, device)
            wd_len_tensor = Data2tensor.idx2tensor(wd_lens, dtype, device)
            batch_size = len(word_ids)
            max_seq_len = seq_len_tensor.max()
            char_tensor = char_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), -1)
            wd_len_tensor = wd_len_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), )
            wd_len_tensor, wdord_tensor = wd_len_tensor.sort(0, descending=True)
            char_tensor = char_tensor[wdord_tensor]
            _, wdord_recover_tensor = wdord_tensor.sort(0, descending=False)
        else:
            char_tensor = None
            wd_len_tensor = None
            wdord_recover_tensor = None
        return word_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor, \
               char_tensor, wd_len_tensor, wdord_recover_tensor

    @staticmethod
    def sort_labelled_tensors(word_ids, seq_lens, char_ids=None, wd_lens=None, label=False, dtype=torch.long,
                              device=torch.device("cpu")):
        word_tensor = Data2tensor.idx2tensor(word_ids, dtype, device)
        seq_len_tensor = Data2tensor.idx2tensor(seq_lens, dtype, device)
        seq_len_tensor, seqord_tensor = seq_len_tensor.sort(0, descending=True)
        word_tensor = word_tensor[seqord_tensor]
        _, seqord_recover_tensor = seqord_tensor.sort(0, descending=False)
        if label:
            input_tensor = word_tensor[:, : -1]
            seq_len_tensor = (input_tensor > 0).sum(dim=1)
            output_tensor = word_tensor[:, 1:]

        if char_ids is not None:
            char_tensor = Data2tensor.idx2tensor(char_ids, dtype, device)
            wd_len_tensor = Data2tensor.idx2tensor(wd_lens, dtype, device)
            if label:
                char_tensor = char_tensor[:, : -1, :]
                wd_len_tensor = wd_len_tensor[:, : -1]
            batch_size = len(word_ids)
            max_seq_len = seq_len_tensor.max()
            char_tensor = char_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), -1)
            wd_len_tensor = wd_len_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), )
            wd_len_tensor, wdord_tensor = wd_len_tensor.sort(0, descending=True)
            char_tensor = char_tensor[wdord_tensor]
            _, wdord_recover_tensor = wdord_tensor.sort(0, descending=False)
        else:
            char_tensor = None
            wd_len_tensor = None
            wdord_recover_tensor = None
        if label:
            return output_tensor, input_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor, \
                   char_tensor, wd_len_tensor, wdord_recover_tensor
        else:
            return word_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor, \
                   char_tensor, wd_len_tensor, wdord_recover_tensor

    @staticmethod
    def set_randseed(seed_num=12345, n_gpu=-1):
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed_num)

