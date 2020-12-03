# -*- coding: utf-8 -*-
"""
Created on 2020-03-27
@author duytinvo
"""
import os
import torch
import logging
import pickle
import argparse
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from mlmodels.utils.BPEtonkenizer import BPE
from mlmodels.utils.trad_tokenizer import Tokenizer
from mlmodels.utils.special_tokens import NULL, UNK
from mlmodels.utils.csvIO import CSV

logger = logging.getLogger(__name__)


def tokens2ids(pretrained_tokenizer):
    """
    :param pretrained_tokenizer: pretrained tokenizer
    :return: a token2index function
    """

    def f(sent):
        tokenized_ids = pretrained_tokenizer.encode(sent).ids
        return tokenized_ids

    return f


def collate_fn(examples, padding_value=0):
    source = pad_sequence([torch.tensor(d[0]) for d in examples], batch_first=True, padding_value=padding_value)
    target = pad_sequence([torch.tensor(d[1]) for d in examples], batch_first=True, padding_value=padding_value)
    return source, target


class IterDataset(IterableDataset):
    r"""
    An iterable dataset to save the data. This dataset supports multi-processing
    to load the data.

    Arguments:
        iterator: the iterator to read data.
        num_lines: the number of lines read by the individual iterator.
    Examples:
        >>> iterdata = CSV.get_csv_iterator(train_data_file, firstline=True, slices=[0, 1])
        >>> num_lines = CSV._len(args.train_data_file, firstline=True)
        >>> train_iterdataset = IterDataset(iterdata, source2idx=source2idx, target2idx=source2idx, num_lines=num_lines)
        >>> train_dataloader = DataLoader(train_iterdataset, pin_memory=True, batch_size=16, collate_fn=collate_fn)
    """

    def __init__(self, iterator, num_lines, source2idx=None, target2idx=None, bpe=False,
                 special_tokens_func=None, label_pad_id=-100):
        super(IterDataset, self).__init__()
        self._num_lines = num_lines
        self._iterator = iterator
        self._setup = False
        self.source2idx = source2idx
        self.target2idx = target2idx
        self.bpe = bpe
        self.special_tokens_func = special_tokens_func
        self.label_pad_id = label_pad_id

    def _setup_iterator(self):
        r"""
        _setup_iterator() function assign the starting line and the number
        of lines to read for the individual worker. Then, send them to the iterator
        to load the data.

        If worker info is not avaialble, it will read all the lines across epochs.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            chunk = int(self._num_lines / worker_info.num_workers)
            start = chunk * worker_info.id
            read = chunk
            if worker_info.id == worker_info.num_workers - 1:
                # The last worker needs to pick up some extra lines
                # if the number of lines aren't exactly divisible
                # by the number of workers.
                # Each epoch we loose an 'extra' number of lines.
                extra = self._num_lines % worker_info.num_workers
                read += extra
        else:
            start = 0
            read = self._num_lines
        self._iterator = self._iterator(start, read)

    def __iter__(self):
        if self._setup is False:
            self._setup_iterator()
            self._setup = True
        for text in self._iterator:
            # always assume that text is in a pair of source and target, which uses None if unused
            if len(text) == 1:
                source = text[0]
                target = None
                score = None
            elif len(text) == 3:
                source, score, target = text[0], text[1], text[-1]
            else:
                source, score, target = text[0], None, text[-1]
            # source, target = text[0], text[-1]
            # print("SOURCE: ", source)
            # print("TARGET: ", target)
            if not self.bpe:
                if self.source2idx is not None:
                    source = self.source2idx(source)
                    if score != None:
                        score = self.source2idx(score, eos=False, sep=True)
                if self.target2idx is not None and target is not None:
                    # print("ERROR: ", target)
                    target = self.target2idx(target)
            else:
                # Todo: tokenizer score for bpe
                if self.source2idx is not None and self.target2idx is not None and target is not None:
                    source_tokens = source.split()
                    target_tokens = target.split()
                    assert len(source_tokens) == len(target_tokens), \
                        "Length of source and target must equal for a sequence labeler task"
                    source = []
                    target = []
                    for i in range(len(source_tokens)):
                        s_tok = self.source2idx(source_tokens[i])
                        if len(s_tok) == 0:
                            s_tok = self.source2idx(UNK)
                        t_tok = self.target2idx(target_tokens[i])
                        assert len(t_tok) == 1, "length of one target token must be 1"
                        # if target_tokens[i].upper() == 'O':
                        #     t_tok += t_tok * (len(s_tok) - 1)
                        # else:
                        #     t_tok += self.target2idx(NULL) * (len(s_tok) - 1)
                        t_tok += [self.label_pad_id] * (len(s_tok) - 1)
                        assert len(s_tok) == len(t_tok), "length of a token and a label should be aligned"
                        source.extend(s_tok)
                        target.extend(t_tok)
                    assert len(source) == len(target), \
                        "Length of source and target must equal for a sequence labeler task"
            if self.special_tokens_func is not None:
                source, target = self.special_tokens_func(source, target)

            yield source, target, score


class MapDataset(Dataset):
    """
    Load full dataset into RAM memory, which maybe inefficient when the data size is too big
    Examples:
        >>> data = CSV.read(train_data_file, firstline=True, slices=[0, 1])
        >>> train_dataset = MapDataset(data, source2idx=source2idx, target2idx=source2idx)
        >>> train_sampler = RandomSampler(train_dataset)
        >>> train_dataloader = DataLoader(train_dataset, sampler=train_sampler, pin_memory=True, batch_size=16, collate_fn=collate_fn)
    """

    def __init__(self, data, source2idx=None, target2idx=None, bpe=False,
                 special_tokens_func=None, label_pad_id=-100):
        super(MapDataset, self).__init__()
        self.examples = []
        self.source2idx = source2idx
        self.target2idx = target2idx
        self.bpe = bpe
        self.special_tokens_func = special_tokens_func
        self.label_pad_id = label_pad_id

        logger.info("Creating features from dataset file", )
        for text in data:
            # always assume that text is in a pair of source and target, which uses None if unused
            if len(text) == 1:
                source = text[0]
                target = None
                score = None
            elif len(text) == 3:
                source, score, target = text[0], text[1], text[-1]
            else:
                source, score, target = text[0], None, text[-1]
            # if self.source2idx is not None:
            #     source = self.source2idx(source)
            # if self.target2idx is not None and target is not None:
            #     target = self.target2idx(target)
            if not self.bpe:
                if self.source2idx is not None:
                    source = self.source2idx(source)
                    if score != None:
                        score = self.source2idx(score, eos=False, sep=True)
                if self.target2idx is not None and target is not None:
                    target = self.target2idx(target)
            else:
                if self.source2idx is not None and self.target2idx is not None and target is not None:
                    source_tokens = source.split()
                    target_tokens = target.split()
                    assert len(source_tokens) == len(target_tokens), \
                        "Length of source and target must equal for a sequence labeler task"
                    source = []
                    target = []
                    for i in range(len(source_tokens)):
                        s_tok = self.source2idx(source_tokens[i])
                        if len(s_tok) == 0:
                            s_tok = self.source2idx(UNK)
                        t_tok = self.target2idx(target_tokens[i])
                        assert len(t_tok) == 1, "length of one target token must be 1"
                        # if target_tokens[i].upper() == 'O':
                        #     t_tok += t_tok * (len(s_tok) - 1)
                        # else:
                        #     t_tok += self.target2idx(NULL) * (len(s_tok) - 1)
                        t_tok += [self.label_pad_id] * (len(s_tok) - 1)
                        assert len(s_tok) == len(t_tok), "length of a token and a label should be aligned"
                        source.extend(s_tok)
                        target.extend(t_tok)
                    assert len(source) == len(target), \
                        "Length of source and target must equal for a sequence labeler task"
            if self.special_tokens_func is not None:
                source, target = self.special_tokens_func(source, target)
            self.examples.append([source, target, score])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", type=str,
                        default="/media/data/review_response/Dev.csv",
                        help="The input training data file (a text file).")
    parser.add_argument("--vocab_file", type=str,
                        default="/media/data/review_response/tokens/bert_level-bpe-vocab.txt",
                        help="Saved vocab file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    tokenizer = BPE.load(args.vocab_file)
    source2idx = tokens2ids(tokenizer)

    # data = CSV.read(args.train_data_file, firstline=True, slices=[0, 1])
    # train_dataset = MapDataset(data, source2idx=source2idx, target2idx=source2idx)
    #
    # # train_sampler = RandomSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, pin_memory=True,
    #                               batch_size=16, collate_fn=collate_fn)
    #
    # for i, batch in enumerate(train_dataloader):
    #     inputs, outputs = batch[0], batch[1]
    #     break

    iterdata = CSV.get_iterator(args.train_data_file, firstline=True)
    num_lines = CSV._len(args.train_data_file, firstline=True)
    train_iterdataset = IterDataset(iterdata, source2idx=source2idx, target2idx=source2idx, num_lines=num_lines)
    train_dataloader = DataLoader(train_iterdataset, pin_memory=True, batch_size=16, collate_fn=collate_fn)

    for i, batch in enumerate(train_dataloader):
        inputs, outputs = batch[0], batch[1]
        break
