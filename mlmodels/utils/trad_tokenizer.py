# -*- coding: utf-8 -*-
"""
Created on 2020-03-30
@author duytinvo
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from random import choices
try:
    from sql_py_antlr.Utilities import Utilities
except:
    pass

from mlmodels.utils.txtIO import TXT
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.special_tokens import PAD, SOT, EOT, UNK, COL, TAB, NULL, NL2LC
from mlmodels.utils.auxiliary import SaveloadHP
sys_tokens = [PAD, SOT, EOT, UNK, NULL, NL2LC]


# ----------------------------------------------------------------------------------------------------------------------
# ======================================== VOCAB-BUILDER FUNCTIONS =====================================================
# ----------------------------------------------------------------------------------------------------------------------
class Tokenizer(object):
    def __init__(self, s_paras, t_paras):
        """
        s_paras = [swl_th=None, swcutoff=1]
        t_paras = [twl_th=None, twcutoff=1]
        """
        # NL query
        self.swl, self.swcutoff = s_paras
        self.sw2i, self.i2sw = {}, {}

        # sql or semQL
        self.twl, self.twcutoff = t_paras
        self.tw2i, self.i2tw = {}, {}

    @staticmethod
    def load_file(files, firstline=True, task=2):
        datasets = []
        for fname in files:
            # Read input files
            if fname.split(".")[-1] == "csv":
                datasets.append(CSV(fname, limit=-1, firstline=firstline, task=task))
            elif fname.split(".")[-1] == "json":
                datasets.append(JSON(fname, limit=-1, task=task))
            elif fname.split(".")[-1] == "txt":
                datasets.append(TXT(fname, limit=-1, firstline=firstline, task=task))
            else:
                raise Exception("Not implement yet")
        return datasets

    @staticmethod
    def load(tokenize_file):
        return SaveloadHP.load(tokenize_file)

    @staticmethod
    def save(tokenizer, tokenize_file):
        SaveloadHP.save(tokenizer, tokenize_file)

    @staticmethod
    def prepare_map(filename, firstline=True, task=2):
        # load datasets to map into indexes
        if filename.split(".")[-1] == "csv":
            data_map = CSV.get_map(filename, firstline=firstline, task=task)
            num_lines = len(data_map)
        elif filename.split(".")[-1] == "json":
            data_map = JSON.get_map(filename, task=task)
            num_lines = len(data_map)
        else:
            raise Exception("Not implement yet")
        return data_map, num_lines

    @staticmethod
    def prepare_iter(filename, firstline=True, task=2):
        # load datasets to map into indexes
        if filename.split(".")[-1] == "csv":
            data_iter = CSV.get_iterator(filename, firstline=firstline, task=task)
            num_lines = CSV._len(filename, firstline=firstline)
        elif filename.split(".")[-1] == "json":
            data_iter = JSON.get_iterator(filename, task=task)
            num_lines = JSON._len(filename)
        elif filename.split(".")[-1] == "txt":
            data_iter = TXT.get_iterator(filename, firstline=firstline, task=task)
            num_lines = TXT._len(filename, firstline=firstline)
        else:
            raise Exception("Not implement yet")
        return data_iter, num_lines

    def get_vocab_size(self):
        return len(self.tw2i)

    def build(self, datasets):
        """
        Read a list of datasets, return vocabulary
        :param datasets: list of file names
        """
        swcnt, swl = Counter(), 0
        twcnt, twl = Counter(), 0
        count = 0
        for dataset in datasets:
            for line in dataset:
                count += 1
                nl, target = line
                # Tokenize nl into tokens
                nl = Tokenizer.process_nl(nl)
                # Tokenize target into tokens
                target = Tokenizer.process_target(target)
                swcnt, swl = Tokenizer.update_sent(nl, swcnt, swl)
                twcnt, twl = Tokenizer.update_sent(target, twcnt, twl)

        swvocab = Tokenizer.update_vocab(swcnt, self.swcutoff, sys_tokens)
        twvocab = Tokenizer.update_vocab(twcnt, self.twcutoff, sys_tokens)

        self.sw2i = swvocab
        self.i2sw = Tokenizer.reversed_dict(swvocab)
        self.swl = swl if self.swl < 0 else min(swl, self.swl)

        self.tw2i = twvocab
        self.i2tw = Tokenizer.reversed_dict(twvocab)
        self.twl = twl if self.twl < 0 else min(twl, self.twl)

        print("\t- Extracting vocabulary: %d total samples" % count)

        print("\t\t- Natural Language Side: ")
        print("\t\t\t- %d total words" % (sum(swcnt.values())))
        print("\t\t\t- %d unique words" % (len(swcnt)))
        print("\t\t\t- %d unique words appearing at least %d times" % (len(swvocab) - len(sys_tokens), self.swcutoff))
        print("\t\t- Label Side: ")
        print("\t\t\t- %d total words" % (sum(twcnt.values())))
        print("\t\t\t- %d unique words" % (len(twcnt)))
        print("\t\t\t- %d unique words appearing at least %d times" % (len(twvocab) - len(sys_tokens), self.twcutoff))

    @staticmethod
    def update_sent(sent, wcnt, wl):
        newsent = []
        for item in sent:
            if isinstance(item, list) or isinstance(item, tuple):
                newsent.extend([tk for tk in item])
            else:
                newsent.append(str(item))
        # newsent = " ".join(newsent).split()
        wcnt.update(newsent)
        wl = max(wl, len(newsent))
        return wcnt, wl

    @staticmethod
    def update_vocab(cnt, cutoff, pads):
        lst = pads + [x for x, y in cnt.most_common() if y >= cutoff]
        vocabs = dict([(y, x) for x, y in enumerate(lst)])
        return vocabs

    @staticmethod
    def reversed_dict(cur_dict):
        inv_dict = {v: k for k, v in cur_dict.items()}
        return inv_dict

    @staticmethod
    def list2dict(data_list):
        data_dict = dict([(y, x) for x, y in enumerate(data_list)])
        return data_dict

    @staticmethod
    def process_target(target):
        target_toks = target.split()
        return target_toks

    @staticmethod
    def tokenize_sql(target):
        target_toks = Utilities.fast_tokenize(target)
        return target_toks

    @staticmethod
    def process_nl(nl):
        nl_toks = nl.split()
        return nl_toks

    @staticmethod
    def collate_fn(padding_value=0, batch_first=True):
        def collate(examples):
            source = pad_sequence([torch.tensor(d[0]) for d in examples],
                                  batch_first=batch_first, padding_value=padding_value)
            target = pad_sequence([torch.tensor(d[1]) if d[1] is not None else torch.empty(0) for d in examples],
                                  batch_first=batch_first, padding_value=padding_value)
            return source, target
        return collate

    @staticmethod
    def lst2idx(tokenizer=None, vocab_words=None, unk_words=True, sos=False, eos=False,
                vocab_chars=None, unk_chars=True, sow=False, eow=False,
                reverse=False):
        """
        Return a function to convert tag2idx or word/word2idx (a list of words comprising characters)
        """

        def f(sequence):
            sent = tokenizer(sequence)
            if vocab_words is not None:
                # SOw,EOw words for  SOW
                word_ids = []
                for word in sent:
                    # ignore words out of vocabulary
                    if word in vocab_words:
                        word_ids += [vocab_words[word]]
                    else:
                        if unk_words:
                            word_ids += [vocab_words[UNK]]
                        else:
                            print("UNK token: ", word)
                            raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct and"
                                            " having {}".format(word))
                if reverse:
                    word_ids = word_ids[::-1]
                if sos:
                    # Add start-of-sentence
                    word_ids = [vocab_words[SOT]] + word_ids

                if eos:
                    # add end-of-sentence
                    word_ids = word_ids + [vocab_words[EOT]]

                if len(word_ids) == 0:
                    word_ids = [[vocab_words[PAD], ]]

            if vocab_chars is not None:
                char_ids = []
                padding = []
                if sow:
                    # add start-of-word
                    padding = [vocab_chars[SOT]] + padding
                if eow:
                    # add end-of-word
                    padding = padding + [vocab_chars[EOT]]

                for word in sent:
                    if word not in sys_tokens:
                        char_id = []
                        for char in word:
                            # ignore chars out of vocabulary
                            if char in vocab_chars:
                                char_id += [vocab_chars[char]]
                            else:
                                if unk_chars:
                                    char_id += [vocab_chars[UNK]]
                                else:
                                    raise Exception(
                                        "Unknow key is not allowed. Check that your vocab (tags?) is correct")
                        if sow:
                            # add start-of-word
                            char_id = [vocab_chars[SOT]] + char_id
                        if eow:
                            # add end-of-word
                            char_id = char_id + [vocab_chars[EOT]]
                        char_ids += [char_id]
                    else:
                        char_ids += [[vocab_chars[word], ]]

                if reverse:
                    char_ids = char_ids[::-1]
                if sos:
                    # add padding start-of-sentence
                    char_ids = [padding] + char_ids
                if eos:
                    # add padding end-of-sentence
                    char_ids = char_ids + [padding]
                if len(char_ids) == 0:
                    char_ids = [padding]

            if vocab_words is not None:
                if vocab_chars is not None:
                    return list(zip(char_ids, word_ids))
                else:
                    return word_ids
            else:
                return char_ids

        return f

    @staticmethod
    def idx2text(pad_ids, i2t, level=2, i2colname=None, i2tabname=None, dbids=None):

        def i2token(token_id, i):
            token = UNK
            if token_id < len(i2t):
                token = i2t.get(token_id, UNK)
            elif token_id >= len(i2t):
                if dbids is not None:
                    if token_id < len(i2t) + len(i2colname[dbids[i]]) - 4:
                        token = i2colname[dbids[i]].get(token_id - len(i2t) + 4, COL)
                    else:
                        token = i2tabname[dbids[i]].get(token_id - len(i2t) - len(i2colname[dbids[i]]) + 8, TAB)
            return token

        if level == 3:
            docs = []
            for i, sent_ids in enumerate(pad_ids):
                sents = []
                for j, token_ids in enumerate(sent_ids):
                    tokens = []
                    for k, token_id in enumerate(token_ids):
                        tokens.append(i2token(token_id, i))
                    sents.append(tokens)
                docs.append(sents)
            return docs
            # return [[[i2t[char] for char in chars] for chars in wds] for wds in pad_ids]
        elif level == 2:
            sents = []
            for i, token_ids in enumerate(pad_ids):
                tokens = []
                for j, token_id in enumerate(token_ids):
                    tokens.append(i2token(token_id, i))
                sents.append(tokens)
            return sents
            # return [[i2t[wd] for wd in wds] for wds in pad_ids]
        else:
            tokens = []
            for i, token_id in enumerate(pad_ids):
                tokens.append(i2token(token_id, i))
            return tokens
            # return [i2t[token] for token in pad_ids]

    @staticmethod
    def decode_batch(pad_ids, i2t, level=2):
        return Tokenizer.idx2text(pad_ids=pad_ids, i2t=i2t, level=level)


if __name__ == '__main__':
    import torch
    from mlmodels.utils.idx2tensor import Data2tensor, seqPAD
    from mlmodels.utils.dataset import IterDataset, collate_fn, tokens2ids
    from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler, TensorDataset
    from mlmodels.utils.BPEtonkenizer import BPE
    from mlmodels.utils.special_tokens import BPAD, PAD, NULL
    from mlmodels.utils.txtIO import TXT

    Data2tensor.set_randseed(12345)
    device = torch.device("cpu")
    dtype = torch.long
    use_cuda = False
    filename = "../../data/reviews/processed_csv/train_res4.csv"
    label_file = "../../data/reviews/processed_csv/labels.txt"
    labels_list = TXT.read(label_file, firstline=False)
    lb2id_dict = Tokenizer.list2dict(sys_tokens + labels_list)
    id2lb_dict = Tokenizer.reversed_dict(lb2id_dict)
    lb2ids = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target, vocab_words=lb2id_dict,
                               unk_words=False, sos=False, eos=False)
    tokenize_type = "bpe"
    if tokenize_type != "bpe":
        # Load datasets to build vocabulary
        data = Tokenizer.load_file([filename], task=2)
        s_paras = [-1,  1]
        t_paras = [-1, 1]
        tokenizer = Tokenizer(s_paras, t_paras)
        tokenizer.build(data)
        nl2ids = Tokenizer.lst2idx(tokenizer=Tokenizer.process_nl, vocab_words=tokenizer.sw2i,
                                   unk_words=True, sos=False, eos=False)
        tokenizer.tw2i = lb2id_dict
        tokenizer.i2tw = id2lb_dict
        tg2ids = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target, vocab_words=tokenizer.tw2i,
                                   unk_words=False, sos=False, eos=False)
        pad_id = tokenizer.sw2i.get(PAD, 0)
    else:
        vocab_file = "/media/data/review_response/tokens/bert_level-bpe-vocab.txt"
        tokenizer = BPE.load(vocab_file)
        tokenizer.add_tokens(sys_tokens)
        nl2ids = BPE.tokens2ids(tokenizer, sos=False, eos=False, add_special_tokens=False)
        tg2ids = BPE.tokens2ids(tokenizer, sos=False, eos=False, add_special_tokens=False)

        pad_id = tokenizer.token_to_id(BPAD) if tokenizer.token_to_id(BPAD) else 0

    collate_fn = BPE.collate_fn(pad_id, True)

    # load datasets to map into indexes
    if filename.split(".")[-1] == "csv":
        train_data = CSV.get_iterator(filename, firstline=True, task=2)
        num_lines = CSV._len(filename)
    elif filename.split(".")[-1] == "json":
        train_data = JSON.get_iterator(filename, task=2)
        num_lines = JSON._len(filename)
    else:
        raise Exception("Not implement yet")

    train_iterdataset = IterDataset(train_data, source2idx=nl2ids, target2idx=lb2ids, num_lines=num_lines, bpe=True)
    train_dataloader = DataLoader(train_iterdataset, pin_memory=True, batch_size=8, collate_fn=collate_fn)

    for i, batch in enumerate(train_dataloader):
        # inputs, outputs = batch[0], batch[1]
        nl_tensor, lb_tensor = batch
        # nl_len_tensor = (nl_tensor > 0).sum(dim=1)
        break

