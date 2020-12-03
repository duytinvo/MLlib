# -*- coding: utf-8 -*-
"""
Created on 25/03/2020
@author duytinvo
"""

import argparse
import glob
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import CharBPETokenizer, ByteLevelBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from mlmodels.utils.special_tokens import unk_token, rep_token, pad_token, suffix_token, \
                                          BCLS, BPAD, BPRE, BSEP, BUNK, BMASK, SOT, EOT


class BPE:
    """
    An implementation of Byte-Pair Encoding (BPE) which supports
    - Character BPE
    - Byte BPE
    - WordPiece BPE
    - SentencePiece BPE
    """
    def __init__(self, args):
        self.args = args
        if self.args.type == "byte":
            self.tokenizer = ByteLevelBPETokenizer(add_prefix_space=True,      # required
                                                   lowercase=True,             # required
                                                   unicode_normalizer=None,     # required
                                                   vocab_file=None, merges_file=None, dropout=None,
                                                   continuing_subword_prefix=None, end_of_word_suffix=None)

        elif self.args.type == "char":
            self.tokenizer = CharBPETokenizer(unk_token=unk_token,            # required
                                              suffix=suffix_token,                # required
                                              lowercase=True,              # required
                                              unicode_normalizer=None,      # required
                                              vocab_file=None, merges_file=None, dropout=None)

        elif self.args.type == "bert":
            self.tokenizer = BertWordPieceTokenizer(clean_text=True,            # required
                                                    handle_chinese_chars=True,  # required
                                                    strip_accents=True,         # required
                                                    lowercase=True,             # required
                                                    vocab_file=None,
                                                    # add_special_tokens=True,
                                                    unk_token=BUNK, sep_token=BSEP,
                                                    cls_token=BCLS, wordpieces_prefix=BPRE)

        elif self.args.type == "sent":
            self.tokenizer = SentencePieceBPETokenizer(add_prefix_space=True,       # required
                                                       unk_token=unk_token, replacement=rep_token,
                                                       vocab_file=None, merges_file=None, dropout=None)

        else:
            raise Exception("Not implement yet")

        pass

    @staticmethod
    def load(vocab_file=None):
        if not os.path.exists(vocab_file):
            raise Exception("{} is not exist".format(vocab_file))
        path, filename = os.path.split(vocab_file)
        ttype = filename.split("_")[0]
        merges_file = os.path.join(path, filename.replace("vocab.json", "merges.txt"))
        if ttype == "byte":
            if not os.path.exists(merges_file):
                raise Exception("{} is not exist".format(merges_file))
            tokenizer = ByteLevelBPETokenizer(
                                                   add_prefix_space=True,      # required
                                                   lowercase=True,             # required
                                                   unicode_normalizer=None,     # required
                                                   vocab_file=vocab_file, merges_file=merges_file, dropout=None,
                                                   continuing_subword_prefix=None, end_of_word_suffix=None)

        elif ttype == "char":
            if not os.path.exists(merges_file):
                raise Exception("{} is not exist".format(merges_file))
            tokenizer = CharBPETokenizer(
                                              unk_token=unk_token,            # required
                                              suffix=suffix_token,                # required
                                              lowercase=True,              # required
                                              unicode_normalizer=None,      # required
                                              vocab_file=vocab_file, merges_file=merges_file, dropout=None)

        elif ttype == "bert":
            tokenizer = BertWordPieceTokenizer(
                                                    clean_text=True,            # required
                                                    handle_chinese_chars=True,  # required
                                                    strip_accents=True,         # required
                                                    lowercase=True,             # required
                                                    vocab_file=vocab_file,
                                                    # add_special_tokens=True,
                                                    unk_token=BUNK, sep_token=BSEP,
                                                    cls_token=BCLS, wordpieces_prefix=BPRE)

        elif ttype == "sent":
            if not os.path.exists(merges_file):
                raise Exception("{} is not exist".format(merges_file))
            tokenizer = SentencePieceBPETokenizer(
                                                       add_prefix_space=True,       # required
                                                       unk_token=unk_token, replacement=rep_token,
                                                       vocab_file=vocab_file, merges_file=merges_file, dropout=None)

        else:
            raise Exception("Not implement yet")

        return tokenizer

    def train(self):
        files, vocab_size, min_frequency = self.args.files, self.args.vocab_size, self.args.min_frequency
        limit_alphabet = self.args.limit_alphabet
        files = glob.glob(files)
        if not files:
            print(f"File does not exist: {args.files}")
            exit(1)

        if self.args.type == "bert":
            # special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            special_tokens = [BPAD, BUNK, BCLS, BSEP, BMASK]
        else:
            # special_tokens = ["<unk>"]
            special_tokens = [pad_token, unk_token]

        if self.args.type == "byte":
            self.tokenizer.train(files=files, vocab_size=vocab_size,
                                 min_frequency=min_frequency, special_tokens=special_tokens,
                                 show_progress=True)

        elif self.args.type == "char":
            self.tokenizer.train(files=files, vocab_size=vocab_size,
                                 min_frequency=min_frequency, special_tokens=special_tokens,
                                 limit_alphabet=limit_alphabet,
                                 initial_alphabet=[],
                                 suffix=suffix_token,
                                 show_progress=True)

        elif self.args.type == "bert":
            self.tokenizer.train(files=files, vocab_size=vocab_size,
                                 min_frequency=min_frequency, special_tokens=special_tokens,
                                 limit_alphabet=limit_alphabet,
                                 initial_alphabet=[],
                                 wordpieces_prefix=BPRE,
                                 show_progress=True)

        elif self.args.type == "sent":
            self.tokenizer.train(files=files, vocab_size=vocab_size,
                                 min_frequency=min_frequency, special_tokens=special_tokens,
                                 limit_alphabet=limit_alphabet,
                                 initial_alphabet=[],
                                 show_progress=True)

        else:
            raise Exception("Not implement yet")

        if not os.path.exists(self.args.out):
            os.mkdir(self.args.out)
        self.tokenizer.save(self.args.out, self.args.type + "_level-bpe")

    @staticmethod
    def tokens2ids(pretrained_tokenizer, sos=False, eos=False, add_special_tokens=False):
        """
        :param pretrained_tokenizer: pretrained tokenizer
        :return: a token2index function
        """

        def f(sent):
            if sos:
                sent = SOT + " " + sent
            if eos:
                sent = sent + " " + EOT
            tokenized_ids = pretrained_tokenizer.encode(sent, add_special_tokens=add_special_tokens).ids
            return tokenized_ids

        return f

    @staticmethod
    def collate_fn(padding_value=0, batch_first=True):
        def collate(examples):
            source = pad_sequence([torch.tensor(d[0]) for d in examples],
                                  batch_first=batch_first, padding_value=padding_value)
            target = pad_sequence([torch.tensor(d[1]) if d[1] is not None else torch.empty(0) for d in examples],
                                  batch_first=batch_first, padding_value=padding_value)
            return source, target

        return collate


def testBPE(vocab_file="/media/data/review_response/tokens/bert_level-bpe-vocab.txt",
            sent="This is an example, where we test BPE standed for Byte-Pair Encoding. Happy coding!!!"):
    # vocab_file = os.path.join(args.out, args.type + "_level-bpe-vocab.txt")
    # sent = "This is an example, where we test BPE standed for Byte-Pair Encoding. Happy coding!!!"
    tokenizer = BPE.load(vocab_file)
    encoded = tokenizer.encode(sent)
    print(encoded.ids)
    print(encoded.tokens)
    return tokenizer


if __name__ == '__main__':
    """
    python mlmodels/utils/BPEtonkenizer.py --type bert --files "input/path/*.txt" --out "output/path/" --vocab_size 30000 --min_frequency 2 --limit_alphabet 1000
    python -m mlmodels.utils.BPEtonkenizer.py --type bert
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", default="/media/data/review_response/rawtxt/*.txt", metavar="path", type=str,
                        help="The files to use as training; accept '**/*.txt' type of patterns if enclosed in quotes")
    parser.add_argument("--out", default="/media/data/review_response/tokens/", type=str,
                        help="Path to the output directory, where the files will be saved")
    parser.add_argument("--type", default="bert", type=str, choices=["char", "byte", "sent", "bert"],
                        help="The name of the output vocab files")
    parser.add_argument("--vocab_size", default=30000, type=int,
                        help="The size of the final vocabulary, including all tokens and alphabet.")
    parser.add_argument("--min_frequency", default=2, type=int,
                        help="The minimum frequency a pair should have in order to be merged.")
    parser.add_argument("--limit_alphabet", default=1000, type=int,
                        help="The maximum different characters to keep in the alphabet.")
    args = parser.parse_args()

    bpe = BPE(args)

    bpe.train()
    
    # tokenizer = testBPE()
