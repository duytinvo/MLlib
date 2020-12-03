# -*- coding: utf-8 -*-
"""
Created on 2020-05-14
@author duytinvo
"""
from collections import Counter
from sql_py_antlr.Utilities import Utilities
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.txtIO import TXT
from mlmodels.utils.trad_tokenizer import Tokenizer
from mlmodels.utils.BPEtonkenizer import BPE


def read_data(filename, firstline=True):
    # load datasets to map into indexes
    if filename.split(".")[-1] == "csv":
        data = CSV.read(filename, firstline=firstline, slices=[0, 1])
    elif filename.split(".")[-1] == "txt":
        data = TXT.read(filename, firstline=firstline)
    elif filename.split(".")[-1] == "json":
        data = JSON.load(filename)
    else:
        raise Exception("Not implement yet")
    return data


def extract_label(inp_file, out_file, tokenizer=Tokenizer.process_target):
    data = read_data(inp_file)
    twcnt, twl = Counter(), 0
    for line in data:
        nl, target = line
        nl = nl.lower()
        target = target.lower()
        # Tokenize target into tokens
        target = tokenizer(target)
        # target = Utilities.fast_tokenize(target)
        twcnt, twl = Tokenizer.update_sent(target, twcnt, twl)
    labels = list(twcnt.keys())
    TXT.write(labels, out_file)


if __name__ == "__main__":
    """
    python -m misc.write_label_file --inp_file /media/data/disambiguator/raw/bluelink_transformed.csv --out_file /media/data/disambiguator/corpus/bluelink_labels.txt --tokenizer split
    """
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--inp_file', help='input (raw) file',
                           default="/media/data/disambiguator/raw/bluelink_transformed.csv",
                           type=str)
    argparser.add_argument('--out_file', help='output label file',
                           default="/media/data/disambiguator/corpus/bluelink_labels.txt",
                           type=str)
    argparser.add_argument('--vocab_file', help='pre-trained bpe vocab file',
                           default="/media/data/review_response/tokens/bert_level-bpe-vocab.txt",
                           type=str)
    argparser.add_argument('--tokenizer', help='tokenizer method', choices=["split", "antlr", "bpe"], default="split",
                           type=str)
    args = argparser.parse_args()

    if args.tokenizer == "split":
        tokenizer = Tokenizer.process_target
    elif args.tokenizer == "antlr":
        tokenizer = Utilities.fast_tokenize
    elif args.tokenizer =="bpe":
        pretrained_tokenizer = BPE.load(args.vocab_file)
        tokenizer = lambda x: pretrained_tokenizer.encode(x).tokens
    else:
        raise Exception("Not implement yet")

    extract_label(inp_file=args.inp_file, out_file=args.out_file, tokenizer=tokenizer)

    # filename = "/media/data/np6/dataset/generated_corpus_stored_train_human.csv"
    # label_file = "/media/data/np6/dataset/labels2.txt"
    # vocab_file = "/media/data/review_response/tokens/bert_level-bpe-vocab.txt"
    # data = read_data(filename)
    # twcnt, twl = Counter(), 0
    # for line in data:
    #     nl, target = line
    #     # Tokenize target into tokens
    #     target = Tokenizer.process_target(target)
    #     target = Utilities.fast_tokenize(target)
    #     twcnt, twl = Tokenizer.update_sent(target, twcnt, twl)
    # labels = list(twcnt.keys())
    # TXT.write(labels, label_file)
