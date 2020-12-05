# -*- coding: utf-8 -*-
"""
Created on 2020-04-20
@author duytinvo
"""
import argparse
from mlmodels.utils.special_tokens import NULL
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.txtIO import TXT
from mlmodels.utils.trad_tokenizer import Tokenizer


def save_label(train_files, label_file, task=2, firstline=True):
    datasets = Tokenizer.load_file(train_files, firstline=firstline, task=task)
    # data = []
    label_set = set()
    for dataset in datasets:
        for nl, label in dataset:
            # data.append(d)
            label_set.update(set(label.split()))
    # label_set.update([NULL])
    TXT.write(label_set, label_file)


if __name__ == "__main__":
    from mlmodels.utils.special_tokens import SENSP, SENGE
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--train_file", type=str,
                        default="/media/data/vnreviews/Product/dataset/train.txt",
                        help="The input training data file")
    parser.add_argument("--label_file", type=str,
                        default="/media/data/vnreviews/Product/dataset/labels.txt",
                        help="The output label file")
    args = parser.parse_args()
    save_label([args.train_file], args.label_file, task=1, firstline=False)
