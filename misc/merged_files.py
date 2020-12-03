# -*- coding: utf-8 -*-
"""
Created on 23/03/2020
@author duytinvo
"""
import numpy as np
from mlmodels.utils.csvIO import CSV


def merge_files(fileone, filetwo, mergedfile):
    dataone = CSV.read(fileone, firstline=False, slices=None)
    datatwo = CSV.read(filetwo, firstline=False, slices=None)
    data = dataone + datatwo
    # data = list(set(data))
    np.random.shuffle(data)
    CSV.write(data, mergedfile)


if __name__ == '__main__':
    fileone = "../data/reviews/processed_csv/train_res4.csv"
    filetwo = "../data/yelp_ner/processed/tag_bioes_train.csv"
    mergedfile = "../data/yelp_ner/merged/merged_train.csv"
    merge_files(fileone, filetwo, mergedfile)

    fileone = "../data/reviews/processed_csv/dev_res4.csv"
    filetwo = "../data/yelp_ner/processed/tag_bioes_val.csv"
    mergedfile = "../data/yelp_ner/merged/merged_dev.csv"
    merge_files(fileone, filetwo, mergedfile)

    fileone = "../data/reviews/processed_csv/test_res4.csv"
    filetwo = "../data/yelp_ner/processed/tag_bioes_test.csv"
    mergedfile = "../data/yelp_ner/merged/merged_test.csv"
    merge_files(fileone, filetwo, mergedfile)