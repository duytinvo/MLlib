# -*- coding: utf-8 -*-
"""
Created on 25/03/2020
@author duytinvo
"""
import torch
from mlmodels.utils.BPEtonkenizer import BPE

vocab_file = "/media/data/review_response/tokens/bert_level-bpe-vocab.txt"

tokenizer = BPE.load(vocab_file)

sent = "This is an example, where we test BPE standed for Byte-Pair Encoding. Happy coding!!!"

encoded = tokenizer.encode(sent)
# from torchtext.experimental.datasets import IMDB
# train1, = IMDB(data_select='train')