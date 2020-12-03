# -*- coding: utf-8 -*-
"""
Created on 25/03/2020
@author duytinvo
"""
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.txtIO import TXT

csvfiles = ["/media/data/review_response/Train.csv", "/media/data/review_response/Dev.csv"]
txtfile = "/media/data/review_response/raw_vocab.txt"

data = []
for csvfile in csvfiles:
    rev = CSV.read(csvfile, True, [0])
    data += rev
    res = CSV.read(csvfile, True, [1])
    data += res

TXT.write(data, txtfile)
