# -*- coding: utf-8 -*-
"""
Created on 2020-07-13
@author duytinvo
"""
from mlmodels.utils.txtIO import TXT
from mlmodels.utils.csvIO import CSV
filename = "/media/data/paraphrase/paralex-evaluation/data/train/paraphrases.txt"

data = TXT.read(filename, False)
newdata = []
for d in data:
    newdata.append(d.split("\t")[:-1])

CSV.write(newdata, "/media/data/paraphrase/paralex.csv")
