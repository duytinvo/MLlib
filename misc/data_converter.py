# -*- coding: utf-8 -*-
"""
Created on 2020-03-06
@author: duytinvo
"""
import csv

trainfile = "../data/disambiguator/train.csv"
devfile = "../data/disambiguator/dev.csv"
wtrainfile = "../data/disambiguator/train_rfm.csv"
wdevfile = "../data/disambiguator/dev_rfm.csv"


def reformat(rfile, wfile):
    data = []
    with open(wfile, "w") as g:
        csvwriter = csv.writer(g)
        with open(rfile, "r") as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            csvwriter.writerow(header)
            for line in csvreader:
                nl = line[0][2:-2].replace("', '", " ")
                lb = line[1][2:-2].replace("', '", " ")
                nlb = []
                for tok in lb.split():
                    if tok == "english":
                        nlb += ["O"]
                    else:
                        nlb += ["S_" + tok]
                csvwriter.writerow([nl, " ".join(nlb)])
                data.append((nl, " ".join(nlb)))


reformat(trainfile, wtrainfile)
reformat(devfile, wdevfile)

