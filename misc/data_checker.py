# -*- coding: utf-8 -*-
"""
Created on 2019-12-20
@author: duytinvo
"""
import copy
import glob
import csv


def label_checker(csvfile, data_dir):
    filenames = glob.glob(data_dir + "*.txt")
    err_cnt = 0
    with open(csvfile, "w") as g:
        csvwriter = csv.writer(g)
        csvwriter.writerow(["review", "target", "sentiment", "combined"])
        data = []
        tg_count = 0
        pos_count = 0
        neg_count = 0
        neu_count = 0
        for filename in filenames:
            cnt = 0
            sent = []
            with open(filename, "r", encoding="utf-8", errors="replace") as f:     # , encoding="utf-8", errors="backslashreplace"
                for line in f:
                    cnt += 1
                    line = line.rstrip().replace("\t", " ").lower()
                    if line.startswith("{'$oid'"):
                        rid = line
                        if len(sent) != 0:
                            try:
                                rv, tar, sen, com = list(zip(*sent))
                                csvwriter.writerow([" ".join(rv), " ".join(tar), " ".join(sen), " ".join(com)])
                                data.append(sent)
                            except:
                                pass
                        sent = []
                    else:
                        if len(line) != 0:
                            token = line.split()[0]
                            lb = " ".join(line.split()[1:])
                            try:
                                tg, pol, sts = lb[1:-1].split(",")
                                tg = tg.strip()
                                pol = pol.strip()
                                sts = sts.strip()
                                if tg == "" and pol == "" and sts == "":
                                    lb = ["O"] * 3
                                elif pol != "u":
                                    if "+" in pol:
                                        l = "positive_" + str(pol.count("+"))
                                        # l = "positive"
                                    elif "-" in pol:
                                        l = "negative_" + str(pol.count("-"))
                                        # l = "negative"
                                    else:
                                        l = "neutral_0"
                                        # l = "neutral"

                                    tg_lst = tg.split()
                                    if sts == "e":
                                        if len(tg_lst) == 1:
                                            lb = ["S", l, "S" + "_" + l.split("_")[0]]  # + " <--> " + tg + " <--> " + pol
                                        else:
                                            idx = tg_lst.index(token)
                                            if idx == 0:
                                                lb = ["B", l, "B" + "_" + l.split("_")[0]]  # + " <--> " + tg + " <--> " + pol
                                            elif idx == len(tg_lst) - 1:
                                                lb = ["E", l, "E" + "_" + l.split("_")[0]]  # + " <--> " + tg + " <--> " + pol
                                            else:
                                                lb = ["I", l, "I" + "_" + l.split("_")[0]]  # + " <--> " + tg + " <--> " + pol
                                    else:
                                        # when target is implicitly mentioned in the review, we deem it as none now
                                        lb = ["O"] * 3
                                        # # TODO: handle implicitly mentioning targets later
                                        # if len(tg_lst) == 1:
                                        #     lb = "Si" + "_" + l  # + " <--> " + tg + " <--> " + pol
                                        # else:
                                        #     idx = tg_lst.index(token)
                                        #     if idx == 0:
                                        #         lb = "Bi" + "_" + l  # + " <--> " + tg + " <--> " + pol
                                        #     elif idx == len(tg_lst) - 1:
                                        #         lb = "Ei" + "_" + l  # + " <--> " + tg + " <--> " + pol
                                        #     else:
                                        #         lb = "Ii" + "_" + l  # + " <--> " + tg + " <--> " + pol
                                    # tg_count += 1
                                else:
                                    # when target is unrelated to the review
                                    lb = ["O"] * 3
                                    # tg_count += 1
                            except:
                                err_cnt += 1
                                print("Labelling error %d at file \"%s\":" % (err_cnt, filename))
                                print("\t- At review id of: ", rid)
                                print("\t- At row number of: ", cnt)
                                print("\t- Line content: ", line)
                                print("\t- Token content: ", token)
                                print("\t- Label content: ", lb)
                                lb = [lb]

                            if "pos" in lb[-1]:
                                pos_count += 1
                            if "neg" in lb[-1]:
                                neg_count += 1
                            if "neu" in lb[-1]:
                                neu_count += 1
                            # print([token], lb)
                            sent.append([token] + lb)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default="/media/data/labeled_reviews/raw/second/", type=str,
                        help="Directory of raw data")
    parser.add_argument("--csvfile", default="/media/data/labeled_reviews/raw/second/data.csv", type=str,
                        help="path of the processed file")
    args = parser.parse_args()

    label_checker(args.csvfile, args.data_dir)
    # filenames = glob.glob("../data/raw_txt/*.txt")
    # csvfile = "../data/processed_csv/res2.csv"
    # # filename = "../../data/raw_txt/1st Final 534 Completed Set.txt"