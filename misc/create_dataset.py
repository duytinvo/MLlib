"""
Created on 2018-09-14
@author: duytinvo
"""
import csv
import numpy as np
from mlmodels.utils.jsonIO import JSON


def write_csv(data, file_name, title=""):
    # data.sort(key=lambda x: len(x[0]), reverse=False)
    file_type = file_name.split(".")[-1]
    if file_type == "json":
        JSON.dump(data, file_name)
    else:
        with open(file_name, "w", newline='') as f:
            if file_name.split(".")[-1] != "csv":
                if len(title) != 0:
                    f.write(title + "\n")
                for line in data:
                    f.write(line + "\n")
            else:
                writer = csv.writer(f, delimiter=",")
                if len(title) != 0:
                    writer.writerow(title)
                writer.writerows(data)


def write_dataset(data_file, train_file, val_file, test_file, tr_ratio=0.9, val_ratio=0.95, shuffle=True,
                  readfirstline=False, writefirstline=False):

    title = ""
    file_type = data_file.split(".")[-1]
    if file_type == "json":
        corpus = JSON.load(data_file)
    else:
        corpus = set()
        with open(data_file, "r") as f:
            if file_type == "csv":
                csvreader = csv.reader(f)
                if readfirstline:
                    title = next(csvreader)
                for line in csvreader:
                    corpus.update([tuple([line[0], line[-1]])])
            else:
                if readfirstline:
                    title = next(f)
                for line in f:
                    corpus.update([line.strip()])
            corpus = list(corpus)
    train_len = int(tr_ratio * len(corpus))
    val_len = int(val_ratio * len(corpus))
    if shuffle:
        np.random.shuffle(corpus)
        train, val, test = np.split(corpus, [train_len, val_len])
        train = train.tolist()
        val = val.tolist()
        test.tolist()
    else:
        train = corpus[:train_len]
        val = corpus[train_len:val_len]
        test = corpus[val_len:]

    if not writefirstline:
        title = ""
    if len(train) != 0:
        write_csv(train, train_file, title)
    if len(val) != 0:
        write_csv(val, val_file, title)
    if len(test) != 0:
        write_csv(test, test_file, title)


if __name__ == "__main__":
    """
    python create_dataset.py --corpus_file ./data/datalake/rawlake/csv/rawlake.csv --train_file ./data/datalake/rawlake/csv/train.csv --val_file ./data/datalake/rawlake/csv/dev.csv --test_file ./data/datalake/rawlake/csv/test.csv --firstline --tr_ratio 0.7 --val_ratio 0.85
    python -m misc.create_dataset --corpus_file /media/data/disambiguator/raw/transformed.csv --train_file /media/data/disambiguator/corpus/train.csv --val_file /media/data/disambiguator/corpus/dev.csv --test_file /media/data/disambiguator/corpus/test.csv --firstline --tr_ratio 0.7 --val_ratio 0.85
    """
    import argparse
    seed_num = 1234
    np.random.seed(seed_num)

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--corpus_file', help='corpus file',
                           default="../data/reviews/processed_csv/res4.csv",
                           type=str)
    argparser.add_argument('--train_file', help='training file',
                           default="../data/reviews/processed_csv/train_res4.csv",
                           type=str)
    argparser.add_argument('--val_file', help='validating file',
                           default="../data/reviews/processed_csv/dev_res4.csv",
                           type=str)
    argparser.add_argument('--test_file', help='testing file',
                           default="../data/reviews/processed_csv/test_res4.csv",
                           type=str)
    argparser.add_argument('--tr_ratio', help='splitting rate of training', default=0.5, type=float)

    argparser.add_argument('--val_ratio', help='splitting rate of validating', default=0.75, type=float)

    argparser.add_argument("--firstline", action='store_true', default=False, help="file header flag")

    argparser.add_argument("--shuffle", action='store_true', default=False, help="shuffling flag")

    args = argparser.parse_args()

    print("Split dataset ...")
    write_dataset(args.corpus_file, args.train_file, args.val_file, args.test_file,
                  args.tr_ratio, args.val_ratio, shuffle=args.shuffle,
                  readfirstline=args.firstline, writefirstline=args.firstline)
