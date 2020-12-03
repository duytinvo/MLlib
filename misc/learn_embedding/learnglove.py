# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:21:18 2015

@author: duytinvo
"""
import argparse
import subprocess


def trainglove(args):
    print('building vocab ...')
    vocabcmd = ["./build/glove/vocab_count -verbose 2", "-min-count", str(args.min_count), "<", args.tr_file,
                ">", args.vocab_file]
    p = subprocess.Popen(' '.join(vocabcmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    # print(out)
    # print(err)
    print('extracting cooccurence words ...')
    cooccurcmd = ["./build/glove/cooccur -verbose 2 -memory 4.0", "-vocab-file", args.vocab_file, "-window-size",
                  str(args.window), "<", args.tr_file, ">", "./results/cooccurrence.bin"]
    p = subprocess.Popen(' '.join(cooccurcmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    # print(out)
    # print(err)
    print('shuffling cooccurence words ...')
    shufflecmd = ["./build/glove/shuffle -verbose 2 -memory 4.0", "<", "./results/cooccurrence.bin",
                  ">", "./results/cooccurrence.shuf.bin"]
    p = subprocess.Popen(' '.join(shufflecmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    # print(out)
    # print(err)
    print("training model ...")
    traincmd = ["./build/glove/glove -verbose 2 -threads 8 -binary 0", "-save-file", args.emb_file,
                "-input-file", "./results/cooccurrence.shuf.bin", "-x-max", str(args.x_max), "-iter", str(args.iters), 
                "-vector-size", str(args.size), "-vocab-file", args.vocab_file, "-write-header 1",
                "-threads", str(args.worker), "-eta", str(args.lr)]
    p = subprocess.Popen(' '.join(traincmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    # print(out)
    # print(err)


if __name__ == '__main__':
    """
    python learnglove.py --tr_file /media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.txt --emb_file ./results/w2v_yelp100.vec
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr_file", help="Directory of Training file.",
                        default='/media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.txt')
    parser.add_argument("--emb_file", help="Embedding file.", default='./results/glove_yelp100.vec')
    parser.add_argument("--vocab_file", help="Vocabulary file.", default='./results/glove_yelp100.vocab')
    parser.add_argument("--min_count", help="Min Count", type=int, default=5)
    parser.add_argument("--window", help="Window Width", type=int, default=5)
    parser.add_argument("--iters", help="number of iterations", type=int, default=5)
    parser.add_argument("--size", help="Embedding Size", type=int, default=100)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.025)
    parser.add_argument("--worker", help="Number of threads", type=int, default=12)
    parser.add_argument("--x_max", help="cutoff in weighting function", type=int, default=100)
    args = parser.parse_args()
    trainglove(args)

