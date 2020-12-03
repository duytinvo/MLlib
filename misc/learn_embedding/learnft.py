# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:21:18 2015

@author: duytinvo
"""
import argparse
import subprocess


def trainfasttext(args):
    print("training model ...")
    if args.mode != "skipgram":
        args.mode = "cbow"
    traincmd = ["./build/fasttext/fasttext", args.mode, "-input", args.tr_file, "-output", args.emb_file,
                "-minn", str(args.minn), "-maxn", str(args.maxn), "-dim", str(args.size), "-epoch", str(args.iters), 
                "-lr", str(args.lr), "-minCount", str(args.min_count), "-ws", str(args.window),
                "-thread", str(args.worker), "-loss", str(args.loss), "-neg", str(args.neg)]
    p = subprocess.Popen(' '.join(traincmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(out)
    print(err)


if __name__ == '__main__':
    """
    python learnft.py --tr_file /media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.txt --emb_file ./results/ft_yelp100.vec
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr_file", help="Directory of Training file.",
                        default='/media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.txt')
    parser.add_argument("--emb_file", help="Embedding file.", default='./results/ft_yelp100.vec')
    parser.add_argument("--min_count", help="Min Count", type=int, default=5)
    parser.add_argument("--window", help="Window Width", type=int, default=5)
    parser.add_argument("--iters", help="number of iterations", type=int, default=5)
    parser.add_argument("--size", help="Embedding Size", type=int, default=100)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.025)
    parser.add_argument("--mode", help="model type", default='skipgram')
    parser.add_argument("--loss", help="loss mode [ns, hs, softmax]", type=str, default="ns")
    parser.add_argument("--neg", help="negative sampling (be used when loss=ns)", type=int, default=5)
    parser.add_argument("--worker", help="Number of threads", type=int, default=12)
    parser.add_argument("--minn", help="minimum sub-word", type=int, default=3)
    parser.add_argument("--maxn", help="maximum sub-word", type=int, default=6)
    args = parser.parse_args()
    trainfasttext(args)

