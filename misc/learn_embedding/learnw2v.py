# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:21:18 2015

@author: duytinvo
"""
import argparse
import itertools
import logging
import os
from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Txtfile(object):
    def __init__(self, fname, firstline=False, limit=-1):
        self.fname = fname
        self.firstline = firstline
        if limit < 0:
            self.limit = None
        else:
            self.limit = limit

    def __iter__(self):
        with open(self.fname, 'r') as f:
            f.seek(0)
            if self.firstline:
                next(f)
            for line in itertools.islice(f, self.limit):
                line = line.strip().split()
                yield line


def savevocab(model, vocab_file):
    with open(vocab_file, 'w') as f:
        for wd in model.wv.index2word:
            count = model.wv.vocab[wd].count
            f.write(wd+' '+str(count)+'\n')


def trainw2v(args):
    model = Word2Vec(min_count=args.min_count, window=args.window, size=args.size,
                     alpha=args.lr, sg=args.sg, hs=args.hs, workers=args.worker, negative=args.negative)

    sents = Txtfile(args.tr_file)

    print('building vocab ...')
    model.build_vocab(sents)

    print('training w2v model ...')
    # sents = Txtfile(args.tr_file)
    model.train(sents, total_examples=model.corpus_count, epochs=args.iters)

    print('writing w2v vectors ...')
    model.wv.save_word2vec_format(args.emb_file)

    # print("saving model ...")
    # model.save(args.mod_file)
    #
    # print('saving vocab ...')
    # savevocab(model, args.vocab_file)


if __name__ == '__main__':
    """
    python learnw2v.py --tr_file /media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.txt --emb_file ./results/w2v_yelp100.vec.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tr_file", help="Directory of Training file.",
                        default='/media/data/restaurants/yelp_dataset/processed/extracted_rev/yelp_data_rev.txt')
    parser.add_argument("--emb_file", help="Embedding file.", default='./results/w2v_yelp100.vec.txt')
    # parser.add_argument("--vocab_file", help="Vocabulary file.", default='./results/w2v_yelp100.vocab')
    # parser.add_argument("--mod_file", help="Saved model file.", default='./results/w2v_yelp100.model')
    parser.add_argument("--min_count", help="Min Count", type=int, default=5)
    parser.add_argument("--window", help="Window Width", type=int, default=5)
    parser.add_argument("--iters", help="number of iterations", type=int, default=5)
    parser.add_argument("--size", help="Embedding Size", type=int, default=100)
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.025)
    parser.add_argument("--sg", action='store_true', default=False, help="skipgram flag (default false: cbow)")
    parser.add_argument("--hs", help="hierarchial sampling", type=int, default=0)
    parser.add_argument("--negative", help="negative sampling (be used when hs=0)", type=int, default=5)
    parser.add_argument("--worker", help="Number of threads", type=int, default=12)
    args = parser.parse_args()
    trainw2v(args)
