# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:51:02 2015

@author: duytinvo
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_embs(fname):
    embs = dict()
    with open(fname, 'r') as f:
        for line in f: 
            p = line.strip().split()
            if len(p) == 2:
                continue
            else:
                w = p[0]
                e = [float(i) for i in p[1:]]
                embs[w] = np.array(e, dtype="float32")
    return embs 


def plot_with_labels(low_dim_embs, labels, filename='tsne2.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


def plot(fname, pname, plot_only=1000):
    embs = load_embs(fname)
    keys = []
    values = []
    for k, v in embs.items():
        keys.append(k)
        values.append(v)
        if len(keys) == plot_only:
            break
    values = np.array(values)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(values)
    plot_with_labels(low_dim_embs, keys, pname)


if __name__=="__main__":
    """
    python plotembs.py --emb_file ./w2vscripts/results/twsamples.process.vec --plot_file ./w2vscripts/results/twsamples.process.vec.png
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", help="Embedding file.", default='./results/twsamples.process.vec')
    parser.add_argument("--plot_file", help="Plotting file.", default='./results/twsamples.process.vec.png')
    parser.add_argument("--plot_only", help="Number of plotting words", type=int, default=1000)
    args = parser.parse_args()
    plot(args.emb_file, args.plot_file, plot_only=args.plot_only)
