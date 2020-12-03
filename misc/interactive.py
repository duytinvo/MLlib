#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 03:38:32 2018

@author: duytinvo
"""
import os
import torch
import argparse
from mlmodels.inference.labeler_serve import TAGGER


def insertion(token, l):
    d = l - len(token)
    lp = d//2
    rp = d - lp
    return " "*lp + token + " "*rp


def word_alignment(words, labels):
    # ml = max(max([len(w) for w in words]), max([len(l) for l in labels]))
    for i in range(len(labels)):
        if len(words[i]) < len(labels[i]):
            words[i] = insertion(words[i], len(labels[i]))
        elif len(words[i]) > len(labels[i]):
            labels[i] = insertion(labels[i], len(words[i]))
    return words, labels


def word_prob_alignment(words, labels, probs):
    ml = max(max([len(w) for w in words]), max([len(l) for l in labels]), max([len(p) for p in probs]))
    for i in range(len(labels)):
        words[i] = insertion(words[i], ml)
        labels[i] = insertion(labels[i], ml)
        probs[i] = insertion(probs[i], ml)
    return words, labels, probs


def word_prob_alignment2(words, labels):
    ml = max(max([len(w) for w in words]), max([len(l) for l in labels]))
    for i in range(len(labels)):
        words[i] = insertion(words[i], ml)
        labels[i] = insertion(labels[i], ml)
    return words, labels


def interactive_shell(args):
    """Creates interactive shell to play with model

    Args:
        model: instance of Classification

    """
    model_api = TAGGER(model_args=args.model_args, model_dir=args.model_dir, use_cuda=args.use_cuda,
                       wombat_path=args.wombat_path)
        
    print("""
To exit, enter 'EXIT'.
Enter a review-level sentence like 
review-sentence> wth is it????""")

    while True:
        # for python 3
        sentence = input("review-sentence> ")

        if sentence == "EXIT":
            break
        entry = model_api.inference(sentence)
        sent_rep = entry["input_review"]
        labels = entry["pred_output"]
        words, labels = word_prob_alignment2(sent_rep, labels, )

        print("INPUT >>>", " ".join(words))
        print("LABEL >>>", " ".join(labels), "\n")
        # print(">>>", " ".join(probs), "\n")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument('--model_args', help='Args file', default="labeler.args",
                           type=str)

    argparser.add_argument('--model_dir', help='Model directory',
                           default="./data/trained_model/2020_03_06_15_09/", type=str)

    argparser.add_argument('--wombat_path', help='wombat directory', type=str,
                           default="/Users/media/data/embeddings/database/glove-sqlite_")
    
    args = argparser.parse_args()
        
    interactive_shell(args)

    # margs = SaveloadHP.load(args.model_args)
    # margs.use_cuda = args.use_cuda
    # i2l = {}
    # for k, v in margs.vocab.l2i.items():
    #     i2l[v] = k
    #
    # #    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    # model_filename = os.path.join(margs.model_dir, margs.model_file)
    # print("Load Model from file: %s" % model_filename)
    # classifier = Labeler_model(margs)
    # classifier.model.load_state_dict(torch.load(model_filename))
    # classifier.model.to(classifier.device)
    #
    # sentence = "the room is small but the balcony is beautiful"
    # label_prob, label_pred = classifier.predict(sentence)
    #
    # labels = [i2l[i.item()] for i in label_pred.squeeze()]
    # probs = ["%.4f" % (p.item()) for p in label_prob.squeeze()]
    #
    # sent_rep = Csvfile.process_sent(sentence)
    # words, labels = word_alignment(sent_rep.split(), labels)
