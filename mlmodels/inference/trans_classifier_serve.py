"""
Created on 2019-02-20
@author: duytinvo
"""
import os
import torch
from mlmodels.utils.csvIO import CSV
from mlmodels.training.trans_classifier_model import TransClassifierModel
from mlmodels.utils.idx2tensor import Data2tensor


class CLASSIFIER(object):
    def __init__(self, args):
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        margs = torch.load(os.path.join(args.model_name_or_path, "training_args.bin"))
        margs.no_cuda = args.no_cuda
        margs.label_file = args.label_file
        margs.model_name_or_path = args.model_name_or_path
        self.tagger = TransClassifierModel(margs)
        self.tagger.model_init(args.model_name_or_path)
        Data2tensor.set_randseed(args.seed)

    @staticmethod
    def prepare_nls(nls):
        data = []
        for nl in nls:
            data.append((CSV.process_nl(nl), None))
        return data

    def batch_inference(self, nls=["I love Chata.ai", "WHO is sucks"], batch_size=4, topk=1):
        nls = self.prepare_nls(nls)
        labels = self.tagger.predict_batch(nls, batch_size, topk=topk)
        return labels

    def inference(self, nl="How many singers do we have?", topk=1):
        nls = self.prepare_nls([nl])
        labels = self.tagger.predict_batch(nls, batch_size=1, topk=topk)
        return labels

    @staticmethod
    def str2label(s):
        if s == '5':
            l = "highly positive"
        elif s =='4':
            l = "positive"
        elif s == '3':
            l = "slightly positive"
        elif s == '2':
            l = "negative"
        else:
            l = "highly negative"
        return l


if __name__ == '__main__':
    import argparse
    from mlmodels.utils.dataset import IterDataset, MapDataset
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
    from tqdm import tqdm, trange
    import numpy as np

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--no_cuda", action='store_false', default=True, help="GPUs Flag (default False)")

    argparser.add_argument("--model_name_or_path", default="/media/data/review_response/toy_datasets/csv/trained_model/",
                           type=str, help="Path to pre-trained model or shortcut name selected in the list")

    argparser.add_argument('--label_file', help='Labeled file', type=str,
                           default="/media/data/review_response/toy_datasets/csv/labels.txt")
    argparser.add_argument('--test_file', help='Test file', type=str,
                           default="/media/data/review_response/customer_data/NH_Collection_Guadalajara_Centro_Historico.csv")
    argparser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = argparser.parse_args()

    model_api = CLASSIFIER(args)
    reviews = ["I love Chata.ai", "WHO is sucks"]
    labels = model_api.batch_inference(reviews, batch_size=16)
    for i in range(len(labels)):
        s2l = [model_api.str2label(j) for j in labels[i][1]]
        labels[i] += (s2l,)

    # data = CSV.read(args.test_file, firstline=True, slices=[0, 1, 2, 3, 4, 5])
    # reviews = [" ".join(d[0].split()) for d in data]
    # toks_list, preds_list, probs_list = model_api.batch_inference(reviews, batch_size=16)
    # labels = [model_api.str2label(s) for s in preds_list]
    # write_data = []
    # for i in range(len(data)):
    #     write_data.append(data[i] + [labels[i]])
    # CSV.write(write_data,
    #           "/media/data/review_response/customer_data/sentiment_NH_Collection_Guadalajara_Centro_Historico.csv")


