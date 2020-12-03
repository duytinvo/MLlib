"""
Created on 2019-02-20
@author: duytinvo
"""
import os
import torch
from mlmodels.training.trans_labeler_model import TransLabelerModel
from mlmodels.utils.auxiliary import SaveloadHP
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.idx2tensor import Data2tensor
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.word_emb_wombat import Wombat
import time


class TAGGER(object):
    def __init__(self, args):
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        margs = torch.load(os.path.join(args.model_name_or_path, "training_args.bin"))
        margs.no_cuda = args.no_cuda
        margs.label_file = args.label_file
        margs.model_name_or_path = args.model_name_or_path
        self.tagger = TransLabelerModel(margs)
        self.tagger.model_init(args.model_name_or_path)
        Data2tensor.set_randseed(args.seed)

    @staticmethod
    def prepare_nls(nls):
        data = []
        for nl in nls:
            data.append((CSV.process_nl(nl), None))
        return data

    def batch_inference(self, nls=["I love Chata.ai", "WHO is sucks"], batch_size=4):
        nls = self.prepare_nls(nls)
        toks_list, preds_list, probs_list = self.tagger.predict_batch(nls, batch_size)
        return toks_list, preds_list, probs_list

    def inference(self, nl="How many singers do we have?"):
        nls = self.prepare_nls([nl])
        toks_list, preds_list, probs_list = self.tagger.predict_batch(nls, batch_size=1)
        return toks_list[0], preds_list[0], probs_list[0]


if __name__ == '__main__':
    import argparse
    from mlmodels.utils.dataset import IterDataset, MapDataset
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
    from tqdm import tqdm, trange
    import numpy as np

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--no_cuda", action='store_false', default=True, help="GPUs Flag (default False)")

    argparser.add_argument("--model_name_or_path", default="../../data/reviews/trained_model", type=str,
                           help="Path to pre-trained model or shortcut name selected in the list")

    argparser.add_argument('--label_file', help='Trained file (semQL) in Json format', type=str,
                           default="../../data/reviews/processed_csv/labels.txt")
    argparser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = argparser.parse_args()

    model_api = TAGGER(args)
    nls = ["I love Chata.ai", "WHO is sucks"]
    toks_list, preds_list, probs_list = model_api.batch_inference(nls, batch_size=16)

    # args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    # margs = torch.load(os.path.join(args.model_name_or_path, "training_args.bin"))
    # margs.no_cuda = args.no_cuda
    # margs.model_name_or_path = args.model_name_or_path
    # margs.label_file = args.label_file
    # tagger = TransLabelerModel(margs)
    # tagger.model_init(args.model_name_or_path)
    # Data2tensor.set_randseed(args.seed)
    #
    # nls = [("I love Chata.ai", None), ("WHO is sucks", None)]
    # self = tagger
    # batch_size = 2
    # eval_dataset = MapDataset(nls, source2idx=self.source2idx, target2idx=self.target2idx, bpe=False,
    #                            special_tokens_func=self.build_inputs_with_special_tokens,
    #                            label_pad_id=self.args.pad_token_label_id)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
    #                              collate_fn=self.collate, num_workers=8)
    # toks_list, preds_list, probs_list = tagger.predict_batch(nls, batch_size=16)
