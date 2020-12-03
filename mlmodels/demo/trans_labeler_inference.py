"""
Created on 2019-02-20
@author: duytinvo
"""
import argparse
from mlmodels.inference.trans_labeler_serve import TAGGER
import mlmodels.demo.trans_labeler_settings as settings


def load_model(args):
    model_api = TAGGER(args)
    return model_api


if __name__ == '__main__':

    import argparse

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
