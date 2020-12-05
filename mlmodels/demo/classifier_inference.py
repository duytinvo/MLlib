
"""
Created on 2019-02-20
@author: duytinvo
"""
import argparse
from mlmodels.inference.classifier_serve import CLASSIFIER
import mlmodels.demo.classifier_settings as settings


def load_model(args):
    model_api = CLASSIFIER(model_args=args.model_args, model_dir=args.model_dir,
                       vocab_file=args.vocab_file, label_file=args.label_file,
                       use_cuda=args.use_cuda, wombat_path=args.wombat_path)
    return model_api


def abs_testing(model_api, test_file, pfile, limit=500, batch_size=8):
    model_api.regression_test(pfile, test_file, limit, batch_size)


def test():
    nl = "all invoice"
    dbid = "locate"
    entry = model_api.single_inference(nl, dbid, bw=1, topk=1, use_sql=True)


if __name__ == '__main__':
    """
    screen python test_demo.py --model_dir ./data/data_locate_nov01/trained_model/2019-11-01T19\:41/ --use_cuda --use_sql --limit -1 --bw 1 --topk 1
    """
    # import csv_settings as settings

    argparser = argparse.ArgumentParser()

    # Use for loading models
    argparser.add_argument('--model_dir', help='Trained model directory', type=str,
                           default="/media/data/vnreviews/Product/dataset/trained_model/")

    argparser.add_argument('--vocab_file', help='file to save a pre-trained tokenizer', type=str,
                           default=settings.vocab_file)
    argparser.add_argument('--label_file', help='File saved all labels', type=str,
                           default="/media/data/vnreviews/Product/dataset/labels.txt")

    argparser.add_argument('--model_args', help='Args file', default=settings.model_args, type=str)

    argparser.add_argument('--wombat_path', help='wombat directory', type=str, default=settings.wombat_path)

    # OPTIONAL: Use for regression
    argparser.add_argument('--regression_file', help='Testing file in Json format', type=str, default=settings.test_file)

    argparser.add_argument('--pfile', help='Predicted text file', type=str, default=settings.pred_test_file)

    argparser.add_argument("--use_cuda", action='store_true', default=settings.use_cuda, help="GPUs Flag (default False)")

    argparser.add_argument('--limit', help='limit', type=int, default=None)

    argparser.add_argument('--batch_size', type=int, help='batch_size', default=settings.batch_size)

    args = argparser.parse_args()

    model_api = load_model(args)

    # Single inference
    nl = "hạt nhỏ, bay mùi hơi lạ."
    nl2 = "tuyệt vời ngoài sức tưởng tượng, chúc shop buôn may bán đắt"
    nls = [nl, nl2]
    entries = model_api.batch_inference(nls)
    # timeit.timeit('test()', number=100, setup="from __main__ import test")

