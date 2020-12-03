"""
Created on 2019-02-20
@author: duytinvo
"""
import os
import torch
from mlmodels.training.labeler_model import Labeler_model
from mlmodels.utils.auxiliary import SaveloadHP
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.word_emb_wombat import Wombat
import time


class TAGGER(object):
    def __init__(self, model_args, model_dir, vocab_file=None, label_file=None,
                 use_cuda=False, wombat_path=None):
        self.wombat_object = None
        if wombat_path is not None:
            self.wombat_object = Wombat(wombat_path)
        margs = SaveloadHP.load(model_dir + model_args)
        margs.use_cuda = use_cuda
        if vocab_file and vocab_file != margs.vocab_file:
            margs.vocab_file = vocab_file
        if label_file and label_file != margs.label_file:
            margs.label_file = label_file
        if model_dir and model_dir != margs.model_dir:
            margs.model_dir = model_dir
        self.tagger = Labeler_model(margs)
        # labeler_filename = os.path.join(margs.model_dir, margs.labeler_file)
        # print("Load Model from file: %s" % labeler_filename)
        # self.tagger.labeler.load_state_dict(torch.load(labeler_filename))
        # self.tagger.labeler.to(self.tagger.device)
        self.tagger.load_parameters(epoch=-1)

    def predict_batch(self, entries):
        entries = self.tagger.predict_batch(entries, self.wombat_object)
        return entries

    def predict_batch_probability(self, entries):
        # predict probability of each token
        entries = self.tagger.predict_batch(entries, self.wombat_object, return_probability=True)
        return entries

    @staticmethod
    def prepare_entry(rv, date=None, rvid=None, rating=None):
        entry = dict()
        # start = time.time()
        if date is not None:
            entry["date"] = date
        if rvid is not None:
            entry["review_id"] = rvid
        if rating is not None:
            entry["review_score"] = rating
        entry['mention'] = rv
        question_toks = CSV.process_nl(rv)
        entry['input_tokens'] = question_toks
        # print("- TIMING: %.4f seconds for NL tokenization" % (time.time() - start))
        return entry

    def batch_inference(self, nls, date=None, rvid=None, rating=None):
        entries = []
        for nl in nls:
            entry = self.prepare_entry(nl, date, rvid, rating)
            entries.append(entry)
        entries = self.predict_batch(entries)
        return entries

    def inference(self, nl="How many singers do we have?", date=None, rvid=None, rating=None):
        entry = self.prepare_entry(nl, date, rvid, rating)
        entry = self.predict_batch([entry])[0]
        return entry

    def batch_inference_probability(self, nls, date=None, rvid=None, rating=None):
        entries = []
        for nl in nls:
            entry = self.prepare_entry(nl, date, rvid, rating)
            entries.append(entry)
        entries = self.predict_batch_probability(entries)
        return entries

    def pack_batch(self, test_file, batch_size=8):
        ftype = test_file.split(".")[-1]
        if ftype == "json":
            data = JSON.load(test_file)
            # random.shuffle(data)
            entries = []
            for entry in data:
                if len(entries) == batch_size:
                    yield entries
                    entries = []
                entries.append(entry)
            if len(entries) != 0:
                yield entries
        elif ftype == "csv":
            data = CSV.read(test_file)
            entries = []
            for row in data:
                if len(entries) == batch_size:
                    yield entries
                    entries = []
                entry = self.prepare_entry(row[0])
                entry["gold_output"] = CSV.process_target(row[-1])
                entries.append(entry)
            if len(entries) != 0:
                yield entries

        else:
            print("not implement yet")
            return

    def regression_test(self, pfile, test_file, limit=None, batch_size=8):
        data_iter = self.pack_batch(test_file, batch_size=batch_size)
        if not os.path.exists(os.path.dirname(pfile)):
            os.mkdir(os.path.dirname(pfile))
        data = []
        reference = []
        candidate = []
        hearder = ["review", "gold_output", "pred_output", "matching"]
        data.append(hearder)
        i = 0
        init = time.time()
        for entries in data_iter:
            entries = self.predict_batch(entries)
            for entry in entries:
                review = entry["input_tokens"]
                gold_output = entry["gold_output"]
                pred_output = " ".join(entry["pred_sequence"])
                # prob_output = str(entry["prob_output"])
                row = [review, gold_output, pred_output, pred_output == gold_output]
                candidate.append(pred_output.split())
                reference.append(gold_output.split())
                data.append(row)
                if i > 0 and i % 2 == 0:
                    now = time.time()
                    print("Processing %d queries in %.4f seconds; Accumulated inference speed: %.4f (queries/second)" %
                          (i, now - init, i/(now-init)))
                i += 1
                if i == limit:
                    return
        metrics = self.tagger.class_metrics(reference, candidate)
        data.append(metrics)
        CSV.write(data, pfile)
        return


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument('--model_args', help='Args file', default="labeler.args",
                           type=str)

    argparser.add_argument('--model_dir', help='Model directory',
                           default="/media/data/disambiguator/traned_model/2020_05_20_10_34/", type=str)

    argparser.add_argument('--vocab_file', help='file to save a pre-trained tokenizer', type=str,
                           default="/media/data/review_response/tokens/bert_level-bpe-vocab.txt")
    argparser.add_argument('--label_file', help='Trained file (semQL) in Json format', type=str,
                           default="/media/data/disambiguator/corpus/labels.txt")

    argparser.add_argument('--wombat_path', help='wombat directory', type=str,
                           default="/media/data/embeddings/database/glove-sqlite_")

    argparser.add_argument('--input_file', help='Input file for regression test', type=str,
                           default="/media/data/disambiguator/test_cases/test_transformed.csv")
    argparser.add_argument('--output_file', help='Output file', type=str,
                           default="/media/data/disambiguator/regression/test_transformed_4layers.csv")

    args = argparser.parse_args()

    model_api = TAGGER(model_args=args.model_args, model_dir=args.model_dir,
                       vocab_file=args.vocab_file, label_file=args.label_file,
                       use_cuda=args.use_cuda, wombat_path=args.wombat_path)

    model_api.regression_test(args.output_file, args.input_file)

    # nl = "hiii justin was an excellent example of what good service is ."
    # nl2 = "very hip atmosphere , great food , friendly staff . kind of expensive but worth it ."
    # nls = [nl, nl2]
    # entries = model_api.batch_inference(nls)

    # model_api.regression_test(pfile="../../data/yelp_ner/test.csv", test_file="../../data/yelp_ner/processed/tag_bioes_test.csv",
    #                           limit=None, batch_size=32)
