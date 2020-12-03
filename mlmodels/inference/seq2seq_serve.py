"""
Created on 2019-02-20
@author: duytinvo
"""
import os
import csv
import time
import torch
import pandas as pd
from mlmodels.para_model import Translator_model
from mlmodels.utils.data_utils import SaveloadHP
from mlmodels.utils.preprocessing_v2 import PREPROCESS, JSON
from mlmodels.utils.sem2sql_v2 import PARSER as sem2sqlParser
from mlmodels.utils.word_emb_wombat import Wombat
import random
from mlmodels.utils.sqltree_parser import sqlParser
from nltk import word_tokenize
from mlmodels.utils.data_utils import SOT, EOT
from sql_py_antlr.Utilities import Utilities


class NP5(object):
    def __init__(self, model_args, model_dir, schema_file,
                 kb_relatedto, kb_isa, use_cuda=False, wombat_path=None):
        self.wombat_object = None
        if wombat_path is not None:
            self.wombat_object = Wombat(wombat_path)
        margs = SaveloadHP.load(model_dir + model_args)
        margs.use_cuda = use_cuda
        self.translator = Translator_model(margs)
        if model_dir and model_dir != margs.model_dir:
            margs.model_dir = model_dir
        seq2seq_filename = os.path.join(margs.model_dir, margs.seq2seq_file)
        print("Load Model from file: %s" % seq2seq_filename)
        self.translator.seq2seq.load_state_dict(torch.load(seq2seq_filename))
        self.translator.seq2seq.to(self.translator.device)

        self.semqlparser = sem2sqlParser(schema_file)
        self.preporcess = PREPROCESS(schema_file, kb_relatedto, kb_isa)

    def greedy_translate(self, entries):
        entries = self.translator.greedy_predict(entries, self.wombat_object)
        return entries

    def beam_translate(self, entries, bw=2, topk=2):
        entries = self.translator.beam_predict(entries, bw, topk, self.wombat_object)
        return entries

    def prepare_entry(self, nl, dbid):
        entry = dict()
        entry["db_id"] = dbid
        entry["question"] = nl
        # start = time.time()
        question_toks = word_tokenize(nl)
        entry['question_toks'] = question_toks
        # print("- TIMING: %.4f seconds for NL tokenization" % (time.time() - start))
        entry = self.preporcess.build_one(entry)
        return entry

    def batch_inference(self, nls, dbid="concert_singer", bw=2, topk=2, use_sql=False):
        entries = []
        for nl in nls:
            entry = self.prepare_entry(nl, dbid)
            entries.append(entry)
        entries = self.beam_translate(entries, bw, topk)[0]
        # print("- TIMING: %.4f seconds for beam search" % (time.time() - start))
        for entry in entries:
            if not use_sql:
                try:
                    entry = self.sem2sql(entry)
                except:
                    print("Unable to map SemQL to SQL; Error is at Sem2sql.py")
                    entry["warning"] = "Unable to map SemQL to SQL; Error is at Sem2sql.py"
                    entry["predicted_query"] = "SELECT * FROM TABLE"
            else:
                if entry["model_result"].startswith(SOT) and entry["model_result"].endswith(EOT):
                    entry["predicted_query"] = entry["model_result"][4: -5]
                elif entry["model_result"].startswith(SOT):
                    entry["predicted_query"] = entry["model_result"][4:]
                elif entry["model_result"].endswith(EOT):
                    entry["predicted_query"] = entry["model_result"][: -5]
                else:
                    entry["predicted_query"] = entry["model_result"]
            entry.pop('nltk_pos')
            entry.pop("question_toks")
            entry.pop("origin_question_toks")
            entry.pop("question_arg")
            entry.pop('question_arg_type')
        return entries

    def inference(self, nl="How many singers do we have?", dbid="concert_singer", bw=2, topk=2, use_sql=False):
        entry = self.prepare_entry(nl, dbid)
        # entry = self.beam_translate([entry], bw, topk)[0]
        if bw == 1:
            entry = self.greedy_translate([entry])[0]
        else:
            entry = self.beam_translate([entry], bw, topk)[0]
        # print("- TIMING: %.4f seconds for beam search" % (time.time() - start))
        if not use_sql:
            try:
                entry = self.sem2sql(entry)
            except:
                print("Unable to map SemQL to SQL; Error is at Sem2sql.py")
                entry["warning"] = "Unable to map SemQL to SQL; Error is at Sem2sql.py"
                entry["predicted_query"] = "SELECT * FROM TABLE"
        else:
            if entry["model_result"].startswith(SOT) and entry["model_result"].endswith(EOT):
                entry["predicted_query"] = entry["model_result"][4: -5]
            elif entry["model_result"].startswith(SOT):
                entry["predicted_query"] = entry["model_result"][4:]
            elif entry["model_result"].endswith(EOT):
                entry["predicted_query"] = entry["model_result"][: -5]
            else:
                entry["predicted_query"] = entry["model_result"]

        entry.pop('nltk_pos')
        entry.pop("question_toks")
        entry.pop("origin_question_toks")
        entry.pop("question_arg")
        entry.pop('question_arg_type')
        return entry

    def sem2sql(self, entry):
        if entry["model_result"].startswith(SOT) and entry["model_result"].endswith(EOT):
            entry["model_result_replace"] = entry["model_result"][4: -5]
        elif entry["model_result"].startswith(SOT):
            entry["model_result_replace"] = entry["model_result"][4:]
        elif entry["model_result"].endswith(EOT):
            entry["model_result_replace"] = entry["model_result"][: -5]
        else:
            entry["model_result_replace"] = entry["model_result"]
        return self.semqlparser.transform(entry)

    def pack_batch(self, test_file, dbid=None, batch_size=8):
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
            data = pd.read_csv(test_file)
            if len(data.columns) == 2:
                data.columns = ['Eng_query', 'SQL_query']  # , 'Eng_Answer']
            else:
                print("not implement yet")
                return
            entries = []
            for i, row in data.iterrows():
                if len(entries) == batch_size:
                    yield entries
                    entries = []
                eng_query = row['Eng_query']
                sql_query = row['SQL_query']
                entry = self.prepare_entry(eng_query, dbid)
                entry["query"] = sql_query
                entries.append(entry)
            if len(entries) != 0:
                yield entries

        else:
            print("not implement yet")
            return

    def regression_json_test(self, pfile, test_file, bw=2, topk=2, use_sql=False, limit=None, batch_size=8):
        data_iter = self.pack_batch(test_file, dbid=None, batch_size=batch_size)
        if not os.path.exists(os.path.dirname(pfile)):
            os.mkdir(os.path.dirname(pfile))
        with open(pfile, "w") as g:
            csvwriter = csv.writer(g)
            hearder = ["db_id", "question", "query", "predicted_query", "pred_prob", "matching"]
            csvwriter.writerow(hearder)
            i = 0
            init = time.time()
            for entries in data_iter:
                i += len(entries)
                if i == limit:
                    return
                bg = time.time()
                if bw == 1:
                    entries = self.greedy_translate(entries)
                else:
                    entries = self.beam_translate(entries, bw, topk)
                for entry in entries:
                    if not use_sql:
                        try:
                            entry = self.sem2sql(entry)
                        except:
                            print("Unable to map SemQL to SQL; Error is at Sem2sql.py")
                            entry["warning"] = "Unable to map SemQL to SQL; Error is at Sem2sql.py"
                            entry["predicted_query"] = "SELECT * FROM TABLE"
                    else:
                        if entry["model_result"].startswith(SOT) and entry["model_result"].endswith(EOT):
                            entry["predicted_query"] = entry["model_result"][4: -5]
                        elif entry["model_result"].startswith(SOT):
                            entry["predicted_query"] = entry["model_result"][4:]
                        elif entry["model_result"].endswith(EOT):
                            entry["predicted_query"] = entry["model_result"][: -5]
                        else:
                            entry["predicted_query"] = entry["model_result"]

                    gquery = " ".join(entry["query_toks_no_value"])
                    pquery = entry["predicted_query"]
                    row = [entry["db_id"], entry["question"], gquery, pquery, entry["pred_prob"], gquery == pquery]
                    csvwriter.writerow(row)

            # pred_data.append(entry)
            if i > 0 and i % 100 == 0:
                now = time.time()
                print("Processing %d queries in %.4f seconds; Inference speed: %.4f (queries/second)" %
                      (i, now - bg, i/(now-init)))
        return

    def regression_csv_test(self, pfile, test_file, dbid,  bw=2, topk=2, use_sql=True, limit=None,
                            nltk_tok=False, batch_size=8):
        data_iter = self.pack_batch(test_file, dbid=dbid, batch_size=batch_size)
        if not os.path.exists(os.path.dirname(pfile)):
            os.mkdir(os.path.dirname(pfile))
        with open(pfile, "w") as f:
            csvwriter = csv.writer(f)
            hearder = ["db_id", "question", "query", "predicted_query", "pred_prob", "matching"]
            csvwriter.writerow(hearder)
            i = 0
            init = time.time()
            for entries in data_iter:
                i += len(entries)
                if i == limit:
                    return
                bg = time.time()
                # print("Entries length:", len(entries))
                if bw == 1:
                    entries = self.greedy_translate(entries)
                else:
                    entries = self.beam_translate(entries, bw, topk)
                for entry in entries:
                    if not use_sql:
                        try:
                            entry = self.sem2sql(entry)
                        except:
                            print("Unable to map SemQL to SQL; Error is at Sem2sql.py")
                            entry["warning"] = "Unable to map SemQL to SQL; Error is at Sem2sql.py"
                            entry["predicted_query"] = "SELECT * FROM TABLE"
                    else:
                        if entry["model_result"].startswith(SOT) and entry["model_result"].endswith(EOT):
                            entry["predicted_query"] = entry["model_result"][4: -5]
                        elif entry["model_result"].startswith(SOT):
                            entry["predicted_query"] = entry["model_result"][4:]
                        elif entry["model_result"].endswith(EOT):
                            entry["predicted_query"] = entry["model_result"][: -5]
                        else:
                            entry["predicted_query"] = entry["model_result"]

                    sql_query = entry["query"]
                    if nltk_tok:
                        query_toks = sqlParser.tokenize(sql_query)
                    else:
                        query_toks = Utilities.tokenize(sql_query)
                    entry["query_toks"] = query_toks
                    entry['query_toks_no_value'] = query_toks
                    gquery = " ".join(entry["query_toks_no_value"])
                    pquery = entry["predicted_query"]
                    row = [entry["db_id"], entry["question"], gquery, pquery, entry["pred_prob"], gquery == pquery]
                    csvwriter.writerow(row)

                # data.append(entry)
                if i > 0 and i % 2 == 0:
                    now = time.time()
                    print("Processing %d queries in %.4f seconds; Accumulated inference speed: %.4f (queries/second)" %
                          (i, now - init, i/(now-init)))
        return


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument('--model_args', help='Args file', default="translator.args",
                           type=str)

    argparser.add_argument('--model_dir', help='Model directory',
                           default="../data/data_locate_toy/trained_model/2019_12_16_16_42/", type=str)

    argparser.add_argument('--schema_file', help='Schema file in Json format', type=str,
                           default="../data/data_locate_toy/schema/json_tables_full.json")

    argparser.add_argument('--kb_relatedto', type=str, help='conceptNet data',
                            default="../data/permanent/english_RelatedTo.pkl")

    argparser.add_argument('--kb_isa', type=str, help='conceptNet data',
                            default="../data/permanent/english_IsA.pkl")

    argparser.add_argument('--wombat_path', help='wombat directory', type=str,
                           default="/Users/media/data/embeddings/database/glove-sqlite_")

    argparser.add_argument("--use_sql", action='store_true', default=False,
                           help="Using sql instead of SemQL")

    args = argparser.parse_args()

    nl = "all invoice"
    dbid = "locate"
    model_api = NP5(model_args=args.model_args, model_dir=args.model_dir, schema_file=args.schema_file,
                    kb_relatedto=args.kb_relatedto, kb_isa=args.kb_isa,
                    use_cuda=args.use_cuda, wombat_path=args.wombat_path)
    entry = dict()
    entry["db_id"] = dbid
    entry["question"] = nl
    question_toks = word_tokenize(nl)
    entry['question_toks'] = question_toks
    entry = model_api.preporcess.build_one(entry)
    # entry = model_api.inference(nl, dbid, bw=2, topk=2, use_sql=True)
