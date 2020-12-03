# -*- coding: utf-8 -*-
"""
Created on 2019-07-02
@author: duytinvo
"""
import settings
import argparse
from np5.utils.json2json import csv2corpus


if __name__ == '__main__':
    """
    python corpus_generator.py --db_id locate --csv_train_file --train_file --csv_dev_file --dev_file --csv_test_file --test_file --schema_file --kb_relatedto --kb_isa --nltk_tok --use_sql
    """

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_mode', help='setlect data to generate (full/partial)', type=str,
                           default=settings.data_mode)

    argparser.add_argument('--db_id', help='DB name for single case', type=str,
                           default=settings.DB_Name)

    argparser.add_argument('--csv_train_file', help='Trained raw file in Json format', type=str,
                           default=settings.csv_train_file)

    argparser.add_argument('--csv_dev_file', help='Validated raw file in Json format', type=str,
                           default=settings.csv_dev_file)

    argparser.add_argument('--csv_test_file', help='Test raw file in Json format', type=str,
                           default=settings.csv_test_file)

    argparser.add_argument('--train_file', help='Trained clean file in Json format', type=str,
                           default=settings.train_file)

    argparser.add_argument('--dev_file', help='Validated clean file in Json format', type=str,
                           default=settings.dev_file)

    argparser.add_argument('--test_file', help='Test clean file in Json format', type=str,
                           default=settings.test_file)

    argparser.add_argument('--schema_file', help='Schema file in Json format', type=str,
                           default=settings.schema_file)

    argparser.add_argument('--kb_relatedto', type=str, help='conceptNet data',
                           default=settings.kb_relatedto)

    argparser.add_argument('--kb_isa', type=str, help='conceptNet data',
                           default=settings.kb_isa)

    argparser.add_argument("--use_sql", action='store_true', default=settings.use_sql,
                           help="Directly predict SQL flag")

    argparser.add_argument("--nltk_tok", action='store_true', default=settings.nltk_tok,
                           help="use nltk to tokenize sqls")

    args = argparser.parse_args()

    csv2corpus(args.train_file, args.csv_train_file, args.schema_file, args.db_id,
               args.kb_relatedto, args.kb_isa, args.use_sql, nltk_tok=args.nltk_tok)

    if args.data_mode == "full":
        csv2corpus(args.dev_file, args.csv_dev_file, args.schema_file, args.db_id,
                   args.kb_relatedto, args.kb_isa, args.use_sql, nltk_tok=args.nltk_tok)

        csv2corpus(args.test_file, args.csv_test_file, args.schema_file, args.db_id,
                   args.kb_relatedto, args.kb_isa, args.use_sql, nltk_tok=args.nltk_tok)

