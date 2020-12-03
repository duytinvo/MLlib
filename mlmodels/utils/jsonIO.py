# -*- coding: utf-8 -*-
"""
Created on 2020-03-05
@author: duytinvo
"""
from mlmodels.utils.special_tokens import NULL, SEP, SENSP, SENGE

import math
import random
import itertools
import json
import os
import copy


class JSON:
    def __init__(self, filename, source2idx=None, target2idx=None, limit=-1, task=2):
        self.data = JSON.load(filename)
        # random.shuffle(self.data)
        self.source2idx = source2idx
        self.target2idx = target2idx
        self.length = len(self.data)
        self.limit = limit if limit > 0 else None
        self.task = task

    def __len__(self):
        return self.length

    def __iter__(self):
        for line in itertools.islice(self.data, self.limit):
            nl, target = JSON.task_parser(line, task=self.task)
            if self.source2idx is not None:
                nl = self.source2idx(nl)
            if self.target2idx is not None:
                target = self.target2idx(target)
            yield nl, target

    @staticmethod
    def task_parser(line, task=2):
        if task == 1:
            nl = line["data"]["text_information"]["comment"]
            target = line["data"]["text_information"]["star"]
        elif task == 2:
            hotel_name = line["data"]["metadata"]["Name_hotel"]
            rver_name = line["data"]["metadata"]["author"]
            # rv_rating = line["data"]["text_information"]["star"]
            # rv_title = line["data"]["text_information"]["title"]
            rv_content = line["data"]["text_information"]["comment"]
            if line["data"].get("rouge_scores", None) is not None:
                rouge_lf = str(round(line["data"]["rouge_scores"]["rouge-l"]["f"], 4))
                # nl = " ".join([rouge_lf, SENSP, rv_rating, SENSP, hotel_name, SENSP, rver_name, SENSP,
                #                rv_title, SENSP, rv_content, SENGE])
                nl = " ".join([rouge_lf, hotel_name, rver_name, rv_content, SENGE])
            else:
                # nl = " ".join([rv_rating, SENSP, hotel_name, SENSP, rver_name, SENSP,
                #                rv_title, SENSP, rv_content, SENGE])
                nl = " ".join([hotel_name, rver_name, rv_content, SENGE])
            target = line["data"]["text_information"]["reply"]
        else:
            raise Exception("Not implemented yet")
        nl = JSON.process_nl(nl)
        target = JSON.process_target(target)
        return nl, target

    @staticmethod
    def null_replacement(text):
        if len(text) == 0:
            return NULL
        else:
            return text

    @staticmethod
    def process_target(target):
        target = target.lower()
        return target

    @staticmethod
    def process_nl(nl):
        nl = nl.lower()
        return nl

    @staticmethod
    def load(_file):
        with open(_file, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def read(data_path, task=2):
        dataset = []
        data = JSON.load(data_path)
        # random.shuffle(data)
        for line in data:
            datum = JSON.task_parser(line, task=task)
            dataset.append(datum)
        return dataset

    @staticmethod
    def _len(_file):
        return len(JSON.load(_file))

    @staticmethod
    def get_map(data_path, task=2):
        r"""
        Generate an iterator to read json file.
        yield texts line-by-line.

        Arguments:
            data_path: a path for the data file.

        """
        data = []
        dataset = JSON.load(data_path)
        for line in dataset:
            data.append(JSON.task_parser(line, task=task))
        return data

    @staticmethod
    def get_iterator(data_path, task=2):
        r"""
        Generate an iterator to read json file.
        yield texts line-by-line.

        Arguments:
            data_path: a path for the data file.

        """

        def iterator(start, num_lines):
            data = JSON.load(data_path)
            for line in data[start: start + num_lines]:
                nl, target = JSON.task_parser(line, task=task)
                yield nl, target
        return iterator

    @staticmethod
    def dump(data, _file):
        if not os.path.exists(os.path.dirname(_file)):
            os.mkdir(os.path.dirname(_file))
        with open(_file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2)


if __name__ == '__main__':
    jsonfile = "/media/data/review_response/Dev.json"
    with open(jsonfile, "r") as f:
        jdata = json.load(f)

    data = []
    data_iter = JSON(jsonfile, source2idx=None, target2idx=None, limit=-1)
    for d in data_iter:
        data.append(d)
        break