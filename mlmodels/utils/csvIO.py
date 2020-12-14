# -*- coding: utf-8 -*-
"""
Created on 2020-03-05
@author: duytinvo
"""
import itertools
import csv
import os
import io
import sys
import six


class CSV(object):
    def __init__(self, filename, source2idx=None, target2idx=None, firstline=True, limit=-1, task=2, delimiter='\t'):
        self.filename = filename
        self.source2idx = source2idx
        self.target2idx = target2idx
        self.length = None
        self.firstline = firstline
        self.limit = limit if limit > 0 else None
        self.task = task
        self.delimiter = delimiter

    def __iter__(self):
        with open(self.filename, newline='', encoding='utf-8') as f:
            f.seek(0)
            csvreader = CSV.unicode_csv_reader(f, delimiter=self.delimiter)
            # csvreader = csv.reader(f)
            if self.firstline:
                # Skip the header
                next(csvreader)
            for line in itertools.islice(csvreader, self.limit):
                nl, target = CSV.task_parser(line, task=self.task)
                if self.source2idx is not None:
                    nl = self.source2idx(nl)
                if self.target2idx is not None:
                    target = self.target2idx(target)
                # TODO: assert len(nl) == len(target)
                yield nl, target

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length

    @staticmethod
    def task_parser(line, task=2):
        score = None
        if task == 1:
            target, nl = line[-2], line[-1]
        elif task == 2:
            nl, target = line[-2], line[-1]
        elif task == 3:
            nl, score, target = line[0], line[1], line[-1]
        else:
            raise Exception("not implement %s yet")
        nl = CSV.process_nl(nl)
        target = CSV.process_target(target)
        if score is not None:
            score = CSV.process_score(score)
            return nl, score, target
        else:
            return nl, target

    @staticmethod
    def process_target(target):
        target = target.lower()
        return target

    @staticmethod
    def process_nl(nl):
        nl = nl.lower()
        return nl

    @staticmethod
    def process_score(score):
        score = str(score).lower()
        return score

    @staticmethod
    def read(_file, firstline=True, slices=None):
        data = []
        with open(_file, 'r', encoding='utf8') as f:
            f.seek(0)
            csvreader = csv.reader(f)
            if firstline:
                # Skip the header
                next(csvreader)
            for line in csvreader:
                if slices is not None:
                    line = [line[i] for i in slices]
                    # if len(line) == 1:
                    #     line = line[0]
                data.append(line)
        return data

    @staticmethod
    def get_map(data_path, firstline=True, task=2, delimiter=','):
        data = []
        with open(data_path, 'r', encoding='utf8') as f:
            f.seek(0)
            csvreader = csv.reader(f, delimiter=delimiter)
            if firstline:
                # Skip the header
                next(csvreader)
            for line in csvreader:
                data.append(CSV.task_parser(line, task=task))
        return data

    @staticmethod
    def get_iterator(data_path, firstline=True, task=2, delimiter=','):
        r"""
        Generate an iterator to read CSV file.
        yield texts line-by-line.

        Arguments:
            data_path: a path for the data file.
            firstline: if using headers

        """

        def iterator(start, num_lines):
            with io.open(data_path, encoding="utf8") as f:
                reader = CSV.unicode_csv_reader(f, delimiter=delimiter)
                if firstline and start == 0:
                    next(reader)
                for i, row in enumerate(reader):
                    if i == start:
                        break
                for _ in range(num_lines):
                    yield CSV.task_parser(row, task=task)
                    try:
                        row = next(reader)
                    except StopIteration:
                        f.seek(0)
                        reader = CSV.unicode_csv_reader(f)
                        if firstline:
                            next(reader)
                        row = next(reader)

        return iterator

    @staticmethod
    def _len(_file, firstline=True, delimiter=','):
        count = 0
        with open(_file, 'r', encoding='utf8') as f:
            f.seek(0)
            csvreader = csv.reader(f, delimiter=delimiter)
            if firstline:
                # Skip the header
                next(csvreader)
            for _ in csvreader:
                count += 1
        return count

    @staticmethod
    def write(data, _file):
        if not os.path.exists(os.path.dirname(_file)):
            os.mkdir(os.path.dirname(_file))
        with open(_file, 'w', encoding='utf8') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerows(data)

    @staticmethod
    def unicode_csv_reader(unicode_csv_data, **kwargs):
        r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
        Borrowed and slightly modified from the Python docs:
        https://docs.python.org/2/library/csv.html#csv-examples

        Arguments:
            unicode_csv_data: unicode csv data (see example below)

        Examples:
            >>> from torchtext.utils import unicode_csv_reader
            >>> import io
            >>> with io.open(data_path, encoding="utf8") as f:
            >>>     reader = unicode_csv_reader(f)

        """

        # Fix field larger than field limit error
        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        csv.field_size_limit(maxInt)

        if six.PY2:
            # csv.py doesn't do Unicode; encode temporarily as UTF-8:
            csv_reader = csv.reader(CSV.utf_8_encoder(unicode_csv_data), **kwargs)
            for row in csv_reader:
                # decode UTF-8 back to Unicode, cell by cell:
                yield [cell.decode('utf-8') for cell in row]
        else:
            for line in csv.reader(unicode_csv_data, **kwargs):
                yield line

    @staticmethod
    def utf_8_encoder(unicode_csv_data):
        for line in unicode_csv_data:
            yield line.encode('utf-8')


if __name__ == "__main__":
    filename = "/media/data/classification/datasets/yelp_review_full_csv/train.csv"
    # csvreader = CSV(filename, firstline=True, limit=100)
    data = CSV.read(filename, slices=[0, 1])
