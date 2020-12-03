# -*- coding: utf-8 -*-
"""
Created on 25/03/2020
@author duytinvo
"""
import os
import itertools


class TXT:
    def __init__(self, filename, source2idx=None, target2idx=None, firstline=True, limit=-1, task=2):
        self.filename = filename
        self.source2idx = source2idx
        self.target2idx = target2idx
        self.length = None
        self.firstline = firstline
        self.limit = limit if limit > 0 else None
        self.task = task

    def __iter__(self):
        with open(self.filename, newline='', encoding='utf-8') as f:
            f.seek(0)
            if self.firstline:
                # Skip the header
                next(f)
            for line in itertools.islice(f, self.limit):
                nl, target = TXT.task_parser(line)
                # assert len(nl) == len(target)
                if self.source2idx is not None:
                    nl = self.source2idx(nl)
                if self.target2idx is not None and target is not None:
                    target = self.target2idx(target)
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
        tokens = line.split()
        if len(tokens) == 1:
            nl = TXT.process_nl(tokens[0])
            target = None
        else:
            if task == 1:
                nl, target = " ".join(tokens[0: -1]), tokens[-1]
            elif task == 2:
                target, nl = tokens[0], " ".join(tokens[1:])
            else:
                raise Exception("not implement %s yet")
            nl = TXT.process_nl(nl)
            target = TXT.process_target(target)
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
    def read(_file, firstline=True):
        if not os.path.exists(os.path.dirname(_file)):
            raise Exception("File does not exist")
        data = []
        with open(_file, 'r', encoding='utf8') as f:
            f.seek(0)
            if firstline:
                # Skip the header
                next(f)
            for line in f:
                data.append(line.strip())
        return data

    @staticmethod
    def _iter(_file, firstline=True):
        if not os.path.exists(os.path.dirname(_file)):
            raise Exception("File does not exist")
        with open(_file, 'r', encoding='utf8') as f:
            f.seek(0)
            if firstline:
                # Skip the header
                next(f)
            for line in f:
                yield line.strip()

    @staticmethod
    def write(data, _file):
        if not os.path.exists(os.path.dirname(_file)):
            os.mkdir(os.path.dirname(_file))
        with open(_file, 'w', encoding='utf8') as f:
            for line in data:
                f.write(line.strip() + "\n")

    @staticmethod
    def get_iterator(data_path, firstline=True, task=2):
        r"""
        Generate an iterator to read CSV file.
        yield texts line-by-line.

        Arguments:
            data_path: a path for the data file.
            firstline: if using headers

        """

        def iterator(start, num_lines):
            with open(data_path, 'r', encoding='utf8') as f:
                if firstline and start == 0:
                    next(f)
                for i, row in enumerate(f):
                    if i == start:
                        break
                for _ in range(num_lines):
                    yield TXT.task_parser(row.strip(), task=task)
                    try:
                        row = next(f)
                    except StopIteration:
                        f.seek(0)
                        if firstline:
                            next(f)
                        row = next(f)

        return iterator

    @staticmethod
    def _len(_file, firstline=True):
        count = 0
        with open(_file, 'r', encoding='utf8') as f:
            f.seek(0)
            if firstline:
                # Skip the header
                next(f)
            for _ in f:
                count += 1
        return count

