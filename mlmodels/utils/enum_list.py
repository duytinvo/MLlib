# -*- coding: utf-8 -*-
"""
Created on 2020-04-16
@author duytinvo
"""
import enum
from mlmodels.utils.special_tokens import *
# from enum import IntEnum


class ParseTask(enum.IntEnum):
    classifier = 1
    labeler = 2
    translator = 3

