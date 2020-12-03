# -*- coding: utf-8 -*-
"""
Created on 2020-05-11
@author duytinvo
"""
from rouge import Rouge
from mlmodels.utils.jsonIO import JSON


def rouge_extraction(read_file, write_file):
    rouge = Rouge()
    data = JSON.load(read_file)
    for line in data:
        scores = rouge.get_scores(line["data"]["text_information"]["comment"],
                                  line["data"]["text_information"]["reply"])[0]
        line["data"]["rouge_scores"] = scores
    JSON.dump(data, write_file)
    return data


if __name__ == "__main__":
    read_file = "/media/data/review_response/Train.json"
    write_file = "/media/data/review_response/Train_rouge.json"
    train_data = rouge_extraction(read_file, write_file)

    read_file = "/media/data/review_response/Dev.json"
    write_file = "/media/data/review_response/Dev_rouge.json"
    dev_data = rouge_extraction(read_file, write_file)

    read_file = "/media/data/review_response/Test.json"
    write_file = "/media/data/review_response/Test_rouge.json"
    test_data = rouge_extraction(read_file, write_file)

