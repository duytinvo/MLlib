# -*- coding: utf-8 -*-
"""
Created on 2019-06-27
@author: duytinvo
"""

data_folder = "./data/reviews/"
# input peripherals
vocab_file = "/media/data/review_response/tokens/bert_level-bpe-vocab.txt"
label_file = data_folder + "processed_csv/labels.txt"
train_file = data_folder + "processed_csv/train_res4.csv"
dev_file = data_folder + "processed_csv/dev_res4.csv"
test_file = data_folder + "processed_csv/test_res4.csv"
pred_test_file = data_folder + "regression/test.csv"
firstline = False

# output peripherals
timestamped_subdir = True
log_file = "logging.txt"
model_dir = data_folder + "trained_model/"
model_args = "labeler.args"
labeler_file = "labeler.m"

# Transfer learning
# tl = False
tlargs = ""
tokenize_type = "splitter"

# vocab & embedding parameters
wl_th = -1
wcutoff = 1
ssos = False
seos = False

swd_embfile = ""
snl_reqgrad = False
wd_dropout = 0.5
wd_padding = False
swd_dim = 300

# Neural Network parameters
ed_mode = "gru"
ed_bidirect = True
ed_outdim = 600
ed_layers = 2
ed_dropout = 0.1

ed_heads = 6
ed_activation = "relu"
# ed_norm = None
ed_hismask = False
if ed_mode == "self_attention":
    assert swd_dim % ed_heads == 0, "input dimension must be divisible by number of heads"

enc_cnn = False  # for CNN at encoder
kernel_size = 3  # for CNN at encoder

final_hidden = 600
final_dropout = 0.5

use_crf = False
se_transitions = False

# Optimizer parameters
max_epochs = 128
batch_size = 32
patience = 64

lr = 0.001
decay_rate = -1.0
clip = -1

optimizer = "RADAM"
metric = "f1"

use_cuda = False

# use pre-trained emb for inference
wombat_path = "/media/data/embeddings/database/glove-sqlite_"
# wombat_path = "/Users/media/data/embeddings/database/glove-sqlite_"

