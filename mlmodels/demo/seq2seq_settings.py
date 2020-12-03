# -*- coding: utf-8 -*-
"""
Created on 2019-06-27
Chata Technologies Inc
"""
import os

data_folder = '/media/data/review_response/'

train_file = os.path.join(data_folder,  'Train.json')
dev_file = os.path.join(data_folder, 'Dev.json')
test_file = os.path.join(data_folder, 'Test.json')

pred_dev_file = data_folder + "regression/Dev.csv"
pred_test_file = data_folder + "regression/Test.csv"

model_dir = data_folder + "trained_model/"
log_file = "logging.txt"
seq2seq_file = "seq2seq.m"
model_args = "translator.args"
tlargs = ""

ed_mode = "gru"
swd_embfile = ""  # '/projects/downloaded_research/syntaxSQL/glove/glove.42B.300d.txt' #''
twd_embfile = ""
optimizer = "RADAM"
metric = "bleu"

wl_th = -1
wcutoff = 50
wd_dropout = 0.5
ed_layers = 2
ed_dropout = 0.1
swd_dim = 100
twd_dim = 100
ed_outdim = 100
final_dropout = 0.5
patience = 32
lr = 0.001
decay_rate = 0.05  # -1.0
max_epochs = 128
batch_size = 16
clip = 5  # -1
teacher_forcing_ratio = 1.0

# CNN at encoder
enc_cnn = False
kernel_size = 3

enc_att = False
ssos = False
seos = False
snl_reqgrad = True  # False

tsos = True
teos = True
twd_reqgrad = True

ed_bidirect = False
wd_padding = False
t_reverse = False
use_cuda = False
tl = False  # Transfer learning
timestamped_subdir = True

# use pre-trained emb for inference
wombat_path = '/projects/wombat_sqlite/glove-sqlite_'
bw = 1
topk = 1
limit = -1
