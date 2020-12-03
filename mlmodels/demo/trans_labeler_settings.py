# -*- coding: utf-8 -*-
"""
Created on 19/03/2020
@author duytinvo
"""
# for k, v in vars(args).items():
#     if isinstance(v, str):
#         print(k, "=", "\"" + v + "\"", " #", type(v))
#     else:
#         print(k, "=", v, " #", type(v))

# inputs
label_file = "./data/reviews/processed_csv/labels.txt"  # <class 'str'>
train_file = "./data/reviews/processed_csv/train_res4.csv"  # <class 'str'>
dev_file = "./data/reviews/processed_csv/dev_res4.csv"  # <class 'str'>
test_file = "./data/reviews/processed_csv/test_res4.csv"  # <class 'str'>
overwrite_cache = False  # <class 'bool'>

firstline = False  # <class 'bool'>

# outputs
output_dir = "./data/reviews/trained_model/"  # <class 'str'>
overwrite_output_dir = False  # <class 'bool'>

# pretrained-model
model_type = "bert"  # <class 'str'>
model_name_or_path = "bert-base-uncased"  # <class 'str'>
config_name = ""  # <class 'str'>
tokenizer_name = ""  # <class 'str'>
cache_dir = ""  # <class 'str'>


# tokentization technique
# do_lower_case = False  # <class 'bool'>
# keep_accents = None  # <class 'NoneType'>
# strip_accents = None  # <class 'NoneType'>
use_fast = None  # <class 'NoneType'>

# Training procedure
do_train = True  # <class 'bool'>
evaluate_during_training = False  # <class 'bool'>
save_steps = 500  # <class 'int'>
do_eval = True  # <class 'bool'>
eval_all_checkpoints = False  # <class 'bool'>
do_predict = False  # <class 'bool'>

# Training setup
max_seq_length = 64  # <class 'int'>
num_train_epochs = 2.0  # <class 'float'>
max_steps = -1  # <class 'int'>

per_gpu_train_batch_size = 128  # <class 'int'>
gradient_accumulation_steps = 1  # <class 'int'>
learning_rate = 5e-05  # <class 'float'>
warmup_steps = 0  # <class 'int'>
weight_decay = 0.0  # <class 'float'>
adam_epsilon = 1e-08  # <class 'float'>
max_grad_norm = 1.0  # <class 'float'>
patience = 2  # <class 'int'>
metric = "f1"  # <class 'str'>
per_gpu_eval_batch_size = 8  # <class 'int'>
no_cuda = False  # <class 'bool'>

seed = 42  # <class 'int'>
fp16 = False  # <class 'bool'>
fp16_opt_level = "O1"  # <class 'str'>
local_rank = -1  # <class 'int'>
# server_ip = ""  # <class 'str'>
# server_port = ""  # <class 'str'>

