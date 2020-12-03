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
label_file = "/media/data/np6/dataset/labels2.txt"  # <class 'str'>
train_file = "/media/data/np6/dataset/train.csv"  # <class 'str'>
dev_file = "/media/data/np6/dataset/valid.csv"  # <class 'str'>
test_file = "/media/data/np6/dataset/corpus_test.csv"  # <class 'str'>

# outputs
output_dir = "/media/data/np6/trained_model/"  # <class 'str'>
overwrite_output_dir = False  # <class 'bool'>

# pretrained-model
model_type = "t5"  # <class 'str'>
model_name_or_path = "t5-small"  # <class 'str'>
# config_name = None  # <class 'str'>
# tokenizer_name = None  # <class 'str'>
cache_dir = None  # <class 'str'>

encoder_layers = 6
decoder_layers = 3
freeze_encoder_embs = False
skip_decoder = False

# Other parameters
mlm = False  # <class 'bool'>
mlm_probability = 0.15  # <class 'float'>
block_size = -1  # <class 'int'>

# Training procedure
should_continue = False  # <class 'bool'>
do_train = False  # <class 'bool'>
evaluate_during_training = False  # <class 'bool'>
save_steps = 10000  # <class 'int'>
do_eval = False  # <class 'bool'>
eval_all_checkpoints = False  # <class 'bool'>
do_predict = False  # <class 'bool'>
do_generate = False  # <class 'bool'>

# Training setup
num_train_epochs = 32  # <class 'float'>
max_steps = -1  # <class 'int'>

per_gpu_train_batch_size = 6  # <class 'int'>
gradient_accumulation_steps = 1  # <class 'int'>
optimizer = "radam"
learning_rate = 1e-3
lr_decay = 0.2  # <class 'float'>
warmup_steps = 0  # <class 'int'>
weight_decay = 0.0  # <class 'float'>
adam_epsilon = 1e-08  # <class 'float'>
max_grad_norm = 5.0  # <class 'float'>
use_scheduler = False

per_gpu_eval_batch_size = 24  # <class 'int'>
save_total_limit = 10  # <class 'int'>
use_cuda = True  # <class 'bool'>

seed = 12345  # <class 'int'>
fp16 = False  # <class 'bool'>
fp16_opt_level = "O1"  # <class 'str'>
local_rank = -1  # <class 'int'>

patience = 4  # <class 'int'>
metric = "loss"  # <class 'str'>
timestamped = False
