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
train_file = "/media/data/review_response/Train.json"  # <class 'str'>
dev_file = "/media/data/review_response/Dev.json"  # <class 'str'>
test_file = "/media/data/review_response/Test.json"  # <class 'str'>
task = "response"

# outputs
output_dir = "/media/data/review_response/trained_model/"  # <class 'str'>
overwrite_output_dir = False  # <class 'bool'>

# pretrained-model
model_type = "t5"  # <class 'str'>
model_name_or_path = "t5-small"  # <class 'str'>
config_name = None  # <class 'str'>
tokenizer_name = None  # <class 'str'>
cache_dir = None  # <class 'str'>


# Other parameters
mlm = False  # <class 'bool'>
mlm_probability = 0.15  # <class 'float'>
block_size = -1  # <class 'int'>

# Training procedure
should_continue = False  # <class 'bool'>
do_train = True  # <class 'bool'>
evaluate_during_training = False  # <class 'bool'>
save_steps = 10000  # <class 'int'>
do_eval = False  # <class 'bool'>
eval_all_checkpoints = False  # <class 'bool'>
do_predict = False  # <class 'bool'>
do_generate = False

# Training setup
num_train_epochs = 2.0  # <class 'float'>
max_steps = -1  # <class 'int'>

per_gpu_train_batch_size = 4  # <class 'int'>
gradient_accumulation_steps = 1  # <class 'int'>
learning_rate = 5e-05  # <class 'float'>
warmup_steps = 0  # <class 'int'>
weight_decay = 0.0  # <class 'float'>
adam_epsilon = 1e-08  # <class 'float'>
max_grad_norm = 1.0  # <class 'float'>

# patience = 2  # <class 'int'>
# metric = "f1"  # <class 'str'>

per_gpu_eval_batch_size = 4  # <class 'int'>
save_total_limit = 10  # <class 'int'>
no_cuda = False  # <class 'bool'>

seed = 42  # <class 'int'>
fp16 = False  # <class 'bool'>
fp16_opt_level = "O1"  # <class 'str'>
local_rank = -1  # <class 'int'>

patience = 8  # <class 'int'>
metric = "f1"  # <class 'str'>
timestamped = False
