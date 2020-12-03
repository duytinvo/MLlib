#!/usr/bin/env bash

echo Pre-process corpus
python corpus_generator.py
#python corpus_generator.py --kb_relatedto --kb_isa --nltk_tok --use_sql --db_id locate --csv_train_file --train_file --csv_dev_file --dev_file --csv_test_file --test_file --schema_file

echo Train the model
python train_demo.py --use_cuda --teacher_forcing_ratio 1.0 --swd_embfile ../embeddings/glove.42B.300d.txt  --optimizer RADAM --swd_inp triple --swd_mode conc --decay_rate 0.05 --clip 5 --lr 0.001 --ed_outdim 600 --sch_att en_hidden --swd_dim 300 --twd_dim 300 --patience 32 --max_epochs 256 --use_sql --batch_size 64 --epochs_per_eval 5 --evaluation_set_ratio 1.0 --epochs_before_first_eval 20 --timestamped_subdir

#--train_file --dev_file --test_file

# echo Regression Test
# python test_demo.py --model_dir ./data/data_locate_nov01/trained_model/ --use_cuda --use_sql --limit -1 --bw 1 --topk 1
