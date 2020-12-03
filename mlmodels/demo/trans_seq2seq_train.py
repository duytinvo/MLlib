# -*- coding: utf-8 -*-
"""
Created on 18/03/2020
@author duytinvo
"""
import sys
import mlmodels.demo.trans_seq2seq_settings as settings
from mlmodels.training.trans_seq2seq_model import main
from mlmodels.utils.idx2tensor import Data2tensor


if __name__ == '__main__':
    """
    python -m mlmodels.demo.trans_language_train --train_file /media/data/paraphrase/distil/toy_paralex_dev.csv --dev_file /media/data/paraphrase/distil/toy_paralex_dev.csv --test_file /media/data/paraphrase/distil/toy_paralex_test.csv --task response --output_dir /media/data/paraphrase/distil/trained_model/ --model_type t5 --model_name_or_path t5-small --num_train_epochs 5 --per_gpu_train_batch_size 16 --seed 12345 --do_train --overwrite_output_dir --evaluate_during_training --metric loss --save_total_limit 3
    """

    Data2tensor.set_randseed(12345)

    # input peripherals
    if '--label_file' not in sys.argv:
        sys.argv.extend(['--label_file', settings.label_file])
    if '--train_file' not in sys.argv:
        sys.argv.extend(['--train_file', settings.train_file])
    if '--dev_file' not in sys.argv:
        sys.argv.extend(['--dev_file', settings.dev_file])
    if '--test_file' not in sys.argv:
        sys.argv.extend(['--test_file', settings.test_file])

    # output peripherals
    if '--output_dir' not in sys.argv:
        sys.argv.extend(['--output_dir', settings.output_dir])
    if '--overwrite_output_dir' not in sys.argv:
        if settings.overwrite_output_dir:
            sys.argv.append('--overwrite_output_dir')

    # pretrained-model
    if '--model_type' not in sys.argv:
        sys.argv.extend(['--model_type', settings.model_type])
    if '--model_name_or_path' not in sys.argv:
        sys.argv.extend(['--model_name_or_path', settings.model_name_or_path])
    # if '--config_name' not in sys.argv and settings.config_name:
    #     sys.argv.extend(['--config_name', settings.config_name])
    # if '--tokenizer_name' not in sys.argv and settings.tokenizer_name:
    #     sys.argv.extend(['--tokenizer_name', settings.tokenizer_name])
    if '--cache_dir' not in sys.argv and settings.cache_dir:
        sys.argv.extend(['--cache_dir', settings.cache_dir])

    if '--encoder_layers' not in sys.argv:
        sys.argv.extend(['--encoder_layers', str(settings.encoder_layers)])
    if '--decoder_layers' not in sys.argv:
        sys.argv.extend(['--decoder_layers', str(settings.decoder_layers)])

    if '--freeze_encoder_embs' not in sys.argv:
        if settings.freeze_encoder_embs:
            sys.argv.append('--freeze_encoder_embs')

    if '--skip_decoder' not in sys.argv:
        if settings.skip_decoder:
            sys.argv.append('--skip_decoder')

    # Other parameters
    if '--mlm' not in sys.argv:
        if settings.mlm:
            sys.argv.append('--mlm')
    if '--mlm_probability' not in sys.argv:
        sys.argv.extend(['--mlm_probability', str(settings.mlm_probability)])
    if '--block_size' not in sys.argv:
        sys.argv.extend(['--block_size', str(settings.block_size)])

    # Training procedure
    if '--should_continue' not in sys.argv:
        if settings.should_continue:
            sys.argv.append('--should_continue')
    if '--do_train' not in sys.argv:
        if settings.do_train:
            sys.argv.append('--do_train')
    if '--evaluate_during_training' not in sys.argv:
        if settings.evaluate_during_training:
            sys.argv.append('--evaluate_during_training')
    if '--save_steps' not in sys.argv:
        sys.argv.extend(['--save_steps', str(settings.save_steps)])

    if '--do_eval' not in sys.argv:
        if settings.do_eval:
            sys.argv.append('--do_eval')
    if '--eval_all_checkpoints' not in sys.argv:
        if settings.eval_all_checkpoints:
            sys.argv.append('--eval_all_checkpoints')

    if '--do_predict' not in sys.argv:
        if settings.do_predict:
            sys.argv.append('--do_predict')

    if '--do_generate' not in sys.argv:
        if settings.do_generate:
            sys.argv.append('--do_generate')

    # Training setup
    if '--num_train_epochs' not in sys.argv:
        sys.argv.extend(['--num_train_epochs', str(settings.num_train_epochs)])
    if '--max_steps' not in sys.argv:
        sys.argv.extend(['--max_steps', str(settings.max_steps)])
    if '--per_gpu_train_batch_size' not in sys.argv:
        sys.argv.extend(['--per_gpu_train_batch_size', str(settings.per_gpu_train_batch_size)])
    if '--gradient_accumulation_steps' not in sys.argv:
        sys.argv.extend(['--gradient_accumulation_steps', str(settings.gradient_accumulation_steps)])
    if '--optimizer' not in sys.argv:
        sys.argv.extend(['--optimizer', settings.optimizer])
    if '--learning_rate' not in sys.argv:
        sys.argv.extend(['--learning_rate', str(settings.learning_rate)])
    if '--lr_decay' not in sys.argv:
        sys.argv.extend(['--lr_decay', str(settings.lr_decay)])
    if '--warmup_steps' not in sys.argv:
        sys.argv.extend(['--warmup_steps', str(settings.warmup_steps)])
    if '--weight_decay' not in sys.argv:
        sys.argv.extend(['--weight_decay', str(settings.weight_decay)])
    if '--adam_epsilon' not in sys.argv:
        sys.argv.extend(['--adam_epsilon', str(settings.adam_epsilon)])
    if '--max_grad_norm' not in sys.argv:
        sys.argv.extend(['--max_grad_norm', str(settings.max_grad_norm)])
    if '--use_scheduler' not in sys.argv:
        if settings.use_scheduler:
            sys.argv.append('--use_scheduler')

    if '--per_gpu_eval_batch_size' not in sys.argv:
        sys.argv.extend(['--per_gpu_eval_batch_size', str(settings.per_gpu_eval_batch_size)])
    if '--save_total_limit' not in sys.argv:
        sys.argv.extend(['--save_total_limit', str(settings.save_total_limit)])

    if '--use_cuda' not in sys.argv:
        if settings.use_cuda:
            sys.argv.append('--use_cuda')

    if '--seed' not in sys.argv:
        sys.argv.extend(['--seed', str(settings.seed)])
    if '--fp16' not in sys.argv:
        if settings.fp16:
            sys.argv.append('--fp16')
    if '--fp16_opt_level' not in sys.argv:
        sys.argv.extend(['--fp16_opt_level', settings.fp16_opt_level])
    if '--local_rank' not in sys.argv:
        sys.argv.extend(['--local_rank', str(settings.local_rank)])

    if '--patience' not in sys.argv:
        sys.argv.extend(['--patience', str(settings.patience)])
    if '--metric' not in sys.argv:
        sys.argv.extend(['--metric', settings.metric])

    if '--timestamped' not in sys.argv:
        if settings.timestamped:
            sys.argv.append('--timestamped')

    main(sys.argv)
