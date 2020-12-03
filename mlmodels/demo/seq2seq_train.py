# -*- coding: utf-8 -*-
"""
Created on 2019-06-27
Chata Technologies Inc
"""
import sys
import mlmodels.demo.seq2seq_settings as np5_settings
from mlmodels.training.seq2seq_model import main


if __name__ == '__main__':
    """
    python train_demo.py --use_cuda --teacher_forcing_ratio .9  --optimizer RADAM --swd_inp dual_tp --swd_mode conc --decay_rate 0.05 --clip 5 --lr 0.001 --ed_outdim 600 --sch_att en_hidden --swd_dim 300 --twd_dim 300 --patience 4 --max_epochs 128 --use_graph --use_pointer --tl --tlargs ./data/trained_model/2019-09-26T22\:13/translator.args   
    """
    from mlmodels.utils.idx2tensor import Data2tensor

    Data2tensor.set_randseed(1234)

    if '--log_file' not in sys.argv:
        sys.argv.extend(['--log_file', np5_settings.log_file])

    if '--train_file' not in sys.argv:
        sys.argv.extend(['--train_file', np5_settings.train_file])
    if '--dev_file' not in sys.argv:
        sys.argv.extend(['--dev_file', np5_settings.dev_file])
    if '--test_file' not in sys.argv:
        sys.argv.extend(['--test_file', np5_settings.test_file])
    if '--pred_dev_file' not in sys.argv:
        sys.argv.extend(['--pred_dev_file', np5_settings.pred_dev_file])
    if '--pred_test_file' not in sys.argv:
        sys.argv.extend(['--pred_test_file', np5_settings.pred_test_file])

    if '--model_dir' not in sys.argv:
        sys.argv.extend(['--model_dir', np5_settings.model_dir])

    if '--model_args' not in sys.argv:
        sys.argv.extend(['--model_args', np5_settings.model_args])

    if '--tlargs' not in sys.argv:
        sys.argv.extend(['--tlargs', np5_settings.tlargs])

    if '--seq2seq_file' not in sys.argv:
        sys.argv.extend(['--seq2seq_file', np5_settings.seq2seq_file])

    if '--ed_mode' not in sys.argv:
        sys.argv.extend(['--ed_mode', np5_settings.ed_mode])
    if '--swd_embfile' not in sys.argv:
        sys.argv.extend(['--swd_embfile', np5_settings.swd_embfile])
    if '--twd_embfile' not in sys.argv:
        sys.argv.extend(['--twd_embfile', np5_settings.twd_embfile])
    if '--optimizer' not in sys.argv:
        sys.argv.extend(['--optimizer', np5_settings.optimizer])
    if '--metric' not in sys.argv:
        sys.argv.extend(['--metric', np5_settings.metric])

    if '--wl_th' not in sys.argv:
        sys.argv.extend(['--wl_th', str(np5_settings.wl_th)])
    if '--wcutoff' not in sys.argv:
        sys.argv.extend(['--wcutoff', str(np5_settings.wcutoff)])
    if '--wd_dropout' not in sys.argv:
        sys.argv.extend(['--wd_dropout', str(np5_settings.wd_dropout)])
    if '--ed_outdim' not in sys.argv:
        sys.argv.extend(['--ed_outdim', str(np5_settings.ed_outdim)])
    if '--ed_layers' not in sys.argv:
        sys.argv.extend(['--ed_layers', str(np5_settings.ed_layers)])
    if '--ed_dropout' not in sys.argv:
        sys.argv.extend(['--ed_dropout', str(np5_settings.ed_dropout)])
    if '--swd_dim' not in sys.argv:
        sys.argv.extend(['--swd_dim', str(np5_settings.swd_dim)])
    if '--twd_dim' not in sys.argv:
        sys.argv.extend(['--twd_dim', str(np5_settings.twd_dim)])
    if '--final_dropout' not in sys.argv:
        sys.argv.extend(['--final_dropout', str(np5_settings.final_dropout)])
    if '--patience' not in sys.argv:
        sys.argv.extend(['--patience', str(np5_settings.patience)])
    if '--lr' not in sys.argv:
        sys.argv.extend(['--lr', str(np5_settings.lr)])
    if '--decay_rate' not in sys.argv:
        sys.argv.extend(['--decay_rate', str(np5_settings.decay_rate)])
    if '--max_epochs' not in sys.argv:
        sys.argv.extend(['--max_epochs', str(np5_settings.max_epochs)])
    if '--batch_size' not in sys.argv:
        sys.argv.extend(['--batch_size', str(np5_settings.batch_size)])

    if '--clip' not in sys.argv:
        sys.argv.extend(['--clip', str(np5_settings.clip)])
    if '--teacher_forcing_ratio' not in sys.argv:
        sys.argv.extend(['--teacher_forcing_ratio', str(np5_settings.teacher_forcing_ratio)])

    if '--enc_cnn' not in sys.argv:
        if np5_settings.enc_cnn:
            sys.argv.append('--enc_cnn')
    if '--kernel_size' not in sys.argv:
        sys.argv.extend(['--kernel_size', str(np5_settings.kernel_size)])

    if '--enc_att' not in sys.argv:
        if np5_settings.enc_att:
            sys.argv.append('--enc_att')

    if '--ssos' not in sys.argv:
        if np5_settings.ssos:
            sys.argv.append('--ssos')
    if '--seos' not in sys.argv:
        if np5_settings.seos:
            sys.argv.append('--seos')

    if '--snl_reqgrad' not in sys.argv:
        if np5_settings.snl_reqgrad:
            sys.argv.append('--snl_reqgrad')

    if '--tsos' not in sys.argv:
        if np5_settings.tsos:
            sys.argv.append('--tsos')
    if '--teos' not in sys.argv:
        if np5_settings.teos:
            sys.argv.append('--teos')
    if '--twd_reqgrad' not in sys.argv:
        if np5_settings.twd_reqgrad:
            sys.argv.append('--twd_reqgrad')

    if '--ed_bidirect' not in sys.argv:
        if np5_settings.ed_bidirect:
            sys.argv.append('--ed_bidirect')
    if '--wd_padding' not in sys.argv:
        if np5_settings.wd_padding:
            sys.argv.append('--wd_padding')
    if '--t_reverse' not in sys.argv:
        if np5_settings.t_reverse:
            sys.argv.append('--t_reverse')
    if '--use_cuda' not in sys.argv:
        if np5_settings.use_cuda:
            sys.argv.append('--use_cuda')
    if '--tl' not in sys.argv:
        if np5_settings.tl:
            sys.argv.append('--tl')
    if '--timestamped_subdir' not in sys.argv:
        if np5_settings.timestamped_subdir:
            sys.argv.append('--timestamped_subdir')

    main(sys.argv)
