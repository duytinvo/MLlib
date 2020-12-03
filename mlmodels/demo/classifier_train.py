# -*- coding: utf-8 -*-
"""
Created on 2019-06-27
@author: duytinvo
"""
import sys
import mlmodels.demo.classifier_settings as settings
from mlmodels.training.classifier_model import main
from mlmodels.utils.idx2tensor import Data2tensor


if __name__ == '__main__':
    """
    python -m mlmodels.demo.labeler_train --use_cuda
    """

    Data2tensor.set_randseed(12345)
    # input peripherals
    if '--vocab_file' not in sys.argv:
        sys.argv.extend(['--vocab_file', settings.vocab_file])
    if '--label_file' not in sys.argv:
        sys.argv.extend(['--label_file', settings.label_file])
    if '--train_file' not in sys.argv:
        sys.argv.extend(['--train_file', settings.train_file])
    if '--dev_file' not in sys.argv:
        sys.argv.extend(['--dev_file', settings.dev_file])
    if '--test_file' not in sys.argv:
        sys.argv.extend(['--test_file', settings.test_file])
    if '--firstline' not in sys.argv:
        if settings.firstline:
            sys.argv.append('--firstline')

    # output peripherals
    if '--timestamped_subdir' not in sys.argv:
        if settings.timestamped_subdir:
            sys.argv.append('--timestamped_subdir')
    if '--log_file' not in sys.argv:
        sys.argv.extend(['--log_file', settings.log_file])
    if '--model_dir' not in sys.argv:
        sys.argv.extend(['--model_dir', settings.model_dir])
    if '--model_args' not in sys.argv:
        sys.argv.extend(['--model_args', settings.model_args])
    if '--labeler_file' not in sys.argv:
        sys.argv.extend(['--labeler_file', settings.labeler_file])

    # Transfer learning
    # if '--tl' not in sys.argv:
    #     if settings.tl:
    #         sys.argv.append('--tl')
    if '--tlargs' not in sys.argv:
        sys.argv.extend(['--tlargs', settings.tlargs])

    # vocab & embedding parameters
    if '--tokenize_type' not in sys.argv:
        sys.argv.extend(['--tokenize_type', settings.tokenize_type])
    if '--wl_th' not in sys.argv:
        sys.argv.extend(['--wl_th', str(settings.wl_th)])
    if '--wcutoff' not in sys.argv:
        sys.argv.extend(['--wcutoff', str(settings.wcutoff)])
    if '--ssos' not in sys.argv:
        if settings.ssos:
            sys.argv.append('--ssos')
    if '--seos' not in sys.argv:
        if settings.seos:
            sys.argv.append('--seos')

    if '--swd_embfile' not in sys.argv:
        sys.argv.extend(['--swd_embfile', settings.swd_embfile])
    if '--snl_reqgrad' not in sys.argv:
        if settings.snl_reqgrad:
            sys.argv.append('--snl_reqgrad')
    if '--wd_dropout' not in sys.argv:
        sys.argv.extend(['--wd_dropout', str(settings.wd_dropout)])
    if '--wd_padding' not in sys.argv:
        if settings.wd_padding:
            sys.argv.append('--wd_padding')
    if '--swd_dim' not in sys.argv:
        sys.argv.extend(['--swd_dim', str(settings.swd_dim)])

    # Neural Network parameters
    if '--ed_mode' not in sys.argv:
        sys.argv.extend(['--ed_mode', settings.ed_mode])
    if '--ed_bidirect' not in sys.argv:
        if settings.ed_bidirect:
            sys.argv.append('--ed_bidirect')
    if '--ed_outdim' not in sys.argv:
        sys.argv.extend(['--ed_outdim', str(settings.ed_outdim)])
    if '--ed_layers' not in sys.argv:
        sys.argv.extend(['--ed_layers', str(settings.ed_layers)])
    if '--ed_dropout' not in sys.argv:
        sys.argv.extend(['--ed_dropout', str(settings.ed_dropout)])

    if '--ed_heads' not in sys.argv:
        sys.argv.extend(['--ed_heads', str(settings.ed_heads)])
    if '--ed_activation' not in sys.argv:
        sys.argv.extend(['--ed_activation', settings.ed_activation])
    if '--ed_hismask' not in sys.argv:
        if settings.ed_hismask:
            sys.argv.append('--ed_hismask')

    if '--enc_cnn' not in sys.argv:
        if settings.enc_cnn:
            sys.argv.append('--enc_cnn')
    if '--kernel_size' not in sys.argv:
        sys.argv.extend(['--kernel_size', str(settings.kernel_size)])

    if '--final_dropout' not in sys.argv:
        sys.argv.extend(['--final_dropout', str(settings.final_dropout)])

    # Optimizer parameters
    if '--max_epochs' not in sys.argv:
        sys.argv.extend(['--max_epochs', str(settings.max_epochs)])
    if '--batch_size' not in sys.argv:
        sys.argv.extend(['--batch_size', str(settings.batch_size)])
    if '--patience' not in sys.argv:
        sys.argv.extend(['--patience', str(settings.patience)])

    if '--lr' not in sys.argv:
        sys.argv.extend(['--lr', str(settings.lr)])
    if '--decay_rate' not in sys.argv:
        sys.argv.extend(['--decay_rate', str(settings.decay_rate)])
    if '--clip' not in sys.argv:
        sys.argv.extend(['--clip', str(settings.clip)])

    if '--optimizer' not in sys.argv:
        sys.argv.extend(['--optimizer', settings.optimizer])
    if '--metric' not in sys.argv:
        sys.argv.extend(['--metric', settings.metric])

    if '--use_cuda' not in sys.argv:
        if settings.use_cuda:
            sys.argv.append('--use_cuda')

    main(sys.argv)
