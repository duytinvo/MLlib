# -*- coding: utf-8 -*-
"""
Created on 2020-03-10
@author: duytinvo
Copy from HuggingFace examples and adapt with NNlib
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys
import random
import argparse
import glob
import logging
import os
import time
from typing import Dict, Tuple
import torch
from tqdm import tqdm, trange
import numpy as np
from datetime import datetime
import shutil

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
)

from mlmodels.utils.auxiliary import Timer
from mlmodels.utils.special_tokens import SENSP, SENGE, SOT, EOT, UNK, PAD, NULL, CLS, SEP, MASK, NL
from mlmodels.utils.idx2tensor import Data2tensor
from mlmodels.utils.trad_tokenizer import Tokenizer
from mlmodels.utils.helper import Helper
from mlmodels.metrics.prf1 import APRF1
from mlmodels.metrics.bleu import compute_bleu
from mlmodels.metrics.string_match import compute_string_match
from mlmodels.modules.t5_multi_sampling import T5ForConditionalGeneration
from mlmodels.metrics.metrics_loader import MetricsFactory

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TransLanguageModel(object):
    def __init__(self, args):
        self.args = args
        self.task = 1 if self.args.task == "sentiment" else (2 if self.args.task == 'response' else 3)

        if self.args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not self.args.mlm:
            raise ValueError("BERT and RoBERTa-like models do not have LM heads but masked LM heads. "
                             "They must be run using the --mlm flag (masked language modeling).")

        if self.args.dev_file is None and self.args.do_eval:
            raise ValueError("Cannot do evaluation without an evaluation data file. "
                             "Either supply a file to --dev_file or remove the --do_eval argument.")

        if self.args.should_continue:
            if self.args.max_steps < 0:
                checkpoint_prefix = "checkpoint_by-epoch"
            else:
                checkpoint_prefix = "checkpoint_by-step"
            # print(checkpoint_prefix)
            sorted_checkpoints = Helper._sorted_checkpoints(self.args.output_dir, checkpoint_prefix=checkpoint_prefix)
            if len(sorted_checkpoints) == 0:
                raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
            else:
                self.args.model_name_or_path = sorted_checkpoints[-1]

        if (os.path.exists(self.args.output_dir) and os.listdir(self.args.output_dir) and
                self.args.do_train and not self.args.overwrite_output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty. "
                             "Use --overwrite_output_dir to overcome.".format(self.args.output_dir))

        # Setup CUDA, GPU & distributed training
        if self.args.local_rank == -1 or self.args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.args.n_gpu = 0 if self.args.no_cuda else torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.args.n_gpu = 1
        self.args.device = device

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                       self.args.local_rank, device, self.args.n_gpu, bool(self.args.local_rank != -1),
                       self.args.fp16, )

        self.config = None
        self.tokenizer = None
        self.model = None

        self.source2idx = None
        self.target2idx = None
        self.collate = None
        self.pad_label_id = torch.nn.CrossEntropyLoss().ignore_index
        self.pad_id = 0

        self.optimizer = None
        self.scheduler = None
        self.tb_writer = None

    def model_init(self):
        # Set seed
        Data2tensor.set_randseed(self.args.seed, self.args.n_gpu)
        # Load pretrained model and tokenizer
        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed training download model & vocab
        # Load the configuration file
        if self.args.config_name:
            # self.config = self.args.config_class.from_pretrained(self.args.config_name,
            #                                                      cache_dir=self.args.cache_dir)
            self.config = AutoConfig.from_pretrained(self.args.config_name, cache_dir=self.args.cache_dir)
        elif self.args.model_name_or_path:
            # self.config = self.args.config_class.from_pretrained(self.args.model_name_or_path,
            #                                                      cache_dir=self.args.cache_dir)
            self.config = AutoConfig.from_pretrained(self.args.model_name_or_path, cache_dir=self.args.cache_dir)
        else:
            # self.config = self.args.config_class()
            raise ValueError(
                "You are instantiating a new config instance from scratch. "
                "This is not supported, but you can do it from another script, "
                "save it, and load it from here, using --config_name")

        # Load the tokenizer file
        if self.args.tokenizer_name:
            # self.tokenizer = self.args.tokenizer_class.from_pretrained(self.args.tokenizer_name,
            #                                                            cache_dir=self.args.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name, cache_dir=self.args.cache_dir)
        elif self.args.model_name_or_path:
            # self.tokenizer = self.args.tokenizer_class.from_pretrained(self.args.model_name_or_path,
            #                                                            cache_dir=self.args.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, cache_dir=self.args.cache_dir)
        else:
            raise ValueError("You are instantiating a new tokenizer from scratch. "
                             "This is not supported, but you can do it from another script, save it,"
                             "and load it from here, using --tokenizer_name")
        # Add special tokens for auto responding
        try:
            if self.tokenizer._bos_token is None:
                self.tokenizer.add_special_tokens({"bos_token": SOT})
        except AssertionError:
            self.tokenizer.add_tokens([SOT])

        try:
            if self.tokenizer._eos_token is None:
                self.tokenizer.add_special_tokens({"eos_token": EOT})
        except AssertionError:
            self.tokenizer.add_tokens([EOT])

        try:
            if self.tokenizer._unk_token is None:
                self.tokenizer.add_special_tokens({"unk_token": UNK})
        except AssertionError:
            self.tokenizer.add_tokens([UNK])

        try:
            if self.tokenizer._sep_token is None:
                self.tokenizer.add_special_tokens({"sep_token": SEP})
        except AssertionError:
            self.tokenizer.add_tokens([SEP])

        try:
            if self.tokenizer._pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": PAD})
        except AssertionError:
            self.tokenizer.add_tokens([PAD])

        try:
            if self.tokenizer._cls_token is None:
                self.tokenizer.add_special_tokens({"cls_token": CLS})
        except AssertionError:
            self.tokenizer.add_tokens([CLS])

        try:
            if self.tokenizer._mask_token is None:
                self.tokenizer.add_special_tokens({"mask_token": MASK})
        except AssertionError:
            self.tokenizer.add_tokens([MASK])

        self.tokenizer.add_tokens([SENSP, SENGE, NULL, NL])

        # Load the model file
        if self.args.model_name_or_path:
            # self.model = self.args.model_class.from_pretrained(self.args.model_name_or_path,
            #                                                    from_tf=bool(".ckpt" in self.args.model_name_or_path),
            #                                                    config=self.config, cache_dir=self.args.cache_dir, )
            if 't5' in self.args.model_name_or_path:
                self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name_or_path)
            else:
                self.model = AutoModelWithLMHead.from_pretrained(self.args.model_name_or_path,
                                                             from_tf=bool(".ckpt" in self.args.model_name_or_path),
                                                             config=self.config, cache_dir=self.args.cache_dir, )
        else:
            logger.info("Training new model from scratch")
            # self.model = self.args.model_class(config=self.config)
            self.model = AutoModelWithLMHead.from_config(self.config)

        self.model.to(self.args.device)

        if self.args.local_rank == 0:
            torch.distributed.barrier()
            # End of barrier to make sure only the first process in distributed training download model & vocab
        if self.args.model_type == "t5":
            if self.args.block_size <= 0:
                self.args.block_size = self.tokenizer.max_len - 1
                # Our input block size will be the max possible for the model
            else:
                self.args.block_size = min(self.args.block_size, self.tokenizer.max_len) - 1
        else:
            if self.args.block_size <= 0:
                self.args.block_size = self.tokenizer.max_len // 2 - 1
                # Our input block size will be the max possible for the model
            else:
                self.args.block_size = min(self.args.block_size, self.tokenizer.max_len // 2) - 1

        data_block_size = self.args.block_size - (self.tokenizer.max_len - self.tokenizer.max_len_single_sentence)
        logger.info("Training/evaluation parameters %s", self.args)

        self.source2idx = TransLanguageModel.tok2id(self.tokenizer, data_block_size, eos=True, special_tokens=True)
        if self.task == 2:
            self.target2idx = self.source2idx
        else:
            self.target2idx = TransLanguageModel.tok2id(self.tokenizer, data_block_size, eos=False,
                                                        special_tokens=False)
        if self.tokenizer.pad_token is not None:
            self.pad_id = self.tokenizer.pad_token_id
        if "gpt2" in self.args.model_type:
            self.pad_label_id = self.pad_id
        self.collate = Helper.collate_fn(padding_value=self.pad_id, target_padding_value=self.pad_label_id,
                                         batch_first=True, task=self.task)
        pass

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "Remove the --mlm flag if you want to use this tokenizer.")

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.args.mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                               for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def train_batch(self, epoch_iterator, tr_loss, logging_loss, global_step=0, steps_trained_in_current_epoch=0):
        self.model.train()
        for step, batch in enumerate(epoch_iterator):
            self.model.zero_grad()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            if self.args.model_type == "t5":
                assert len(batch) == 2, "Must return a pair of input-output"
                inputs, labels = batch[0], batch[1]
            else:
                if len(batch) == 2:
                    # batch, _ = batch
                    batch = torch.cat((batch[0], batch[1]), dim=1)
                inputs, labels = self.mask_tokens(batch) if self.args.mlm else (batch, batch)
            inputs = inputs.to(self.args.device)
            labels = labels.to(self.args.device)

            if self.args.model_type == "t5":
                outputs = self.model(input_ids=inputs, lm_labels=labels)
            else:
                outputs = self.model(inputs, masked_lm_labels=labels) if self.args.mlm else self.model(inputs,
                                                                                                       labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                global_step += 1

                if self.args.local_rank in [-1, 0] and self.args.max_steps > 0 \
                        and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    # Log metrics
                    if self.args.local_rank == -1 and self.args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.evaluate_batch(dev_file=self.args.dev_file)
                        for key, value in results.items():
                            self.tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    self.tb_writer.add_scalar("lr", self.scheduler.get_lr()[0], global_step)
                    self.tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.args.save_steps, global_step)
                    logging_loss = tr_loss

                    # Save model checkpoint
                    checkpoint_prefix = "checkpoint_by-step"
                    output_dir = os.path.join(self.args.output_dir, "{}_{}".format(checkpoint_prefix, global_step))
                    checkpoints_to_be_deleted = Helper.save_model(do_train=self.args.do_train,
                                                                  local_rank=self.args.local_rank,
                                                                  output_dir=output_dir,
                                                                  model=self.model, tokenizer=self.tokenizer,
                                                                  optimizer=self.optimizer, scheduler=self.scheduler,
                                                                  save_total_limit=self.args.save_total_limit,
                                                                  args=self.args,
                                                                  checkpoint_prefix=checkpoint_prefix)

                    logger.info("Saving model checkpoint to %s", output_dir)
                    logger.info("Saving optimizer and scheduler states to %s\n", output_dir)
                    for checkpoint in checkpoints_to_be_deleted:
                        logger.info("Order checkpoint [{}] deleted due to args.save_total_limit".format(checkpoint))
                        logger.info("Older checkpoint [{}] deleted due to args.save_total_limit".format(checkpoint))

            if global_step > self.args.max_steps > 0:
                epoch_iterator.close()
                break
        return tr_loss, logging_loss, global_step, steps_trained_in_current_epoch

    def train(self):
        """
        Train the model
        """
        self.model_init()

        if self.args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter()

        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)

        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed training process the dataset,
            # and the others will use the cache

        train_dataloader, train_numlines = Helper.build_dataloader(file=self.args.train_file, task=self.task,
                                                                   source2idx=self.source2idx,
                                                                   target2idx=self.target2idx,
                                                                   batch_size=self.args.train_batch_size,
                                                                   firstline=self.args.firstline, collate=self.collate)

        if self.args.local_rank == 0:
            torch.distributed.barrier()
            # End of barrier to make sure only the first process in distributed training process the dataset,
            # and the others will use the cache

        num_batchs = (train_numlines // self.args.train_batch_size) + 1 \
            if train_numlines % self.args.train_batch_size != 0 else train_numlines // self.args.train_batch_size
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    num_batchs // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = num_batchs // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        self.optimizer, self.scheduler = Helper.load_optimizer(
            optimizer=self.args.optimizer, model_named_parameters=self.model.named_parameters(),
            learning_rate=self.args.learning_rate, adam_epsilon=self.args.adam_epsilon, t_total=t_total,
            warmup_steps=self.args.warmup_steps, weight_decay=self.args.weight_decay,
            model_name_or_path=self.args.model_name_or_path)

        # check apex
        if self.args.fp16:
            try:
                global apex
                import apex
                # from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer,
                                                             opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                                   output_device=self.args.local_rank,
                                                                   find_unused_parameters=True)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_numlines)
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.args.train_batch_size * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if self.args.model_name_or_path and os.path.exists(self.args.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.args.model_name_or_path.split("-")[-1].split("/")[0]
                if checkpoint_suffix.startswith("step"):
                    global_step = int(checkpoint_suffix.split("_")[-1])
                    epochs_trained = global_step // (num_batchs // self.args.gradient_accumulation_steps)
                    steps_trained_in_current_epoch = global_step % (num_batchs // self.args.gradient_accumulation_steps)
                else:
                    epochs_trained = int(checkpoint_suffix.split("_")[-1])
                    global_step = epochs_trained * (num_batchs // self.args.gradient_accumulation_steps)
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the %d epoch", steps_trained_in_current_epoch,
                            epochs_trained+1)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0

        # model_to_resize = self.model.module if hasattr(self.model, "module") else self.model
        # Take care of distributed/parallel training
        # model_to_resize.resize_token_embeddings(len(self.tokenizer))

        train_iterator = trange(epochs_trained, int(self.args.num_train_epochs), desc="Epoch",
                                disable=self.args.local_rank not in [-1, 0])
        Data2tensor.set_randseed(self.args.seed, self.args.n_gpu)  # Added here for reproducibility
        nepoch_no_imprv = 0
        best_dev = np.inf if self.args.metric == "loss" else -np.inf
        ep_count = epochs_trained
        if self.args.overwrite_output_dir and not self.args.should_continue and os.path.exists(self.args.output_dir):
            shutil.rmtree(self.args.output_dir)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0],
                                  total=num_batchs)
            tr_loss, logging_loss, \
            global_step, steps_trained_in_current_epoch = self.train_batch(epoch_iterator,
                                                                           tr_loss,
                                                                           logging_loss,
                                                                           global_step,
                                                                           steps_trained_in_current_epoch)
            ep_count += 1
            if self.args.local_rank in [-1, 0] and self.args.max_steps < 0:
                self.tb_writer.add_scalar("lr", self.scheduler.get_lr()[0], global_step)
                self.tb_writer.add_scalar("loss", tr_loss / global_step, global_step)

                # Save model checkpoint
                checkpoint_prefix = "checkpoint_by-epoch"
                output_dir = os.path.join(self.args.output_dir, "{}_{}".format(checkpoint_prefix, ep_count))
                checkpoints_to_be_deleted = Helper.save_model(do_train=self.args.do_train,
                                                              local_rank=self.args.local_rank, output_dir=output_dir,
                                                              model=self.model, tokenizer=self.tokenizer,
                                                              optimizer=self.optimizer,
                                                              scheduler=self.scheduler,
                                                              save_total_limit=self.args.save_total_limit,
                                                              args=self.args,
                                                              checkpoint_prefix=checkpoint_prefix)
                logger.info("Saving model checkpoint to %s", output_dir)
                logger.info("Saving optimizer and scheduler states to %s\n", output_dir)
                for checkpoint in checkpoints_to_be_deleted:
                    logger.info("Older checkpoint [{}] deleted due to args.save_total_limit".format(checkpoint))

                # Only evaluate when single GPU otherwise metrics may not average well
                if self.args.local_rank == -1 and self.args.evaluate_during_training:
                    results = self.evaluate_batch(self.args.dev_file,
                                                  prefix="of {} used the epoch_{} model".format(self.args.dev_file,
                                                                                                ep_count))
                    for key, value in results.items():
                        self.tb_writer.add_scalar("eval_{}".format(key), value, ep_count)

                    dev_metric = results["perplexity"] if self.args.metric == "loss" else results["f1"]
                    cond = dev_metric < best_dev if self.args.metric == "loss" else dev_metric > best_dev
                    if cond:
                        logger.info("New improvement at %d", ep_count)
                        # Save the best model
                        # TODO: copying the folder instead of saving
                        checkpoints_to_be_deleted = Helper.save_model(do_train=self.args.do_train,
                                                                      local_rank=self.args.local_rank,
                                                                      output_dir=self.args.output_dir,
                                                                      model=self.model,
                                                                      tokenizer=self.tokenizer,
                                                                      optimizer=self.optimizer,
                                                                      scheduler=self.scheduler,
                                                                      save_total_limit=self.args.save_total_limit,
                                                                      args=self.args)
                        logger.info("Saving the best model to %s from checkpoint %s", self.args.output_dir, output_dir)

                        best_dev = dev_metric
                        nepoch_no_imprv = 0
                    else:
                        nepoch_no_imprv += 1
                        if nepoch_no_imprv >= self.args.patience:
                            # Testing
                            if self.args.do_predict:
                                if not os.path.exists(self.args.output_dir):
                                    logger.error("%s does not exist", self.args.output_dir)

                                self.model, self.tokenizer, self.optimizer, self.scheduler = Helper.load_model(
                                    output_dir=self.args.output_dir,
                                    device=self.args.device,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler)
                                test_results = self.evaluate_batch(self.args.test_file,
                                                                   prefix="of {} used the epoch_{} model".
                                                                   format(self.args.test_file, ep_count))
                                for key, value in test_results.items():
                                    self.tb_writer.add_scalar("eval_{}".format(key), value, ep_count)
                            self.tb_writer.close()
                            return global_step, tr_loss / global_step

            if global_step > self.args.max_steps > 0:
                train_iterator.close()
                break

        if self.args.local_rank in [-1, 0]:
            if not self.args.evaluate_during_training:
                # Save the model of the last epoch as the best model
                # TODO: copying the folder instead of saving
                checkpoints_to_be_deleted = Helper.save_model(do_train=self.args.do_train,
                                                              local_rank=self.args.local_rank,
                                                              output_dir=self.args.output_dir,
                                                              model=self.model,
                                                              tokenizer=self.tokenizer,
                                                              optimizer=self.optimizer,
                                                              scheduler=self.scheduler,
                                                              save_total_limit=self.args.save_total_limit,
                                                              args=self.args)
                logger.info("Saving the current model as the best model to %s", self.args.output_dir)
            if self.args.do_predict:
                if not os.path.exists(self.args.output_dir):
                    logger.error("%s does not exist", self.args.output_dir)
                self.model, self.tokenizer, self.optimizer, self.scheduler = Helper.load_model(
                    output_dir=self.args.output_dir,
                    device=self.args.device,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler)
                test_results = self.evaluate_batch(self.args.test_file,
                                                   prefix="of {} used the epoch_{} model".
                                                   format(self.args.test_file, ep_count))
                for key, value in test_results.items():
                    self.tb_writer.add_scalar("eval_{}".format(key), value, ep_count)
            self.tb_writer.close()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)
        return global_step, tr_loss / global_step

    def evaluate_batch(self, dev_file, prefix="") -> Dict:
        eval_output_dir = self.args.output_dir
        if self.args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly

        eval_dataloader, num_lines = Helper.build_dataloader(file=dev_file, task=self.task, source2idx=self.source2idx,
                                                             target2idx=self.target2idx,
                                                             batch_size=self.args.eval_batch_size,
                                                             firstline=self.args.firstline, collate=self.collate)
        num_batchs = (num_lines // self.args.eval_batch_size) + 1 \
            if num_lines % self.args.eval_batch_size != 0 else num_lines // self.args.eval_batch_size
        # multi-gpu evaluate
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", num_lines)
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        nl_tokens = []
        candidate = []
        reference = []
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating", total=num_batchs):
            if self.args.model_type == "t5":
                assert len(batch) == 2, "Must return a pair of input-output"
                inputs, labels = batch[0], batch[1]
            else:
                if len(batch) == 2:
                    # batch, _ = batch
                    batch = torch.cat((batch[0], batch[1]), dim=1)
                inputs, labels = self.mask_tokens(batch) if self.args.mlm else (batch, batch)

            inputs = inputs.to(self.args.device)
            attention_mask = inputs != self.pad_id
            inputs_len = attention_mask.sum(dim=-1)
            labels = labels.to(self.args.device)
            labels_mask = labels != self.pad_label_id
            labels_len = labels_mask.sum(dim=-1)

            with torch.no_grad():
                if self.args.model_type == "t5":
                    outputs = self.model(input_ids=inputs, lm_labels=labels)
                else:
                    outputs = self.model(inputs, masked_lm_labels=labels) \
                        if self.args.mlm else self.model(inputs, labels=labels)
                lm_loss = outputs[0]
                logits = outputs[1]
                pred_labels = logits.argmax(dim=-1)
                eval_loss += lm_loss.mean().item()
            preds = [ids[:l] for ids, l in zip(pred_labels.tolist(), labels_len.tolist())]
            out_label_ids = [ids[:l] for ids, l in zip(labels.tolist(), labels_len.tolist())]
            nl_list = [ids[:l] for ids, l in zip(inputs.tolist(), inputs_len.tolist())]
            if self.task == 1:
                candidate.extend(preds)
                reference.extend(out_label_ids)
            else:
                label_words = self.tokenizer.batch_decode(out_label_ids)
                predict_words = self.tokenizer.batch_decode(preds)

                nl_token = self.tokenizer.batch_decode(nl_list)
                # reference = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                reference.extend([words.split() for words in label_words])
                # candidate = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                candidate.extend([words.split() for words in predict_words])
                nl_tokens.extend([words.split() for words in nl_token])

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity.item()}
        if self.task == 1:
            precision, recall, f1, acc = APRF1.sklearn(reference, candidate)
            result.update({"precision": precision, "recall": recall, "f1": f1, "accuracy": acc})
        else:
            rand_idx = random.randint(0, len(reference) - 1)
            print("\nRANDOMLY sampling: ")
            print("\t- A NL question: ", " ".join(nl_tokens[rand_idx]))
            print("\t- A LABEL query: ", " ".join(reference[rand_idx]))
            print("\t- A PREDICTED query: ", " ".join(candidate[rand_idx]), "\n")
            bleu_score = compute_bleu(list(zip(reference)), candidate)
            result.update({"bleu_score": bleu_score[0]})

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        return result

    def evaluate(self, eval_output_file="eval_results.txt"):
        # Evaluation
        results = {}
        checkpoints = [self.args.output_dir]
        if self.args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in
                               sorted(glob.glob(self.args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoint.split("-")) > 1 else "best"
            self.model, self.tokenizer, self.optimizer, self.scheduler = Helper.load_model(
                output_dir=checkpoint,
                device=self.args.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler)
            result = self.evaluate_batch(dev_file=self.args.dev_file,
                                         prefix="of {} used the {} model".format(self.args.dev_file, global_step))
            # if global_step != "best":
            #     result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update({global_step: result})
        output_dev_file = os.path.join(self.args.output_dir, eval_output_file)
        with open(output_dev_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("- {}:\n".format(key))
                for k in sorted(results[key].keys()):
                    writer.write("\t{} = {}\n".format(k, str(results[key][k])))
        return results

    @staticmethod
    def tok2id(pretrained_tokenizer, block_size=512, eos=True, special_tokens=True):
        """
        :param pretrained_tokenizer: pretrained tokenizer
        :param block_size: max length of a sequence
        :param eos: add an end of sequence token
        :param special_tokens: add specific token from the pretrained tokenizer
        :return: a token2index function
        """

        def f(sequence, eos=eos, sep=False):
            # TODO: add more code to handle special tokens
            tokens = pretrained_tokenizer.tokenize(sequence)[:block_size]
            if eos:
                assert pretrained_tokenizer.eos_token, "There is no END OF SEQUENCE token"
                tokens += [pretrained_tokenizer.eos_token]
            if sep:
                assert pretrained_tokenizer.sep_token, "There is no SEP OF SEQUENCE token"
                tokens += [pretrained_tokenizer.sep_token]
            tokenized_ids = pretrained_tokenizer.convert_tokens_to_ids(tokens)
            if special_tokens:
                tokenized_ids = pretrained_tokenizer.build_inputs_with_special_tokens(tokenized_ids)
            return tokenized_ids

        return f

    def generate(self, eval_file, eval_output_file="generate_eval_results.txt"):
        # Evaluation
        results = {}
        checkpoints = [self.args.output_dir]
        if self.args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in
                               sorted(glob.glob(self.args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Generate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoint.split("-")) > 1 else "best"

            # Check output directory
            if not os.path.exists(checkpoint):
                logger.error("%s does not exist", checkpoint)

            self.model, self.tokenizer, self.optimizer, self.scheduler = Helper.load_model(
                output_dir=checkpoint,
                device=self.args.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler)
            result = self.generate_batch(dev_file=eval_file,
                                         prefix="of {} used the {} model".format(eval_file, global_step))
            # if global_step != "best":
            #     result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update({global_step: result})
        output_dev_file = os.path.join(self.args.output_dir, eval_output_file)
        with open(output_dev_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("- {}:\n".format(key))
                for k in sorted(results[key].keys()):
                    writer.write("\t{} = {}\n".format(k, str(results[key][k])))
        return results

    def generate_batch(self, dev_file, prefix="") -> Dict:
        eval_output_dir = self.args.output_dir
        if self.args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir, exist_ok=True)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_dataloader, num_lines = Helper.build_dataloader(file=dev_file, task=self.task, source2idx=self.source2idx,
                                                             target2idx=self.target2idx,
                                                             batch_size=self.args.eval_batch_size,
                                                             firstline=self.args.firstline, collate=self.collate)
        num_batchs = (num_lines // self.args.eval_batch_size) + 1 \
            if num_lines % self.args.eval_batch_size != 0 else num_lines // self.args.eval_batch_size
        # multi-gpu evaluate
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running generation {} *****".format(prefix))
        logger.info("  Num examples = %d", num_lines)
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        nb_eval_steps = 0
        nl_tokens = []
        reference = []
        candidate = []
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Generating", total=num_batchs):
            if self.args.model_type == "t5":
                assert len(batch) == 2, "Must return a pair of input-output"
                inputs, labels = batch[0], batch[1]
            else:
                if len(batch) == 2:
                    # batch, _ = batch
                    batch = torch.cat((batch[0], batch[1]), dim=1)
                inputs, labels = self.mask_tokens(batch) if self.args.mlm else (batch, batch)

            inputs = inputs.to(self.args.device)
            attention_mask = inputs != self.pad_id
            # inputs_len = attention_mask.sum(dim=-1)
            labels = labels.to(self.args.device)
            # labels = labels.clone()
            decoder_attention_mask = labels != self.pad_label_id
            labels_len = decoder_attention_mask.sum(dim=-1)
            max_length = int(1.3 * labels_len.max().item())
            with torch.no_grad():
                outputs = self.model.generate(input_ids=inputs,
                                              max_length=max_length,
                                              temperature=self.config.temperature,
                                              top_k=self.config.top_k,
                                              top_p=self.config.top_p,
                                              repetition_penalty=self.config.repetition_penalty,
                                              num_beams=1,
                                              do_sample=self.config.do_sample,
                                              num_return_sequences=self.config.num_return_sequences,
                                              bos_token_id=self.tokenizer.convert_tokens_to_ids(
                                                  self.tokenizer.eos_token),
                                              # pad_token_id=self.pad_token_id,
                                              eos_token_id=self.tokenizer.convert_tokens_to_ids(
                                                  self.tokenizer.bos_token),
                                              )

                pred_labels = outputs
                # probs = outputs[1]
                # TODO: fix this bug IndexError: list index out of range
                preds = pred_labels.detach().cpu().tolist()
                out_label_ids = labels.detach().cpu().tolist()
                nl_list = inputs.detach().cpu().tolist()

                self.tokenizer.i2tw = Tokenizer.reversed_dict(self.tokenizer.get_vocab())
                label_words = Tokenizer.decode_batch(out_label_ids, self.tokenizer.i2tw, 2)
                label_words = [words[:i] if EOT not in words else words[: words.index(EOT)]
                               for words, i in zip(label_words, labels_len.tolist())]

                predict_words = Tokenizer.decode_batch(preds, self.tokenizer.i2tw, 2)
                # Remove SOT
                predict_words = [words if words[0] != SOT else words[1:]
                                 for words in predict_words]
                predict_words = [words if EOT not in words else words[: words.index(EOT)]
                                 for words in predict_words]
                nl_token = self.tokenizer.batch_decode(nl_list, skip_special_tokens=True)
                # reference = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                reference.extend(label_words)
                # candidate = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                candidate.extend(predict_words)
                nl_tokens.extend(nl_token)

            nb_eval_steps += 1

        result = {}
        rand_idx = random.randint(0, len(reference) - 1)
        print("\nRANDOMLY sampling: ")
        print("\t- A NL question: ", nl_tokens[rand_idx])
        print("\t- A LABEL query: ", " ".join(reference[rand_idx]))
        print("\t- A PREDICTED query: ", " ".join(candidate[rand_idx]), "\n")
        bleu_score = compute_bleu(list(zip(reference)), candidate)
        string_match = compute_string_match(reference, candidate)

        result.update({"bleu_"
                       "score": bleu_score[0]})
        result.update({"string_match_score": string_match})

        logger.info("***** Generate results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        return result


def main(argv):
    now = time.time()
    parser = argparse.ArgumentParser(argv)
    # input peripherals
    parser.add_argument("--train_file", default="../../../media/data/paraphrase/manual_paraphrases_dev.csv", type=str,
                        help="The input training data file (a text file)."
                        )
    parser.add_argument("--dev_file", default="../../../media/data/paraphrase/manual_paraphrases_dev.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).", )
    parser.add_argument("--test_file", default="../../../media/data/paraphrase/manual_paraphrases_dev.csv", type=str,
                        help="An optional input test data file to test the perplexity on (a text file).", )
    parser.add_argument("--task", default="conditional", type=str, choices=["response", "sentiment", "conditional"],
                        help="Select a task to solve",)
    parser.add_argument("--firstline", action='store_true', default=False,
                        help="labelled files having a header" )
    parser.add_argument("--conditional_metric", default='rouge', type=str, choices=["rouge"],
                        help="Metric to condition the input sequence during training and leave it None if using "
                             "precomputed value")

    # output peripherals
    parser.add_argument("--output_dir", default="/media/data/review_response/trained_model", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    # pretrained-model
    parser.add_argument("--model_type", type=str, default="t5",
                        help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--model_name_or_path", default="t5-small", type=str,
                        help="The model checkpoint for weights initialization. "
                             "Leave None if you want to train a model from scratch.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path. "
                             "If both are None, initialize a new config.", )
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. "
                             "If both are None, initialize a new tokenizer.", )
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 "
                             "(instead of the default one)", )

    # Other parameters
    parser.add_argument("--mlm", action="store_true",
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. "
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    # Training procedure
    parser.add_argument("--should_continue", action="store_true",
                        help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.", default=True)
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_eval", action="store_true", help="Whether to run metrics on the dev set.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as "
                             "model_name_or_path ending and ending with step number")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_generate", action="store_true",
                        help="Whether to run generation on both dev and test sets.")

    # Training setup
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimizer", default="adamw", type=str, help="An optimizer method", )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, "
                             "does not delete by default")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping if no improvement after patience epoches")
    parser.add_argument("--metric", type=str, default="loss", choices=["f1", "loss"],
                        help="Optimized criterion (loss or f1)")
    parser.add_argument("--timestamped", action='store_true', default=False,
                        help="Save models in timestamped subdirectory")
    args = parser.parse_args()

    if args.conditional_metric != None:
        logger.info("Computing {} for the dataset".format(args.conditional_metric))
        metric_factory = MetricsFactory()
        metric = metric_factory.create_metrics(args.conditional_metric, '../metrics/ROUGE-1.5.5/' )
        from mlmodels.utils.csvIO import CSV
        files = {'train_file': args.train_file, 'dev_file': args.dev_file, 'test_file': args.test_file}
        new_files_dir = {}
        for file_name, file in files.items():
            new_data = []
            for data in CSV.read(file):
                score = metric.evaluate_example(data[0], data[1])
                new_data.append([data[0], score, data[1]])
                break
            start_index = args.train_file.rfind('/')
            new_file_name = file[:start_index] + '/conditional_file' + file[start_index:]
            CSV.write(new_data, new_file_name)
            new_files_dir[file_name] = new_file_name
        args.train_file = new_files_dir['train_file']
        args.dev_file = new_files_dir['dev_file']
        args.test_file = new_files_dir['test_file']

    if args.timestamped and args.do_train:
        # args.output_dir = os.path.abspath(os.path.join(args.output_dir, ".."))
        sub_folder = datetime.now().isoformat(sep='-', timespec='minutes').replace(":", "-").replace("-", "_")
        args.output_dir = os.path.join(args.output_dir, sub_folder)

    lm_model = TransLanguageModel(args)

    # Training
    train_start = time.time()
    if args.do_train:
        global_step, tr_loss = lm_model.train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info("Training time = %s", Timer.asHours(time.time() - train_start))

    # Evaluation
    evaluate_start = time.time()
    results = {}
    # Run evaluation separately
    if args.do_eval and args.local_rank in [-1, 0]:
        results = lm_model.evaluate(eval_output_file="dev_results.txt")
    logger.info("Evaluate time = %s", Timer.asHours(time.time() - evaluate_start))

    # Testing
    test_start = time.time()
    # Running testing separately
    if args.do_predict and args.local_rank in [-1, 0]:
        results = lm_model.evaluate(eval_output_file="test_results.txt")
    logger.info("Test time = %s", Timer.asHours(time.time() - test_start))

    # Testing
    generate_start = time.time()
    if args.do_generate and args.local_rank in [-1, 0]:
        _ = lm_model.generate(eval_file=args.dev_file, eval_output_file="generate_dev_results.txt")
        _ = lm_model.generate(eval_file=args.test_file, eval_output_file="generate_test_results.txt")
    logger.info("Generate time = %s", Timer.asHours(time.time() - generate_start))

    total_time = Timer.asHours(time.time() - now)
    logger.info("Total time = %s", total_time)

    return results


if __name__ == "__main__":
    main(sys.argv)
