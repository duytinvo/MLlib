# -*- coding: utf-8 -*-
"""
Created on 2020-03-12
@author: duytinvo
"""
import argparse
import glob
import logging
import os
import sys
import random
import re
import shutil
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from mlmodels.utils.txtIO import TXT
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.trad_tokenizer import Tokenizer, sys_tokens
from mlmodels.utils.idx2tensor import Data2tensor
from mlmodels.utils.dataset import IterDataset, MapDataset
from mlmodels.metrics.prf1 import NER_metrics
from mlmodels.utils.idx2tensor import seqPAD

logger = logging.getLogger(__name__)


class TransLabelerModel(object):
    def __init__(self, args):
        self.args = args
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
                       self.args.local_rank, device, self.args.n_gpu, bool(self.args.local_rank != -1), self.args.fp16)
        self.args.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.args.cls_token_at_end = False
        self.args.sep_token_extra = False
        self.num_labels = 0
        self.pad_id = 0

        self.config = None
        self.tokenizer = None
        self.model = None

        self.source2idx = None
        self.target2idx = None
        self.collate = None
        self.build_inputs_with_special_tokens = None

        self.optimizer = None
        self.scheduler = None
        self.tb_writer = None

        pass

    @staticmethod
    def tok2id(pretrained_tokenizer, block_size=512):
        """
        :param pretrained_tokenizer: pretrained tokenizer
        :param block_size: max length of a sequence
        :return: a token2index function
        """

        def f(sequence):
            # TODO: add more code to handle special tokens
            tokens = pretrained_tokenizer.tokenize(sequence)[:block_size]
            # assert pretrained_tokenizer.eos_token, "There is no END OF SEQUENCE token"
            # tokens += [pretrained_tokenizer.eos_token]
            tokenized_ids = pretrained_tokenizer.convert_tokens_to_ids(tokens)
            # tokenized_ids = pretrained_tokenizer.build_inputs_with_special_tokens(tokenized_ids)
            return tokenized_ids
        return f

    @staticmethod
    def collate_fn(padding_value=0, target_padding_value=None, batch_first=True):
        def collate(examples):
            source = pad_sequence([torch.tensor(d[0]) for d in examples],
                                  batch_first=batch_first, padding_value=padding_value)
            target = pad_sequence([torch.tensor(d[1]) if d[1] is not None else torch.empty(0) for d in examples],
                                  batch_first=batch_first,
                                  padding_value=target_padding_value if target_padding_value else padding_value)
            return source, target
        return collate

    @staticmethod
    def add_special_tokens(
            max_seq_length,
            special_tokens_count,
            cls_token_id,
            sep_token_id,
            cls_token_at_end=False,
            sep_token_extra=False,
            pad_token_label_id=-100):
        def f(tokens_ids, label_ids=None):
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            if len(tokens_ids) > max_seq_length - special_tokens_count:
                tokens_ids = tokens_ids[: (max_seq_length - special_tokens_count)]
                if label_ids is not None:
                    label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens_ids += [sep_token_id]
            if label_ids is not None:
                label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_ids += [sep_token_id]
                if label_ids is not None:
                    label_ids += [pad_token_label_id]
            # segment_ids = [sequence_a_segment_id] * len(tokens_ids)

            if cls_token_at_end:
                tokens_ids += [cls_token_id]
                if label_ids is not None:
                    label_ids += [pad_token_label_id]
                # segment_ids += [cls_token_segment_id]
            else:
                tokens_ids = [cls_token_id] + tokens_ids
                if label_ids is not None:
                    label_ids = [pad_token_label_id] + label_ids
                # segment_ids = [cls_token_segment_id] + segment_ids
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            # input_mask = [1 if mask_padding_with_zero else 0] * len(tokens_ids)
            return tokens_ids, label_ids

        return f

    def model_init(self, model_name_or_path):
        # Set seed
        Data2tensor.set_randseed(self.args.seed, self.args.n_gpu)
        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed training download model & vocab
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                           cache_dir=self.args.cache_dir if self.args.cache_dir else None,
                                                           use_fast=self.args.use_fast)

            labels_list = TXT.read(self.args.label_file, firstline=False)
            self.tokenizer.tw2i = Tokenizer.list2dict(sys_tokens + labels_list)
            self.tokenizer.i2tw = Tokenizer.reversed_dict(self.tokenizer.tw2i)
            self.pad_id = 0 if self.tokenizer._pad_token is None else self.tokenizer.pad_token_id
            self.num_labels = len(self.tokenizer.tw2i)

            if self.args.max_seq_length <= 0:
                self.args.max_seq_length = self.tokenizer.max_len
                # Our input block size will be the max possible for the model
            else:
                self.args.max_seq_length = min(self.args.max_seq_length, self.tokenizer.max_len)

            data_block_size = self.args.max_seq_length - (
                        self.tokenizer.max_len - self.tokenizer.max_len_single_sentence)
            logger.info("Training/evaluation parameters %s", self.args)

            self.source2idx = TransLabelerModel.tok2id(self.tokenizer, data_block_size)
            self.target2idx = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target,
                                                vocab_words=self.tokenizer.tw2i,
                                                unk_words=False, sos=False, eos=False)

            self.build_inputs_with_special_tokens = TransLabelerModel.add_special_tokens(
                max_seq_length=self.args.max_seq_length,
                special_tokens_count=self.tokenizer.num_special_tokens_to_add(),
                cls_token_id=self.tokenizer.cls_token_id,
                sep_token_id=self.tokenizer.sep_token_id,
                cls_token_at_end=self.args.cls_token_at_end,
                sep_token_extra=self.args.sep_token_extra,
                pad_token_label_id=self.args.pad_token_label_id)

            self.collate = TransLabelerModel.collate_fn(padding_value=self.pad_id,
                                                        target_padding_value=self.args.pad_token_label_id,
                                                        batch_first=True)

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.tokenizer.i2tw,
            label2id=self.tokenizer.tw2i,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        self.model = AutoModelForTokenClassification.from_pretrained(
                                        model_name_or_path,
                                        from_tf=bool(".ckpt" in self.args.model_name_or_path),
                                        config=self.config,
                                        cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        self.model.to(self.args.device)

        if self.args.local_rank == 0:
            torch.distributed.barrier()
            # Make sure only the first process in distributed training will download model & vocab
        pass

    def train_batch(self, train_dataloader, tr_loss, logging_loss, global_step=0, steps_trained_in_current_epoch=0):
        # self.model.zero_grad()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            nl = batch[0]
            nl_mask = (nl != self.pad_id).to(dtype=nl.dtype)
            segment_id = (nl == self.pad_id).to(dtype=nl.dtype)
            labels = batch[1]
            inputs = {"input_ids": nl, "attention_mask": nl_mask, "labels": labels}
            if self.args.model_type != "distilbert":
                inputs["token_type_ids"] = (segment_id if self.args.model_type in ["bert", "xlnet"] else None)
                # XLM and RoBERTa don"t use segment_ids

            outputs = self.model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

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
                self.model.zero_grad()
                global_step += 1

                if self.args.local_rank in [-1, 0] and self.args.max_steps > 0 \
                        and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.tb_writer.add_scalar("lr", self.scheduler.get_lr()[0], global_step)
                    self.tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.args.save_steps, global_step)
                    logging_loss = tr_loss

                    # Save model checkpoint
                    checkpoint_prefix = "checkpoint_by-step"
                    output_dir = os.path.join(self.args.output_dir, "{}_{}".format(checkpoint_prefix, global_step))
                    self.save_model(output_dir, checkpoint_prefix=checkpoint_prefix)

                    if self.args.local_rank == -1 and self.args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.evaluate_batch(self.args.dev_file)
                        for key, value in results.items():
                            self.tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            if global_step > self.args.max_steps > 0:
                epoch_iterator.close()
                break
        return tr_loss, logging_loss, global_step, steps_trained_in_current_epoch

    def train(self):
        self.model_init(self.args.model_name_or_path)

        """ Train the model """
        if self.args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter()

        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)

        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_data, train_numlines = Tokenizer.prepare_iter(self.args.train_file, firstline=self.args.firstline, task=2)
        train_dataset = IterDataset(train_data, source2idx=self.source2idx, target2idx=self.target2idx,
                                    num_lines=train_numlines,
                                    bpe=True, special_tokens_func=self.build_inputs_with_special_tokens,
                                    label_pad_id=self.args.pad_token_label_id)
        train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=self.args.train_batch_size,
                                      collate_fn=self.collate, num_workers=8)
        if self.args.local_rank == 0:
            torch.distributed.barrier()
            # End of barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        num_batchs = (train_numlines // self.args.train_batch_size) + 1 \
            if train_numlines % self.args.train_batch_size != 0 else train_numlines // self.args.train_batch_size
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (num_batchs // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = num_batchs // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_steps,
                                                         num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.args.model_name_or_path, "optimizer.pt")) and \
                os.path.isfile(os.path.join(self.args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "scheduler.pt")))

        if self.args.fp16:
            try:
                global apex
                import apex
                # from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=self.args.fp16_opt_level)

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
                    self.args.train_batch_size * self.args.gradient_accumulation_steps *
                    (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                global_step = int(self.args.model_name_or_path.split("-")[-1].split("/")[0])
            except ValueError:
                global_step = 0
            epochs_trained = global_step // (num_batchs // self.args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (num_batchs // self.args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(epochs_trained, int(self.args.num_train_epochs), desc="Epoch",
                                disable=self.args.local_rank not in [-1, 0])
        # Added here for reproducibility
        Data2tensor.set_randseed(self.args.seed, self.args.n_gpu)

        nepoch_no_imprv = 0
        best_dev = np.inf if self.args.metric == "loss" else -np.inf
        ep_count = 0
        for _ in train_iterator:
            tr_loss, logging_loss, global_step, steps_trained_in_current_epoch = self.train_batch(train_dataloader,
                                                                                                  tr_loss, logging_loss,
                                                                                                  global_step,
                                                                                                  steps_trained_in_current_epoch)
            ep_count += 1
            if self.args.local_rank in [-1, 0] and self.args.max_steps < 0:
                self.tb_writer.add_scalar("lr", self.scheduler.get_lr()[0], global_step)
                self.tb_writer.add_scalar("loss", tr_loss / global_step, global_step)

                # Save model checkpoint
                checkpoint_prefix = "checkpoint_by-epoch"
                output_dir = os.path.join(self.args.output_dir, "{}_{}".format(checkpoint_prefix, ep_count))
                self.save_model(output_dir, checkpoint_prefix=checkpoint_prefix)
                # Only evaluate when single GPU otherwise metrics may not average well
                if self.args.local_rank == -1 and self.args.evaluate_during_training:
                    results, _, _ = self.evaluate_batch(self.args.dev_file,
                                                        prefix="of dev_set used the epoch_{} model".format(ep_count))
                    for key, value in results.items():
                        self.tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                    dev_metric = results["loss"] if self.args.metric == "loss" else results["f1"]
                    cond = dev_metric < best_dev if self.args.metric == "loss" else dev_metric > best_dev
                    if cond:
                        logger.info("New improvement at %d", ep_count)
                        # Save model checkpoint
                        self.save_model(self.args.output_dir)
                        best_dev = dev_metric
                        nepoch_no_imprv = 0
                    else:
                        nepoch_no_imprv += 1
                        if nepoch_no_imprv >= self.args.patience:
                            # Testing
                            if self.args.do_predict:
                                self.predict()
                            self.tb_writer.close()
                            return global_step, tr_loss / global_step

            if global_step > self.args.max_steps > 0:
                train_iterator.close()
                break

        if self.args.local_rank in [-1, 0]:
            if self.args.do_predict:
                self.predict()
            self.tb_writer.close()

        return global_step, tr_loss / global_step

    def save_model(self, output_dir, checkpoint_prefix=""):
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if self.args.do_train and self.args.local_rank in [-1, 0]:
            # Create output directory if needed
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
        # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if len(checkpoint_prefix) != 0:
            self._rotate_checkpoints(checkpoint_prefix)
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s\n", output_dir)

    def load_model(self, output_dir, checkpoint=False):
        # Check output directory
        if not os.path.exists(output_dir):
            logger.error("%s does not exist", output_dir)
        # Load a trained model and vocabulary that you have fine-tuned

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(output_dir)

            labels_list = TXT.read(self.args.label_file, firstline=False)
            self.tokenizer.tw2i = Tokenizer.list2dict(sys_tokens + labels_list)
            self.tokenizer.i2tw = Tokenizer.reversed_dict(self.tokenizer.tw2i)
            self.pad_id = 0 if self.tokenizer._pad_token is None else self.tokenizer.pad_token_id
            self.num_labels = len(self.tokenizer.tw2i)

            if self.args.max_seq_length <= 0:
                self.args.max_seq_length = self.tokenizer.max_len
                # Our input block size will be the max possible for the model
            else:
                self.args.max_seq_length = min(self.args.max_seq_length, self.tokenizer.max_len)

            data_block_size = self.args.max_seq_length - (
                    self.tokenizer.max_len - self.tokenizer.max_len_single_sentence)
            logger.info("Training/evaluation parameters %s", self.args)

            self.source2idx = TransLabelerModel.tok2id(self.tokenizer, data_block_size)
            self.target2idx = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target,
                                                vocab_words=self.tokenizer.tw2i,
                                                unk_words=False, sos=False, eos=False)

            self.build_inputs_with_special_tokens = TransLabelerModel.add_special_tokens(
                max_seq_length=self.args.max_seq_length,
                special_tokens_count=self.tokenizer.num_special_tokens_to_add(),
                cls_token_id=self.tokenizer.cls_token_id,
                sep_token_id=self.tokenizer.sep_token_id,
                cls_token_at_end=self.args.cls_token_at_end,
                sep_token_extra=self.args.sep_token_extra,
                pad_token_label_id=self.args.pad_token_label_id)

            self.collate = TransLabelerModel.collate_fn(padding_value=self.pad_id,
                                                        target_padding_value=self.args.pad_token_label_id,
                                                        batch_first=True)
        self.model = AutoModelForTokenClassification.from_pretrained(output_dir)
        self.model.to(self.args.device)
        if checkpoint:
            self.optimizer.load_state_dict(torch.load(os.path.join(output_dir, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(output_dir, "scheduler.pt")))
        pass

    @staticmethod
    def _sorted_checkpoints(output_dir, checkpoint_prefix="checkpoint", use_mtime=False):
        ordering_and_checkpoint_path = []

        glob_checkpoints = glob.glob(os.path.join(output_dir, "{}_*".format(checkpoint_prefix)))

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, checkpoint_prefix="checkpoint", use_mtime=False):
        if not self.args.save_total_limit:
            return
        if self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(self.args.output_dir, checkpoint_prefix, use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate_batch(self, eval_file, prefix=""):
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        eval_dataset, eval_numlines = Tokenizer.prepare_iter(eval_file, firstline=self.args.firstline,
                                                             task=2)
        eval_dataset = IterDataset(eval_dataset, source2idx=self.source2idx, target2idx=self.target2idx,
                                   num_lines=eval_numlines, bpe=True,
                                   special_tokens_func=self.build_inputs_with_special_tokens,
                                   label_pad_id=self.args.pad_token_label_id)
        eval_dataloader = DataLoader(eval_dataset, pin_memory=True, batch_size=self.args.eval_batch_size,
                                     collate_fn=self.collate, num_workers=8)

        num_batchs = (eval_numlines // self.args.eval_batch_size) + 1 \
            if eval_numlines % self.args.eval_batch_size != 0 else eval_numlines // self.args.eval_batch_size

        # multi-gpu evaluate
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", eval_numlines)
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        nl_tokens = []
        reference = []
        candidate = []
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating", total=num_batchs):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                nl = batch[0]
                nl_mask = (nl != self.pad_id).to(dtype=nl.dtype)
                segment_id = (nl == self.pad_id).to(dtype=nl.dtype)
                labels = batch[1]
                inputs = {"input_ids": nl, "attention_mask": nl_mask, "labels": labels}
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        segment_id if self.args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            preds = logits.argmax(dim=-1).detach().cpu().tolist()
            out_label_ids = inputs["labels"].detach().cpu().tolist()
            nl_list = inputs["input_ids"].detach().cpu().tolist()

            for i in range(len(out_label_ids)):
                pred_seq = []
                gold_seq = []
                nl_seq = []
                for j in range(len(out_label_ids[i])):
                    if out_label_ids[i][j] != self.args.pad_token_label_id:
                        gold_seq.append(self.tokenizer.i2tw[out_label_ids[i][j]])
                        pred_seq.append(self.tokenizer.i2tw[preds[i][j]])
                        nl_seq.append(self.tokenizer._convert_id_to_token(nl_list[i][j]))
                reference.append(gold_seq)
                candidate.append(pred_seq)
                nl_tokens.append(nl_seq)

        eval_loss = eval_loss / nb_eval_steps
        # print(candidate, reference)
        if len(candidate) != 0 and len(reference) != 0:
            assert len(candidate) == len(reference)
            # Randomly sample one pair
            rand_idx = random.randint(0, len(reference) - 1)
            print("\nRANDOMLY sampling: ")
            print("\t- An Input Sequence: ", " ".join(nl_tokens[rand_idx]))
            print("\t- A LABEL query: ", " ".join(reference[rand_idx]))
            print("\t- A PREDICTED query: ", " ".join(candidate[rand_idx]))
            # print("\t- A PREDICTED prob: ", predict_probs[rand_idx], "\n\n")
            metrics = TransLabelerModel.class_metrics(reference, candidate)
        else:
            metrics = [0., 0., 0.]

        results = {"loss": eval_loss, "precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        return results, candidate, nl_tokens

    @staticmethod
    def class_metrics(reference, candidate):
        P, R, F1 = NER_metrics.span_metrics(reference, candidate)
        return P, R, F1

    def evaluate(self):
        results = {}
        checkpoints = [self.args.output_dir]
        if self.args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c)
                               for c in sorted(glob.glob(self.args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        # sorted_checkpoints = self._sorted_checkpoints(self.args.output_dir, "checkpoint", "False")
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoint.split("-")) > 1 else "best"
            self.load_model(checkpoint)
            result, _, _ = self.evaluate_batch(self.args.dev_file, prefix="of dev_set used the {} model".format(global_step))
            # result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update({global_step: result})
        output_dev_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_dev_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("- {}:\n".format(key))
                for k in sorted(results[key].keys()):
                    writer.write("\t{} = {}\n".format(k, str(results[key][k])))
        return results

    def predict(self):
        self.load_model(self.args.output_dir)
        result, predictions, nl_tokens = self.evaluate_batch(self.args.test_file, prefix="of test_set used the best model")
        # Save results
        output_test_results_file = os.path.join(self.args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(self.args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            for i, tokens in enumerate(nl_tokens):
                assert len(tokens) == len(predictions[i]), print(tokens, predictions[i])
                for wd, lb in zip(tokens, predictions[i]):
                    output_line = wd + "\t" + lb + "\n"
                    writer.write(output_line)
                writer.write("\n")

    def predict_batch(self, data, batch_size=1):
        eval_dataset = MapDataset(data, source2idx=self.source2idx, target2idx=self.target2idx, bpe=False,
                                  special_tokens_func=self.build_inputs_with_special_tokens,
                                  label_pad_id=self.args.pad_token_label_id)
        eval_dataloader = DataLoader(eval_dataset, pin_memory=True, batch_size=batch_size,
                                     collate_fn=self.collate, num_workers=8)
        num_batchs = (len(data) // batch_size) + 1 if len(data) % batch_size != 0 else len(data) // batch_size
        # multi-gpu evaluate
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        # Eval!
        preds = None
        probs = None
        output_len = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating", total=num_batchs):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                nl = batch[0]
                nl_mask = (nl != self.pad_id).to(dtype=nl.dtype)
                segment_id = (nl == self.pad_id).to(dtype=nl.dtype)
                inputs = {"input_ids": nl, "attention_mask": nl_mask}
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        segment_id if self.args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = self.model(**inputs)
                logits_prob, logits_idx = torch.topk(torch.softmax(outputs[0], dim=-1), k=1)
                logits_prob.squeeze_(-1)
                logits_idx.squeeze_(-1)
                if preds is None:
                    preds = logits_idx.detach().cpu().numpy()
                    probs = logits_prob.detach().cpu().numpy()
                    output_len = inputs["attention_mask"].detach().sum(dim=-1).cpu().numpy()
                else:
                    preds = np.append(preds, logits_idx.detach().cpu().numpy(), axis=0)
                    probs = np.append(probs, logits_prob.detach().cpu().numpy(), axis=0)
                    output_len = np.append(output_len, inputs["attention_mask"].detach().sum(dim=-1).cpu().numpy(),
                                           axis=0)

        preds_list = [[] for _ in range(len(eval_dataset))]
        probs_list = [[] for _ in range(len(eval_dataset))]
        toks_list = [[] for _ in range(len(eval_dataset))]
        for i in range(len(eval_dataset)):
            toks_list[i].extend(self.tokenizer.convert_ids_to_tokens(eval_dataset[i][0][:output_len[i]]))
            for j in range(output_len[i]):
                preds_list[i].append(self.tokenizer.i2tw[preds[i][j]])
                probs_list[i].append(probs[i, j])
        return toks_list, preds_list, probs_list


def main(argv):
    parser = argparse.ArgumentParser(argv)
    # Required parameters
    parser.add_argument('--label_file', help='Trained file (semQL) in Json format', type=str,
                           default="./data/reviews/processed_csv/labels.txt")
    parser.add_argument('--train_file', help='Trained file (semQL) in Json format', type=str,
                           default="./data/reviews/processed_csv/train_res4.csv")
    parser.add_argument('--dev_file', help='Validated file (semQL) in Json format', type=str,
                           default="./data/reviews/processed_csv/dev_res4.csv")
    parser.add_argument('--test_file', help='Tested file (semQL) in Json format', type=str,
                           default="./data/reviews/processed_csv/test_res4.csv")
    parser.add_argument("--firstline", action='store_true', default=False,
                           help="labelled files having a header")

    parser.add_argument("--output_dir", default="./data/reviews/trained_model", type=str,
                        # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.",)

    # Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        # required=True,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        # required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than "
                             "this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run metrics on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.",)
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, "
                             "does not delete by default")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as "
                             "model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping if no improvement after patience epoches")
    parser.add_argument("--metric", type=str, default="f1", help="Optimized criterion (loss or f1)")
    args = parser.parse_args()
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    trans_ner_model = TransLabelerModel(args)
    # Training
    if args.do_train:
        global_step, tr_loss = trans_ner_model.train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        results = trans_ner_model.evaluate()

    # Testing
    if args.do_predict and args.local_rank in [-1, 0]:
        trans_ner_model.predict()

    return results


if __name__ == "__main__":
    """
    python3 -m mlmodels.training.trans_labeler_model_rev2 --train_file ./data/ner/processed/train.txt --dev_file ./data/ner/processed/dev.txt --test_file ./data/ner/processed/test.txt --model_type bert --labels ./data/ner/processed/labels.txt --model_name_or_path bert-base-multilingual-cased --output_dir ./data/ner/trained_model/ --max_seq_length  128 --num_train_epochs 3 --per_gpu_train_batch_size 32 --save_steps 750 --seed 12345 --do_train --do_eval --overwrite_output_dir --evaluate_during_training
    """
    results = main(sys.argv)
