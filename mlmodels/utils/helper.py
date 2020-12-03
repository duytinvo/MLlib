import os
from logging import Logger
from typing import List, Tuple, Generator, Callable
import glob
import re
import shutil
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mlmodels.modules.init_optimizers import optimizers
from mlmodels.utils.dataset import IterDataset
from mlmodels.utils.trad_tokenizer import Tokenizer


class Helper:

    @staticmethod
    def save_model(do_train: bool, local_rank: int, output_dir: str, model, tokenizer, optimizer,
                   scheduler, save_total_limit, args=None, checkpoint_prefix=""):
        """

        @param do_train: check if the path exists
        @param local_rank: check for distributed training
        @param output_dir: output dir to save
        @param model: model to save
        @param tokenizer: tokenizer to save
        @param optimizer: optimizer to save
        @param scheduler: scheduler to save
        @param save_total_limit: total number of checkpoint to save
        @param args: args to save
        @param checkpoint_prefix: checkpoint prefix to save
        @return: list of deleted checkpoints
        """
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if do_train and local_rank in [-1, 0]:
            # Create output directory if needed
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, "module") else model)
        # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        checkpoints_to_be_deleted = []
        # Good practice: save your training arguments together with the trained model
        if args:
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
        if len(checkpoint_prefix) != 0:
            checkpoints_to_be_deleted = Helper._rotate_checkpoints(save_total_limit=save_total_limit,
                                                                   output_dir=os.path.dirname(output_dir),
                                                                   checkpoint_prefix=checkpoint_prefix)
            # TODO: there is a bug if using overwrite_output_dir without should_continue
            Helper.save_optimizer(optimizer, scheduler, output_dir)
        return checkpoints_to_be_deleted

    @staticmethod
    def save_optimizer(optimizer, scheduler, output_dir):
        """

        @param optimizer: optimizer to save
        @param scheduler: scheduler to save
        @param output_dir: location to save
        """
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    @staticmethod
    def load_model(output_dir, device, optimizer, scheduler, checkpoint=False):
        """

        @param output_dir: directory to load the model from
        @param device: model the model to device
        @param optimizer: default optimizer if no checkpoint
        @param scheduler: default scheduler if no scheduler
        @param checkpoint: checkpoint to look for
        @return: model, tokenizer, optimizer, scheduler
        """
        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model.to(device)
        if checkpoint:
            optimizer.load_state_dict(torch.load(os.path.join(output_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(output_dir, "scheduler.pt")))
        return model, tokenizer, optimizer, scheduler

    @staticmethod
    def load_optimizer(optimizer: str, model_named_parameters: Generator, learning_rate: float, adam_epsilon: float,
                       t_total: float, warmup_steps: int, weight_decay: float, model_name_or_path=None):
        """

        @param optimizer: optimizer to load
        @param model_named_parameters: generator of model parameters
        @param learning_rate: learning rate
        @param adam_epsilon: adam epsilon value
        @param t_total: total number of steps
        @param warmup_steps: warmup after
        @param weight_decay: weight decay
        @param model_name_or_path: model name or path to load for optimizer and scheduler
        @return: optimizer and scheduler
        """
        # TO-DO: Add return by factory pattern
        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer = optimizers.init_optimizers(optimizer=optimizer, model_named_parameters=model_named_parameters,
                                               learning_rate=learning_rate, adam_epsilon=adam_epsilon,
                                               weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
                model_name_or_path
                and os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_name_or_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))

        return optimizer, scheduler

    @staticmethod
    def _sorted_checkpoints(output_dir, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
        """

        @param output_dir: the output directory of saved file
        @param checkpoint_prefix: the prefix for the checkpoint
        @param use_mtime:  use time for sorting
        @return: sorted checkpoint
        """
        ordering_and_checkpoint_path = []

        glob_checkpoints = glob.glob(os.path.join(output_dir, "{}_*".format(checkpoint_prefix)))

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(".*{}_([0-9]+)".format(checkpoint_prefix), path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    @staticmethod
    def _rotate_checkpoints(save_total_limit, output_dir: str, checkpoint_prefix="checkpoint", use_mtime=False) -> List:
        """

        @param save_total_limit: Total number of checkpoint to retain
        @param output_dir: The output dir
        @param checkpoint_prefix: checkpoint prefix name
        @param use_mtime:  use time for sorting
        @return: return a list of deleted checkpoint
        """
        checkpoints_to_be_deleted = []
        if not save_total_limit:
            return checkpoints_to_be_deleted
        if save_total_limit <= 0:
            return checkpoints_to_be_deleted

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = Helper._sorted_checkpoints(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix,
                                                        use_mtime=use_mtime)
        if len(checkpoints_sorted) <= save_total_limit:
            return checkpoints_to_be_deleted

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            shutil.rmtree(checkpoint)
        return checkpoints_to_be_deleted

    @staticmethod
    def collate_fn(padding_value=0, target_padding_value=None, batch_first=True, task = 1) -> Callable:
        """

        @param padding_value: Source padding value
        @param target_padding_value: Target padding value
        @param batch_first: True if the batch is first index
        @return: return the iterator
        """

        def collate(examples):
            """

            @param examples: batch of source and target sequence to pad
            @return: source and target sequence
            """
            if task == 3:
                try:
                    source = pad_sequence([torch.tensor(d[-1] + d[0]) for d in examples],
                                          batch_first=batch_first, padding_value=padding_value)
                    target = pad_sequence(
                        [torch.tensor(d[1]) if d[1] is not None else torch.empty(0) for d in examples],
                        batch_first=batch_first,
                        padding_value=target_padding_value if target_padding_value else padding_value)
                except:
                    if len(examples[0]) < 3:
                        print('Incorrect task id')
                    else:
                        print('Error occured during iteration of batches ')
            else:
                source = pad_sequence([torch.tensor(d[0]) for d in examples],
                                      batch_first=batch_first, padding_value=padding_value)
                target = pad_sequence([torch.tensor(d[1]) if d[1] is not None else torch.empty(0) for d in examples],
                                      batch_first=batch_first,
                                      padding_value=target_padding_value if target_padding_value else padding_value)
            return source, target

        return collate

    @staticmethod
    def build_dataloader(file: str, task: int, source2idx: Callable, target2idx: Callable, batch_size: int,
                         firstline: bool,
                         collate: Callable) -> Tuple[DataLoader, int]:
        """

        @param task: Choose the respective task
        @param source2idx: source tokenizer
        @param target2idx: target tokenizer
        @param batch_size: batch size
        @param firstline: if first line is header
        @param collate: collate function for sequence conversion
        @return: Dataloader and the file size
        """
        iterdata, num_lines = Tokenizer.prepare_iter(file, firstline=firstline,
                                                     task=task)
        dataset = IterDataset(iterdata, source2idx=source2idx, target2idx=target2idx,
                              num_lines=num_lines)
        dataloader = DataLoader(dataset, pin_memory=True, batch_size=batch_size, collate_fn=collate,
                                num_workers=8)
        return dataloader, num_lines
