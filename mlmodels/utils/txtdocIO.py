# -*- coding: utf-8 -*-
"""
Created on 2020-03-09
@author: duytinvo
This code is copied and modified from run_language_modeling.py
"""
import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import Dict, List, Tuple
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


class TextDataset(Dataset):
    def __init__(self, file_path: str, source2idx=None,
                 # target2idx=None,
                 tokenizer=None,
                 model_type="", overwrite_cache=True,
                 block_size=512, read_line=True):
        self.examples = []
        self.source2idx = source2idx
        # if tokenizer is not None:
        #     block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
        #     if len(model_type) != 0:
        #         self.source2idx = self.tok2id(tokenizer, block_size)

        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, model_type + "_cached_lm_" + filename)

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                if isinstance(self.examples[0][0], str) and self.source2idx is not None:
                    for i in range(len(self.examples)):
                        self.examples[i] = self.source2idx(self.examples[i])
                    logger.info("Saving features into cached file %s", cached_features_file)
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            if not read_line:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                # tokenized_text = tokenizer.tokenize(text)
                tokenized_text = self.process_nl(text, tokenizer)

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    tokens = tokenized_text[i:i+block_size]
                    if self.source2idx is not None:
                        tokens = self.source2idx(tokens)
                    self.examples.append(tokens)
                # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.
            else:
                with open(file_path, encoding="utf-8") as f:
                    for text in f.read().splitlines():
                        if len(text) > 0 and not text.isspace():
                            # tokens = self.process_nl(text, tokenizer.tokenize)
                            tokens = self.process_nl(text, tokenizer)
                            if self.source2idx is not None:
                                tokens = self.source2idx(tokens)
                            self.examples.append(tokens)
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.source2idx is not None:
            return torch.tensor(self.examples[item], dtype=torch.long)
        else:
            return self.examples[item]

    @staticmethod
    def process_nl(nl, tokenizer=None):
        if tokenizer is not None:
            nl_toks = tokenizer(nl)
        else:
            nl_toks = nl.lower().split()
        return nl_toks

    @staticmethod
    def tok2id(pretrained_tokenizer, block_size=512):
        """
        :param pretrained_tokenizer: pretrained tokenizer
        :param block_size: max length of a sequence
        :return: a token2index function
        """

        def f(tokens):
            # TODO: add more code to handle special tokens
            tokenized_ids = pretrained_tokenizer.convert_tokens_to_ids(tokens)[:block_size]
            tokenized_ids = pretrained_tokenizer.build_inputs_with_special_tokens(tokenized_ids)
            return tokenized_ids

        return f


def set_seed(seed=12345, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", type=str,
                        default="../../data/wikitext2/raw/wiki.train.raw",
                        # required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", type=str,
                        default="../../wikitext2/trained_model",
                        # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--model_type", type=str,
                        default="gpt2",
                        # required=True,
                        help="The model architecture to be trained or fine-tuned.",)

    # Other parameters
    parser.add_argument(
        "--line_by_line",
        action="store_false",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )

    parser.add_argument("--model_name_or_path", type=str,
        default="gpt2",
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_false", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    logger.info("Training/evaluation parameters %s", args)

    # train_dataset = TextDataset(file_path=file_path, tokenizer=tokenizer, model_type=args.model_type,
    #                             overwrite_cache=args.overwrite_cache, block_size=args.block_size,
    #                             read_line=args.line_by_line)

    block_size = args.block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
    source2idx = TextDataset.tok2id(tokenizer, block_size)
    train_dataset = TextDataset(file_path=args.train_data_file, source2idx=source2idx,
                                tokenizer=tokenizer.tokenize,
                                block_size=block_size,
                                model_type=args.model_type,
                                overwrite_cache=args.overwrite_cache,
                                read_line=args.line_by_line)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=16, collate_fn=collate)

    for i, batch in enumerate(train_dataloader):
        inputs = batch
        break