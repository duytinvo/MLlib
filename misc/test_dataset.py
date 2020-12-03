# -*- coding: utf-8 -*-
"""
Created on 2020-04-02
@author duytinvo
"""
import logging
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.txtIO import TXT
from mlmodels.utils.dataset import MapDataset, IterDataset
from mlmodels.utils.trad_tokenizer import Tokenizer, sys_tokens

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer
)

from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoModelForTokenClassification
)

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    AutoModelForSequenceClassification
)


def tok2id(pretrained_tokenizer, block_size=512, eos=True, special_tokens=True):
    """
    :param pretrained_tokenizer: pretrained tokenizer
    :param block_size: max length of a sequence
    :param eos: add an end of sequence token
    :param special_tokens: add specific token from the pretrained tokenizer
    :return: a token2index function
    """

    def f(sequence):
        # TODO: add more code to handle special tokens
        tokens = pretrained_tokenizer.tokenize(sequence)[:block_size]
        if eos:
            assert pretrained_tokenizer.eos_token, "There is no END OF SEQUENCE token"
            tokens += [pretrained_tokenizer.eos_token]
        tokenized_ids = pretrained_tokenizer.convert_tokens_to_ids(tokens)
        if special_tokens:
            tokenized_ids = pretrained_tokenizer.build_inputs_with_special_tokens(tokenized_ids)
        return tokenized_ids

    return f


def collate_func(padding_value=0, target_padding_value=None, batch_first=True):
    def collate(examples):
        source = pad_sequence([torch.tensor(d[0]) for d in examples],
                              batch_first=batch_first, padding_value=padding_value)
        target = pad_sequence([torch.tensor(d[1]) if d[1] is not None else torch.empty(0) for d in examples],
                              batch_first=batch_first,
                              padding_value=target_padding_value if target_padding_value else padding_value)
        return source, target
    return collate

# def collate_fn(examples, source_padding_value=0, target_padding_value=-100):
#     source = pad_sequence([torch.tensor(d[0]) for d in examples], batch_first=True, padding_value=source_padding_value)
#     target = pad_sequence([torch.tensor(d[1]) if d[1] is not None else torch.empty(0) for d in examples],
#                           batch_first=True, padding_value=target_padding_value)
#     return source, target


def add_special_tokens(
                        max_seq_length,
                        special_tokens_count,
                        cls_token_id,
                        sep_token_id,
                        cls_token_at_end=False,
                        sep_token_extra=False,
                        pad_token_label_id=-100):
    def f(tokens_ids, label_ids):
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if len(tokens_ids) > max_seq_length - special_tokens_count:
            tokens_ids = tokens_ids[: (max_seq_length - special_tokens_count)]
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
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens_ids += [sep_token_id]
            label_ids += [pad_token_label_id]
        # segment_ids = [sequence_a_segment_id] * len(tokens_ids)

        if cls_token_at_end:
            tokens_ids += [cls_token_id]
            label_ids += [pad_token_label_id]
            # segment_ids += [cls_token_segment_id]
        else:
            tokens_ids = [cls_token_id] + tokens_ids
            label_ids = [pad_token_label_id] + label_ids
            # segment_ids = [cls_token_segment_id] + segment_ids
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_mask = [1 if mask_padding_with_zero else 0] * len(tokens_ids)
        return tokens_ids, label_ids
    return f


if __name__ == "__main__":
    from rouge import Rouge
    from mlmodels.metrics.bleu import compute_bleu
    from mlmodels.metrics.prf1 import APRF1
    from mlmodels.utils.special_tokens import SENSP, SENGE, SOT, EOT, UNK, CLS, SEP, PAD, MASK, NULL, NL

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model_name = "bert-base-uncased"
    # model_name = "t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add specific tokens
    try:
        if tokenizer._bos_token is None:
            tokenizer.add_special_tokens({"bos_token": SOT})
    except:
        tokenizer.add_tokens([SOT])

    try:
        if tokenizer._eos_token is None:
            tokenizer.add_special_tokens({"eos_token": EOT})
    except:
        tokenizer.add_tokens([EOT])

    try:
        if tokenizer._unk_token is None:
            tokenizer.add_special_tokens({"unk_token": UNK})
    except:
        tokenizer.add_tokens([UNK])

    try:
        if tokenizer._sep_token is None:
            tokenizer.add_special_tokens({"sep_token": SEP})
    except:
        tokenizer.add_tokens([SEP])

    try:
        if tokenizer._pad_token is None:
            tokenizer.add_special_tokens({"pad_token": PAD})
    except:
        tokenizer.add_tokens([PAD])

    try:
        if tokenizer._cls_token is None:
            tokenizer.add_special_tokens({"cls_token": CLS})
    except:
        tokenizer.add_tokens([CLS])

    try:
        if tokenizer._mask_token is None:
            tokenizer.add_special_tokens({"mask_token": MASK})
    except:
        tokenizer.add_tokens([MASK])

    tokenizer.add_tokens([SENSP, SENGE, NL, NULL])

    special_tokens_count = tokenizer.num_special_tokens_to_add()
    max_seq_length = tokenizer.max_len
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token_at_end = False
    sep_token_extra = False
    pad_token_label_id = -100

    label_file = "/media/data/review_response/labels.txt"
    labels_list = TXT.read(label_file, firstline=False)
    tokenizer.tw2i = Tokenizer.list2dict(sys_tokens + labels_list)
    tokenizer.i2tw = Tokenizer.reversed_dict(tokenizer.tw2i)
    lb2ids = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target, vocab_words=tokenizer.tw2i,
                               unk_words=False, sos=False, eos=False)
    pad_id = 0 if tokenizer._pad_token is None else tokenizer.pad_token_id
    num_labels = len(tokenizer.tw2i)

    build_inputs_with_special_tokens = add_special_tokens(max_seq_length=max_seq_length,
                                                          special_tokens_count=special_tokens_count,
                                                          cls_token_id=cls_token_id,
                                                          sep_token_id=sep_token_id,
                                                          cls_token_at_end=cls_token_at_end,
                                                          sep_token_extra=sep_token_extra,
                                                          pad_token_label_id=pad_token_label_id)

    source2idx = tok2id(tokenizer, block_size=510, eos=False, special_tokens=True)
    target2idx = tok2id(tokenizer, block_size=510, eos=False, special_tokens=False)

    collate_fn = collate_func(padding_value=pad_id, target_padding_value=-100, batch_first=True)

    config = AutoConfig.from_pretrained(model_name,
                                        num_labels=num_labels,
                                        id2label=tokenizer.i2tw,
                                        label2id=tokenizer.tw2i)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               from_tf=bool(".ckpt" in model_name),
                                                               config=config)


    # data = JSON.read(args.train_data_file)
    # train_dataset = MapDataset(data, source2idx=source2idx, target2idx=source2idx)
    # # train_sampler = RandomSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, pin_memory=True,
    #                               batch_size=16, collate_fn=collate_fn)

    data_file = "/media/data/review_response/Dev.json"
    iterdata, num_lines = Tokenizer.prepare_iter(data_file, firstline=True, task=1)

    train_iterdataset = IterDataset(iterdata, source2idx=source2idx, target2idx=lb2ids, num_lines=num_lines)
                                    # bpe=True, special_tokens_func=build_inputs_with_special_tokens)

    # avg_scores = {'f': 0., 'p': 0., 'r': 0.}
    # rouge_lf = []
    # rouge = Rouge()
    # for line in iterdata(0, num_lines):
    #     scores = rouge.get_scores(line[0], line[1])[0]
    #     rouge_lf.append(scores['rouge-l']['f'])
    #     break
    #     avg_scores['f'] += scores['rouge-l']['f']
    #     avg_scores['p'] += scores['rouge-l']['p']
    #     avg_scores['r'] += scores['rouge-l']['r']
    # avg_scores['f'] = avg_scores['f']/num_lines
    # avg_scores['p'] = avg_scores['p']/num_lines
    # avg_scores['r'] = avg_scores['r']/num_lines

    train_dataloader = DataLoader(train_iterdataset, batch_size=4, collate_fn=collate_fn)

    candidate = []
    reference = []
    cnt = 0
    for i, batch in enumerate(train_dataloader):
        cnt += 1
        inputs, labels = batch[0], batch[1].squeeze()
        labels_mask = labels != -100
        labels_len = labels_mask.sum(dim=-1)
        outputs = model(input_ids=inputs, labels=labels)
        loss, logits = outputs[:2]
        pred_labels = logits.argmax(dim=-1)
        #
        # candidate.extend([ids[:l] for ids, l in zip(pred_labels.tolist(), labels_len.tolist())])
        # reference.extend([ids[:l] for ids, l in zip(labels.tolist(), labels_len.tolist())])
        if cnt == 10:
            break
    # results = compute_bleu(list(zip(reference)), candidate)
