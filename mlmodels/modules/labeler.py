# -*- coding: utf-8 -*-
"""
Created on 2020-02-24
@author: duytinvo
"""
import math
import torch
import torch.nn as nn
from mlmodels.modules.embeddings import Emb_layer, PositionalEncoding
from mlmodels.modules.encoders import Word_Encoder
from mlmodels.modules.crf import NN_CRF
from mlmodels.modules.transformer_base import Encoder_base


class Labeler(nn.Module):
    def __init__(self, nlemb_HPs, enc_HPs, crf_HPs, drop_rate=0.5, num_labels=None):
        super(Labeler, self).__init__()
        print("- Use the {} model".format(enc_HPs[0]))
        if enc_HPs[0] == "self_attention":
            self.use_selfatt = True
            print("\t- Add word embedding module")
            self.sembedding = PositionalEncoding(nlemb_HPs)
            print("\t- Add encoding modules")
            self.encoder = Encoder_base(enc_HPs)
            self.fn_dim = enc_HPs[1]
            self.norm_dim = enc_HPs[2]
        else:
            self.use_selfatt = False
            print("\t- Add word embedding module")
            self.sembedding = Emb_layer(nlemb_HPs)
            print("\t- Add encoding modules")
            self.encoder = Word_Encoder(enc_HPs)
            self.fn_dim = enc_HPs[2]

        self.finaldrop_layer = nn.Dropout(drop_rate)

        self.num_labels = num_labels
        self.use_crf = crf_HPs[0]
        if num_labels > 2:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, num_labels)
            if self.use_crf:
                self.crf_layer = NN_CRF(crf_HPs)
            else:
                self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.hidden2tag_layer.bias.data.zero_()
    #     self.hidden2tag_layer.weight.data.uniform_(-initrange, initrange)

    @staticmethod
    def sort_tensors(word_tensor, seq_len_tensor):
        seq_len_tensor, seqord_tensor = seq_len_tensor.sort(0, descending=True)
        word_tensor = word_tensor[seqord_tensor]
        _, seqord_recover_tensor = seqord_tensor.sort(0, descending=False)
        return word_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor

    @staticmethod
    def reorder_tensor(inp_tensor, new_order_tensor, dim=0):
        num_dim = inp_tensor.dim()
        if isinstance(inp_tensor, tuple):
            if new_order_tensor is not None:
                if dim == 0 and dim < num_dim:
                    if inp_tensor.size(0) != 1 and inp_tensor.size(0) == new_order_tensor.size(0):
                        inp_tensor = tuple(tensor[new_order_tensor] for tensor in inp_tensor)
                elif dim == 1 and dim < num_dim:
                    if inp_tensor.size(1) != 1 and inp_tensor.size(1) == new_order_tensor.size(0):
                        inp_tensor = tuple(tensor[:, new_order_tensor, :] for tensor in inp_tensor)
                elif dim == 2 and dim < num_dim:
                    if inp_tensor.size(2) != 1 and inp_tensor.size(2) == new_order_tensor.size(0):
                        inp_tensor = tuple(tensor[:, :, new_order_tensor] for tensor in inp_tensor)
                else:
                    raise RuntimeError("Not implemented yet")
        else:
            if new_order_tensor is not None:
                if dim == 0 and dim < num_dim:
                    if inp_tensor.size(0) != 1 and inp_tensor.size(0) == new_order_tensor.size(0):
                        inp_tensor = inp_tensor[new_order_tensor]
                elif dim == 1 and dim < num_dim:
                    if inp_tensor.size(1) != 1 and inp_tensor.size(1) == new_order_tensor.size(0):
                        inp_tensor = inp_tensor[:, new_order_tensor, :]
                elif dim == 2 and dim < num_dim:
                    if inp_tensor.size(2) != 1 and inp_tensor.size(2) == new_order_tensor.size(0):
                        inp_tensor = inp_tensor[:, :, new_order_tensor]
                else:
                    raise RuntimeError("Not implemented yet")
        return inp_tensor

    def scoring(self, rnn_out):
        rnn_score = self.hidden2tag_layer(rnn_out)
        rnn_score = self.finaldrop_layer(rnn_score)
        return rnn_score

    def NLL_loss(self, label_score, label_tensor, label_mask):
        if self.num_labels > 2:
            if self.use_crf:
                de_loss = self.crf_layer.NLL_loss(label_score, label_tensor, label_mask)
            else:
                label_score = label_score[label_mask]
                label_tensor = label_tensor[label_mask]
                # label_score = [B, C]; label_tensor = [B, ]
                de_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1,))
        else:
            label_score = label_score[label_mask]
            label_tensor = label_tensor[label_mask]
            # label_score = [B, *]; label_tensor = [B, *]
            de_loss = self.lossF(label_score, label_tensor.float().view(-1, 1))
        return de_loss

    def forward(self, nl_tensor, nl_len_tensor, wombat_tensor=None):
        if self.use_selfatt:
            key_padding_mask = (nl_tensor == 0)
            en_inp = self.sembedding(nl_tensor).transpose(0, 1) * math.sqrt(self.norm_dim)
            if wombat_tensor is not None:
                en_inp += wombat_tensor
            en_out = self.encoder(en_inp, None, key_padding_mask).transpose(0, 1)
            en_score = self.scoring(en_out)
        else:
            device = nl_tensor.device
            # sort lengths of input tensors in the descending mode
            nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = self.sort_tensors(nl_tensor, nl_len_tensor)
            # en_inp = [batch, nl_len, nl_emb]
            en_inp = self.sembedding(nl_tensor)
            if wombat_tensor is not None:
                wombat_tensor = self.reorder_tensor(wombat_tensor, nl_ord_tensor, dim=0)
                en_inp += wombat_tensor
            # en_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
            # en_hidden = (h_n,c_n) ---> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
            en_out, en_hidden = self.encoder(en_inp, nl_len_tensor)
            en_score = self.scoring(en_out)
            # recover the original order of outputs to compute loss
            en_score = self.reorder_tensor(en_score, nl_recover_ord_tensor, dim=0)
        return en_score

    def inference(self, label_score, label_mask=None):
        if self.num_labels > 2:
            if self.use_crf:
                best_paths = self.crf_layer.inference(label_score, label_mask)
                # label_prob, label_pred = best_paths
                label_pred, label_prob = list(zip(*best_paths))
            else:
                label_prob = torch.softmax(label_score, dim=-1)
                label_prob, label_pred = label_prob.data.topk(1)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5)
        return label_prob, label_pred


if __name__ == '__main__':
    import torch
    from mlmodels.utils.idx2tensor import Data2tensor, seqPAD
    from mlmodels.utils.dataset import IterDataset, collate_fn, tokens2ids
    from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler, TensorDataset
    from mlmodels.utils.BPEtonkenizer import BPE
    from mlmodels.utils.special_tokens import BPAD, PAD, NULL, EOT
    from mlmodels.utils.txtIO import TXT
    from mlmodels.utils.trad_tokenizer import Tokenizer, sys_tokens
    from mlmodels.utils.jsonIO import JSON
    from mlmodels.utils.csvIO import CSV

    Data2tensor.set_randseed(12345)
    device = torch.device("cpu")
    dtype = torch.long
    use_cuda = False
    filename = "../../data/reviews/processed_csv/train_res4.csv"
    label_file = "../../data/reviews/processed_csv/labels.txt"
    labels_list = TXT.read(label_file, firstline=False)
    lb2id_dict = Tokenizer.list2dict(sys_tokens + labels_list)
    id2lb_dict = Tokenizer.reversed_dict(lb2id_dict)
    lb2ids = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target, vocab_words=lb2id_dict,
                               unk_words=False, sos=False, eos=False)
    tokenize_type = "bpe"
    if tokenize_type != "bpe":
        # Load datasets to build vocabulary
        data = Tokenizer.load_file([filename], task=2)
        s_paras = [-1, 1]
        t_paras = [-1, 1]
        tokenizer = Tokenizer(s_paras, t_paras)
        tokenizer.build(data)
        nl2ids = Tokenizer.lst2idx(tokenizer=Tokenizer.process_nl, vocab_words=tokenizer.sw2i,
                                   unk_words=True, sos=False, eos=False)
        tokenizer.tw2i = lb2id_dict
        tokenizer.i2tw = id2lb_dict
        tg2ids = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target, vocab_words=tokenizer.tw2i,
                                   unk_words=False, sos=False, eos=False)
        pad_id = tokenizer.sw2i.get(PAD, 0)
        sw_size = len(tokenizer.sw2i)
        tw_size = len(tokenizer.tw2i)
        collate_fn = Tokenizer.collate_fn(pad_id, True)
    else:
        vocab_file = "/media/data/review_response/tokens/bert_level-bpe-vocab.txt"
        tokenizer = BPE.load(vocab_file)
        tokenizer.add_tokens(sys_tokens)
        nl2ids = BPE.tokens2ids(tokenizer, sos=False, eos=False, add_special_tokens=False)
        tg2ids = BPE.tokens2ids(tokenizer, sos=False, eos=False, add_special_tokens=False)

        pad_id = tokenizer.token_to_id(BPAD) if tokenizer.token_to_id(BPAD) is not None else tokenizer.token_to_id(PAD)
        sw_size = tokenizer.get_vocab_size()
        tw_size = tokenizer.get_vocab_size()
        collate_fn = BPE.collate_fn(pad_id, True)

    # load datasets to map into indexes
    if filename.split(".")[-1] == "csv":
        train_data = CSV.get_iterator(filename, firstline=True, task=2)
        num_lines = CSV._len(filename)
    elif filename.split(".")[-1] == "json":
        train_data = JSON.get_iterator(filename, task=2)
        num_lines = JSON._len(filename)
    else:
        raise Exception("Not implement yet")

    train_iterdataset = IterDataset(train_data, source2idx=nl2ids, target2idx=lb2ids, num_lines=num_lines, bpe=True)
    train_dataloader = DataLoader(train_iterdataset, pin_memory=True, batch_size=8, collate_fn=collate_fn)

    for i, batch in enumerate(train_dataloader):
        # inputs, outputs = batch[0], batch[1]
        nl_tensor, lb_tensor = batch
        nl_len_tensor = (nl_tensor != pad_id).sum(dim=1)
        break

    use_selfatt = True
    if use_selfatt:
        # use the maximum length 5 times larger than input length
        nlemb_HPs = [sw_size, 50, None, 0.5, True, True, 1000]
        # nn_mode, ninp, nhid, nlayers, nhead, dropout, activation, norm, his_mask
        enc_HPs = ["self_attention", 50, 200, 6, 10, 0.5, "relu", None, False]
    else:
        nlemb_HPs = [sw_size, 50, None, 0.5, True, True]
        enc_HPs = ["lstm", 50, 200, 2, True, 0.5]
    crf_HPs = [False, len(lb2id_dict), True]
    labeler = Labeler(nlemb_HPs, enc_HPs, crf_HPs, drop_rate=0.5, num_labels=len(lb2id_dict))
    de_score = labeler(nl_tensor, nl_len_tensor)
    output_idx = de_score.max(-1)[1]
    label_mask = nl_tensor != pad_id
    de_loss = labeler.NLL_loss(de_score, lb_tensor, label_mask)


    reference = []
    candidate = []
    label_words = Tokenizer.decode_batch(lb_tensor.tolist(), id2lb_dict, 2)
    label_words = [words[:i] for words, i in zip(label_words, label_mask.sum(dim=1).tolist())]
    predict_words = Tokenizer.decode_batch(output_idx.tolist(), id2lb_dict, 2)
    predict_words = [words[:i] for words, i in zip(predict_words, label_mask.sum(dim=1).tolist())]
    if tokenize_type != "bpe":
        nl_token = tokenizer.decode_batch(nl_tensor.tolist(), tokenizer.i2sw, 2)
        nl_token = [words[:i] if EOT not in words else words[: words.index(EOT)]
                    for words, i in zip(nl_token, (nl_tensor > 0).sum(dim=1).tolist())]
    else:
        nl_token = tokenizer.decode_batch(nl_tensor.tolist())
        # nl_token = [enc_words.tokens for enc_words in self.args.vocab.encode_batch(nl_token)]
        nl_token = [words[0: words.find(EOT)].split() for words in nl_token]
        pass
    # reference = [[w1, ..., EOT], ..., [w1, ..., EOT]]
    reference.extend(label_words)
    candidate.extend(predict_words)
    # test inference
    label_prob, label_pred = labeler.inference(de_score, label_mask)
    predict_words = Tokenizer.decode_batch(label_pred.squeeze(-1).tolist(), id2lb_dict, 2)
    predict_words = [words[:i] for words, i in zip(predict_words, label_mask.sum(dim=1).tolist())]
