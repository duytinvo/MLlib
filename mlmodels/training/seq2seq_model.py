"""
Created on 2018-11-27
@author: duytinvo
"""
import gc
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from datetime import datetime
from mlmodels.metrics.bleu import compute_bleu
from mlmodels.metrics.string_match import compute_string_match_score
from mlmodels.utils.special_tokens import PAD, SOT, EOT, COL, TAB, UNK, PAD_id, SOT_id, EOT_id, UNK_id, COL_id, TAB_id, BPAD, BUNK
from mlmodels.utils.auxiliary import Progbar, Timer, SaveloadHP
from mlmodels.modules.encoder_decoder import EncDec as Seq2seq
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.trad_tokenizer import Tokenizer, sys_tokens
from mlmodels.utils.idx2tensor import Data2tensor, seqPAD
from mlmodels.utils.emb_loader import Embeddings
from mlmodels.modules.radam import RAdam
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler, TensorDataset
from mlmodels.utils.dataset import IterDataset
from mlmodels.utils.BPEtonkenizer import BPE


Data2tensor.set_randseed(1234)


class Translator_model(object):
    def __init__(self, args=None):
        print("INFO: - Load the pre-built tokenizer...")
        if args.tokenize_type != "bpe":
            tokenizer = Tokenizer.load(os.path.join(args.model_dir, "tokenizer.vocab"))
        else:
            tokenizer = BPE.load(args.vocab_file)
            tokenizer.add_tokens(sys_tokens)
            tokenizer.tw2i = tokenizer.get_vocab()
            tokenizer.i2tw = Tokenizer.reversed_dict(tokenizer.tw2i)
        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0" if self.args.use_cuda else "cpu")
        # Include SOt, EOt if set set_words, else Ignore SOt, EOt
        # self.num_labels = len(self.tokenizer.tw2i)
        self.num_labels = self.tokenizer.get_vocab_size()
        if self.num_labels > 2:
            self.lossF = nn.CrossEntropyLoss().to(self.device)
        else:
            self.lossF = nn.BCEWithLogitsLoss().to(self.device)

        # Hyper-parameters at source language
        if self.args.tokenize_type != "bpe":
            self.source2idx = Tokenizer.lst2idx(tokenizer=self.tokenizer.process_nl,
                                                vocab_words=self.tokenizer.sw2i, unk_words=True,
                                                sos=self.args.ssos, eos=self.args.seos)

            # Hyper-parameters at target language
            self.target2idx = Tokenizer.lst2idx(tokenizer=self.tokenizer.process_target,
                                                vocab_words=self.tokenizer.tw2i, unk_words=True,
                                                sos=self.args.tsos, eos=self.args.teos)
            self.pad_id = self.tokenizer.sw2i.get(PAD, 0)
            self.unk_id = self.tokenizer.sw2i.get(UNK, UNK_id)
            sw_size = len(self.tokenizer.sw2i)
            # tw_size = len(self.tokenizer.tw2i)
            self.collate_fn = Tokenizer.collate_fn(self.pad_id, True)
        else:
            self.source2idx = BPE.tokens2ids(self.tokenizer, sos=self.args.ssos, eos=self.args.seos)
            self.target2idx = BPE.tokens2ids(self.tokenizer, sos=self.args.tsos, eos=self.args.teos)
            self.pad_id = self.tokenizer.token_to_id(BPAD) if self.tokenizer.token_to_id(BPAD) is not None \
                else self.tokenizer.token_to_id(PAD)
            self.unk_id = self.tokenizer.token_to_id(BUNK) if self.tokenizer.token_to_id(BUNK) is not None \
                else self.tokenizer.token_to_id(UNK)
            sw_size = self.tokenizer.get_vocab_size()
            # tw_size = self.tokenizer.get_vocab_size()
            self.collate_fn = BPE.collate_fn(self.pad_id, True)

        # Hyper-parameters at word-level source language
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        nlemb_HPs = [sw_size, self.args.swd_dim, self.args.swd_pretrained,
                     self.args.wd_dropout, self.args.wd_padding, self.args.snl_reqgrad]
        # NL inputs
        # Encoder
        # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        if self.args.enc_cnn:
            enc_HPs = ["cnn", self.args.swd_dim, self.args.ed_outdim,
                       self.args.ed_layers, self.args.ed_bidirect, self.args.kernel_size]
        else:
            enc_HPs = [self.args.ed_mode, self.args.swd_dim, self.args.ed_outdim,
                       self.args.ed_layers, self.args.ed_bidirect, self.args.ed_dropout]

        # Decoder
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs

        temb_HPs = [self.num_labels, self.args.twd_dim, self.args.twd_pretrained,
                    self.args.wd_dropout, self.args.wd_padding, self.args.twd_reqgrad]

        # Hyper-parameters at word-level target language
        dec_HPs = [self.args.ed_mode, self.args.twd_dim, self.args.ed_outdim,
                   self.args.ed_layers, self.args.ed_bidirect, self.args.ed_dropout]
        dec_HPs = [temb_HPs, dec_HPs]

        print("INFO: - Build model...")
        # self.seq2seq = Seq2seq(semb_HPs, sch_HPs, enc_HPs, dec_HPs, drop_rate=self.args.final_dropout,
        #                        num_labels=self.num_labels, enc_att=self.args.enc_att).to(self.device)
        self.seq2seq = Seq2seq(nlemb_HPs, enc_HPs, dec_HPs, drop_rate=self.args.final_dropout,
                               num_labels=self.num_labels, enc_att=self.args.enc_att)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.seq2seq = nn.DataParallel(self.seq2seq)
        self.seq2seq.to(self.device)

        self.seq2seq_optimizer = None
        if self.args.optimizer.lower() == "adamax":
            self.init_optimizers(optim.Adamax)

        elif self.args.optimizer.lower() == "adam":
            self.init_optimizers(optim.Adam)

        elif self.args.optimizer.lower() == "radam":
            self.init_optimizers(RAdam)

        elif self.args.optimizer.lower() == "adadelta":
            self.init_optimizers(optim.Adadelta)

        elif self.args.optimizer.lower() == "adagrad":
            self.init_optimizers(optim.Adagrad)

        else:
            self.init_optimizers(optim.SGD)

    def init_optimizers(self, opt_method=optim.SGD):
        self.seq2seq_optimizer = opt_method(self.seq2seq.parameters(), lr=self.args.lr)

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            # label_score = [B, C]; label_tensor = [B, ]
            de_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1,))
        else:
            # label_score = [B, *]; label_tensor = [B, *]
            de_loss = self.lossF(label_score, label_tensor.float().view(-1, 1))
        return de_loss

    def greedy_predict(self, entries, wombat_object=None, maxlen=2000):
        nl = []
        wd_tokens = []
        for entry in entries:
            wd_tokens.append(entry["question_arg"])
            nl.append(self.source2idx(entry["question_arg"]))
        self.seq2seq.eval()
        with torch.no_grad():
            nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=self.pad_id, nlevels=1)
            nl_tensor = Data2tensor.idx2tensor(nl_pad_ids, dtype=torch.long, device=self.device)
            nl_len_tensor = Data2tensor.idx2tensor(nl_lens, dtype=torch.long, device=self.device)
            # wombat_tensor = [batch, nl_len, emb_dim]
            wombat_tensor = torch.zeros(nl_tensor.shape + (self.args.swd_dim,), dtype=torch.float32, device=self.device)
            wombat_idx = (nl_tensor == self.unk_id).nonzero()
            if wombat_object is not None:
                for t, (i, j) in enumerate(wombat_idx.tolist()):
                    wombat_emb = wombat_object.get(wd_tokens[t][i][j])
                    if wombat_emb is not None:
                        wombat_tensor[i, j] = torch.from_numpy(wombat_emb)

            pred_outputs, acc_prob = self.seq2seq.greedy_predict(nl_tensor, nl_len_tensor,
                                                                 maxlen=maxlen, wombat_tensor=wombat_tensor)
            if self.args.tokenize_type != "bpe":
                predict_words = self.tokenizer.decode_batch(pred_outputs.tolist(), self.tokenizer.i2tw, 2)
                predict_words = [words if EOT not in words else words[: words.index(EOT) + 1] for words in
                                 predict_words]
            else:
                predict_words = self.tokenizer.decode_batch(pred_outputs.tolist())
                predict_words = [words[0: words.find(EOT)].split() for words in predict_words]
            # predict_prob = acc_prob.prod(dim=-1).tolist()
            predict_prob = acc_prob.squeeze().tolist()
        for i, entry in enumerate(entries):
            entry['model_result'] = " ".join(predict_words[i])
            entry['pred_prob'] = predict_prob[i]
        return entries

    def beam_predict(self, entries, bw=2, topk=2, wombat_object=None, maxlen=2000):
        nl = []
        wd_tokens = []
        for entry in entries:
            wd_tokens.append(entry["question_arg"])
            nl.append(self.source2idx(entry["question_arg"]))
        self.seq2seq.eval()
        with torch.no_grad():
            nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=self.pad_id, nlevels=1)
            nl_tensor = Data2tensor.idx2tensor(nl_pad_ids, dtype=torch.long, device=self.device)
            nl_len_tensor = Data2tensor.idx2tensor(nl_lens, dtype=torch.long, device=self.device)

            # wombat_tensor = [batch, nl_len, emb_dim]
            wombat_tensor = torch.zeros(nl_tensor.shape + (self.args.swd_dim,), dtype=torch.float32, device=self.device)
            wombat_idx = (nl_tensor == self.unk_id).nonzero()
            if wombat_object is not None:
                for t, (i, j) in enumerate(wombat_idx.tolist()):
                    wombat_emb = wombat_object.get(wd_tokens[t][i][j])
                    if wombat_emb is not None:
                        wombat_tensor[i, j] = torch.from_numpy(wombat_emb)

            pred_outputs, predict_prob = self.seq2seq.beam_predict(nl_tensor, nl_len_tensor,
                                                                   minlen=1, maxlen=maxlen,
                                                                   bw=bw, n_best=topk, wombat_tensor=wombat_tensor)
            if self.args.tokenize_type != "bpe":
                predict_words = self.tokenizer.decode_batch(pred_outputs, self.tokenizer.i2tw, 3)
                predict_words = [words if EOT not in words else words[: words.index(EOT) + 1] for words in
                                 predict_words]
                predict_words = [[" ".join(words) for words in topk_outputs] for topk_outputs in predict_words]
            else:
                predict_words = [self.tokenizer.decode_batch(topk_outputs) for topk_outputs in pred_outputs]
                predict_words = [[words[0: words.find(EOT)] for words in topk_outputs] for topk_outputs in predict_words]
        for i, entry in enumerate(entries):
            entry['model_result'] = predict_words[i][0]
            entry['pred_prob'] = predict_prob[i][0]
            entry['decoded_batch'] = list(zip(predict_words[i], predict_prob[i]))
        return entries

    def evaluate_batch(self, eva_data, num_eva, pred_file):
        start = time.time()
        self.seq2seq.eval()
        nl_tokens = []
        reference = []
        candidate = []
        dev_loss = []
        total_tokens = 0
        eva_iterdataset = IterDataset(eva_data, source2idx=self.source2idx,
                                      target2idx=self.target2idx, num_lines=num_eva)
        eva_dataloader = DataLoader(eva_iterdataset, pin_memory=True,
                                    batch_size=self.args.batch_size, collate_fn=self.collate_fn)
        with torch.no_grad():
            for i, d in enumerate(eva_dataloader):
                d = tuple(t.to(self.device) for t in d)
                nl_tensor, lb_tensor = d
                nl_len_tensor = (nl_tensor != self.pad_id).sum(dim=1)

                random_force = False
                de_score = self.seq2seq(nl_tensor, nl_len_tensor, lb_tensor, random_force)
                olb_tensor = lb_tensor[:, 1:]
                label_mask = olb_tensor != self.pad_id
                # total_lossoss = total_loss.mean()
                # TODO: can move NLL into seq2seq for multigpu
                total_loss = self.NLL_loss(de_score[label_mask], olb_tensor[label_mask])
                dev_loss.append(total_loss.item())
                total_tokens += label_mask.sum()

                output_idx = de_score.max(-1)[1]
                if self.args.tokenize_type != "bpe":
                    label_words = self.tokenizer.decode_batch(olb_tensor.tolist(), self.tokenizer.i2tw, 2)
                    label_words = [words[:i] if EOT not in words else words[: words.index(EOT)]
                                   for words, i in zip(label_words, label_mask.sum(dim=1).tolist())]

                    predict_words = self.tokenizer.decode_batch(output_idx.tolist(), self.tokenizer.i2tw, 2)
                    predict_words = [words[:i] if EOT not in words else words[: words.index(EOT)]
                                     for words, i in zip(predict_words, label_mask.sum(dim=1).tolist())]

                    nl_token = self.tokenizer.decode_batch(nl_tensor.tolist(), self.tokenizer.i2sw, 2)
                    nl_token = [words[:i] if EOT not in words else words[: words.index(EOT)]
                                for words, i in zip(nl_token, (nl_tensor > 0).sum(dim=1).tolist())]
                else:
                    label_words = self.tokenizer.decode_batch(olb_tensor.tolist())
                    # label_words = [enc_words.tokens for enc_words in self.tokenizer.encode_batch(label_words)]
                    label_words = [words[0: words.find(EOT)].split() for words in label_words]

                    predict_words = self.tokenizer.decode_batch(output_idx.tolist())
                    # predict_words = [enc_words.tokens for enc_words in self.tokenizer.encode_batch(predict_words)]
                    predict_words = [words[0: words.find(EOT)].split() for words in predict_words]

                    nl_token = self.tokenizer.decode_batch(nl_tensor.tolist())
                    # nl_token = [enc_words.tokens for enc_words in self.tokenizer.encode_batch(nl_token)]
                    nl_token = [words[0: words.find(EOT)].split() for words in nl_token]

                # reference = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                reference.extend(label_words)

                if sum([len(k) for k in predict_words]) != 0:
                    # candidate = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                    candidate.extend(predict_words)

                nl_tokens.extend(nl_token)
                del nl_tensor, nl_len_tensor, lb_tensor, random_force, de_score, olb_tensor, label_mask
                # gc.collect()
                # torch.cuda.empty_cache()
        bleu_score = [0.0]
        string_match_score = 0.0
        if len(candidate) != 0:
            # Randomly sample one pair
            rand_idx = random.randint(0, len(reference) - 1)
            print("\nRANDOMLY sampling: ")
            print("\t- A LABEL query: ", " ".join(reference[rand_idx]))
            print("\t- A PREDICTED query: ", " ".join(candidate[rand_idx]), "\n")
            bleu_score, string_match_score = Translator_model.class_metrics(list(zip(reference)), candidate,
                                                                            nl_tokens, pred_file)
        end = time.time() - start
        speed = total_tokens / end
        return np.mean(dev_loss), bleu_score, string_match_score, speed

    def train_batch(self, train_data, num_train):
        clip_rate = self.args.clip
        batch_size = self.args.batch_size
        # num_train = train_data.length
        total_batch = num_train // batch_size + 1
        prog = Progbar(target=total_batch)
        # set model in train model
        self.seq2seq.train()
        train_loss = []
        train_iterdataset = IterDataset(train_data, source2idx=self.source2idx,
                                        target2idx=self.target2idx, num_lines=num_train)
        train_dataloader = DataLoader(train_iterdataset, pin_memory=True,
                                      batch_size=batch_size, collate_fn=self.collate_fn)
        for i, d in enumerate(train_dataloader):
            self.seq2seq.zero_grad()
            d = tuple(t.to(self.device) for t in d)
            nl_tensor, lb_tensor = d
            nl_len_tensor = (nl_tensor != self.pad_id).sum(dim=1)

            random_force = True if random.random() < self.args.teacher_forcing_ratio else False
            # print("\nMODEL INPUTs: ", nl_tensor.shape, "\n")
            de_score = self.seq2seq(nl_tensor, nl_len_tensor, lb_tensor, random_force)
            olb_tensor = lb_tensor[:, 1:]
            label_mask = olb_tensor != self.pad_id
            # total_lossoss = total_loss.mean()
            # TODO: can move NLL into seq2seq for multigpu
            total_loss = self.NLL_loss(de_score[label_mask], olb_tensor[label_mask])
            train_loss.append(total_loss.item())
            # Compute gradients wrt all parameters that are connected to the graph
            total_loss.backward()
            # clip gradient if employing
            if clip_rate > 0:
                torch.nn.utils.clip_grad_norm_(self.seq2seq.parameters(), clip_rate)
            # update parameters in all sub-graphs
            self.seq2seq_optimizer.step()
            prog.update(i + 1, [("Train loss", total_loss.item())])
            del nl_tensor, nl_len_tensor, lb_tensor, random_force, de_score, olb_tensor, label_mask
            # gc.collect()
            # torch.cuda.empty_cache()
        return np.mean(train_loss)

    def appendfile(self, line):
        with open(self.args.log_file, "a") as f:
            f.write(line)

    def train(self):
        # training result is returned after training to inform calling code of the outcome of training
        # Values: Matching threshold reached (success): 0, Otherwise: 1
        # training_result = 1
        train_data, train_numlines = Tokenizer.prepare_iter(self.args.train_file, firstline=self.args.firstline, task=2)
        dev_data, dev_numlines = Tokenizer.prepare_iter(self.args.dev_file, firstline=self.args.firstline, task=2)
        test_data, test_numlines = Tokenizer.prepare_iter(self.args.test_file, firstline=self.args.firstline, task=2)

        saved_epoch = 0
        nepoch_no_imprv = 0
        epoch_start = time.time()
        max_epochs = self.args.max_epochs
        best_dev = -np.inf if self.args.metric == "bleu" else np.inf

        if self.args.tl:
            # 1. Load pre-trained model from previous model_dir
            print("INFO: - Load transfer learning models")
            self.load_transferlearning(epoch=-1)
            # 2. Update model_dir to the new one
            if self.args.timestamped_subdir:
                self.args.model_dir = os.path.abspath(os.path.join(self.args.model_dir, ".."))
                sub_folder = datetime.now().isoformat(sep='-', timespec='minutes').replace(":", "-").replace("-", "_")
            else:
                sub_folder = ''
            if not os.path.exists(os.path.join(self.args.model_dir, sub_folder)):
                os.mkdir(os.path.join(self.args.model_dir, sub_folder))
            self.args.model_dir = os.path.join(self.args.model_dir, sub_folder)
            # 3. Update logfile dir
            self.args.log_file = os.path.join(self.args.model_dir, self.args.log_file)
            with open(self.args.log_file, "w") as f:
                f.write("START TRAINING\n")
            # 4. save updated arguments and log file to the new folder
            print("INFO: - Save new argument file")
            SaveloadHP.save(self.args, os.path.join(self.args.model_dir, self.args.model_args))

            dev_loss, dev_bleu, dev_string_match, dev_speed = self.evaluate_batch(dev_data, dev_numlines, self.args.pred_dev_file)
            best_dev = dev_bleu[0] if self.args.metric == "bleu" else dev_loss
            print("INFO: - Transfer learning performance")
            print("         - Current Dev loss: %.4f; Current Dev bleu: %.4f; Current Dev string match: %.4f; Dev speed: %.2f(tokens/s)" %
                  (dev_loss, dev_bleu[0], dev_string_match, dev_speed))
            self.appendfile("\t- Transfer learning performance")
            self.appendfile("\t\t- Current Dev loss: %.4f; Current Dev bleu: %.4f; Current Dev string match: %.4f; Dev speed: %.2f(tokens/s)\n" %
                            (dev_loss, dev_bleu[0], dev_string_match, dev_speed))

            # print("INFO: - Save transfer learning models")
            # self.save_parameters(epoch=0)
            # suppose the transfered model is the best one and save in the main dir
            self.save_parameters(epoch=-1)
        else:
            with open(self.args.log_file, "w") as f:
                f.write("START TRAINING\n")

        print('Dev metric:', self.args.metric)
        for epoch in range(1, max_epochs + 1):
            print("Epoch: %s/%s" % (epoch, max_epochs))
            stime = time.time()
            train_loss = self.train_batch(train_data, train_numlines)
            print("BONUS: Training time of %.4f" % (time.time() - stime))
            # Save the  model
            # print("INFO: - Frequently save models to checkpoint folders")
            # self.save_parameters(epoch=epoch)
            # set the first model as the best one and save to the main dir
            # evaluate on developing data

            dev_loss, dev_bleu, dev_string_match, dev_speed = self.evaluate_batch(dev_data, dev_numlines,
                                                                                  self.args.pred_dev_file)

            dev_metric = dev_bleu[0] if self.args.metric == "bleu" else dev_loss
            cond = dev_metric > best_dev if self.args.metric == "bleu" else dev_loss < best_dev
            if cond:
                nepoch_no_imprv = 0
                saved_epoch = epoch
                best_dev = dev_metric
                print("UPDATES: - New improvement")
                print("         - Train loss: %.4f" % train_loss)
                print("         - Dev loss: %.4f; Dev bleu: %.4f; Dev string match: %.4f; Dev speed: %.2f(tokens/s)" %
                      (dev_loss, dev_bleu[0], dev_string_match, dev_speed))
                self.appendfile("\t- New improvement at epoch %d:\n" % saved_epoch)
                self.appendfile("\t\t- Dev loss: %.4f; Dev bleu: %.4f; Dev string match: %.4f; Dev speed: %.2f(tokens/s)\n" %
                                (dev_loss, dev_bleu[0], dev_string_match, dev_speed))
                print("INFO: - Save best models")
                self.save_parameters(epoch=-1)

                # if dev_string_match >= self.args.matching_threshold:
                #     # TODO: automatically load models to gcp
                #     training_result = 0
                #     break

            else:
                print("UPDATES: - No improvement")
                print("         - Train loss: %.4f" % train_loss)
                print("         - Dev loss: %.4f; Dev bleu: %.4f; Dev string match: %.4f; Dev speed: %.2f(tokens/s)" %
                      (dev_loss, dev_bleu[0], dev_string_match, dev_speed))
                nepoch_no_imprv += 1
                # Decay learning_rate if no improvement
                if self.args.decay_rate > 0:
                    self.lr_decay(epoch)

                if nepoch_no_imprv >= self.args.patience:
                    # Load the current best models
                    print("INFO: - Load best models")
                    self.load_parameters(epoch=-1)

                    test_loss, test_bleu, test_string_match, test_speed = self.evaluate_batch(test_data, test_numlines,
                                                                                              self.args.pred_test_file)
                    print("SUMMARY: - Early stopping after %d epochs without improvements" % nepoch_no_imprv)
                    print("         - Dev metric (%s): %.4f" % (self.args.metric, best_dev))
                    print("         - Test loss: %.4f; Test bleu: %.4f; Test string match: %.4f; Test speed: %.2f(tokens/s)" %
                          (test_loss, test_bleu[0], test_string_match, test_speed))

                    self.appendfile("STOP TRAINING at epoch %s/%s\n" % (epoch, max_epochs))
                    self.appendfile("\t- Testing the best model at epoch %d:\n" % saved_epoch)
                    self.appendfile("\t\t- Test loss: %.4f; Test bleu: %.4f; Test speed: %.2f(tokens/s)\n" %
                                    (test_loss, test_bleu[0], test_speed))
                    return test_bleu[0]

            epoch_finish, epoch_remain = Timer.timeEst2(epoch_start, epoch / max_epochs)
            print("INFO: - Trained time for %d epochs: %s" % (epoch, epoch_finish))
            print("\t- Remained time for %d epochs (est): %s\n" % (max_epochs - epoch, epoch_remain))

        # print("INFO: - Save best models")
        # self.save_parameters(epoch=-1)
        print("INFO: - Load best models")
        self.load_parameters(epoch=-1)

        test_loss, test_bleu, test_string_match, test_speed = self.evaluate_batch(test_data, test_numlines,
                                                                                  self.args.pred_test_file)
        print("SUMMARY: - Completed %d epoches" % max_epochs)
        print("         - Dev metric (%s): %.4f" % (self.args.metric, best_dev))
        print("         - Test loss: %.4f; Test bleu: %.4f; Test string match: %.4f; Test speed: %.2f(tokens/s)" %
              (test_loss, test_bleu[0], test_string_match, test_speed))
        self.appendfile("STOP TRAINING at epoch %s/%s\n" % (epoch, max_epochs))
        self.appendfile("\t- Testing the best model at epoch %d:\n" % saved_epoch)
        self.appendfile("\t\t- Test loss: %.4f; Test bleu: %.4f; Test string match: %.4f; Test speed: %.2f(tokens/s)\n" %
                        (test_loss, test_bleu[0], test_string_match, test_speed))
        return test_bleu[0]

    def save_parameters(self, epoch=1):
        # Convert model to CPU to avoid out of GPU memory
        epoch_dir = os.path.join(self.args.model_dir, 'epoch_' + str(epoch)) if epoch >= 0 else self.args.model_dir
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)

        seq2seq_filename = os.path.join(epoch_dir, self.args.seq2seq_file)
        print("\t- Save seq2seq_filename model to file: %s" % seq2seq_filename)
        self.seq2seq.to("cpu")
        torch.save(self.seq2seq.state_dict(), seq2seq_filename)
        self.seq2seq.to(self.device)

    def load_parameters(self, epoch=1):
        epoch_dir = os.path.join(self.args.model_dir, 'epoch_' + str(epoch)) if epoch >= 0 else self.args.model_dir
        assert os.path.exists(epoch_dir), "Not found epoch dir %s" % epoch_dir

        seq2seq_filename = os.path.join(epoch_dir, self.args.seq2seq_file)
        print("\t- Load enc-token-emb model from file: %s" % seq2seq_filename)
        self.seq2seq.load_state_dict(torch.load(seq2seq_filename))
        self.seq2seq.to(self.device)

    def load_transferlearning(self, epoch=-1):
        epoch_dir = os.path.join(self.args.model_dir, 'epoch_' + str(epoch)) if epoch >= 0 else self.args.model_dir
        try:
            seq2seq_filename = os.path.join(epoch_dir, self.args.seq2seq_file)
            print("\t- Load enc-token-emb model from file: %s" % seq2seq_filename)
            self.seq2seq.load_state_dict(torch.load(seq2seq_filename))
            self.seq2seq.to(self.device)
        except:
            print("\t- UNABLE to load seq2seq_filename model from file: %s" % seq2seq_filename)

    def lr_decay(self, epoch):
        lr = self.args.lr / (1 + self.args.decay_rate * epoch)
        print("INFO: - No improvement; Learning rate is setted as: %f" % lr)
        for param_group in self.seq2seq_optimizer.param_groups:
            param_group['lr'] = lr

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    @staticmethod
    def filter_pad(label_words, seq_len):
        # label_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
        label_words = list(zip(*label_words))
        # label_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
        # ignore EOT (i-1)
        filter_words = [words[:i] if EOT not in words else words[: words.index(EOT)]
                        for words, i in zip(label_words, seq_len)]
        # print("Sequence length: ", seq_len)
        # print("Before filter: ", label_words)
        # print("After filter: ", filter_words)
        return filter_words

    @staticmethod
    def class_metrics(reference, candidate, nl_tokens, pred_file):
        bleu_score = compute_bleu(reference, candidate)
        string_match_score = compute_string_match_score(reference, candidate, nl_tokens, pred_file)
        return bleu_score, string_match_score

    @staticmethod
    def build_data(args):
        if not args.tl:
            if not os.path.exists(args.model_dir):
                os.mkdir(args.model_dir)
            if args.timestamped_subdir:
                sub_folder = datetime.now().isoformat(sep='-', timespec='minutes').replace(":", "-").replace("-", "_")
            else:
                sub_folder = ''
            if not os.path.exists(os.path.join(args.model_dir, sub_folder)):
                os.mkdir(os.path.join(args.model_dir, sub_folder))
            args.model_dir = os.path.join(args.model_dir, sub_folder)
            args.log_file = os.path.join(args.model_dir, args.log_file)
            if args.tokenize_type != "bpe":
                s_paras = [args.wl_th, args.wcutoff]
                t_paras = [args.wl_th, args.wcutoff]
                print("INFO: - Build vocabulary...")

                tokenizer = Tokenizer(s_paras, t_paras)
                files = [args.train_file]
                if args.train_file != args.dev_file:
                    files.append(args.dev_file)
                # Load datasets to build vocabulary
                data = Tokenizer.load_file(files, task=2)
                tokenizer.build(datasets=data)
                sw2i = tokenizer.sw2i
                tw2i = tokenizer.tw2i
                print("INFO: - Save vocabulary...")
                Tokenizer.save(tokenizer, os.path.join(args.model_dir, "tokenizer.vocab"))
            else:
                print("INFO: - Load vocabulary...")
                tokenizer = BPE.load(args.vocab_file)
                tokenizer.add_tokens(sys_tokens)
                sw2i = tokenizer.get_vocab()
                tw2i = tokenizer.get_vocab()

            # args.tokenizer = tokenizer
            # Source language
            args.swd_pretrained = None
            args.twd_pretrained = None
            if len(args.swd_embfile) != 0:
                scale = np.sqrt(3.0 / args.swd_dim)
                emb_reader = Embeddings(args.swd_embfile)
                args.swd_pretrained = emb_reader.get_W(args.swd_dim, sw2i, scale)
                if args.twd_embfile == args.swd_embfile:
                    scale = np.sqrt(3.0 / args.twd_dim)
                    args.twd_pretrained = emb_reader.get_W(args.twd_dim, tw2i, scale)

            # Target language
            if len(args.twd_embfile) != 0:
                scale = np.sqrt(3.0 / args.twd_dim)
                if args.twd_pretrained is None:
                    emb_reader = Embeddings(args.swd_embfile)
                args.twd_pretrained = emb_reader.get_W(args.twd_dim, tw2i, scale)

            # directly integrate transfer learning if no updating new words
            SaveloadHP.save(args, os.path.join(args.model_dir, args.model_args))
            return args
        else:
            print("INFO: - Use transfer learning technique")
            assert os.path.exists(args.tlargs), print("\t - There is no pre-trained argument file")
            # load pre-trained argument file from a previous training folder
            margs = SaveloadHP.load(args.tlargs)
            # margs.tl = args.tl
            # margs.log_file = args.log_file

            # TODO update new vocab and all other new arguments used for new training
            # 0. Read vocab
            # 1. Update schema
            # 2. Update vocab
            # args.tokenizer = margs.tokenizer
            # 3. Use all model file directory of previous train
            args.model_dir = margs.model_dir
            args.seq2seq_file = margs.seq2seq_file
            # 4. Keep the remaining current arguments
            # add a constraint at the loading time that if fail to load any model, just skip it
            args.swd_pretrained = margs.swd_pretrained
            args.twd_pretrained = margs.twd_pretrained
            return args


def main(argv):
    argparser = argparse.ArgumentParser(argv)

    argparser.add_argument("--timestamped_subdir", action='store_true', default=False,
                           help="Save models in timestamped subdirectory")

    argparser.add_argument("--log_file", type=str, default="logging.txt", help="log_file")

    argparser.add_argument('--vocab_file', help='file to save a pre-trained tokenizer', type=str,
                           default="/media/data/review_response/tokens/bert_level-bpe-vocab.txt")

    argparser.add_argument('--train_file', help='Trained file (semQL) in Json format', type=str,
                           default="/media/data/review_response/Dev.json")

    argparser.add_argument('--dev_file', help='Validated file (semQL) in Json format', type=str,
                           default="/media/data/review_response/Dev.json")

    argparser.add_argument('--test_file', help='Tested file (semQL) in Json format', type=str,
                           default="/media/data/review_response/Test.json")

    argparser.add_argument('--pred_dev_file', help='Regression test on the valuation file', type=str,
                           default="/media/data/review_response/regression/Dev.json")

    argparser.add_argument('--pred_test_file', help='Regression test on the valuation file', type=str,
                           default="/media/data/review_response/regression/Test.json")

    # Language parameters
    argparser.add_argument("--tokenize_type", type=str, default="bpe", help="tokenize type", choices=["bpe", "splitter"])

    argparser.add_argument("--wl_th", type=int, default=-1, help="Word length threshold")

    argparser.add_argument("--wcutoff", type=int, default=10, help="Prune words occurring <= wcutoff")

    argparser.add_argument("--wd_dropout", type=float, default=0.5,
                           help="Dropout rate at word-level embedding")

    argparser.add_argument("--wd_padding", action='store_true', default=False,
                           help="Flag to set all padding tokens to zero during training at word level")

    argparser.add_argument("--ed_bidirect", action='store_true', default=False,
                           help="Word-level NN Bi-directional flag")

    argparser.add_argument("--ed_mode", type=str, default="lstm", help="Word-level neural network type")

    argparser.add_argument("--ed_outdim", type=int, default=300,
                           help="Source Word-level neural network dimension")

    argparser.add_argument("--ed_layers", type=int, default=2, help="Source Number of NN layers at word level")

    argparser.add_argument("--ed_dropout", type=float, default=0.5,
                           help="Dropout rate at the encoder-decoder layer")

    argparser.add_argument("--enc_cnn", action='store_true', default=False,
                           help="Encoder-level CNN flag")

    argparser.add_argument("--kernel_size", type=int, default=3, help="kernel_size of CNN (ks = 2*pad + 1)")

    argparser.add_argument("--enc_att", action='store_true', default=False,
                           help="Encoder-level NN attentional mechanism flag")

    # Source language parameters
    argparser.add_argument("--ssos", action='store_true', default=False,
                           help="Start padding flag at a source sentence level")

    argparser.add_argument("--seos", action='store_true', default=False,
                           help="End padding flag at a source sentence level (True)")

    argparser.add_argument("--swd_embfile", type=str, help="Source Word embedding file", default="")

    argparser.add_argument("--swd_dim", type=int, default=300, help="Source Word embedding size")

    # argparser.add_argument("--swd_reqgrad", action='store_true', default=False,
    #                        help="Either freezing or unfreezing pretrained embedding")
    argparser.add_argument("--snl_reqgrad", action='store_true', default=False,
                           help="Either freezing or unfreezing word pretrained embedding")

    # Target language parameters
    argparser.add_argument("--tsos", action='store_true', default=False,
                           help="Start padding flag at a target sentence level")

    argparser.add_argument("--teos", action='store_true', default=False,
                           help="End padding flag at a target sentence level (True)")

    argparser.add_argument("--t_reverse", action='store_true', default=False,
                           help="Reversing flag (reverse the sequence order of target language)")

    argparser.add_argument("--twd_embfile", type=str, help="Target Word embedding file", default="")

    argparser.add_argument("--twd_dim", type=int, default=300, help="Target Word embedding size")

    argparser.add_argument("--twd_reqgrad", action='store_true', default=False,
                           help="Either freezing or unfreezing pretrained embedding")

    # Other parameters
    argparser.add_argument("--final_dropout", type=float, default=0.5, help="Dropout rate at the last layer")

    argparser.add_argument("--patience", type=int, default=32,
                           help="Early stopping if no improvement after patience epoches")

    argparser.add_argument("--optimizer", type=str, default="ADAM", help="Optimized method (adagrad, sgd, ...)")

    argparser.add_argument("--metric", type=str, default="bleu", help="Optimized criterion (loss or bleu)")

    argparser.add_argument("--lr", type=float, default=0.001, help="Learning rate (ADAM: 0.001)")

    argparser.add_argument("--decay_rate", type=float, default=0.05, help="Decay rate (0.05)")

    argparser.add_argument("--max_epochs", type=int, default=256, help="Maximum trained epochs")

    argparser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")

    argparser.add_argument('--clip', default=5, type=int, help='Clipping value (5)')

    argparser.add_argument('--model_dir', help='Model directory',
                           default="/media/data/review_response/trained_model/", type=str)

    argparser.add_argument('--seq2seq_file', help='Trained seq2seq_file filename',
                           default="seq2seq.m", type=str)

    argparser.add_argument('--model_args', help='Trained argument filename', default="translator.args", type=str)

    argparser.add_argument("--use_cuda", action='store_false', default=True, help="GPUs Flag (default False)")

    argparser.add_argument('--teacher_forcing_ratio', help='teacher forcing ratio', default=1.0, type=float)

    argparser.add_argument("--tl", action='store_true', default=False, help="transfer learning (default False)")

    argparser.add_argument('--tlargs', help='Trained argument transfer learning filename', default="", type=str)

    # argparser.add_argument('--matching_threshold', help='matching acc threshold', default=0.9999, type=float)

    args, unknown = argparser.parse_known_args()

    args = Translator_model.build_data(args)

    translator = Translator_model(args)

    training_result = translator.train()
    return training_result


if __name__ == '__main__':
    """
    python seq2seq_model.py --use_cuda --teacher_forcing_ratio 0.8
    """
    main(sys.argv)
