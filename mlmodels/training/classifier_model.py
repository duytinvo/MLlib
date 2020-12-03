"""
Created on 2018-11-27
@author: duytinvo
"""
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from datetime import datetime
from mlmodels.metrics.prf1 import APRF1
from mlmodels.utils.special_tokens import PAD, SOT, EOT, UNK, PAD_id, SOT_id, EOT_id, UNK_id, NULL, BPAD, BUNK
from mlmodels.utils.auxiliary import Progbar, Timer, SaveloadHP
from mlmodels.modules.classifier import Classifier
# from mlmodels.utils.vocab_builder import Vocab
from mlmodels.utils.idx2tensor import Data2tensor, seqPAD
from mlmodels.utils.emb_loader import Embeddings
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.txtIO import TXT
from mlmodels.modules.radam import RAdam
from mlmodels.utils.trad_tokenizer import Tokenizer, sys_tokens
from mlmodels.utils.BPEtonkenizer import BPE
from mlmodels.utils.dataset import IterDataset
from torch.utils.data import DataLoader
import torch.nn as nn

Data2tensor.set_randseed(12345)


class Classifier_model(object):
    def __init__(self, args=None):
        print("INFO: - Load the pre-built tokenizer...")
        if args.tokenize_type != "bpe":
            tokenizer = Tokenizer.load(os.path.join(args.model_dir, "tokenizer.vocab"))
        else:
            tokenizer = BPE.load(args.vocab_file)
            tokenizer.add_tokens(sys_tokens)

        labels_list = TXT.read(args.label_file, firstline=False)
        tokenizer.tw2i = Tokenizer.list2dict(sys_tokens + labels_list)
        tokenizer.i2tw = Tokenizer.reversed_dict(tokenizer.tw2i)
        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0" if self.args.use_cuda else "cpu")
        self.num_labels = len(self.tokenizer.tw2i)
        # Hyper-parameters at target language
        self.target2idx = Tokenizer.lst2idx(tokenizer=Tokenizer.process_target,
                                            vocab_words=self.tokenizer.tw2i, unk_words=True,
                                            sos=self.args.ssos, eos=self.args.seos)

        if self.args.tokenize_type != "bpe":
            # Hyper-parameters at source language
            self.source2idx = Tokenizer.lst2idx(tokenizer=Tokenizer.process_nl,
                                                vocab_words=self.tokenizer.sw2i, unk_words=True,
                                                sos=self.args.ssos, eos=self.args.seos)

            self.pad_id = self.tokenizer.sw2i.get(PAD, PAD_id)
            self.unk_id = self.tokenizer.sw2i.get(UNK, UNK_id)
            sw_size = len(self.tokenizer.sw2i)
            # tw_size = len(self.tokenizer.tw2i)
            self.collate_fn = Tokenizer.collate_fn(self.pad_id, True)
        else:
            self.source2idx = BPE.tokens2ids(self.tokenizer, sos=self.args.ssos, eos=self.args.seos)
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

        # Encoder
        # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        if self.args.enc_cnn:
            enc_HPs = ["cnn", self.args.swd_dim, self.args.ed_outdim,
                       self.args.ed_layers, self.args.ed_bidirect, self.args.kernel_size]
        else:
            if self.args.ed_mode == "self_attention":
                # use the maximum length 5 times larger than input length
                nlemb_HPs += [self.tokenizer.swl*5]
                # nn_mode, ninp, nhid, nlayers, nhead, dropout, activation, norm, his_mask
                enc_HPs = [self.args.ed_mode, self.args.swd_dim, self.args.ed_outdim, self.args.ed_layers,
                           self.args.ed_heads, self.args.ed_dropout, self.args.ed_activation, None, self.args.ed_hismask]
            else:
                enc_HPs = [self.args.ed_mode, self.args.swd_dim, self.args.ed_outdim,
                           self.args.ed_layers, self.args.ed_bidirect, self.args.ed_dropout]

        print("INFO: - Build model...")
        self.classifier = Classifier(nlemb_HPs, enc_HPs, drop_rate=self.args.final_dropout, num_labels=self.num_labels)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.classifier = nn.DataParallel(self.classifier)
        self.classifier.to(self.device)

        self.classifier_optimizer = None
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
        self.classifier_optimizer = opt_method(self.classifier.parameters(), lr=self.args.lr)

    def predict_batch(self, entries, wombat_object=None):
        nl = []
        wd_tokens = []
        for entry in entries:
            input_tokens = entry["input_tokens"]
            ids = self.source2idx(input_tokens)
            nl.append(ids)
            if self.args.tokenize_type != "bpe":
                entry['input_list'] = self.tokenizer.process_nl(input_tokens)
            else:
                entry['input_list'] = self.tokenizer.encode(input_tokens, add_special_tokens=False).tokens
            wd_tokens.append(entry['input_list'])
        self.classifier.eval()
        with torch.no_grad():
            nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=self.pad_id, nlevels=1)
            nl_tensor = Data2tensor.idx2tensor(nl_pad_ids, dtype=torch.long, device=self.device)
            nl_len_tensor = Data2tensor.idx2tensor(nl_lens, dtype=torch.long, device=self.device)
            # wombat_tensor = [batch, nl_len, emb_dim]
            wombat_tensor = torch.zeros(nl_tensor.shape + (self.args.swd_dim,), dtype=torch.float32, device=self.device)
            wombat_idx = (nl_tensor == self.unk_id).nonzero()
            if wombat_object is not None:
                for t, (i, j) in enumerate(wombat_idx.tolist()):
                    word_to_lookup = wd_tokens[i][j]
                    print('Looking up Wombat for:', word_to_lookup)
                    wombat_emb = wombat_object.get(word_to_lookup)
                    if wombat_emb is not None:
                        print('Found Wombat embedding for:', word_to_lookup)
                        wombat_tensor[i, j] = torch.from_numpy(wombat_emb)
            de_score = self.classifier(nl_tensor, nl_len_tensor, wombat_tensor=wombat_tensor)
            label_mask = nl_tensor > 0
            output_prob, output_idx = self.classifier.inference(de_score)
            # output_idx = de_score.max(-1)[1]
            predict_words = Tokenizer.decode_batch(output_idx.squeeze(-1).tolist(), self.tokenizer.i2tw, 1)
            # predict_prob = acc_prob.prod(dim=-1).tolist()
            predict_prob = output_prob.squeeze(-1).tolist()

        for i, entry in enumerate(entries):
            # entry["pred_pair"] = list(zip(entry["input_review"], predict_words[i]))
            entry['pred_sequence'] = predict_words[i]
            entry['prob_sequence'] = predict_prob[i]
        return entries

    def evaluate_batch(self, eva_data, num_eva):
        start = time.time()
        self.classifier.eval()
        nl_tokens = []
        reference = []
        candidate = []
        predict_probs = []
        dev_loss = []
        total_docs = 0
        eva_iterdataset = IterDataset(eva_data, source2idx=self.source2idx,
                                      target2idx=self.target2idx, num_lines=num_eva,
                                      bpe=False)
        eva_dataloader = DataLoader(eva_iterdataset, pin_memory=True,
                                    batch_size=self.args.batch_size, collate_fn=self.collate_fn)
        with torch.no_grad():
            for i, d in enumerate(eva_dataloader):
                # nl, target = list(zip(*d))
                d = tuple(t.to(self.device) for t in d)
                nl_tensor, lb_tensor = d
                nl_len_tensor = (nl_tensor != self.pad_id).sum(dim=1)

                de_score = self.classifier(nl_tensor, nl_len_tensor)
                # TODO: can move NLL into seq2seq for multigpu
                total_loss = self.classifier.NLL_loss(de_score, lb_tensor)

                dev_loss.append(total_loss.item())
                total_docs += nl_tensor.size(0)
                output_prob, output_idx = self.classifier.inference(de_score)

                label_words = Tokenizer.decode_batch(lb_tensor.squeeze(-1).tolist(), self.tokenizer.i2tw, 1)
                reference.extend(label_words)
                predict_words = Tokenizer.decode_batch(output_idx.squeeze(-1).tolist(), self.tokenizer.i2tw, 1)
                # predict_prob = acc_prob.prod(dim=-1).tolist()
                predict_probs += output_prob.squeeze(-1).tolist()
                # if sum([len(k) for k in predict_words]) != 0:
                # candidate = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                candidate.extend(predict_words)

                if self.args.tokenize_type != "bpe":
                    nl_token = self.tokenizer.decode_batch(nl_tensor.tolist(), self.tokenizer.i2sw, 2)
                    nl_token = [words[:i] if EOT not in words else words[: words.index(EOT)]
                                for words, i in zip(nl_token, (nl_tensor > 0).sum(dim=1).tolist())]
                else:
                    nl_token = self.tokenizer.decode_batch(nl_tensor.tolist())
                    # nl_token = [enc_words.tokens for enc_words in self.tokenizer.encode_batch(nl_token)]
                    nl_token = [words[0: words.find(EOT)].split() for words in nl_token]
                nl_tokens.extend(nl_token)
                del nl_tensor, nl_len_tensor, lb_tensor, de_score
                # gc.collect()
                # torch.cuda.empty_cache()

        if len(candidate) != 0 and len(reference) != 0:
            assert len(candidate) == len(reference)
            # Randomly sample one pair
            rand_idx = random.randint(0, len(reference) - 1)
            print("\nRANDOMLY sampling: ")
            print("\t- An Input Sequence: ", " ".join(nl_tokens[rand_idx]))
            print("\t- A LABEL query: ", " ".join(reference[rand_idx]))
            print("\t- A PREDICTED query: ", " ".join(candidate[rand_idx]))
            print("\t- A PREDICTED prob: ", predict_probs[rand_idx], "\n\n")
            metrics = Classifier_model.class_metrics(reference, candidate)
        else:
            metrics = [0., 0., 0., 0.]
        end = time.time() - start
        speed = total_docs / end
        return sum(dev_loss)/len(dev_loss), metrics, speed

    def train_batch(self, train_data, num_train):
        clip_rate = self.args.clip
        batch_size = self.args.batch_size
        # num_train = len(train_data)
        total_batch = num_train // batch_size + 1
        prog = Progbar(target=total_batch)
        # set model in train model
        self.classifier.train()
        train_loss = []
        total_tokens = 0
        train_iterdataset = IterDataset(train_data, source2idx=self.source2idx,
                                        target2idx=self.target2idx, num_lines=num_train,
                                        bpe=False)
        train_dataloader = DataLoader(train_iterdataset, pin_memory=True,
                                      batch_size=batch_size, collate_fn=self.collate_fn)

        for i, d in enumerate(train_dataloader):
            self.classifier.zero_grad()
            # nl, target = list(zip(*d))
            d = tuple(t.to(self.device) for t in d)
            nl_tensor, lb_tensor = d
            # assert nl_tensor.shape == lb_tensor.shape
            nl_len_tensor = (nl_tensor != self.pad_id).sum(dim=1)

            # print("\nMODEL INPUTs: ", nl_tensor.shape, "\n")
            de_score = self.classifier(nl_tensor, nl_len_tensor)
            label_mask = nl_tensor > 0
            total_tokens += label_mask.sum()
            # TODO: can move NLL into seq2seq for multigpu
            total_loss = self.classifier.NLL_loss(de_score, lb_tensor)
            train_loss.append(total_loss.item())
            # Compute gradients wrt all parameters that are connected to the graph
            total_loss.backward()
            # clip gradient if employing
            if clip_rate > 0:
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), clip_rate)
            # update parameters in all sub-graphs
            self.classifier_optimizer.step()
            prog.update(i + 1, [("Train loss", total_loss.item())])
            del nl_tensor, nl_len_tensor, lb_tensor, de_score, label_mask
            # gc.collect()
            # torch.cuda.empty_cache()
        return sum(train_loss)/len(train_loss)

    def appendfile(self, line):
        with open(self.args.log_file, "a") as f:
            f.write(line)

    def train(self):
        train_data, train_numlines = Tokenizer.prepare_iter(self.args.train_file, firstline=self.args.firstline, task=1)
        dev_data, dev_numlines = Tokenizer.prepare_iter(self.args.dev_file, firstline=self.args.firstline, task=1)
        test_data, test_numlines = Tokenizer.prepare_iter(self.args.test_file, firstline=self.args.firstline, task=1)

        saved_epoch = 0
        nepoch_no_imprv = 0
        epoch_start = time.time()
        max_epochs = self.args.max_epochs
        # best_dev = -np.inf if self.args.metric == "f1" else np.inf
        best_dev = np.inf if self.args.metric == "loss" else -np.inf

        with open(self.args.log_file, "w") as f:
            f.write("START TRAINING\n")
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

            # 4. save updated arguments and log file to the new folder
            print("INFO: - Save new argument file")
            SaveloadHP.save(self.args, os.path.join(self.args.model_dir, self.args.model_args))

            dev_loss, dev_metrics, dev_speed = self.evaluate_batch(dev_data, dev_numlines)
            # best_dev = dev_metrics[2] if self.args.metric == "f1" else dev_loss
            best_dev = dev_loss if self.args.metric == "loss" else dev_metrics[2]
            print("INFO: - Transfer learning performance")
            print("         - Current Dev loss: %.4f; Current Dev P: %.4f; Current Dev R: %.4f; "
                  "Current Dev F1: %.4f; Dev speed: %.2f(tokens/s)" %
                  (dev_loss, dev_metrics[0], dev_metrics[1], dev_metrics[2], dev_speed))
            self.appendfile("\t- Transfer learning performance")
            self.appendfile("\t\t- Current Dev loss: %.4f; Current Dev P: %.4f; Current Dev R: %.4f; "
                            "Current Dev F1: %.4f; Dev speed: %.2f(tokens/s)" %
                            (dev_loss, dev_metrics[0], dev_metrics[1], dev_metrics[2], dev_speed))

            # print("INFO: - Save transfer learning models")
            # self.save_parameters(epoch=0)
            # suppose the transfered model is the best one and save in the main dir
            self.save_parameters(epoch=-1)

        for epoch in range(1, max_epochs + 1):
            print("Epoch: %s/%s" % (epoch, max_epochs))
            stime = time.time()
            train_loss = self.train_batch(train_data, train_numlines)
            print("BONUS: Training time of %.4f" % (time.time() - stime))
            # Save the  model
            # print("INFO: - Frequently save models to checkpoint folders")
            # self.save_parameters(epoch=epoch)
            # evaluate on developing data
            dev_loss, dev_metrics, dev_speed = self.evaluate_batch(dev_data, dev_numlines)
            # dev_metric = dev_metrics[2] if self.args.metric == "f1" else dev_loss
            dev_metric = dev_loss if self.args.metric == "loss" else dev_metrics[2]
            # cond = dev_metric > best_dev if self.args.metric == "f1" else dev_loss < best_dev
            cond = dev_loss < best_dev if self.args.metric == "loss" else dev_metric > best_dev
            if cond:
                nepoch_no_imprv = 0
                saved_epoch = epoch
                best_dev = dev_metric
                print("UPDATES: - New improvement")
                print("         - Train loss: %.4f" % train_loss)
                print("         - Current Dev loss: %.4f; Current Dev P: %.4f; Current Dev R: %.4f; "
                      "Current Dev F1: %.4f; Dev speed: %.2f(tokens/s)" %
                      (dev_loss, dev_metrics[0], dev_metrics[1], dev_metrics[2], dev_speed))
                self.appendfile("\t- New improvement at epoch %d:\n" % saved_epoch)
                self.appendfile("\t\t- Current Dev loss: %.4f; Current Dev P: %.4f; Current Dev R: %.4f; "
                                "Current Dev F1: %.4f; Dev speed: %.2f(tokens/s)\n" %
                                (dev_loss, dev_metrics[0], dev_metrics[1], dev_metrics[2], dev_speed))
                print("INFO: - Save best models")
                self.save_parameters(epoch=-1)

            else:
                print("UPDATES: - No improvement")
                print("         - Train loss: %.4f" % train_loss)
                print("         - Current Dev loss: %.4f; Current Dev P: %.4f; Current Dev R: %.4f; "
                      "Current Dev F1: %.4f; Dev speed: %.2f(tokens/s)" %
                      (dev_loss, dev_metrics[0], dev_metrics[1], dev_metrics[2], dev_speed))
                nepoch_no_imprv += 1
                # Decay learning_rate if no improvement
                if self.args.decay_rate > 0:
                    self.lr_decay(epoch)

                if nepoch_no_imprv >= self.args.patience:
                    # Load the current best models
                    print("INFO: - Load best models")
                    self.load_parameters(epoch=-1)

                    test_loss, test_metrics, test_speed = self.evaluate_batch(test_data, test_numlines)
                    print("SUMMARY: - Early stopping after %d epochs without improvements" % nepoch_no_imprv)
                    print("         - Dev metric (%s): %.4f" % (self.args.metric, best_dev))
                    print("         - Test loss: %.4f; Test P: %.4f; Test R: %.4f; "
                          "Test F1: %.4f; Test speed: %.2f(tokens/s)" %
                          (test_loss, test_metrics[0], test_metrics[1], test_metrics[2], test_speed))

                    self.appendfile("STOP TRAINING at epoch %s/%s\n" % (epoch, max_epochs))
                    self.appendfile("\t- Testing the best model at epoch %d:\n" % saved_epoch)
                    self.appendfile("\t\t- Test loss: %.4f; Test P: %.4f; Test R: %.4f; "
                                    "Test F1: %.4f; Test speed: %.2f(tokens/s)\n" %
                                    (test_loss, test_metrics[0], test_metrics[1], test_metrics[2], test_speed))
                    return test_metrics

            epoch_finish, epoch_remain = Timer.timeEst2(epoch_start, epoch / max_epochs)
            print("INFO: - Trained time for %d epochs: %s" % (epoch, epoch_finish))
            print("\t- Remained time for %d epochs (est): %s\n" % (max_epochs - epoch, epoch_remain))

        print("INFO: - Load best models")
        self.load_parameters(epoch=-1)

        test_loss, test_metrics, test_speed = self.evaluate_batch(test_data, test_numlines)
        print("SUMMARY: - Completed %d epoches" % max_epochs)
        print("         - Dev metric (%s): %.4f" % (self.args.metric, best_dev))
        print("         - Test loss: %.4f; Test P: %.4f; Test R: %.4f; "
              "Test F1: %.4f; Test speed: %.2f(tokens/s)" %
              (test_loss, test_metrics[0], test_metrics[1], test_metrics[2], test_speed))
        self.appendfile("STOP TRAINING at epoch %s/%s\n" % (epoch, max_epochs))
        self.appendfile("\t- Testing the best model at epoch %d:\n" % saved_epoch)
        self.appendfile("\t\t- Test loss: %.4f; Test P: %.4f; Test R: %.4f; "
                        "Test F1: %.4f; Test speed: %.2f(tokens/s)\n" %
                        (test_loss, test_metrics[0], test_metrics[1], test_metrics[2], test_speed))
        return test_metrics

    def save_parameters(self, epoch=1):
        # Convert model to CPU to avoid out of GPU memory
        epoch_dir = os.path.join(self.args.model_dir, 'epoch_' + str(epoch)) if epoch >= 0 else self.args.model_dir
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)

        classifier_filename = os.path.join(epoch_dir, self.args.classifier_file)
        print("\t- Save model to file: %s" % classifier_filename)
        self.classifier.to("cpu")
        torch.save(self.classifier.state_dict(), classifier_filename)
        self.classifier.to(self.device)

    def load_parameters(self, epoch=1):
        epoch_dir = os.path.join(self.args.model_dir, 'epoch_' + str(epoch)) if epoch >= 0 else self.args.model_dir
        assert os.path.exists(epoch_dir), "Not found epoch dir %s" % epoch_dir

        classifier_filename = os.path.join(epoch_dir, self.args.classifier_file)
        print("\t- Load model from file: %s" % classifier_filename)
        self.classifier.load_state_dict(torch.load(classifier_filename))
        self.classifier.to(self.device)

    def load_transferlearning(self, epoch=-1):
        epoch_dir = os.path.join(self.args.model_dir, 'epoch_' + str(epoch)) if epoch >= 0 else self.args.model_dir
        try:
            classifier_filename = os.path.join(epoch_dir, self.args.classifier_file)
            print("\t- Load model from file: %s" % classifier_filename)
            self.classifier.load_state_dict(torch.load(classifier_filename))
            self.classifier.to(self.device)
        except:
            print("\t- UNABLE to load model from file: %s" % self.args.classifier_file)

    def lr_decay(self, epoch):
        lr = self.args.lr / (1 + self.args.decay_rate * epoch)
        print("INFO: - No improvement; Learning rate is setted as: %f" % lr)
        for param_group in self.classifier_optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def class_metrics(reference, candidate):
        P, R, F1, ACC = APRF1.sklearn(reference, candidate)
        return P, R, F1, ACC

    @staticmethod
    def build_data(args):
        args.tl = True if len(args.tlargs) != 0 else False
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
                data = Tokenizer.load_file(files, task=1)
                tokenizer.build(datasets=data)
                sw2i = tokenizer.sw2i
                print("INFO: - Save vocabulary...")
                Tokenizer.save(tokenizer, os.path.join(args.model_dir, "tokenizer.vocab"))
            else:
                print("INFO: - Load vocabulary...")
                tokenizer = BPE.load(args.vocab_file)
                tokenizer.add_tokens(sys_tokens)
                sw2i = tokenizer.get_vocab()
                # tw2i = tokenizer.get_vocab()
            # labels_list = TXT.read(args.label_file, firstline=False)
            # lb2id_dict = Tokenizer.list2dict(sys_tokens + labels_list)
            # id2lb_dict = Tokenizer.reversed_dict(lb2id_dict)
            # tokenizer.tw2i = lb2id_dict
            # tokenizer.i2tw = id2lb_dict
            # args.tokenizer = tokenizer
            # Source language
            args.swd_pretrained = None
            if len(args.swd_embfile) != 0:
                scale = np.sqrt(3.0 / args.swd_dim)
                emb_reader = Embeddings(args.swd_embfile)
                args.swd_pretrained = emb_reader.get_W(args.swd_dim, sw2i, scale)

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
            args.classifier_file = margs.classifier_file
            # 4. Keep the remaining current arguments
            # add a constraint at the loading time that if fail to load any model, just skip it
            args.swd_pretrained = margs.swd_pretrained
            return args


def main(argv):
    argparser = argparse.ArgumentParser(argv)
    # input peripherals
    argparser.add_argument('--vocab_file', help='file to save a pre-trained tokenizer', type=str,
                           default="/media/data/review_response/tokens/bert_level-bpe-vocab.txt")
    argparser.add_argument('--label_file', help='Trained file (semQL) in Json format', type=str,
                           default="/media/data/classification/datasets/yelp_review_full_csv/labels.txt")
    argparser.add_argument('--train_file', help='Trained file (semQL) in Json format', type=str,
                           default="/media/data/classification/datasets/yelp_review_full_csv/dev.s.csv")
    argparser.add_argument('--dev_file', help='Validated file (semQL) in Json format', type=str,
                           default="/media/data/classification/datasets/yelp_review_full_csv/dev.s.csv")
    argparser.add_argument('--test_file', help='Tested file (semQL) in Json format', type=str,
                           default="/media/data/classification/datasets/yelp_review_full_csv/test.csv")
    argparser.add_argument("--firstline", action='store_true', default=False,
                           help="labelled files having a header")

    # output peripherals
    argparser.add_argument("--timestamped_subdir", action='store_true', default=False,
                           help="Save models in timestamped subdirectory")
    argparser.add_argument("--log_file", type=str, default="logging.txt", help="log_file")
    argparser.add_argument('--model_dir', help='Model directory',
                           default="./data/reviews/trained_model/", type=str)
    argparser.add_argument('--model_args', help='Trained argument filename', default="classifier.args", type=str)
    argparser.add_argument('--classifier_file', help='Trained classifier_file filename',
                           default="classifier.m", type=str)

    # Transfer learning
    argparser.add_argument('--tlargs', help='Trained argument transfer learning filename', default="", type=str)

    # vocab & embedding parameters
    argparser.add_argument("--tokenize_type", type=str, default="splitter",
                           help="tokenize type", choices=["bpe", "splitter"])
    argparser.add_argument("--wl_th", type=int, default=-1, help="Word length threshold")
    argparser.add_argument("--wcutoff", type=int, default=1, help="Prune words occurring <= wcutoff")
    argparser.add_argument("--ssos", action='store_true', default=False,
                           help="Start padding flag at a source sentence level")
    argparser.add_argument("--seos", action='store_true', default=False,
                           help="End padding flag at a source sentence level (True)")

    argparser.add_argument("--swd_embfile", type=str, help="Source Word embedding file", default="")
    argparser.add_argument("--snl_reqgrad", action='store_true', default=False,
                           help="Either freezing or unfreezing word pretrained embedding")
    argparser.add_argument("--wd_dropout", type=float, default=0.5,
                           help="Dropout rate at word-level embedding")
    argparser.add_argument("--wd_padding", action='store_true', default=False,
                           help="Flag to set all padding tokens to zero during training at word level")
    argparser.add_argument("--swd_dim", type=int, default=300, help="Source Word embedding size")

    # Neural Network parameters
    argparser.add_argument("--ed_mode", type=str, default="lstm", help="Word-level neural network type")
    argparser.add_argument("--ed_bidirect", action='store_true', default=False,
                           help="Word-level NN Bi-directional flag")
    argparser.add_argument("--ed_outdim", type=int, default=600,
                           help="Source Word-level neural network dimension")
    argparser.add_argument("--ed_layers", type=int, default=2, help="Source Number of NN layers at word level")
    argparser.add_argument("--ed_dropout", type=float, default=0.5,
                           help="Dropout rate at the encoder-decoder layer")
    # self-attention parameters
    argparser.add_argument("--ed_heads", type=int, default=6, help="number of heads")
    argparser.add_argument("--ed_activation", type=str, default="relu",
                           help="Activiation function used in self-transformer")
    argparser.add_argument("--ed_hismask", action='store_true', default=False,
                           help="Set True to mask future inputs")

    argparser.add_argument("--enc_cnn", action='store_true', default=False,
                           help="Encoder-level CNN flag")
    argparser.add_argument("--kernel_size", type=int, default=3, help="kernel_size of CNN (ks = 2*pad + 1)")

    argparser.add_argument("--final_dropout", type=float, default=0.5, help="Dropout rate at the last layer")

    # Optimizer parameters
    argparser.add_argument("--max_epochs", type=int, default=256, help="Maximum trained epochs")
    argparser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    argparser.add_argument("--patience", type=int, default=32,
                           help="Early stopping if no improvement after patience epoches")

    argparser.add_argument("--lr", type=float, default=0.001, help="Learning rate (ADAM: 0.001)")
    argparser.add_argument("--decay_rate", type=float, default=-1.0, help="Decay rate (0.05)")
    argparser.add_argument('--clip', default=-1, type=int, help='Clipping value (5)')

    argparser.add_argument("--optimizer", type=str, default="ADAM", help="Optimized method (adagrad, sgd, ...)")
    argparser.add_argument("--metric", type=str, default="f1", help="Optimized criterion (loss or f1)")

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    args, unknown = argparser.parse_known_args()

    args = Classifier_model.build_data(args)

    translator = Classifier_model(args)

    training_result = translator.train()

    return training_result


if __name__ == '__main__':
    """
    python classifier_model.py --use_cuda --teacher_forcing_ratio 0.8
    """
    main(sys.argv)
