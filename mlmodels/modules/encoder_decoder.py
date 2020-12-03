# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import random
import torch
import torch.nn as nn
from mlmodels.modules.embeddings import Emb_layer
from mlmodels.modules.encoders import Word_Encoder
from mlmodels.modules.decoders import Word_Decoder_v2
from mlmodels.modules.attention import GlobalAttention
from mlmodels.modules.beam_search import BeamSearch, GNMTGlobalScorer
from mlmodels.utils.special_tokens import PAD_id, SOT_id, EOT_id, UNK_id, COL_id, TAB_id, PAD
from mlmodels.utils.idx2tensor import Data2tensor

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncDec(nn.Module):
    def __init__(self, semb_HPs, enc_HPs, dec_HPs, drop_rate=0.5, num_labels=None, enc_att=False):
        super(EncDec, self).__init__()
        print("\t- Add NL embedding modules:")
        self.sembedding = Emb_layer(semb_HPs)
        assert enc_HPs[1] == semb_HPs[1]
        print("\t- Add encoding modules:")
        print("\t\t- Add encoder module")
        self.encoder = Word_Encoder(enc_HPs)
        print("\t- Add decoding modules:")
        self.decoder = Word_Decoder_v2(dec_HPs)

        self.enc_cnn = enc_HPs[0]
        self.ed_mode = dec_HPs[1][0]
        self.ed_bidirect = dec_HPs[1][4]

        self.enc_att = enc_att
        hidden_dim = dec_HPs[1][2]
        self.fn_dim = hidden_dim

        if self.enc_att:
            self.enc_attention = GlobalAttention(hidden_dim, hidden_dim)
            self.fn_dim += hidden_dim

        self.finaldrop_layer = nn.Dropout(drop_rate)

        self.num_labels = num_labels
        if num_labels > 2:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss(reduction='none')
        else:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()

    def forward(self, nl_tensor, nl_len_tensor, lb_tensor, teacher_force):
        device = nl_tensor.device
        # print("\n\t- INSIDE MODEL INPUTs: ", nl_tensor.shape, device, "\n")
        # sort lengths of input tensors in the descending mode
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = self.sort_tensors(nl_tensor, nl_len_tensor)
        # en_inp = [batch, nl_len, nl_emb]
        en_inp = self.sembedding(nl_tensor)
        # en_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # en_hidden = (h_n,c_n) ---> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
        en_out, en_hidden = self.encoder(en_inp, nl_len_tensor)
        if self.enc_cnn == "cnn" and self.ed_mode != "lstm":
            en_hidden = en_hidden[0]
        # en_hn = tensor(batch_size, num_directions * rnn_dim)
        # en_hn = self.encoder.get_last_hiddens(en_hidden)
        # recover the original order of inputs
        en_out = self.reorder_tensor(en_out, nl_recover_ord_tensor, dim=0)
        de_hidden = self.reorder_tensor(en_hidden, nl_recover_ord_tensor, dim=1)
        # en_hn = self.reorder_tensor(en_hn, nl_recover_ord_tensor, dim=0)
        nl_len_tensor = self.reorder_tensor(nl_len_tensor, nl_recover_ord_tensor, dim=0)

        # sort lengths of output tensors in the descending mode
        ilb_tensor = lb_tensor[:, : -1]
        lb_len_tensor = (ilb_tensor > 0).sum(dim=1)
        # olb_tensor = lb_tensor[:, 1:]
        # label_mask = olb_tensor > 0

        ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor = self.sort_tensors(ilb_tensor, lb_len_tensor)
        # reorder lengths of inputs following lengths of outputs
        en_out = self.reorder_tensor(en_out, lb_ord_tensor, dim=0)
        de_hidden = self.reorder_tensor(de_hidden, lb_ord_tensor, dim=1)
        # en_hn = self.reorder_tensor(en_hn, lb_ord_tensor, dim=0)
        nl_len_tensor = self.reorder_tensor(nl_len_tensor, lb_ord_tensor, dim=0)
        en_mask = None
        if nl_len_tensor.size(0) > 1:
            en_mask = torch.arange(en_out.size(1), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:, None]

        if teacher_force:
            # de_out = [batch, seq_len, hd_dim]
            # de_hidden = (h_n,c_n) ---> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
            de_out, de_hidden = self.decoder(ilb_tensor, lb_len_tensor, de_hidden)
            enc_context, enc_align = None, None
            if self.enc_att:
                # enc_context: [batch, seq_length2, hidden_dim]
                enc_context, enc_align = self.enc_attention(en_out, de_out, en_mask)
                # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)

            if enc_context is not None:
                de_out = torch.cat((de_out, enc_context), dim=-1)

            # de_score = [batch, seq_len, num_labels]
            de_score = self.scoring(de_out)

        else:
            # first input to the decoder is the <sos> token
            batch_size = ilb_tensor.shape[0]
            max_len = ilb_tensor.shape[1]
            num_outputs = self.num_labels
            de_score = torch.zeros(batch_size, max_len, num_outputs, device=device)
            output = ilb_tensor[:, 0].view(batch_size, 1)
            for t in range(max_len):
                # de_out = [batch, 1, hd_dim]
                # de_hidden = (h_n,c_n) ---> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
                de1_out, de_hidden = self.decoder(output, None, de_hidden)
                enc1_context, enc1_align = None, None
                if self.enc_att:
                    # enc_context: [batch, seq_length2, hidden_dim]
                    enc1_context, enc1_align = self.enc_attention(en_out, de1_out, en_mask)
                    # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
                if enc1_context is not None:
                    de1_out = torch.cat((de1_out, enc1_context), dim=-1)
                # de_score = [batch, 1, num_labels]
                de1_score = self.scoring(de1_out)
                de_score[:, t] = de1_score[:, 0, :]
                # TODO handle out of vocab
                output = de1_score.max(-1)[1].detach().clone()
        # recover the original order of outputs to compute loss
        de_score = self.reorder_tensor(de_score, lb_recover_ord_tensor, dim=0)
        return de_score

    def greedy_predict(self, nl_tensor, nl_len_tensor, maxlen=500, wombat_tensor=None):
        device = nl_len_tensor.device
        # sort lengths of input tensors in the descending mode
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = self.sort_tensors(nl_tensor, nl_len_tensor)

        en_inp = self.sembedding(nl_tensor)
        if wombat_tensor is not None:
            wombat_tensor = self.reorder_tensor(wombat_tensor, nl_ord_tensor, dim=0)
            en_inp += wombat_tensor
        en_out, en_hidden = self.encoder(en_inp, nl_len_tensor)
        if self.enc_cnn == "cnn" and self.ed_mode != "lstm":
            en_hidden = en_hidden[0]
        # en_hn = tensor(batch_size, num_directions * rnn_dim)
        en_hn = self.encoder.get_last_hiddens(en_hidden)
        # recover the original order of inputs
        en_out = self.reorder_tensor(en_out, nl_recover_ord_tensor, dim=0)
        de_hidden = self.reorder_tensor(en_hidden, nl_recover_ord_tensor, dim=1)
        en_hn = self.reorder_tensor(en_hn, nl_recover_ord_tensor, dim=0)
        nl_len_tensor = self.reorder_tensor(nl_len_tensor, nl_recover_ord_tensor, dim=0)
        en_mask = None
        if nl_len_tensor.size(0) > 1:
            en_mask = torch.arange(en_out.size(1), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:, None]

        batch_size = nl_tensor.shape[0]
        output = Data2tensor.idx2tensor([[SOT_id]] * batch_size, dtype=torch.long, device=device)
        pred_outputs = []
        acc_prob = Data2tensor.idx2tensor([[0.0]] * batch_size, dtype=torch.float32, device=device)
        EOT_tensor = Data2tensor.idx2tensor([[False]] * batch_size, dtype=torch.bool, device=device)
        count = 0
        while True:
            count += 1
            pred_outputs.append(output)
            de_out, de_hidden = self.decoder(output, None, de_hidden)
            enc_context, enc_align = None, None
            if self.enc_att:
                # enc_context: [batch, seq_length2, hidden_dim]
                enc_context, enc_align = self.enc_attention(en_out, de_out, en_mask)
                # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
            if enc_context is not None:
                de_out = torch.cat((de_out, enc_context), dim=-1)

            # de_score = [batch, 1, num_labels]
            de_score = self.scoring(de_out)
            log_probs = torch.nn.functional.log_softmax(de_score, dim=-1)
            top1_scores, top1_ids = torch.topk(log_probs,  1, dim=-1)

            # pred_prob, pred_label = self.inference(de_score)
            raw_output = top1_ids.squeeze(-1)
            acc_prob += top1_scores.squeeze(-1)
            EOT_tensor = EOT_tensor | (raw_output == EOT_id)
            # TODO: change to tensor.all()
            if EOT_tensor.all() or count > maxlen:
                # extend EOT to outputs
                pred_outputs.append(raw_output)
                break

            output = raw_output.detach().clone()

        pred_outputs = torch.cat(pred_outputs, dim=-1)
        # acc_prob = torch.cat(acc_prob, dim=-1)
        return pred_outputs, acc_prob.exp()

    def beam_predict(self, nl_tensor, nl_len_tensor, minlen=1, maxlen=500, bw=2, n_best=2, wombat_tensor=None):
        device = nl_len_tensor.device
        # sort lengths of input tensors in the descending mode
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = self.sort_tensors(nl_tensor, nl_len_tensor)
        # wombat_tensor = [batch, nl_len, emb_dim]
        en_inp = self.sembedding(nl_tensor)
        if wombat_tensor is not None:
            wombat_tensor = self.reorder_tensor(wombat_tensor, nl_ord_tensor, dim=0)
            en_inp += wombat_tensor
        en_out, en_hidden = self.encoder(en_inp, nl_len_tensor)
        if self.enc_cnn == "cnn" and self.ed_mode != "lstm":
            en_hidden = en_hidden[0]
        # en_hn = tensor(batch_size, num_directions * rnn_dim)
        en_hn = self.encoder.get_last_hiddens(en_hidden)
        # recover the original order of inputs
        en_out = self.reorder_tensor(en_out, nl_recover_ord_tensor, dim=0)
        de_hidden = self.reorder_tensor(en_hidden, nl_recover_ord_tensor, dim=1)
        en_hn = self.reorder_tensor(en_hn, nl_recover_ord_tensor, dim=0)
        nl_len_tensor = self.reorder_tensor(nl_len_tensor, nl_recover_ord_tensor, dim=0)

        # (0) Initialize a beam node
        batch_size = nl_tensor.shape[0]
        beam_node = BeamSearch(batch_size=batch_size, beam_size=bw, pad=PAD_id, bos=SOT_id, eos=EOT_id,
                               min_length=minlen, max_length=maxlen, n_best=n_best,
                               global_scorer=GNMTGlobalScorer(alpha=0., beta=0., length_penalty=None, coverage_penalty=None),
                               device=device)
        # (1) Prepare input tensors with new_batch = batch * bw
        en_out = beam_node.fn_map_state(en_out, dim=0)
        if isinstance(de_hidden, tuple):
            de_hidden = tuple(beam_node.fn_map_state(x, dim=1) for x in de_hidden)
        else:
            de_hidden = beam_node.fn_map_state(de_hidden, dim=1)
        en_hn = beam_node.fn_map_state(en_hn, dim=0)
        nl_len_tensor = beam_node.fn_map_state(nl_len_tensor, dim=0)
        en_mask = None
        if nl_len_tensor.size(0) > 1:
            en_mask = torch.arange(en_out.size(1), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:, None]
        count = 0
        while True:
            count += 1
            # (2) Predict token by token
            decoder_input = beam_node.current_predictions.view(-1, 1).detach().clone()
            de_out, de_hidden = self.decoder(decoder_input, None, de_hidden)
            enc_context, enc_align = None, None
            if self.enc_att:
                # enc_context: [batch, seq_length2, hidden_dim]
                # print("ENCODER:")
                enc_context, enc_align = self.enc_attention(en_out, de_out, en_mask)
                # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
            if enc_context is not None:
                de_out = torch.cat((de_out, enc_context), dim=-1)
            de_score = self.scoring(de_out)
            # (3) update current topk
            log_probs = torch.nn.functional.log_softmax(de_score, dim=-1)
            beam_node.advance(log_probs.squeeze(1))
            # (4) Check if finishing
            any_finished = beam_node.is_finished.any()
            if any_finished:
                beam_node.update_finished()
                if beam_node.done or count > maxlen:
                    break
            # (5) filter out completed sequence and create new batch
            select_indices = beam_node.select_indices
            if any_finished:
                # Reorder states.
                en_out = en_out.index_select(0, select_indices)
                if isinstance(de_hidden, tuple):
                    de_hidden = tuple(x.index_select(1, select_indices) for x in de_hidden)
                else:
                    de_hidden = de_hidden.index_select(1, select_indices)

                en_hn = en_hn.index_select(0, select_indices)
                nl_len_tensor = nl_len_tensor.index_select(0, select_indices)
                en_mask = None
                if len(nl_len_tensor) >= 1:
                    en_mask = torch.arange(en_out.size(1), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:,
                                                                                                   None]

        return beam_node.predictions, beam_node.scores

    def scoring(self, rnn_out):
        de_score = self.hidden2tag_layer(rnn_out)
        de_score = self.finaldrop_layer(de_score)
        return de_score

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            # label_score = [B, C]; label_tensor = [B, ]
            de_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1, ))

        else:
            # label_score = [B, *]; label_tensor = [B, *]
            de_loss = self.lossF(label_score, label_tensor.float().view(-1, 1))
        return de_loss

    def inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = torch.softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred

    def logsm_inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = torch.nn.functional.log_softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred

    def count_params(self):
        from functools import reduce
        "count number trainable parameters in a pytorch model"
        total_params = sum(reduce(lambda a, b: a * b, x.size()) for x in self.parameters())
        return total_params

    @staticmethod
    def reorder_tensor(inp_tensor, new_order_tensor, dim=0):
        if isinstance(inp_tensor, tuple):
            num_dim = inp_tensor[0].dim()
            if new_order_tensor is not None:
                if dim == 0 and dim < num_dim:
                    if inp_tensor[0].size(0) != 1 and inp_tensor[0].size(0) == new_order_tensor.size(0):
                        inp_tensor = tuple(tensor[new_order_tensor] for tensor in inp_tensor)
                elif dim == 1 and dim < num_dim:
                    if inp_tensor[0].size(1) != 1 and inp_tensor[0].size(1) == new_order_tensor.size(0):
                        inp_tensor = tuple(tensor[:, new_order_tensor, :] for tensor in inp_tensor)
                elif dim == 2 and dim < num_dim:
                    if inp_tensor[0].size(2) != 1 and inp_tensor[0].size(2) == new_order_tensor.size(0):
                        inp_tensor = tuple(tensor[:, :, new_order_tensor] for tensor in inp_tensor)
                else:
                    raise RuntimeError("Not implemented yet")
        else:
            num_dim = inp_tensor.dim()
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

    @staticmethod
    def sort_tensors(word_tensor, seq_len_tensor):
        seq_len_tensor, seqord_tensor = seq_len_tensor.sort(0, descending=True)
        word_tensor = word_tensor[seqord_tensor]
        _, seqord_recover_tensor = seqord_tensor.sort(0, descending=False)
        return word_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor


if __name__ == '__main__':
    import torch
    from mlmodels.utils.idx2tensor import Data2tensor
    from mlmodels.utils.dataset import IterDataset
    from torch.utils.data import DataLoader
    from mlmodels.utils.BPEtonkenizer import BPE
    from mlmodels.utils.special_tokens import BPAD, PAD, SOT, EOT, NULL
    from mlmodels.utils.trad_tokenizer import Tokenizer
    from mlmodels.utils.jsonIO import JSON

    Data2tensor.set_randseed(12345)
    device = torch.device("cpu")
    dtype = torch.long
    use_cuda = False
    filename = "/media/data/review_response/Dev.json"
    tokenize_type = "bpe"
    if tokenize_type != "bpe":
        # Load datasets to build vocabulary
        data = Tokenizer.load_file([filename], task=2)
        s_paras = [-1, 1]
        t_paras = [-1, 1]
        vocab = Tokenizer(s_paras, t_paras)
        vocab.build(data)
        nl2ids = Tokenizer.lst2idx(tokenizer=vocab.process_nl, vocab_words=vocab.sw2i,
                                   unk_words=True, eos=True)
        tg2ids = Tokenizer.lst2idx(tokenizer=vocab.process_target, vocab_words=vocab.tw2i,
                                   unk_words=False, sos=True, eos=True)
        pad_id = vocab.sw2i.get(PAD, 0)
        sw_size = len(vocab.sw2i)
        tw_size = len(vocab.tw2i)
    else:
        vocab_file = "/media/data/review_response/tokens/bert_level-bpe-vocab.txt"
        vocab = BPE.load(vocab_file)
        vocab.add_tokens([SOT, EOT, NULL])
        nl2ids = BPE.tokens2ids(vocab)
        tg2ids = BPE.tokens2ids(vocab)

        pad_id = vocab.token_to_id(BPAD) if vocab.token_to_id(BPAD) else 0
        sw_size = vocab.get_vocab_size()
        tw_size = vocab.get_vocab_size()

    collate_fn = BPE.collate_fn(pad_id, True)
    # load datasets to map into indexes
    train_data = JSON.get_iterator(filename)
    num_lines = JSON._len(filename)
    # train_data = CSV.get_iterator(filename, firstline=True)
    # num_lines = CSV._len(filename)
    train_iterdataset = IterDataset(train_data, source2idx=nl2ids, target2idx=tg2ids, num_lines=num_lines)
    train_dataloader = DataLoader(train_iterdataset, pin_memory=True, batch_size=8, collate_fn=collate_fn)

    for i, batch in enumerate(train_dataloader):
        # inputs, outputs = batch[0], batch[1]
        nl_tensor, lb_tensor = batch
        # nl_len_tensor = (nl_tensor > 0).sum(dim=1)
        break

    # COPY FROM DATA_UTILS
    emb_size = sw_size
    emb_dim = 50
    emb_pretrained = None
    emb_drop_rate = 0.5
    emb_zero_padding = True
    requires_grad = True

    nn_mode = "lstm"
    nn_inp_dim = 50
    nn_out_dim = 100
    nn_layers = 2
    nn_bidirect = True
    nn_dropout = 0.5

    fn_dropout = 0.5
    teacher_forcing_ratio = 0.5
    enc_att = True

    # # NL inputs
    nlemb_HPs = [emb_size, emb_dim, None, emb_drop_rate, emb_zero_padding, requires_grad]
    # source_emb = Emb_layer(nlemb_HPs)

    # encoder
    enc_HPs = [nn_mode, emb_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout]
    encoder = Word_Encoder(word_HPs=enc_HPs)

    # decoder
    # Hyper-parameters at word-level target language
    temb_HPs = [tw_size, emb_dim, None, emb_drop_rate, emb_zero_padding, requires_grad]
    # Hyper-parameters at word-level target language
    dec_HPs = [nn_mode, emb_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout]

    dec_HPs = [temb_HPs, dec_HPs]
    # decoder = Word_Decoder_v2(word_HPs=dec_HPs)


    # # encoder attention
    # if enc_att:
    #     hidden_dim = dec_HPs[1][2]
    #     enc_attention = GlobalAttention(hidden_dim, hidden_dim)
    #
    # hidden_dim = dec_HPs[1][2]
    # fn_dim = hidden_dim
    # if enc_att:
    #     fn_dim += hidden_dim
    #
    # hidden2tag = nn.Linear(fn_dim, len(vocab.tw2i))
    seq2seq = EncDec(nlemb_HPs, enc_HPs, dec_HPs, drop_rate=fn_dropout, num_labels=tw_size, enc_att=enc_att)
    nl_len_tensor = (nl_tensor > pad_id).sum(dim=1)
    random_force = True if random.random() < teacher_forcing_ratio else False
    # print("\nMODEL INPUTs: ", nl_tensor.shape, "\n")
    de_score = seq2seq(nl_tensor, nl_len_tensor, lb_tensor, random_force)
    olb_tensor = lb_tensor[:, 1:]
    label_mask = olb_tensor > 0

    total_loss = seq2seq.NLL_loss(de_score[label_mask], olb_tensor[label_mask]).mean()

    output_idx = de_score.max(-1)[1]
    if tokenize_type != "bpe":
        label_words = vocab.decode_batch(olb_tensor.tolist(), vocab.i2tw, 2)
        label_words = [words[:i] if EOT not in words else words[: words.index(EOT)]
                       for words, i in zip(label_words, label_mask.sum(dim=1).tolist())]

        predict_words = vocab.decode_batch(output_idx.tolist(), vocab.i2tw, 2)
        predict_words = [words[:i] if EOT not in words else words[: words.index(EOT)]
                         for words, i in zip(predict_words, label_mask.sum(dim=1).tolist())]

        nl_token = vocab.decode_batch(nl_tensor.tolist(), vocab.i2sw, 2)
        nl_token = [words[:i] if EOT not in words else words[: words.index(EOT)]
                    for words, i in zip(nl_token, (nl_tensor > 0).sum(dim=1).tolist())]
    else:
        label_words = vocab.decode_batch(olb_tensor.tolist())
        # label_words = [enc_words.tokens for enc_words in self.args.vocab.encode_batch(label_words)]
        label_words = [words[0: words.find(EOT)].split() for words in label_words]

        predict_words = vocab.decode_batch(output_idx.tolist())
        # predict_words = [enc_words.tokens for enc_words in self.args.vocab.encode_batch(predict_words)]
        predict_words = [words[0: words.find(EOT)].split() for words in predict_words]

        nl_token = vocab.decode_batch(nl_tensor.tolist())
        # nl_token = [enc_words.tokens for enc_words in self.args.vocab.encode_batch(nl_token)]
        nl_token = [words[0: words.find(EOT)].split() for words in nl_token]

    # Test greedy
    pred_outputs, acc_prob = seq2seq.greedy_predict(nl_tensor, nl_len_tensor)
    if tokenize_type != "bpe":
        predict_words = vocab.decode_batch(pred_outputs.tolist(), vocab.i2tw, 2)
        predict_words = [words if EOT not in words else words[: words.index(EOT) + 1] for words in
                         predict_words]
    else:
        predict_words = vocab.decode_batch(pred_outputs.tolist())
        predict_words = [words[0: words.find(EOT)].split() for words in predict_words]
    predict_prob = acc_prob.squeeze().tolist()

    # test beam search
    pred_outputs, predict_prob = seq2seq.beam_predict(nl_tensor, nl_len_tensor, bw=3, n_best=2)

    if tokenize_type != "bpe":
        predict_words = vocab.decode_batch(pred_outputs, vocab.i2tw, 3)
        predict_words = [words if EOT not in words else words[: words.index(EOT) + 1] for words in
                         predict_words]
        predict_words = [[" ".join(words) for words in topk_outputs] for topk_outputs in predict_words]
    else:
        predict_words = [vocab.decode_batch(topk_outputs) for topk_outputs in pred_outputs]
        predict_words = [[words[0: words.find(EOT)] for words in topk_outputs] for topk_outputs in predict_words]