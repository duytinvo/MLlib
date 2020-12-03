"""
Created on 2018-11-27
@author: duytinvo
"""
import random
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from np5.utils.data_utils import EOT, COL, TAB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Emb_layer(nn.Module):
    """
    This module take (characters or words) indices as inputs and outputs (characters or words) embedding
    """

    def __init__(self, HPs):
        super(Emb_layer, self).__init__()
        [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        self.zero_padding = zero_padding
        self.embeddings = nn.Embedding(size, dim, padding_idx=0)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))
        if not requires_grad:
            print("Fixed pre-trained embeddings")
            self.embeddings.weight.requires_grad = requires_grad
        self.drop = nn.Dropout(drop_rate)

    def forward(self, inputs, auxiliary_embs=None):
        return self.get_embs(inputs, auxiliary_embs)

    def get_embs(self, inputs, auxiliary_embs=None):
        """
        embs.shape([0, 1]) == auxiliary_embs.shape([0, 1])
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0, 1, 2, 3])
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        if auxiliary_embs is not None:
            assert embs_drop.shape[:-1] == auxiliary_embs.shape[:-1]
            embs_drop = torch.cat((embs_drop, auxiliary_embs), -1)
        return embs_drop

    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index, :] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs

    def set_zeros(self, idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)


class CNN_layer(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an CNN layer to extract:
        - all hidden features
        - last hidden features
    """
    def __init__(self, HPs):
        super(CNN_layer, self).__init__()
        [_, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, kernel_size] = HPs
        nn_cnn_dim = nn_out_dim // 2 if nn_bidirect else nn_out_dim
        # mlmodels.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation, groups, bias, padding_mode)
        # L_out = [L_in + 2 × padding − dilation × (kernel_size−1) − 1] / stride + 1; dilation=1, stride=1
        # L_out = L_in + 2 × padding −  kernel_size + 1
        # L_out = L_in ==> 2 × padding −  kernel_size + 1 = 0
        # kernel_size = 3 ==> padding = 1; kernel_size = 5 ==> padding = 2
        padding = int((kernel_size-1)/2)
        self.nn_bidirect = nn_bidirect
        self.nn_layers = nn_layers
        self.fw_layers = nn.ModuleList()
        self.fw_layers.append(nn.Conv1d(nn_inp_dim, nn_cnn_dim, int(kernel_size), padding=padding))
        for i in range(nn_layers-1):
            self.fw_layers.append(nn.Conv1d(nn_cnn_dim, nn_cnn_dim, int(kernel_size), padding=padding))
        if nn_bidirect:
            self.bw_layers = nn.ModuleList()
            self.bw_layers.append(nn.Conv1d(nn_inp_dim, nn_cnn_dim, int(kernel_size), padding=padding))
            for i in range(nn_layers - 1):
                self.bw_layers.append(nn.Conv1d(nn_cnn_dim, nn_cnn_dim, int(kernel_size), padding=padding))

    def forward(self, emb_inputs, input_lengths=None, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, input_lengths, init_hidden)

    def get_all_hiddens(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
            input:
                emb_inputs: tensor(batch_size, seq_length, emb_dim)
            output:
                tensor(batch_size, seq_length, hidden_dim)
                hn: final feature vector of a sequence
                hc: max-pooling feature vector
        """
        # emb_inputs: (batch_size, seq_length, emb_dim) --> (batch_size, emb_dim, seq_length)
        emb_inputs = emb_inputs.transpose(1, -1)
        # w_0 --> w_n
        inp_fwcnn = emb_inputs
        hn_fw = []
        hc_fw = []
        out_fwconv = torch.empty_like(inp_fwcnn)
        for i in range(self.nn_layers):
            # out_fwconv: (batch_size, hidden_dim, seq_length)
            out_fwconv = self.fw_layers[i](inp_fwcnn)
            # hn_bw, hc_bw: (batch_size, hidden_dim)
            hn_fw.append(out_fwconv[:, :, -1])
            # equivalent with max pooling
            hc_fw.append(out_fwconv.max(-1)[0])
            inp_fwcnn = out_fwconv
        # hn: (num_layers, batch_size, hidden_dim)
        h_n = torch.stack(hn_fw)
        h_c = torch.stack(hc_fw)
        # cnn_out: (batch_size, hidden_dim, seq_length)
        cnn_out = out_fwconv

        if self.nn_bidirect:
            # reverse seq_length w_n --> w_0
            inp_bwconv = emb_inputs.flip(-1)
            hn_bw = []
            hc_bw = []
            out_bwconv = torch.empty_like(inp_bwconv)
            for i in range(self.nn_layers):
                # out_bwconv: (batch_size, hidden_dim, seq_length)
                out_bwconv = self.bw_layers[i](inp_bwconv)

                # hn_bw, hc_bw: (batch_size, hidden_dim)
                hn_bw.append(out_bwconv[:, :, -1])
                # equivalent with max pooling
                hc_bw.append(out_bwconv.max(-1)[0])
                inp_bwconv = out_bwconv
            # hn_bw: (num_layers, batch_size, hidden_dim)
            hn_bw = torch.stack(hn_bw)
            hc_bw = torch.stack(hc_bw)
            # h_n: (num_layers * num_directions, batch_size, hidden_dim)
            h_n = torch.cat([h_n, hn_bw], dim=0)
            h_c = torch.cat([h_c, hc_bw], dim=0)
            # cnn_out: (batch_size, num_directions*hidden_dim, seq_length)
            # reverse back w_0 --> w_n before concatenating
            out_bwconv = out_bwconv.flip(-1)
            cnn_out = torch.cat([cnn_out, out_bwconv], dim=1)

        # cnn_out: (batch_size, seq_length, num_directions*hidden_dim)
        cnn_out.transpose_(1, -1)
        return cnn_out, (h_n, h_c)

    def get_last_hiddens(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
            input:
                emb_inputs: tensor(batch_size, seq_length, emb_dim)
            output:
                tensor(batch_size, seq_length, hidden_dim)
        """
        # emb_inputs: (batch_size, seq_length, emb_dim) --> (batch_size, emb_dim, seq_length)
        emb_inputs.transpose_(1, -1)
        inp_fwcnn = emb_inputs
        hn_fw = []
        out_fwconv = torch.empty_like(inp_fwcnn)
        for i in range(self.nn_layers):
            # out_fwconv: (batch_size, hidden_dim, seq_length)
            out_fwconv = self.fw_layers[i](inp_fwcnn)
            # hn_bw, hc_bw: (batch_size, hidden_dim)
            hn_fw.append(out_fwconv[:, :, -1])
            inp_fwcnn = out_fwconv
        # hn: (batch_size, hidden_dim)
        h_n = hn_fw[-1]
        # cnn_out: (batch_size, hidden_dim, seq_length)
        cnn_out = out_fwconv

        if self.nn_bidirect:
            # reverse seq_length w_n --> w_0
            inp_bwconv = emb_inputs.flip(-1)
            hn_bw = []
            out_bwconv = torch.empty_like(inp_bwconv)
            for i in range(self.nn_layers):
                # out_bwconv: (batch_size, hidden_dim, seq_length)
                out_bwconv = self.bw_layers[i](inp_bwconv)

                # hn_bw, hc_bw: (batch_size, hidden_dim)
                hn_bw.append(out_bwconv[:, :, -1])
                inp_bwconv = out_bwconv
            # hn_bw: (num_layers, batch_size, hidden_dim)
            hn_bw = torch.stack(hn_bw)
            # h_n: (batch_size, hidden_dim * num_directions)
            h_n = torch.cat([h_n, hn_bw[-1]], dim=-1)
            # cnn_out: (batch_size, num_directions*hidden_dim, seq_length)
            # reverse back w_0 --> w_n before concatenating
            out_bwconv = out_bwconv.flip(-1)
            cnn_out = torch.cat([cnn_out, out_bwconv], dim=1)

        # cnn_out: (batch_size, seq_length, num_directions*hidden_dim)
        cnn_out.transpose_(1, -1)
        return h_n


class RNN_layer(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer to extract:
        - all hidden features
        - last hidden features
        - all attentional hidden features
        - last attentional hidden features
    """
    def __init__(self, HPs):
        super(RNN_layer, self).__init__()
        [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        nn_rnn_dim = nn_out_dim // 2 if nn_bidirect else nn_out_dim
        if nn_mode == "rnn":
            if nn_layers == 1:
                self.hidden_layer = nn.RNN(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers,
                                           batch_first=True, bidirectional=nn_bidirect)
            else:
                self.hidden_layer = nn.RNN(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                           batch_first=True, bidirectional=nn_bidirect)

        elif nn_mode == "gru":
            if nn_layers == 1:
                self.hidden_layer = nn.GRU(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers,
                                           batch_first=True, bidirectional=nn_bidirect)
            else:
                self.hidden_layer = nn.GRU(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                           batch_first=True, bidirectional=nn_bidirect)
        else:
            if nn_layers == 1:
                self.hidden_layer = nn.LSTM(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers,
                                            batch_first=True, bidirectional=nn_bidirect)
            else:
                self.hidden_layer = nn.LSTM(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                            batch_first=True, bidirectional=nn_bidirect)
            # # Set the bias of forget gate to 1.0
            # for names in self.hidden_layer._all_weights:
            #     for name in filter(lambda n: "bias" in n, names):
            #         bias = getattr(self.hidden_layer, name)
            #         n = bias.size(0)
            #         start, end = n // 4, n // 2
            #         bias.data[start:end].fill_(1.)

    def forward(self, emb_inputs, input_lengths, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, input_lengths, init_hidden)

    def get_last_hiddens(self, emb_inputs, input_lengths, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, hidden_dim)
        """
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # hc_n = (h_n,c_n)
        # h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
        rnn_out, hc_n = self.get_all_hiddens(emb_inputs, input_lengths, init_hidden=init_hidden)
        # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
        if type(hc_n) == tuple:
            h_n = torch.cat((hc_n[0][-2, :, :], hc_n[0][-1, :, :]), -1)
        else:
            h_n = torch.cat((hc_n[-2, :, :], hc_n[-1, :, :]), -1)
        return h_n

    def get_all_hiddens(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, seq_length, hidden_dim)
        """
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # hc_n = (h_n,c_n);
        # h_n = tensor(num_layers*num_directions, batch_size, rnn_dim)
        if input_lengths is not None:
            total_length = emb_inputs.size(1)
            # pack_input = pack_padded_sequence(emb_inputs, input_lengths.cpu().numpy(), True)
            pack_input = pack_padded_sequence(emb_inputs, input_lengths, True)
            self.hidden_layer.flatten_parameters()
            rnn_out, hc_n = self.hidden_layer(pack_input, init_hidden)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=total_length)
        else:
            rnn_out, hc_n = self.hidden_layer(emb_inputs, init_hidden)
        return rnn_out, hc_n


class ColPredictor(nn.Module):
    """
    This model is used to simultaneously predict the number of elements and a set of these elements.
    It could be used in:
        - col (Column) classifier model
            + #num = 6
                -- {[], [vl_i], [vl_i, vl_j], [vl_i, vl_j, vl_k], [vl_i, vl_jk, vl_l], [vl_i, vl_jkl, vl_m]};
                -- Note: no [] in corpus
            + #labels = 1
                -- directly and dynamically pass #columns (126 --> [0, 125])
    """
    def __init__(self,  HPs, use_hs=True, num_labels=4, labels=1,
                 dtype=torch.long, device=torch.device("cpu")):
        super(ColPredictor, self).__init__()
        self.dtype = dtype
        self.device = device
        self.use_hs = use_hs

        self.q_lstm = RNN_layer(HPs)
        self.hs_lstm = RNN_layer(HPs)
        self.col_lstm = RNN_layer(HPs)

        N_h = HPs[2]
        self.q_num_att = nn.Linear(N_h, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(N_h, N_h)
        self.col_num_out_hs = nn.Linear(N_h, N_h)

        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, num_labels))  # num of cols: 1-4

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out_hs = nn.Linear(N_h, N_h)

        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, labels))  # labels = 1

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        # Multi-label classification
        self.CE = nn.CrossEntropyLoss()
        # Binary classification
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, q_emb_var, q_len, q_recover,
                hs_emb_var, hs_len, hs_recover,
                col_emb_var, col_len, col_recover,
                gt_col=None):
        # q_enc: (B, max_q_len, N_h)
        q_enc, _ = self.q_lstm(q_emb_var, q_len)
        q_enc = q_enc[q_recover]
        # hs_enc: (B, max_hs_len, N_h)
        hs_enc, _ = self.hs_lstm(hs_emb_var, hs_len)
        hs_enc = hs_enc[hs_recover]
        # col_enc: (B, max_col_len, N_h)
        col_enc, _ = self.col_lstm(col_emb_var, col_len)
        col_enc = col_enc[col_recover]

        q_mask = torch.arange(q_enc.size(1), dtype=self.dtype, device=self.device)[None, :] < q_len[q_recover][:, None]
        hs_mask = torch.arange(hs_enc.size(1), dtype=self.dtype, device=self.device)[None, :] < hs_len[hs_recover][:, None]
        col_mask = torch.arange(col_enc.size(1), dtype=self.dtype, device=self.device)[None, :] < col_len[col_recover][:, None]
        # ---------------------------------------------------------------------------------
        #                       Predict column number
        # ---------------------------------------------------------------------------------
        # att_val_qc_num: (B, max_col_len, max_q_len)
        # col_enc: (B, max_col_len, N_h); q_enc: (B, max_q_len, N_h) --MLP(T(1,2))--> (B, N_h, max_q_len)
        att_val_qc_num = torch.bmm(col_enc, self.q_num_att(q_enc).transpose(1, 2))
        # _qc_mask: (B, max_col_len, max_q_len)
        _qc_mask = torch.bmm(col_mask.to(dtype=torch.float).unsqueeze(-1), q_mask.to(dtype=torch.float).unsqueeze(1))

        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_qc_num[~_qc_mask.to(dtype=torch.uint8)] = -100
        # att_prob_qc_num: (B, max_col_len, max_q_len)
        att_prob_qc_num = self.softmax(att_val_qc_num)
        # q_weighted_num: (B, N_h)
        # q_enc.unsqueeze(1): (B, 1, max_q_len, N_h); att_prob_qc_num.unsqueeze(3): (B, max_col_len, max_q_len, 1)
        # TODO: instead of using .sum(1) on max_col_len, could use one more attention layer
        q_weighted_num = (q_enc.unsqueeze(1) * att_prob_qc_num.unsqueeze(3)).sum(2).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        # att_val_hc_num: (B, max_col_len, max_hs_len)
        # col_enc: (B, max_col_len, N_h); hs_enc: (B, max_hs_len, N_h) --MLP(T(1,2))--> (B, N_h, max_hs_len)
        att_val_hc_num = torch.bmm(col_enc, self.hs_num_att(hs_enc).transpose(1, 2))
        # _hc_mask: (B, max_col_len, max_hs_len)
        _hc_mask = torch.bmm(col_mask.to(dtype=torch.float).unsqueeze(-1), hs_mask.to(dtype=torch.float).unsqueeze(1))

        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_hc_num[~_hc_mask.to(dtype=torch.uint8)] = -100
        # att_prob_hc_num: (B, max_col_len, max_hs_len)
        att_prob_hc_num = self.softmax(att_val_hc_num)
        # TODO: instead of using .sum(1) on max_col_len, could use one more attention layer
        # (hs_enc.unsqueeze(1): (B, 1, max_hs_len, N_h)
        # att_prob_hc_num.unsqueeze(3): (B, max_col_len, max_hs_len, 1);
        # hs_weighted_num: (B, N_h)
        hs_weighted_num = (hs_enc.unsqueeze(1) * att_prob_hc_num.unsqueeze(3)).sum(2).sum(1)
        # col_num_score: (B, num_labels)
        col_num_score = self.col_num_out(self.col_num_out_q(q_weighted_num) +
                                         int(self.use_hs) * self.col_num_out_hs(hs_weighted_num))

        # ---------------------------------------------------------------------------------
        #                               Predict columns
        # ---------------------------------------------------------------------------------
        # att_val_qc: (B, max_col_len, max_q_len)
        # col_enc: (B, max_col_len, N_h); q_enc: (B, max_q_len, N_h) --MLP(T(1,2))--> (B, N_h, max_q_len)
        att_val_qc = torch.bmm(col_enc, self.q_att(q_enc).transpose(1, 2))
        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_qc[~_qc_mask.to(dtype=torch.uint8)] = -100

        # att_prob_qc: (B, max_col_len, max_q_len)
        att_prob_qc = self.softmax(att_val_qc)
        # q_weighted: (B, max_col_len, N_h)
        # q_enc.unsqueeze(1): (B, 1, max_q_len, N_h)
        # att_prob_qc.unsqueeze(3): (B, max_col_len, max_q_len, 1)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qc.unsqueeze(3)).sum(2)

        # Same as the above, compute SQL history embedding weighted by column attentions
        # att_val_hc: (B, max_col_len, max_hs_len)
        # col_enc: (B, max_col_len, N_h); hs_enc: (B, max_hs_len, N_h) --MLP(T(1,2))--> (B, N_h, max_hs_len)
        att_val_hc = torch.bmm(col_enc, self.hs_att(hs_enc).transpose(1, 2))
        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_hc[~_hc_mask.to(dtype=torch.uint8)] = -100

        # att_prob_hc: (B, max_col_len, max_hs_len)
        att_prob_hc = self.softmax(att_val_hc)
        # hs_weighted: (B, max_col_len, N_h)
        # hs_enc.unsqueeze(1): (B, 1, max_hs_len, N_h); att_prob_hc.unsqueeze(3): (B, max_col_len, max_hs_len, 1)
        hs_weighted = (hs_enc.unsqueeze(1) * att_prob_hc.unsqueeze(3)).sum(2)
        # Compute prediction scores
        # col_score: (B, max_col_len)
        # q_weighted --MLP--> (B, max_col_len, N_h)
        # hs_weighted --MLP--> (B, max_col_len, N_h)
        # col_enc --MLP--> (B, max_col_len, N_h)
        col_score = self.col_out(self.col_out_q(q_weighted) +
                                 int(self.use_hs) * self.col_out_hs(hs_weighted) +
                                 self.col_out_c(col_enc)).squeeze(-1)
        # col_num_score: (B, num_labels); col_score: (B, max_col_len)
        # score = (col_num_score, col_score)
        return col_num_score, col_score

    def NLL_loss(self, score_tensor, truth_tensor):
        loss = 0
        # col_num_score: (B, num_labels); col_score: (B, max_col_len)
        col_num_score, col_score = score_tensor
        # truth_num_var: (B, ); truth_var: (B, max_col_len)
        truth_num_var, truth_var = truth_tensor
        # --------------------------------------------------------------------------------
        #                       loss for the column number
        # --------------------------------------------------------------------------------
        loss += self.CE(col_num_score, truth_num_var)
        # --------------------------------------------------------------------------------
        #                           loss for the key words
        # --------------------------------------------------------------------------------
        pred_prob = self.sigmoid(col_score)
        bce_loss = torch.mean(- 3 * (truth_var * torch.log(pred_prob + 1e-10))
                              - (1 - truth_var) * torch.log(1 - pred_prob + 1e-10))
        loss += bce_loss
        # loss += self.BCE(col_score, truth_var)
        return loss

    def norm_scores(self, score_tensor):
        # col_num_score: (B, num_labels); col_score: (B, max_col_len)
        col_num_score, col_score = score_tensor
        return torch.softmax(col_num_score, dim=-1).data.cpu().numpy(), torch.sigmoid(col_score).data.cpu().numpy()

    def check_acc(self, score_tensor, truth_labels):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth_labels)
        pred = []
        col_num_score, col_score = self.norm_scores(score_tensor)
        for b in range(B):
            cur_pred = {}
            # when col_num_score[b] = 0 --> select 1 element
            # col_num = [1, num_labels]
            # col_num = np.argmax(col_num_score[b]) + 1  # idx starting from 0 --> a[idx+1] = max(a)
            col_num = np.argmax(col_num_score[b])
            cur_pred['col_num'] = col_num
            cur_pred['col'] = np.argsort(-col_score[b])[:col_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth_labels)):
            col_num, col = p['col_num'], p['col']
            flag = True
            if col_num != len(t):  # double check truth format and for test cases
                num_err += 1
                flag = False
            # to metrics col predicts, if the gold sql has JOIN and foreign key col, then both fks are acceptable
            fk_list = []
            regular = []
            for l in t:
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)

            if flag:  # double check
                for c in col:
                    for fk in fk_list:
                        if c in fk:
                            fk_list.remove(fk)
                    for r in regular:
                        if c == r:
                            regular.remove(r)

                if len(fk_list) != 0 or len(regular) != 0:
                    err += 1
                    flag = False

            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))


class Word_Encoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs):
        super(Word_Encoder, self).__init__()
        if word_HPs[0] == "cnn":
            self.word_nn_embs = CNN_layer(word_HPs)
        else:
            self.word_nn_embs = RNN_layer(word_HPs)

    def forward(self, emb_inputs, word_lengths, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, word_lengths, init_hidden)

    def get_all_hiddens(self, emb_inputs, word_lengths, init_hidden=None):
        rnn_out, hidden_out = self.word_nn_embs(emb_inputs, word_lengths, init_hidden)
        return rnn_out, hidden_out

    def get_last_hiddens(self, emb_inputs, word_lengths, init_hidden=None):
        h_n = self.word_nn_embs.get_last_hiddens(emb_inputs, word_lengths, init_hidden)
        return h_n


class Word_alignment(nn.Module):
    def __init__(self, in_features, out_features):
        super(Word_alignment, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.xavier_uniform_(self.weight)

    def forward(self, input1, input2, input_mask=None):
        """

        :param input1: [batch, seq_length1, in_features]
        :param input2: [batch, seq_length2, out_features]
        :param input_mask: mask of input1
        :return:
        """
        out1 = F.linear(input1, self.weight)
        # out1: [batch, seq_length1, out_features]
        # input2: [batch, seq_length2, out_features]
        out2 = torch.matmul(out1, input2.transpose(1, -1))
        if input_mask is not None and out2.size(0) == input_mask.size(0):
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, seq_length2]
        # input1: [batch, seq_length1, in_features]
        satt = torch.matmul(F.softmax(out2, dim=1).transpose(1, -1), input1)
        # satt: [batch, seq_length2, in_features]
        return satt


class Col_awareness(nn.Module):
    def __init__(self, col_features, enc_features):
        super(Col_awareness, self).__init__()
        self.enc_features = enc_features
        self.col_features = col_features
        self.weight = Parameter(torch.Tensor(enc_features, col_features))
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.xavier_uniform_(self.weight)

    def forward(self, input1, input2, input_mask=None):
        """
        :param input1: [batch, seq_length1, col_features]
        :param input2: [batch, enc_features]
        :param input_mask: mask of input1
        :return:
        """
        # input1: [batch, seq_length1, col_features]
        # weight: [enc_features, col_features]
        # out1: [batch, seq_length1, enc_features]
        out1 = F.linear(input1, self.weight)
        # out1: [batch, seq_length1, enc_features]
        # input2: [batch, enc_features] -- > input2.unsqueeze(-1): [batch, enc_features, 1]
        out2 = torch.matmul(out1, input2.unsqueeze(-1))
        if input_mask is not None and out2.size(0) == input_mask.size(0):
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, 1] --> out2.transpose(1, -1): [batch, 1, seq_length1]
        # input1: [batch, seq_length1, col_features]
        satt = torch.matmul(F.softmax(out2.transpose(1, -1), dim=2), input1)
        # satt: [batch, 1, col_features]
        return satt


class Pointer_net(nn.Module):
    def __init__(self, in_features, out_features):
        super(Pointer_net, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.lossF = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.xavier_uniform_(self.weight)

    def forward(self, input1, input2, input_mask=None):
        pointer_score = self.scoring(input1, input2, input_mask)
        return pointer_score

    def scoring(self, input1, input2, input_mask=None):
        """
        :param input1: [batch, seq_length1, in_features]
        :param input2: [batch, seq_length2, out_features]
        :param input_mask: mask of input1
        """
        out1 = F.linear(input1, self.weight)
        # out1: [batch, seq_length1, out_features]
        # input2: [batch, seq_length2, out_features]
        out2 = torch.matmul(out1, input2.transpose(1, -1))
        # TODO: use mask tensor to filter out padding in out2[:,seq_length1,:]
        if input_mask is not None and out2.size(0) == input_mask.size(0):
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, seq_length2]
        # pointers: [batch, seq_length1, seq_length2]
        # pointers = F.softmax(out2, dim=1)
        # out2: [batch, seq_length2, seq_length1]
        return out2.transpose(1, -1)

    def inference(self, label_score, k=1):
        # label_score: [batch, seq_length2, seq_length1]
        # pointers: [batch, seq_length2, seq_length1]
        pointers = F.softmax(label_score, dim=-1)
        label_prob, label_pred = pointers.data.topk(k, dim=-1)
        return label_prob, label_pred

    def logsm_inference(self, label_score, k=1):
        # label_score: [batch, seq_length2, seq_length1]
        label_prob = F.log_softmax(label_score, dim=-1)
        label_prob, label_pred = label_prob.data.topk(k, dim=-1)
        return label_prob, label_pred

    def NLL_loss(self, label_score, label_tensor):
        # label_score = [B, C]; label_tensor = [B, ]
        batch_loss = self.lossF(label_score, label_tensor)
        return batch_loss


class Word_Decoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs, drop_rate=0.5, num_labels=None, enc_att=False, sch_att='',
                 col_dim=-1, tab_dim=-1):
        super(Word_Decoder, self).__init__()
        self.embedding = Emb_layer(word_HPs[0])
        self.decoder = Word_Encoder(word_HPs[1])
        self.num_labels = num_labels
        self.enc_att = enc_att
        self.sch_att = sch_att
        hidden_dim = word_HPs[1][2]
        self.fn_dim = hidden_dim
        self.finaldrop_layer = nn.Dropout(drop_rate)
        if self.enc_att:
            self.enc_attention = Word_alignment(hidden_dim, hidden_dim)
            self.fn_dim += hidden_dim
        if self.sch_att == "en_hidden":
            assert col_dim > 0
            self.col_attention = Col_awareness(col_dim, hidden_dim)
            self.fn_dim += col_dim

            assert tab_dim > 0
            self.tab_attention = Col_awareness(tab_dim, hidden_dim)
            self.fn_dim += tab_dim
        elif self.sch_att == "de_hidden":
            assert col_dim > 0
            self.col_attention = Word_alignment(col_dim, hidden_dim)
            self.fn_dim += col_dim

            assert tab_dim > 0
            self.tab_attention = Word_alignment(tab_dim, hidden_dim)
            self.fn_dim += tab_dim
        else:
            pass

        if num_labels > 2:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()

    def forward(self, word_inputs, word_lengths, init_hidden=None,
                enc_out=None, enc_mask=None, enc_hn=None,
                colemb=None, tabemb=None, colmask=None, tabmask=None):
        rnn_out, hidden_out = self.get_all_hiddens(word_inputs, word_lengths, init_hidden, enc_out, enc_mask, enc_hn,
                                                    colemb, tabemb, colmask, tabmask)
        de_score = self.scoring(rnn_out)
        return de_score, rnn_out, hidden_out

    def get_all_hiddens(self, word_inputs, word_lengths, init_hidden, enc_out, enc_mask, enc_hn,
                        colemb, tabemb, colmask=None, tabmask=None):
        emb_inputs = self.embedding(word_inputs)
        rnn_out, hidden_out = self.decoder(emb_inputs, word_lengths, init_hidden)
        enc_context = None
        if self.enc_att:
            # enc_context: [batch, seq_length2, hidden_dim]
            enc_context = self.enc_attention(enc_out, rnn_out, enc_mask)
            # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
        if self.sch_att == "en_hidden":
            # enc_hn: [batch, hidden_dim]
            # colemb: [batch, num_col, col_features]
            # col_context: [batch, 1, col_features]
            col_context = self.col_attention(colemb, enc_hn, colmask)
            # col_context: [batch, seq_length2, col_features]
            col_context = col_context.expand(-1, rnn_out.size(1), -1)

            # tab_context: [batch, 1, tab_features]
            tab_context = self.tab_attention(tabemb, enc_hn, tabmask)
            # tab_context: [batch, seq_length2, tab_features]
            tab_context = tab_context.expand(-1, rnn_out.size(1), -1)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        elif self.sch_att == "de_hidden":
            # col_context: [batch, seq_length2, col_dim]
            col_context = self.col_attention(colemb, rnn_out, colmask)

            # tab_context: [batch, seq_length2, tab_dim]
            tab_context = self.tab_attention(tabemb, rnn_out, tabmask)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        else:
            col_context = None
            tab_context = None

        if enc_context is not None:
            rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
        if col_context is not None:
            rnn_out = torch.cat((rnn_out, col_context), dim=-1)
        if tab_context is not None:
            rnn_out = torch.cat((rnn_out, tab_context), dim=-1)
        return rnn_out, hidden_out

    def scoring(self, rnn_out):
        de_score = self.hidden2tag_layer(rnn_out)
        de_score = self.finaldrop_layer(de_score)
        return de_score

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            # label_score = [B, C]; label_tensor = [B, ]
            de_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1,))
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


class Word_Decoder_v2(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs):
        super(Word_Decoder_v2, self).__init__()
        self.embedding = Emb_layer(word_HPs[0])
        self.decoder = Word_Encoder(word_HPs[1])

    def forward(self, word_inputs, word_lengths, init_hidden=None):
        rnn_out, hidden_out = self.get_all_hiddens(word_inputs, word_lengths, init_hidden)
        return rnn_out, hidden_out

    def get_all_hiddens(self, word_inputs, word_lengths, init_hidden):
        emb_inputs = self.embedding(word_inputs)
        rnn_out, hidden_out = self.decoder(emb_inputs, word_lengths, init_hidden)
        return rnn_out, hidden_out


class Source_Emb(nn.Module):
    def __init__(self, semb_HPs):
        super(Source_Emb, self).__init__()
        nlemb_HPs, tpemb_HPs, posemb_HPs, swd_inp, swd_mode = semb_HPs
        self.swd_inp = swd_inp
        self.swd_mode = swd_mode
        self.sembedding = Emb_layer(nlemb_HPs)
        if swd_inp.startswith("dual_tp"):
            self.tpembedding = Emb_layer(tpemb_HPs)

        if self.swd_inp.startswith("dual_pos"):
            self.posembedding = Emb_layer(posemb_HPs)

        if self.swd_inp.startswith("triple"):
            self.tpembedding = Emb_layer(tpemb_HPs)
            self.posembedding = Emb_layer(posemb_HPs)

        if self.swd_inp.startswith("triple") and self.swd_mode.startswith("conc"):
            self.inp_dim = nlemb_HPs[1] + tpemb_HPs[1] + posemb_HPs[1]
        elif self.swd_inp.startswith("dual_pos") and self.swd_mode.startswith("conc"):
            self.inp_dim = nlemb_HPs[1] + posemb_HPs[1]
        elif self.swd_inp.startswith("dual_tp") and self.swd_mode.startswith("conc"):
            self.inp_dim = nlemb_HPs[1] + tpemb_HPs[1]
        else:
            self.inp_dim = nlemb_HPs[1]

    def forward(self, nl_tensor, tp_tensor=None, pos_tensor=None):
        return self.inp_composition(nl_tensor, tp_tensor, pos_tensor)

    def inp_composition(self, nl_tensor, tp_tensor=None, pos_tensor=None):
        # nlemb: (batch, q_len, emb_size)
        nlemb = self.sembedding(nl_tensor)
        # if wombat_object is not None:
        #     for k, v in enumerate(nl_pad_ids[0]):
        #         if v == self.args.vocab.sw2i[UNK]:
        #             wombat_emb = wombat_object.get(nl[k])
        #             if wombat_emb is not None:
        #                 print("Use external Pre-trained emb for {}".format(entry["question_arg"][k]))
        #                 nlemb[0][k] = torch.from_numpy(wombat_emb)
        if self.swd_inp.startswith("triple"):
            # tpemb: (batch, h_len, emb_size)
            tpemb = self.tpembedding(tp_tensor)
            # posemb: (batch, h_len, emb_size)
            posemb = self.posembedding(pos_tensor)
            if self.swd_mode.startswith("conc"):
                en_inp = torch.cat([nlemb, tpemb, posemb], dim=-1)
            elif self.swd_mode.startswith("avg"):
                en_inp = (nlemb + tpemb + posemb) / 3
            else:
                en_inp = nlemb + tpemb + posemb
        elif self.swd_inp.startswith("dual_pos"):
            # posemb: (batch, h_len, emb_size)
            posemb = self.posembedding(pos_tensor)
            if self.swd_mode.startswith("conc"):
                en_inp = torch.cat([nlemb, posemb], dim=-1)
            elif self.swd_mode.startswith("avg"):
                en_inp = (nlemb + posemb) / 2
            else:
                en_inp = nlemb + posemb
        elif self.swd_inp.startswith("dual_tp"):
            # tpemb: (batch, h_len, emb_size)
            tpemb = self.tpembedding(tp_tensor)
            if self.swd_mode.startswith("conc"):
                en_inp = torch.cat([nlemb, tpemb], dim=-1)
            elif self.swd_mode.startswith("avg"):
                en_inp = (nlemb + tpemb) / 2
            else:
                en_inp = nlemb + tpemb
        else:
            en_inp = nlemb
        return en_inp


class Schema_Ptr(nn.Module):
    def __init__(self, ptr_HPs):
        super(Schema_Ptr, self).__init__()
        col_dim, tab_dim, fn_dim = ptr_HPs
        self.col_pointer = Pointer_net(col_dim, fn_dim)
        self.tab_pointer = Pointer_net(tab_dim, fn_dim)


class Schema_Att(nn.Module):
    def __init__(self, sch_HPs):
        super(Schema_Att, self).__init__()
        colemb_HPs, tabemb_HPs, gnn_HPs, att_HPs = sch_HPs

        self.col_dim = colemb_HPs[1]
        self.tab_dim = tabemb_HPs[1]

        self.colembedding = Emb_layer(colemb_HPs)
        self.tabembedding = Emb_layer(tabemb_HPs)

        self.use_graph = gnn_HPs[0]
        if self.use_graph:
            assert self.col_dim == self.tab_dim, print("Column emb and table emb must have the same dimension")
            _, num_timesteps, num_edge_types, dropout = gnn_HPs
            self.gnn = GatedGraphConv(self.col_dim, num_timesteps, num_edge_types, dropout)

        use_transformer, sch_att, hidden_dim = att_HPs
        self.use_transformer = att_HPs[0]
        self.sch_att = att_HPs[1]
        if self.sch_att == "en_hidden":
            assert self.col_dim > 0
            self.col_attention = Col_awareness(self.col_dim, hidden_dim)

            assert self.tab_dim > 0
            self.tab_attention = Col_awareness(self.tab_dim, hidden_dim)
        elif self.sch_att == "de_hidden":
            assert self.col_dim > 0
            self.col_attention = Word_alignment(self.col_dim, hidden_dim)

            assert self.tab_dim > 0
            self.tab_attention = Word_alignment(self.tab_dim, hidden_dim)
        else:
            pass

    def forward(self, col_tensor, tab_tensor, colmask, tabmask, enc_hn=None, dec_out=None, edge_indexes=None):
        if self.use_transformer:
            colemb = self.colembedding(col_tensor)
            tabemb = self.tabembedding(tab_tensor)
        else:
            colemb = self.colembedding(col_tensor).sum(dim=-2)
            tabemb = self.tabembedding(tab_tensor).sum(dim=-2)

        if self.use_graph:
            graph_data_list = []
            num_tab = tabemb.size(1)
            num_col = colemb.size(1)
            num_nodes = num_tab + num_col
            graphemb = torch.cat([tabemb, colemb], dim=1)
            for gidx in range(len(edge_indexes)):
                graph_data = Data(graphemb[gidx])
                for eidx, l in enumerate(edge_indexes[gidx]):
                    if l:
                        edge_ = Data2tensor.idx2tensor(l, device=self.device,
                                                       dtype=torch.long).transpose(0, 1).contiguous()
                    else:
                        edge_ = Data2tensor.idx2tensor(l, device=self.device, dtype=torch.long)
                    graph_data[f'edge_index_type_{eidx}'] = edge_
                graph_data_list.append(graph_data)

            graph_batch = Batch.from_data_list(graph_data_list)

            gnn_output = self.gnn(graph_batch.x, [graph_batch[f'edge_index_type_{eidx}'] or eidx in range(self.gnn.num_edge_types)])
            gnn_output = gnn_output.view(len(edge_indexes), num_nodes, -1)
            tabemb = gnn_output[:, :num_tab, :]
            colemb = gnn_output[:, num_tab:, :]

        if self.sch_att == "en_hidden":
            # enc_hn: [batch, hidden_dim]
            # colemb: [batch, num_col, col_features]
            # col_context: [batch, 1, col_features]
            col_context = self.col_attention(colemb, enc_hn, colmask)
            # col_context: [batch, seq_length2, col_features]
            col_context = col_context.expand(-1, dec_out.size(1), -1)

            # tab_context: [batch, 1, tab_features]
            tab_context = self.tab_attention(tabemb, enc_hn, tabmask)
            # tab_context: [batch, seq_length2, tab_features]
            tab_context = tab_context.expand(-1, dec_out.size(1), -1)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        elif self.sch_att == "de_hidden":
            # col_context: [batch, seq_length2, col_dim]
            col_context = self.col_attention(colemb, dec_out, colmask)

            # tab_context: [batch, seq_length2, tab_dim]
            tab_context = self.tab_attention(tabemb, dec_out, tabmask)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        else:
            col_context = None
            tab_context = None
        return col_context, tab_context


class Seq2seq_v2(nn.Module):
    def __init__(self, semb_HPs, sch_HPs, enc_HPs, dec_HPs, drop_rate=0.5, num_labels=None, enc_att=False):
        super(Seq2seq_v2, self).__init__()
        self.source_emb = Source_Emb(semb_HPs)
        self.schema_att = Schema_Att(sch_HPs)
        enc_HPs[1] = self.source_emb.inp_dim
        self.encoder = Word_Encoder(enc_HPs)
        self.decoder = Word_Decoder_v2(dec_HPs)

        self.use_transformer = sch_HPs[3][0]
        self.enc_cnn = enc_HPs[0]
        self.ed_mode = dec_HPs[1][0]
        self.ed_bidirect = dec_HPs[1][4]

        self.enc_att = enc_att
        hidden_dim = dec_HPs[1][2]
        self.fn_dim = hidden_dim

        if self.enc_att:
            self.enc_attention = Word_alignment(hidden_dim, hidden_dim)
            self.fn_dim += hidden_dim

        self.sch_att = self.schema_att.sch_att
        if self.sch_att == "en_hidden" or self.sch_att == "de_hidden":
            self.fn_dim += self.schema_att.col_dim + self.schema_att.tab_dim

        self.finaldrop_layer = nn.Dropout(drop_rate)

        self.num_labels = num_labels
        if num_labels > 2:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()

    def reorder_nl(self, en_out, en_hidden, nl_len_tensor, nl_recover_ord_tensor, lb_ord_tensor):
        """
            Reordering the order of nls in a batch to follow SQL orders
        """
        if self.enc_cnn == "cnn" and self.ed_mode != "lstm":
            en_hidden = en_hidden[0]
        if isinstance(en_hidden, tuple):
            en_hidden = tuple(hidden[:, nl_recover_ord_tensor, :] for hidden in en_hidden)
            en_hidden = tuple(hidden[:, lb_ord_tensor, :] for hidden in en_hidden)
            if self.ed_bidirect:
                en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
            else:
                en_hn = en_hidden[0][-1, :, :]
        else:
            en_hidden = en_hidden[:, nl_recover_ord_tensor, :]
            en_hidden = en_hidden[:, lb_ord_tensor, :]
            if self.ed_bidirect:
                en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
            else:
                en_hn = en_hidden[-1, :, :]

        en_out = en_out[nl_recover_ord_tensor, :, :]
        en_out = en_out[lb_ord_tensor, :, :]
        nl_len_tensor = nl_len_tensor[nl_recover_ord_tensor]
        nl_len_tensor = nl_len_tensor[lb_ord_tensor]
        return en_out, en_hidden, en_hn, nl_len_tensor

    def reorder_schema(self, col_tensor, col_len_tensor, tab_tensor, tab_len_tensor, lb_ord_tensor, nl_size):
        """
            Reordering the order of nls in a batch to follow SQL orders
        """
        if self.use_transformer:
            if col_tensor.size(0) != 1 and col_tensor.size(0) == lb_ord_tensor.size(0):
                col_tensor = col_tensor[lb_ord_tensor, :]
                col_len_tensor = col_len_tensor[lb_ord_tensor]

            if tab_tensor.size(0) != 1 and tab_tensor.size(0) == lb_ord_tensor.size(0):
                tab_tensor = tab_tensor[lb_ord_tensor, :]
                tab_len_tensor = tab_len_tensor[lb_ord_tensor]
        else:
            if col_tensor.size(0) != 1 and col_tensor.size(0) == lb_ord_tensor.size(0):
                col_tensor = col_tensor[lb_ord_tensor, :, :]
                col_len_tensor = col_len_tensor[lb_ord_tensor]

            if tab_tensor.size(0) != 1 and tab_tensor.size(0) == lb_ord_tensor.size(0):
                tab_tensor = tab_tensor[lb_ord_tensor, :, :]
                tab_len_tensor = tab_len_tensor[lb_ord_tensor]
        return col_tensor, col_len_tensor, tab_tensor, tab_len_tensor

    def forward(self, nl_tensor, nl_len_tensor, nl_recover_ord_tensor,
                ilb_tensor, lb_len_tensor, lb_ord_tensor, teacher_force,
                col_tensor, tab_tensor, col_len_tensor, tab_len_tensor, edge_indexes,
                tp_tensor=None, pos_tensor=None):
        en_inp = self.source_emb(nl_tensor, tp_tensor, pos_tensor)
        en_out, en_hidden = self.encoder(en_inp, nl_len_tensor)

        en_out, de_hidden, en_hn, nl_len_tensor = self.reorder_nl(en_out, en_hidden,
                                                                  nl_len_tensor, nl_recover_ord_tensor, lb_ord_tensor)
        en_mask = torch.arange(max(nl_len_tensor), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:, None]

        col_tensor, col_len_tensor, tab_tensor, tab_len_tensor = self.reorder_schema(col_tensor, col_len_tensor,
                                                                                     tab_tensor, tab_len_tensor,
                                                                                     lb_ord_tensor, nl_tensor.size(0))
        colmask = torch.arange(max(col_len_tensor), dtype=torch.long, device=device)[None, :] < col_len_tensor[:, None]
        tabmask = torch.arange(max(tab_len_tensor), dtype=torch.long, device=device)[None, :] < tab_len_tensor[:, None]
        if teacher_force:
            de_out, de_hidden = self.decoder(ilb_tensor, lb_len_tensor, de_hidden)
            enc_context = None
            if self.enc_att:
                # enc_context: [batch, seq_length2, hidden_dim]
                enc_context = self.enc_attention(en_out, de_out, en_mask)
                # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
            col_context, tab_context = None, None
            if self.sch_att:
                col_context, tab_context = self.schema_att(col_tensor, tab_tensor, colmask, tabmask, en_hn, de_out, edge_indexes)

            if enc_context is not None:
                de_out = torch.cat((de_out, enc_context), dim=-1)
            if col_context is not None:
                de_out = torch.cat((de_out, col_context), dim=-1)
            if tab_context is not None:
                de_out = torch.cat((de_out, tab_context), dim=-1)
            de_score = self.scoring(de_out)
            return de_score, de_out, de_hidden
        else:
            # first input to the decoder is the <sos> token
            batch_size = ilb_tensor.shape[0]
            max_len = ilb_tensor.shape[1]
            de_outputs = torch.zeros(batch_size, max_len, self.num_labels, device=device)
            output = ilb_tensor[:, 0].view(batch_size, 1)
            for t in range(max_len):
                de_out, de_hidden = self.decoder(output, None, de_hidden)
                enc_context = None
                if self.enc_att:
                    # enc_context: [batch, seq_length2, hidden_dim]
                    enc_context = self.enc_attention(en_out, de_out, en_mask)
                    # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)

                col_context, tab_context = self.schema_att(col_tensor, tab_tensor, colmask, tabmask, en_hn, de_out, edge_indexes)
                if enc_context is not None:
                    de_out = torch.cat((de_out, enc_context), dim=-1)
                if col_context is not None:
                    de_out = torch.cat((de_out, col_context), dim=-1)
                if tab_context is not None:
                    de_out = torch.cat((de_out, tab_context), dim=-1)
                de_score = self.scoring(de_out)
                de_outputs[:, t] = de_score[:, 0, :]
                output = de_score.max(-1)[1].detach()
            return de_outputs, de_out, de_hidden

    def scoring(self, rnn_out):
        de_score = self.hidden2tag_layer(rnn_out)
        de_score = self.finaldrop_layer(de_score)
        return de_score

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            # label_score = [B, C]; label_tensor = [B, ]
            de_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1,))
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


class LoadBert(nn.Module):
    def __init__(self, model_name):
        super(LoadBert, self).__init__()
        from pytorch_pretrained_bert import BertModel
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, input_tensors):
        return self.get_emb(input_tensors)

    def get_emb(self, input_tensors):
        # Predict hidden states features for each layer
        self.model.eval()
        # fit all BERT parameters without fine-tuning
        for param in self.model.parameters():
            param.requires_grad = False
        encoded_layers, _ = self.model(input_tensors)
        return encoded_layers[-1]


def multigpu_feed(inp_tensor, device):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        inp_tensor = nn.DataParallel(inp_tensor)
    return inp_tensor.to(device)


class Seq2seq(nn.Module):
    def __init__(self, args):
        super(Seq2seq, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")
        # Include SOt, EOt if set set_words, else Ignore SOt, EOt
        self.num_labels = len(self.args.vocab.tw2i)

        # Hyper-parameters at source language
        self.schema2idx = self.args.vocab.hierlst2idx(vocab_words=self.args.vocab.sw2i, unk_words=True,
                                                      sos=False, eos=False)

        self.source2idx = self.args.vocab.lst2idx(vocab_words=self.args.vocab.sw2i, unk_words=True,
                                                  sos=self.args.ssos, eos=self.args.seos)

        self.tp2idx = self.args.vocab.lst2idx(vocab_words=self.args.vocab.tp2i, unk_words=True,
                                              sos=self.args.ssos, eos=self.args.seos)

        self.pos2idx = self.args.vocab.lst2idx(vocab_words=self.args.vocab.pos2i, unk_words=True,
                                               sos=self.args.ssos, eos=self.args.seos)

        # Hyper-parameters at target language
        self.target2idx = self.args.vocab.lst2idx(vocab_words=self.args.vocab.tw2i, unk_words=False,
                                                  sos=self.args.tsos, eos=self.args.teos,
                                                  reverse=self.args.t_reverse)

        # Hyper-parameters at word-level source language
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        semb_HPs = [len(self.args.vocab.sw2i), self.args.swd_dim, self.args.swd_pretrained,
                    self.args.wd_dropout, self.args.wd_padding, self.args.swd_reqgrad]
        self.sembedding = Emb_layer(semb_HPs)

        # Hyper-parameters at hidden-level source language
        if self.args.swd_inp.startswith("dual_tp"):
            tpemb_HPs = [len(self.args.vocab.tp2i), self.args.swd_dim, None,
                         self.args.wd_dropout, self.args.wd_padding, self.args.swd_reqgrad]
            self.tpembedding = Emb_layer(tpemb_HPs)

        if self.args.swd_inp.startswith("dual_pos"):
            posemb_HPs = [len(self.args.vocab.pos2i), self.args.swd_dim, None,
                          self.args.wd_dropout, self.args.wd_padding, self.args.swd_reqgrad]
            self.posembedding = Emb_layer(posemb_HPs)

        if self.args.swd_inp.startswith("triple"):
            tpemb_HPs = [len(self.args.vocab.tp2i), self.args.swd_dim, None,
                         self.args.wd_dropout, self.args.wd_padding, self.args.swd_reqgrad]
            self.tpembedding = Emb_layer(tpemb_HPs)

            posemb_HPs = [len(self.args.vocab.pos2i), self.args.swd_dim, None,
                          self.args.wd_dropout, self.args.wd_padding, self.args.swd_reqgrad]
            self.posembedding = Emb_layer(posemb_HPs)

        if self.args.use_transformer:
            self.col2idx = self.args.schema_reader.Trans2idx(self.args.schema_reader.col2i)
            col_dim = self.args.col_pretrained.shape[1]
            colemb_HPs = [len(self.args.schema_reader.col2i), col_dim, self.args.col_pretrained,
                          self.args.wd_dropout, self.args.wd_padding, False]
            self.colembedding = Emb_layer(colemb_HPs)

            self.tab2idx = self.args.schema_reader.Trans2idx(self.args.schema_reader.tab2i)
            tab_dim = self.args.tab_pretrained.shape[1]
            tabemb_HPs = [len(self.args.schema_reader.tab2i), tab_dim, self.args.tab_pretrained,
                          self.args.wd_dropout, self.args.wd_padding, False]
            self.tabembedding = Emb_layer(tabemb_HPs)
        else:
            self.col2idx = self.schema2idx
            self.colembedding = self.sembedding
            col_dim = self.args.swd_dim

            self.tab2idx = self.schema2idx
            self.tabembedding = self.sembedding
            tab_dim = self.args.swd_dim

        # Hyper-parameters at hidden-level source language
        if self.args.swd_inp.startswith("triple") and self.args.swd_mode.startswith("conc"):
            inp_dim = self.args.swd_dim * 3
        elif self.args.swd_inp.startswith("dual") and self.args.swd_mode.startswith("conc"):
            inp_dim = self.args.swd_dim * 2
        else:
            inp_dim = self.args.swd_dim
        # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        enc_HPs = [self.args.ed_mode, inp_dim, self.args.ed_outdim,
                   self.args.ed_layers, self.args.ed_bidirect, self.args.ed_dropout]

        self.encoder = Word_Encoder(word_HPs=enc_HPs)

        # Hyper-parameters at word-level target language
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        temb_HPs = [len(self.args.vocab.tw2i), self.args.twd_dim, self.args.twd_pretrained,
                    self.args.wd_dropout, self.args.wd_padding, self.args.twd_reqgrad]

        # Hyper-parameters at word-level target language
        dec_HPs = [self.args.ed_mode, self.args.twd_dim, self.args.ed_outdim,
                   self.args.ed_layers, self.args.ed_bidirect, self.args.ed_dropout]

        self.decoder = Word_Decoder(word_HPs=[temb_HPs, dec_HPs], drop_rate=self.args.final_dropout,
                                    num_labels=self.num_labels, enc_att=self.args.enc_att, sch_att=self.args.sch_att,
                                    col_dim=col_dim, tab_dim=tab_dim)

        if self.args.use_pointer:
            self.col_ids = list(self.args.vocab.col2i_ord.keys())
            self.tab_ids = list(self.args.vocab.tab2i_ord.keys())

            self.col_pointer = Pointer_net(col_dim, self.decoder.fn_dim)
            self.tab_pointer = Pointer_net(tab_dim, self.decoder.fn_dim)

        if self.args.use_graph:
            assert col_dim == tab_dim, print("Column emb and table emb must have the same dimension")
            self.gnn = GatedGraphConv(col_dim, num_timesteps=self.args.graph_timesteps,
                                      num_edge_types=self.args.graph_edge_types,
                                      dropout=self.args.graph_dropout)

    def forward(self, batch_data, mode="train"):
        # set model in train model
        dbid, nl, tp, pos, target = list(zip(*batch_data))
        if len(set(dbid)) == 1:
            col, tab, edge_index = self.args.schema_reader.getts(dbid[0], col2idx=self.col2idx, tab2idx=self.tab2idx)
            cols = [col]
            tabs = [tab]
            edge_indexes = [edge_index]
        else:
            cols = []
            tabs = []
            edge_indexes = []
            for db in dbid:
                col, tab, edge_index = self.args.schema_reader.getts(db, col2idx=self.col2idx, tab2idx=self.tab2idx)
                cols.append(col)
                tabs.append(tab)
                edge_indexes.append(edge_index)

        nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=self.args.vocab.sw2i[PAD], nlevels=1)
        nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=self.device)
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

        tp_pad_ids, tp_lens = seqPAD.pad_sequences(tp, pad_tok=self.args.vocab.tp2i[PAD], nlevels=1)
        tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=self.device)
        tp_tensor = tp_tensor[nl_ord_tensor]

        pos_pad_ids, pos_lens = seqPAD.pad_sequences(pos, pad_tok=self.args.vocab.pos2i[PAD], nlevels=1)
        pos_tensor = Data2tensor.idx2tensor(pos_pad_ids, dtype=torch.long, device=self.device)
        pos_tensor = pos_tensor[nl_ord_tensor]
        assert tp_lens == nl_lens

        lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=self.args.vocab.tw2i[PAD], nlevels=1)
        lb_tensors = Data2tensor.sort_labelled_tensors(lb_pad_ids, lb_lens, label=True, dtype=torch.long,
                                                       device=self.device)
        olb_tensor, ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor, _, _, _ = lb_tensors

        # nlemb: (batch, q_len, emb_size)
        nlemb = self.sembedding(nl_tensor)
        tpemb = None
        posemb = None
        if self.args.swd_inp.startswith("dual_tp"):
            # tpemb: (batch, h_len, emb_size)
            tpemb = self.tpembedding(tp_tensor)

        if self.args.swd_inp.startswith("dual_pos"):
            # posemb: (batch, h_len, emb_size)
            posemb = self.posembedding(pos_tensor)
        if self.args.swd_inp.startswith("triple"):
            # tpemb: (batch, h_len, emb_size)
            tpemb = self.tpembedding(tp_tensor)
            # posemb: (batch, h_len, emb_size)
            posemb = self.posembedding(pos_tensor)

        # en_inp = nlemb + tpemb + posemb
        en_inp = self.inp_composition(nlemb, tpemb, posemb)

        if self.args.use_transformer:
            col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=self.args.schema_reader.col2i[PAD],
                                                         nlevels=1)
            col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=self.device)
            col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=self.device)
            # if len(set(dbid)) != 1:
            #     col_tensor = col_tensor[nl_ord_tensor]
            # colemb: (batch, col_len, emb_size)
            colemb = self.colembedding(col_tensor)

            tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=self.args.schema_reader.tab2i[PAD],
                                                         nlevels=1)
            tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=self.device)
            tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=self.device)

            tabemb = self.tabembedding(tab_tensor)
        else:
            col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=self.args.vocab.sw2i[PAD], nlevels=3)
            col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=self.device)
            col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=self.device)

            colemb = self.colembedding(col_tensor).sum(dim=-2).sum(dim=-2)

            tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)
            tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=self.device)
            tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=self.device)

            tabemb = self.tabembedding(tab_tensor).sum(dim=-2)

        if self.args.use_graph:
            graph_data_list = []
            num_tab = tabemb.size(1)
            num_col = colemb.size(1)
            num_nodes = num_tab + num_col
            graphemb = torch.cat([tabemb, colemb], dim=1)
            for gidx in range(len(edge_indexes)):
                graph_data = Data(graphemb[gidx])
                for eidx, l in enumerate(edge_indexes[gidx]):
                    if l:
                        edge_ = Data2tensor.idx2tensor(l, device=self.device,
                                                       dtype=torch.long).transpose(0, 1).contiguous()
                    else:
                        edge_ = Data2tensor.idx2tensor(l, device=self.device, dtype=torch.long)
                    graph_data[f'edge_index_type_{eidx}'] = edge_
                graph_data_list.append(graph_data)

            graph_batch = Batch.from_data_list(graph_data_list)

            gnn_output = self.gnn(graph_batch.x, [graph_batch[f'edge_index_type_{eidx}']
                                                  for eidx in range(self.gnn.num_edge_types)])
            gnn_output = gnn_output.view(len(edge_indexes), num_nodes, -1)

            tabemb = gnn_output[:, :num_tab, :]
            colemb = gnn_output[:, num_tab:, :]

        if len(cols) == 1:
            colemb = colemb.expand(nlemb.size(0), -1, -1)
            col_len_tensor = col_len_tensor.expand(nlemb.size(0))
        if len(tabs) == 1:
            tabemb = tabemb.expand(nlemb.size(0), -1, -1)
            tab_len_tensor = tab_len_tensor.expand(nlemb.size(0))

        # source --> encoder --> source_hidden
        en_out, en_hidden = self.encoder(en_inp, nl_len_tensor)
        if isinstance(en_hidden, tuple):
            de_hidden = tuple(hidden[:, nl_recover_ord_tensor, :] for hidden in en_hidden)
            de_hidden = tuple(hidden[:, lb_ord_tensor, :] for hidden in de_hidden)
            if self.args.ed_bidirect:
                en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
            else:
                en_hn = en_hidden[0][-1, :, :]
        else:
            de_hidden = en_hidden[:, nl_recover_ord_tensor, :]
            de_hidden = de_hidden[:, lb_ord_tensor, :]
            if self.args.ed_bidirect:
                en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
            else:
                en_hn = en_hidden[-1, :, :]

        en_out = en_out[nl_recover_ord_tensor, :, :]
        en_out = en_out[lb_ord_tensor, :, :]
        nl_len_tensor = nl_len_tensor[nl_recover_ord_tensor]
        nl_len_tensor = nl_len_tensor[lb_ord_tensor]
        # colemb = (batch, col_len, dim); tabemb = (batch, tab_len, dim)
        colemb = colemb[lb_ord_tensor, :, :]
        tabemb = tabemb[lb_ord_tensor, :, :]
        col_len_tensor = col_len_tensor[lb_ord_tensor]
        tab_len_tensor = tab_len_tensor[lb_ord_tensor]

        en_mask = torch.arange(max(nl_len_tensor),
                               dtype=torch.long, device=self.device)[None, :] < nl_len_tensor[:, None]
        col_len_mask = torch.arange(max(col_len_tensor),
                                    dtype=torch.long, device=self.device)[None, :] < col_len_tensor[:, None]
        tab_len_mask = torch.arange(max(tab_len_tensor),
                                    dtype=torch.long, device=self.device)[None, :] < tab_len_tensor[:, None]

        if mode == "train":
            total_loss = 0
            use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False
            if use_teacher_forcing:  # directly feed the ground-truth target language to the decoder
                label_mask = olb_tensor > 0
                if self.args.use_pointer:
                    # map C(id), T(id) of inputs into column <COL>  and table <TAB> placeholders
                    mapilb_tensor = ilb_tensor.clone()
                    mapilb_tensor = mapilb_tensor.to("cpu").apply_(lambda x: self.args.vocab.coltab2i_ph.get(x, x))
                    mapilb_tensor = mapilb_tensor.to(self.device)
                    de_score, de_out, de_hidden = self.decoder(mapilb_tensor, lb_len_tensor, de_hidden, en_out,
                                                               en_mask, en_hn, colemb, tabemb, col_len_mask, tab_len_mask)
                    # colpt_score = [batch, nl_len, col_len]
                    colpt_score = self.col_pointer(colemb, de_out, col_len_mask)
                    # [batch, nl_len, tab_len]
                    tabpt_score = self.tab_pointer(tabemb, de_out, tab_len_mask)

                    # TODO double-check for cuda
                    mask_col = (olb_tensor == self.col_ids[0])
                    for colid in self.col_ids[1:]:
                        mask_col = (olb_tensor == colid) | mask_col
                    # filter out all columns
                    colord_tensor = olb_tensor[mask_col]
                    colpt_loss = 0
                    if colord_tensor.nelement() != 0:
                        # map C(id) of outputs into column <COL> placeholders
                        colord_tensor = colord_tensor.to("cpu").apply_(lambda x: self.args.vocab.col2i_ord[x])
                        # colord_tensor.to(self.device)
                        gold_label = colord_tensor.to(self.device)
                        pred_label = colpt_score[mask_col].view(gold_label.shape[0], -1)

                        colpt_loss = self.col_pointer.NLL_loss(pred_label, gold_label)

                    # TODO double check for cuda
                    mask_tab = olb_tensor == self.tab_ids[0]
                    for tabid in self.tab_ids[1:]:
                        mask_tab = (olb_tensor == tabid) | mask_tab
                    tabord_tensor = olb_tensor[mask_tab]
                    tabpt_loss = 0
                    if tabord_tensor.nelement() != 0:
                        tabord_tensor = tabord_tensor.to("cpu").apply_(lambda x: self.args.vocab.tab2i_ord[x])
                        # tabord_tensor.to(self.device)
                        gold_label = tabord_tensor.to(self.device)
                        pred_label = tabpt_score[mask_tab].view(gold_label.shape[0], -1)
                        tabpt_loss = self.tab_pointer.NLL_loss(pred_label, gold_label)
                    # map C(id), T(id) of OUTPUTs into column and table placeholders <COL> and <TAB>
                    mapolb_tensor = olb_tensor.clone()
                    mapolb_tensor = mapolb_tensor.to("cpu").apply_(lambda x: self.args.vocab.coltab2i_ph.get(x, x))
                    mapolb_tensor = mapolb_tensor.to(self.device)
                    mask_noncoltab = label_mask ^ mask_col ^ mask_tab
                    mapolb_tensor = mapolb_tensor[label_mask]
                    # mapolb_tensor = mapolb_tensor[mask_noncoltab]
                    de_loss = 0
                    if mapolb_tensor.nelement() != 0:
                        de_loss = self.decoder.NLL_loss(de_score[label_mask], mapolb_tensor)
                        # de_loss = self.decoder.NLL_loss(de_score[mask_noncoltab], maskolb_tensor)
                    total_loss = de_loss + colpt_loss + tabpt_loss
                else:
                    de_score, de_out, de_hidden = self.decoder(ilb_tensor, lb_len_tensor, de_hidden, en_out, en_mask,
                                                               en_hn, colemb, tabemb, col_len_mask, tab_len_mask)
                    total_loss = self.decoder.NLL_loss(de_score[label_mask], olb_tensor[label_mask])
            else:
                batch_size, target_length = ilb_tensor.size()
                # Extract the first target word (tsos or SOT) to feed to decoder
                # ilb1_tensor = [batch_size, 1]
                ilb1_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                     device=self.device)
                # lb1_len_tensor = [batch_size]
                lb1_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
                count_tokens = 0
                for j in range(target_length):
                    # olb_tensor = (batch, len)
                    olb1_tensor = olb_tensor[:, j]
                    label1_mask = olb1_tensor > 0
                    if self.args.use_pointer:
                        # map C(id), T(id) of inputs into column and table placeholders <COL> and <TAB>
                        mapilb1_tensor = ilb1_tensor.clone()
                        mapilb1_tensor = mapilb1_tensor.to("cpu").apply_(
                            lambda x: self.args.vocab.coltab2i_ph.get(x, x))
                        mapilb1_tensor = mapilb1_tensor.to(self.device)
                        # de1_out = [batch_size, 1, hidden_dim]
                        de1_score, de1_out, de_hidden = self.decoder.get_all_hiddens(mapilb1_tensor, lb1_len_tensor,
                                                                                     de_hidden, en_out, en_mask, en_hn,
                                                                                     colemb, tabemb, col_len_mask, tab_len_mask)
                        # label1_prob = label1_pred = [batch_size, 1, 1]
                        label1_prob, label1_pred = self.decoder.inference(de1_score, 1)
                        # ilb1_tensor = [batch_size, 1]
                        ilb1_tensor = label1_pred.squeeze(-1).detach()  # detach from history as input

                        colpt1_score = self.col_pointer(colemb, de1_out, col_len_mask)
                        # TODO double check for cuda
                        mask1_col = olb1_tensor == self.col_ids[0]
                        for colid in self.col_ids[1:]:
                            mask1_col = (olb1_tensor == colid) | mask1_col
                        colord1_tensor = olb1_tensor[mask1_col]
                        colpt1_loss = 0
                        if colord1_tensor.nelement() != 0:
                            colord1_tensor = colord1_tensor.to("cpu").apply_(lambda x: self.args.vocab.col2i_ord[x])
                            # colord1_tensor.to(self.device)
                            gold1_label = colord1_tensor.to(self.device)
                            pred1_label = colpt1_score[mask1_col].view(gold1_label.shape[0], -1)
                            colpt1_loss = self.col_pointer.NLL_loss(pred1_label, gold1_label)

                        tabpt1_score = self.tab_pointer(tabemb, de1_out, tab_len_mask)
                        # TODO double check for cuda
                        mask1_tab = olb1_tensor == self.tab_ids[0]
                        for tabid in self.tab_ids[1:]:
                            mask1_tab = (olb1_tensor == tabid) | mask1_tab
                        tabord1_tensor = olb1_tensor[mask1_tab]
                        tabpt1_loss = 0
                        if tabord1_tensor.nelement() != 0:
                            tabord1_tensor = tabord1_tensor.to("cpu").apply_(lambda x: self.args.vocab.tab2i_ord[x])
                            # tabord1_tensor.to(self.device)
                            gold1_label = tabord1_tensor.to(self.device)
                            pred1_label = tabpt1_score[mask1_tab].view(gold1_label.shape[0], -1)
                            tabpt1_loss = self.tab_pointer.NLL_loss(pred1_label, gold1_label)

                        # map C(id), T(id) of OUTPUTs into column and table placeholders <COL> and <TAB>
                        mapolb1_tensor = olb1_tensor.clone()
                        mapolb1_tensor = mapolb1_tensor.to("cpu").apply_(
                            lambda x: self.args.vocab.coltab2i_ph.get(x, x))
                        mapolb1_tensor = mapolb1_tensor.to(self.device)
                        mask1_noncoltab = label1_mask ^ mask1_col ^ mask1_tab
                        mapolb1_tensor = mapolb1_tensor[label1_mask]
                        # mapolb1_tensor = mapolb1_tensor[mask1_noncoltab]
                        de1_loss = 0
                        if mapolb1_tensor.nelement() != 0:
                            de1_loss = self.decoder.NLL_loss(de1_score[label1_mask], mapolb1_tensor)
                            # de1_loss = self.decoder.NLL_loss(de1_score[mask1_noncoltab], maskolb1_tensor)
                        total_loss = total_loss + de1_loss + colpt1_loss + tabpt1_loss
                        count_tokens += label1_mask.sum().item()
                    else:
                        # de_out = [batch_size, 1, hidden_dim]
                        de1_score, de1_out, de_hidden = self.decoder(ilb1_tensor, lb1_len_tensor, de_hidden,
                                                                     en_out, en_mask, en_hn, colemb, tabemb,
                                                                     col_len_mask, tab_len_mask)
                        # label1_prob = label1_pred = [batch_size, 1, 1]
                        label1_prob, label1_pred = self.decoder.inference(de1_score, 1)
                        # ilb1_tensor = [batch_size, 1]
                        ilb1_tensor = label1_pred.squeeze(-1).detach()  # detach from history as input
                        total_loss += self.decoder.NLL_loss(de1_score[label1_mask], olb1_tensor[label1_mask])
                        count_tokens += label1_mask.sum().item()
                total_loss = total_loss / count_tokens
            return total_loss
        else:
            batch_size, target_length = ilb_tensor.size()
            # Extract the first target word (tsos or SOT) to feed to decoder
            # ilb1_tensor = [batch_size, 1]
            ilb1_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                 device=self.device)
            # lb1_len_tensor = [batch_size]
            lb1_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
            count_tokens = 0
            total_loss = 0
            # Ignore SOT
            label_words = []
            predict_words = []
            for j in range(target_length):
                olb1_tensor = olb_tensor[:, j]
                label1_mask = olb1_tensor > 0
                if self.args.use_pointer:
                    # map C(id), T(id) of inputs into column and table placeholders <COL> and <TAB>
                    mapilb1_tensor = ilb1_tensor.clone()
                    mapilb1_tensor = mapilb1_tensor.to("cpu").apply_(
                        lambda x: self.args.vocab.coltab2i_ph.get(x, x))
                    mapilb1_tensor = mapilb1_tensor.to(self.device)
                    # de_out = [batch_size, 1, hidden_dim]
                    de1_score, de1_out, de_hidden = self.decoder(mapilb1_tensor, lb1_len_tensor, de_hidden, en_out,
                                                                 en_mask, en_hn, colemb, tabemb, col_len_mask, tab_len_mask)
                    colpt1_score = self.col_pointer(colemb, de1_out, col_len_mask)
                    tabpt1_score = self.tab_pointer(tabemb, de1_out, tab_len_mask)
                    # label1_prob = label1_pred = [batch_size, 1, 1]
                    label1_prob, label1_pred = self.decoder.inference(de1_score, 1)
                    # ilb1_tensor = [batch_size, 1]
                    ilb1_tensor = label1_pred.squeeze(-1).detach()  # detach from history as input
                    predict_word = Vocab.idx2text(ilb1_tensor.squeeze(-1).tolist(), self.args.vocab.i2tw, 1)
                    col1_label_prob, col1_label_pred = self.col_pointer.inference(colpt1_score, k=1)
                    predict_col = col1_label_pred.squeeze(-1).squeeze(-1).tolist()
                    tab1_label_prob, tab1_label_pred = self.tab_pointer.inference(tabpt1_score, k=1)
                    predict_tab = tab1_label_pred.squeeze(-1).squeeze(-1).tolist()
                    for idx in range(len(predict_word)):
                        if predict_word[idx] == COL:
                            predict_word[idx] = "C(" + str(predict_col[idx]) + ")"
                        if predict_word[idx] == TAB:
                            predict_word[idx] = "T(" + str(predict_tab[idx]) + ")"
                    predict_words += [predict_word]

                    # TODO double check for cuda
                    # col_ids = list(self.args.vocab.col2i_ord.keys())
                    mask1_col = olb1_tensor == self.col_ids[0]
                    for colid in self.col_ids[1:]:
                        mask1_col = (olb1_tensor == colid) | mask1_col
                    colord1_tensor = olb1_tensor[mask1_col]
                    colpt1_loss = 0
                    if colord1_tensor.nelement() != 0:
                        colord1_tensor = colord1_tensor.to("cpu").apply_(lambda x: self.args.vocab.col2i_ord[x])
                        gold1_label = colord1_tensor.to(self.device)
                        pred1_label = colpt1_score[mask1_col].view(gold1_label.shape[0], -1)
                        colpt1_loss = self.col_pointer.NLL_loss(pred1_label, gold1_label)

                    # TODO double check for cuda
                    mask1_tab = olb1_tensor == self.tab_ids[0]
                    for tabid in self.tab_ids[1:]:
                        mask1_tab = (olb1_tensor == tabid) | mask1_tab
                    tabord1_tensor = olb1_tensor[mask1_tab]
                    tabpt1_loss = 0
                    if tabord1_tensor.nelement() != 0:
                        tabord1_tensor = tabord1_tensor.to("cpu").apply_(lambda x: self.args.vocab.tab2i_ord[x])
                        gold1_label = tabord1_tensor.to(self.device)
                        pred1_label = tabpt1_score[mask1_tab].view(gold1_label.shape[0], -1)
                        tabpt1_loss = self.tab_pointer.NLL_loss(pred1_label, gold1_label)
                    # label_words = predict_words = [batch_size]
                    mapolb1_tensor = olb1_tensor.clone()
                    mapolb1_tensor = mapolb1_tensor.to("cpu").apply_(lambda x: self.args.vocab.coltab2i_ph.get(x, x))
                    mapolb1_tensor = mapolb1_tensor.to(self.device)
                    label_words += [Vocab.idx2text(olb1_tensor.tolist(), self.args.vocab.i2tw, 1)]

                    mask1_noncoltab = label1_mask ^ mask1_col ^ mask1_tab
                    mapolb1_tensor = mapolb1_tensor[label1_mask]
                    # mapolb1_tensor = mapolb1_tensor[mask1_noncoltab]
                    de1_loss = 0
                    if mapolb1_tensor.nelement() != 0:
                        pred1_label = de1_score[label1_mask]
                        # pred1_label = de1_score[mask1_noncoltab]
                        gold1_label = mapolb1_tensor
                        de1_loss = self.decoder.NLL_loss(pred1_label, gold1_label)
                    total_loss = total_loss + de1_loss + colpt1_loss + tabpt1_loss
                    count_tokens += label1_mask.sum().item()
                else:
                    # de1_out = [batch_size, 1, hidden_dim]
                    de1_score, de1_out, de_hidden = self.decoder(ilb1_tensor, lb1_len_tensor, de_hidden, en_out,
                                                                 en_mask, en_hn, colemb, tabemb,
                                                                 col_len_mask, tab_len_mask)
                    # label1_prob = label1_pred = [batch_size, 1, 1]
                    label1_prob, label1_pred = self.decoder.inference(de1_score, 1)
                    # ilb1_tensor = [batch_size, 1]
                    ilb1_tensor = label1_pred.squeeze(-1).detach()  # detach from history as input
                    # label_words = predict_words = [batch_size]
                    label_words += [Vocab.idx2text(olb1_tensor.tolist(), self.args.vocab.i2tw, 1)]
                    predict_words += [Vocab.idx2text(ilb1_tensor.squeeze(-1).tolist(), self.args.vocab.i2tw, 1)]
                    total_loss += self.decoder.NLL_loss(de1_score[label1_mask], olb1_tensor[label1_mask])
                    count_tokens += label1_mask.sum().item()
            total_loss = total_loss / count_tokens
            # label_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
            # -->list(zip(*label_words)) --> [[w1, ..., EOT], ..., [w1, ..., EOT]]
            label_words = self.filter_pad(label_words, (olb_tensor > 0).sum(dim=1).tolist())
            # predict_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
            # --> list(zip(*predict_words)) --> predict_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
            predict_words = self.filter_pad(predict_words, (olb_tensor > 0).sum(dim=1).tolist())

            return total_loss, label_words, predict_words, count_tokens

    def filter_pad(self, label_words, seq_len):
        # label_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
        label_words = list(zip(*label_words))
        # label_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
        # ignore EOT (i-1)
        filter_words = [words[:i - 1] if EOT not in words else words[: words.index(EOT)]
                        for words, i in zip(label_words, seq_len)]
        # print("Sequence length: ", seq_len)
        # print("Before filter: ", label_words)
        # print("After filter: ", filter_words)
        return filter_words

    def inp_composition(self, nlemb, tpemb=None, posemb=None):
        if self.args.swd_inp.startswith("triple"):
            if self.args.swd_mode.startswith("conc"):
                en_inp = torch.cat([nlemb, tpemb, posemb], dim=-1)
            elif self.args.swd_mode.startswith("avg"):
                en_inp = (nlemb + tpemb + posemb) / 3
            else:
                en_inp = nlemb + tpemb + posemb
        elif self.args.swd_inp.startswith("dual_pos"):
            if self.args.swd_mode.startswith("conc"):
                en_inp = torch.cat([nlemb, posemb], dim=-1)
            elif self.args.swd_mode.startswith("avg"):
                en_inp = (nlemb + posemb) / 2
            else:
                en_inp = nlemb + posemb
        elif self.args.swd_inp.startswith("dual_tp"):
            if self.args.swd_mode.startswith("conc"):
                en_inp = torch.cat([nlemb, tpemb], dim=-1)
            elif self.args.swd_mode.startswith("avg"):
                en_inp = (nlemb + tpemb) / 2
            else:
                en_inp = nlemb + tpemb
        else:
            en_inp = nlemb
        return en_inp


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from pytorch_transformers import BertTokenizer, XLNetTokenizer
    from np5.utils.data_utils import Schema_reader, Jsonfile, Vocab, seqPAD, Data2tensor, PAD, SOT, SaveloadHP
    from np5.modules.gated_graph_conv import GatedGraphConv
    device = torch.device("cpu")
    dtype = torch.long
    use_sql = True
    use_transformer = False
    use_pointer = False
    use_graph = False
    use_cuda = False
    db_file = "../../data/data_locate_toy/schema/json_tables_full.json"
    filename = "../../data/data_locate_toy/corpus/train.json"

    s_paras = [-1,  1]
    t_paras = [-1, 1]
    transformer_mode = "xlnet"
    if transformer_mode == "xlnet":
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased') if use_transformer else None
    else:   # "bert"
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') if use_transformer else None
    schema_reader = Schema_reader(db_file, tokenizer, transformer_mode=transformer_mode)

    vocab = Vocab(s_paras, t_paras, use_sql=use_sql, use_pointer=use_pointer)
    vocab.build([filename], schema_reader)

    nl2ids = vocab.lst2idx(vocab_words=vocab.sw2i, unk_words=True, eos=True)
    tp2ids = vocab.lst2idx(vocab_words=vocab.tp2i, unk_words=True, eos=True)
    pos2ids = vocab.lst2idx(vocab_words=vocab.pos2i, unk_words=True, eos=True)

    col2ids = vocab.hierlst2idx(vocab_words=vocab.sw2i, unk_words=True, eos=False)
    tab2ids = vocab.hierlst2idx(vocab_words=vocab.sw2i, unk_words=True, eos=False)

    col2ids = schema_reader.Trans2idx(vocab.col2i) if use_transformer else col2ids
    tab2ids = schema_reader.Trans2idx(vocab.tab2i) if use_transformer else tab2ids

    tg2ids = vocab.lst2idx(vocab_words=vocab.tw2i, unk_words=False, sos=True, eos=True)

    train_data = Jsonfile(filename, source2idx=nl2ids, tp2id=tp2ids, pos2id=pos2ids, target2idx=tg2ids,
                          use_sql=use_sql, use_pointer=use_pointer, schema_reader=schema_reader)

    data_idx = []
    batch = 8
    for d in vocab.minibatches(train_data, batch):
        data_idx.append(d)
        dbid, nl, tp, pos, target = list(zip(*d))

        nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=vocab.sw2i[PAD], nlevels=1)
        nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=device)
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

        tp_pad_ids, tp_lens = seqPAD.pad_sequences(tp, pad_tok=vocab.tp2i[PAD], nlevels=1)
        tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=device)
        tp_tensor = tp_tensor[nl_ord_tensor]

        pos_pad_ids, pos_lens = seqPAD.pad_sequences(pos, pad_tok=vocab.pos2i[PAD], nlevels=1)
        pos_tensor = Data2tensor.idx2tensor(pos_pad_ids, dtype=torch.long, device=device)
        pos_tensor = pos_tensor[nl_ord_tensor]
        assert tp_lens == nl_lens

        lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=vocab.tw2i[PAD], nlevels=1)
        lb_tensors = Data2tensor.sort_labelled_tensors(lb_pad_ids, lb_lens, label=True, dtype=torch.long,
                                                       device=device)
        olb_tensor, ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor, _, _, _ = lb_tensors

        cols = []
        tabs = []
        edge_indexes = []
        if len(set(dbid)) == 1:
            col, tab, edge_index = schema_reader.getts(dbid[0], col2idx=col2ids, tab2idx=tab2ids)
            cols = [col]
            tabs = [tab]
            edge_indexes = [edge_index]
            # tbcol_edge, coltb_edge, forpricol_edge, priforcol_edge, forpritab_edge, prifortab_edge = edge_index
        else:
            for db in dbid:
                col, tab, edge_index = schema_reader.getts(db, col2idx=col2ids, tab2idx=tab2ids)
                cols.append(col)
                tabs.append(tab)
                edge_indexes.append(edge_index)
                # tbcol_edge, coltb_edge, forpricol_edge, priforcol_edge, forpritab_edge, prifortab_edge = edge_index

        if use_transformer:
            col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=vocab.col2i[PAD], nlevels=1)
            col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=device)
            col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=device)

            tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=vocab.tab2i[PAD], nlevels=1)
            tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=device)
            tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=device)

        else:
            col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=vocab.sw2i[PAD], nlevels=2)
            col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=device)
            col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=device)

            tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=vocab.sw2i[PAD], nlevels=2)
            tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=device)
            tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=device)
        break

    # COPY FROM DATA_UTILS
    emb_size = len(vocab.sw2i)
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
    col_emb_file = "../../data/embeddings/schema.col.xlnet-large-cased"
    tab_emb_file = "../../data/embeddings/schema.tab.xlnet-large-cased"
    module_test = "partial2"

    if module_test == "separate":
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        emb_hps = [emb_size, emb_dim, emb_pretrained, emb_drop_rate, emb_zero_padding, requires_grad]
        sembedding = Emb_layer(emb_hps)
        # nlemb: (batch, q_len, emb_size)
        nlemb = sembedding(nl_tensor)

        tpemb_hps = [len(vocab.tp2i), emb_dim, None, emb_drop_rate, emb_zero_padding, requires_grad]
        tpembedding = Emb_layer(tpemb_hps)
        # tpemb: (batch, q_len, emb_size)
        tpemb = tpembedding(tp_tensor)

        posemb_hps = [len(vocab.pos2i), emb_dim, None, emb_drop_rate, emb_zero_padding, requires_grad]
        posembedding = Emb_layer(posemb_hps)
        # tpemb: (batch, q_len, emb_size)
        posemb = posembedding(pos_tensor)

        # en_emb: (batch, q_len, emb_size)
        en_emb = nlemb + tpemb + posemb
        # colemb: (batch, col_len, emb_size)
        if use_transformer:

            W = SaveloadHP.load(col_emb_file)
            colembedding = nn.Embedding(len(vocab.col2i), 1024, padding_idx=0)
            colembedding.weight.data.copy_(torch.from_numpy(W))
            colemb = colembedding(col_tensor)


            W = SaveloadHP.load(tab_emb_file)
            tabembedding = nn.Embedding(len(vocab.tab2i), 1024, padding_idx=0)
            tabembedding.weight.data.copy_(torch.from_numpy(W))
            tabemb = tabembedding(tab_tensor)

        else:
            colembedding = sembedding
            colemb = colembedding(col_tensor).sum(dim=-2).sum(dim=-2)

            tabembedding = sembedding
            tabemb = tabembedding(tab_tensor).sum(dim=-2)

        if use_graph:
            graph_data_list = []
            num_tab = tabemb.size(1)
            num_col = colemb.size(1)
            num_nodes = num_tab + num_col
            graphemb = torch.cat([tabemb, colemb], dim=1)
            for gidx in range(len(edge_indexes)):
                graph_data = Data(graphemb[gidx])
                for eidx, l in enumerate(edge_indexes[gidx]):
                    if l:
                        edge_ = Data2tensor.idx2tensor(l, device=device, dtype=torch.long).transpose(0, 1).contiguous()
                    else:
                        edge_ = Data2tensor.idx2tensor(l, device=device, dtype=torch.long)
                    graph_data[f'edge_index_type_{eidx}'] = edge_
                graph_data_list.append(graph_data)

            graph_batch = Batch.from_data_list(graph_data_list)
            gnn = GatedGraphConv(50, num_timesteps=3, num_edge_types=6, dropout=0.5)
            gnn_output = gnn(graph_batch.x, [graph_batch[f'edge_index_type_{eidx}'] for eidx in range(6)])
            gnn_output = gnn_output.view(len(edge_indexes), num_nodes, -1)

            tabemb = gnn_output[:, :num_tab, :]
            colemb = gnn_output[:, num_tab:, :]

        if len(set(dbid)) == 1:
            # # repeat for copying
            # colemb = colemb.repeat(nlemb.size(0), 1, 1)
            # expand for single view memory
            colemb = colemb.expand(nlemb.size(0), -1, -1)
            col_len_tensor = col_len_tensor.expand(nlemb.size(0))

            tabemb = tabemb.expand(nlemb.size(0), -1, -1)
            tab_len_tensor = tab_len_tensor.expand(nlemb.size(0))

        # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        # ernn_hps = [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout]
        # ernn_hps = [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, kernel_size]
        kernel_size = 3
        ernn_hps = ["cnn", nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, kernel_size]
        encoder = Word_Encoder(ernn_hps)
        en_out, en_hidden = encoder(en_emb, nl_len_tensor)

        if isinstance(en_hidden, tuple):
            # last hidden vectors if encoder --> initial hidden vectors for decoder
            de_hidden = tuple(hidden[:, nl_recover_ord_tensor, :] for hidden in en_hidden)
            de_hidden = tuple(hidden[:, lb_ord_tensor, :] for hidden in de_hidden)
            if nn_bidirect:
                en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
            else:
                en_hn = en_hidden[0][-1, :, :]
        else:
            de_hidden = en_hidden[:, nl_recover_ord_tensor, :]
            de_hidden = de_hidden[:, lb_ord_tensor, :]
            if nn_bidirect:
                en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
            else:
                en_hn = en_hidden[-1, :, :]

        # de_hidden = en_hidden_re_sort
        en_out = en_out[nl_recover_ord_tensor, :, :]
        en_out = en_out[lb_ord_tensor, :, :]
        # enc_out = en_output_sort
        nl_len_tensor = nl_len_tensor[nl_recover_ord_tensor]
        nl_len_tensor = nl_len_tensor[lb_ord_tensor]
        colemb = colemb[lb_ord_tensor, :, :]
        tabemb = tabemb[lb_ord_tensor, :, :]
        col_len_tensor = col_len_tensor[lb_ord_tensor]
        tab_len_tensor = tab_len_tensor[lb_ord_tensor]

        en_mask = torch.arange(max(nl_len_tensor))[None, :] < nl_len_tensor[:, None]
        col_len_mask = torch.arange(max(col_len_tensor))[None, :] < col_len_tensor[:, None]
        tab_len_mask = torch.arange(max(tab_len_tensor))[None, :] < tab_len_tensor[:, None]
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        emb_hps = [emb_size, emb_dim, emb_pretrained, emb_drop_rate, emb_zero_padding, requires_grad]
        # tembeddings = Emb_layer(emb_hps)
        # lb_emb = tembeddings(ilb_tensor)
        # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        drnn_hps = [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout]

        col_dim = colemb.size(-1)
        tab_dim = tabemb.size(-1)
        num_labels = len(vocab.tw2i)
        decoder = Word_Decoder([emb_hps, drnn_hps], drop_rate=fn_dropout, num_labels=num_labels, enc_att=True,
                               sch_att="en_hidden", col_dim=col_dim, tab_dim=tab_dim)

        if use_pointer:
            col_pointer = Pointer_net(col_dim, decoder.fn_dim)
            tab_pointer = Pointer_net(tab_dim, decoder.fn_dim)

        label_mask = olb_tensor > 0
        if use_pointer:
            # map C(id), T(id) of inputs into column and table placeholders <COL> and <TAB>
            ilb_tensor = ilb_tensor.clone()
            ilb_tensor.apply_(lambda x: vocab.coltab2i_ph.get(x, x))

            de_out, _ = decoder.get_all_hiddens(ilb_tensor, lb_len_tensor, de_hidden, en_out, en_mask, en_hn,
                                                colemb, tabemb, col_len_mask, tab_len_mask)
            de_score = decoder.scoring(de_out)

            colpt_score = col_pointer(colemb, de_out, col_len_mask)
            tabpt_score = tab_pointer(tabemb, de_out, tab_len_mask)

            # TODO double check for cuda
            # mask_col = torch.zeros_like(olb_tensor, dtype=torch.uint8)
            col_ids = list(vocab.col2i_ord.keys())
            mask_col = olb_tensor == col_ids[0]
            for colid in col_ids[1:]:
                mask_col = (olb_tensor == colid) | mask_col
            colord_tensor = olb_tensor[mask_col]
            colord_tensor.apply_(lambda x: vocab.col2i_ord[x])
            colpt_loss = 0
            if colord_tensor.nelement() != 0:
                colpt_loss = col_pointer.NLL_loss(colpt_score[mask_col], colord_tensor)

            # TODO double check for cuda
            # mask_tab = torch.zeros_like(olb_tensor, dtype=torch.uint8)
            tab_ids = list(vocab.tab2i_ord.keys())
            mask_tab = olb_tensor == tab_ids[0]
            for tabid in tab_ids[1:]:
                mask_tab = (olb_tensor == tabid) | mask_tab
            tabord_tensor = olb_tensor[mask_tab]
            tabord_tensor.apply_(lambda x: vocab.tab2i_ord[x])
            tabpt_loss = 0
            if tabord_tensor.nelement() != 0:
                tabpt_loss = tab_pointer.NLL_loss(tabpt_score[mask_tab], tabord_tensor)

            # map C(id), T(id) of OUTPUTs into column and table placeholders <COL> and <TAB>
            olbph_tensor = olb_tensor.clone()
            olbph_tensor.apply_(lambda x: vocab.coltab2i_ph.get(x, x))
            de_loss = decoder.NLL_loss(de_score[label_mask], olbph_tensor[label_mask])
            total_loss = de_loss + colpt_loss + tabpt_loss

            col_label_prob, col_label_pred = col_pointer.inference(colpt_score)
            tab_label_prob, tab_label_pred = tab_pointer.inference(tabpt_score)
            label_prob, label_pred = decoder.inference(de_score, 1)
        else:
            de_out, _ = decoder.get_all_hiddens(ilb_tensor, lb_len_tensor, de_hidden, en_out, en_mask, en_hn,
                                                colemb, tabemb, col_len_mask, tab_len_mask)
            de_score = decoder.scoring(de_out)
            total_loss = decoder.NLL_loss(de_score[label_mask], olb_tensor[label_mask])

        # example of one token/step
        batch_size, target_length = ilb_tensor.size()
        # Extract the first target word (tsos or SOT) to feed to decoder
        j = 0
        ilb1_tensor = Data2tensor.idx2tensor([[vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                             device=device)
        # t_seq_tensor = [batch_size, 1]
        lb1_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=device)
        # t_seq_tensor = [batch_size, 1]
        olb1_tensor = olb_tensor[:, j]
        label1_mask = olb1_tensor > 0

        mask1_col = torch.zeros_like(label1_mask, dtype=torch.bool)
        for idx in vocab.col2i_ord:
            mask1_col = (olb1_tensor == idx) | mask1_col
        mask1_tab = torch.zeros_like(label1_mask, dtype=torch.bool)
        for idx in vocab.tab2i_ord:
            mask1_tab = (olb1_tensor == idx) | mask1_tab
        mask1_noncoltab = label1_mask ^ mask1_col ^ mask1_tab

        de1_out, de_hidden = decoder.get_all_hiddens(ilb1_tensor, lb1_len_tensor, de_hidden, en_out, en_mask, en_hn,
                                                     colemb, tabemb, col_len_mask, tab_len_mask)
        de1_score = decoder.scoring(de1_out)

        de1_loss = decoder.NLL_loss(de1_score[label1_mask], olb1_tensor[label1_mask])

        colord1_tensor = olb1_tensor[mask1_col].apply_(lambda x: vocab.col2i_ord[x])
        if use_pointer:
            colpt1_score = col_pointer.scoring(colemb, de1_out, col_len_mask)
            tabpt1_score = tab_pointer.scoring(tabemb, de1_out, tab_len_mask)
            colpt1_loss = 0
            if colord1_tensor.nelement() != 0:
                colpt1_loss = col_pointer.NLL_loss(colpt1_score[mask1_col], colord1_tensor)

            batch1_loss = colpt1_loss + de1_loss

            col1_label_prob, col1_label_pred = col_pointer.inference(colpt1_score)

        label1_prob, label1_pred = decoder.inference(de1_score, 1)
        ilb1_tensor = label1_pred.squeeze(-1).detach()  # detach from history as input
        predict1_words = Vocab.idx2text(ilb1_tensor.squeeze().tolist(), vocab.i2tw, 1)
    else:
        import settings as args

        nlemb_HPs = [len(vocab.sw2i), args.swd_dim, None,
                     args.wd_dropout, args.wd_padding, args.snl_reqgrad]
        tpemb_HPs, posemb_HPs = None, None
        # Hyper-parameters at hidden-level source language
        if args.swd_inp.startswith("dual_tp"):
            tpemb_HPs = [len(vocab.tp2i), args.swd_dim, None,
                         args.wd_dropout, args.wd_padding, args.stp_reqgrad]
        if args.swd_inp.startswith("dual_pos"):
            posemb_HPs = [len(vocab.pos2i), args.swd_dim, None,
                          args.wd_dropout, args.wd_padding, args.spos_reqgrad]
        if args.swd_inp.startswith("triple"):
            tpemb_HPs = [len(vocab.tp2i), args.swd_dim, None,
                         args.wd_dropout, args.wd_padding, args.stp_reqgrad]
            posemb_HPs = [len(vocab.pos2i), args.swd_dim, None,
                          args.wd_dropout, args.wd_padding, args.spos_reqgrad]

        # NL inputs
        semb_HPs = [nlemb_HPs, tpemb_HPs, posemb_HPs, args.swd_inp, args.swd_mode]
        source_emb = Source_Emb(semb_HPs)

        # encoder
        inp_dim = source_emb.inp_dim
        if args.enc_cnn:
            enc_HPs = ["cnn", inp_dim, args.ed_outdim,
                       args.ed_layers, args.ed_bidirect, args.kernel_size]
        else:
            enc_HPs = [args.ed_mode, inp_dim, args.ed_outdim,
                       args.ed_layers, args.ed_bidirect, args.ed_dropout]
        encoder = Word_Encoder(word_HPs=enc_HPs)

        # decoder
        # Hyper-parameters at word-level target language
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        temb_HPs = [len(vocab.tw2i), args.twd_dim, None,
                    args.wd_dropout, args.wd_padding, args.twd_reqgrad]

        # Hyper-parameters at word-level target language
        dec_HPs = [args.ed_mode, args.twd_dim, args.ed_outdim,
                   args.ed_layers, args.ed_bidirect, args.ed_dropout]

        dec_HPs = [temb_HPs, dec_HPs]

        decoder = Word_Decoder_v2(word_HPs=dec_HPs)

        # encoder attention
        if args.enc_att:
            hidden_dim = dec_HPs[1][2]
            enc_attention = Word_alignment(hidden_dim, hidden_dim)

        # tables
        colemb_HPs = [len(vocab.sw2i), args.swd_dim, None, args.wd_dropout, args.wd_padding, args.snl_reqgrad]
        col_dim = args.swd_dim
        tabemb_HPs = [len(vocab.sw2i), args.swd_dim, None, args.wd_dropout, args.wd_padding, args.snl_reqgrad]
        tab_dim = args.swd_dim

        gnn_HPs = None
        if args.use_graph:
            assert col_dim == tab_dim, print("Column emb and table emb must have the same dimension")
            gnn_HPs = [args.use_graph, args.graph_timesteps, args.graph_edge_types, args.graph_dropout]
        else:
            gnn_HPs = [False, args.graph_timesteps, args.graph_edge_types,
                       args.graph_dropout]

        att_HPs = [args.use_transformer, args.sch_att, args.ed_outdim]

        sch_HPs = [colemb_HPs, tabemb_HPs, gnn_HPs, att_HPs]
        schema_att = Schema_Att(sch_HPs)
        if module_test == 'partial':
            en_inp = source_emb(nl_tensor, tp_tensor, pos_tensor)

            en_out, en_hidden = encoder(en_inp, nl_len_tensor)

            if enc_HPs[0] == 'cnn' and args.ed_mode != "lstm":
                en_hidden = en_hidden[0]
            if isinstance(en_hidden, tuple):
                de_hidden = tuple(hidden[:, nl_recover_ord_tensor, :] for hidden in en_hidden)
                de_hidden = tuple(hidden[:, lb_ord_tensor, :] for hidden in de_hidden)
                if args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
                else:
                    en_hn = en_hidden[0][-1, :, :]
            else:
                de_hidden = en_hidden[:, nl_recover_ord_tensor, :]
                de_hidden = de_hidden[:, lb_ord_tensor, :]
                if args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
                else:
                    en_hn = en_hidden[-1, :, :]

            en_out = en_out[nl_recover_ord_tensor, :, :]
            en_out = en_out[lb_ord_tensor, :, :]
            nl_len_tensor = nl_len_tensor[nl_recover_ord_tensor]
            nl_len_tensor = nl_len_tensor[lb_ord_tensor]

            de_out, de_hidden = decoder(ilb_tensor, lb_len_tensor, de_hidden)

            en_mask = torch.arange(max(nl_len_tensor), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:,
                                                                                                   None]
            enc_context = None
            if args.enc_att:
                # enc_context: [batch, seq_length2, hidden_dim]
                enc_context = enc_attention(en_out, de_out, en_mask)
                # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)

            # if args.use_transformer:
            #     if col_tensor.size(0) == 1:
            #         # # repeat for copying
            #         # col_tensor = col_tensor.repeat(nlemb.size(0), 1)
            #         # # expand for single view memory
            #         col_tensor = col_tensor.expand(nl_tensor.size(0), -1)
            #         col_len_tensor = col_len_tensor.expand(nl_tensor.size(0))
            #     else:
            #         col_tensor = col_tensor[lb_ord_tensor]
            #         col_len_tensor = col_len_tensor[lb_ord_tensor]
            #
            #     if tab_tensor.size(0) == 1:
            #         tab_tensor = tab_tensor.expand(nl_tensor.size(0), -1)
            #         tab_len_tensor = tab_len_tensor.expand(nl_tensor.size(0))
            #     else:
            #         tab_tensor = tab_tensor[lb_ord_tensor]
            #         tab_len_tensor = tab_len_tensor[lb_ord_tensor]
            # else:
            #     if col_tensor.size(0) == 1:
            #         col_tensor = col_tensor.expand(nl_tensor.size(0), -1, -1)
            #         col_len_tensor = col_len_tensor.expand(nl_tensor.size(0))
            #     else:
            #         col_tensor = col_tensor[lb_ord_tensor]
            #         col_len_tensor = col_len_tensor[lb_ord_tensor]
            #
            #     if tab_tensor.size(0) == 1:
            #         tab_tensor = tab_tensor.expand(nl_tensor.size(0), -1, -1)
            #         tab_len_tensor = tab_len_tensor.expand(nl_tensor.size(0))
            #     else:
            #         tab_tensor = tab_tensor[lb_ord_tensor]
            #         tab_len_tensor = tab_len_tensor[lb_ord_tensor]

            colmask = torch.arange(max(col_len_tensor), dtype=torch.long, device=device)[None, :] < col_len_tensor[:, None]
            tabmask = torch.arange(max(tab_len_tensor), dtype=torch.long, device=device)[None, :] < tab_len_tensor[:, None]

            col_context, tab_context = schema_att(col_tensor, tab_tensor, colmask, tabmask, en_hn, de_out, edge_indexes)
        else:
            seq2seq = Seq2seq_v2(semb_HPs, sch_HPs, enc_HPs, dec_HPs, drop_rate=0.5,
                                 num_labels=len(vocab.tw2i), enc_att=args.enc_att)
            random_force = random.random() < 0.5
            de_score, dec_out, de_hidden = seq2seq(nl_tensor, nl_len_tensor, nl_recover_ord_tensor,
                                                   ilb_tensor, lb_len_tensor, lb_ord_tensor, random_force,
                                                   col_tensor, tab_tensor, col_len_tensor, tab_len_tensor, edge_indexes,
                                                   tp_tensor, pos_tensor)

            label_mask = olb_tensor > 0
            total_loss = seq2seq.NLL_loss(de_score[label_mask], olb_tensor[label_mask])
            pass

        pass

