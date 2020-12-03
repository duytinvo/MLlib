# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import torch
import numpy as np
import torch.nn as nn


def multigpu_feed(inp_tensor, device):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        inp_tensor = nn.DataParallel(inp_tensor)
    return inp_tensor.to(device)


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
