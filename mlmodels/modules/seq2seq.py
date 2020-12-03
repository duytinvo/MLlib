# -*- coding: utf-8 -*-
"""
Created on 2019-12-11
@author: duytinvo
"""
import random
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from mlmodels.modules.embeddings import Emb_layer
from mlmodels.modules.pointers import Pointer_net
from mlmodels.modules.encoders import Word_Encoder
from mlmodels.modules.decoders import Word_Decoder, Word_Decoder_v2
from mlmodels.modules.attention import GlobalAttention, Word_alignment, Col_awareness
from mlmodels.modules.beam_search import BeamSearch, GNMTGlobalScorer
from mlmodels.utils.data_utils import Data2tensor, PAD_id, SOT_id, EOT_id, UNK_id, COL_id, TAB_id

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Source_Emb(nn.Module):
    def __init__(self, semb_HPs):
        super(Source_Emb, self).__init__()
        nlemb_HPs, tpemb_HPs, posemb_HPs, swd_inp, swd_mode = semb_HPs
        self.swd_inp = swd_inp
        self.swd_mode = swd_mode
        print("\t\t- Add word embedding module")
        self.sembedding = Emb_layer(nlemb_HPs)
        if swd_inp.startswith("dual_tp"):
            print("\t\t- Add type embedding module")
            self.tpembedding = Emb_layer(tpemb_HPs)

        if self.swd_inp.startswith("dual_pos"):
            print("\t\t- Add POS embedding module")
            self.posembedding = Emb_layer(posemb_HPs)

        if self.swd_inp.startswith("triple"):
            print("\t\t- Add type embedding module")
            self.tpembedding = Emb_layer(tpemb_HPs)
            print("\t\t- Add POS embedding module")
            self.posembedding = Emb_layer(posemb_HPs)

        if self.swd_inp.startswith("triple") and self.swd_mode.startswith("conc"):
            self.inp_dim = nlemb_HPs[1] + tpemb_HPs[1] + posemb_HPs[1]
        elif self.swd_inp.startswith("dual_pos") and self.swd_mode.startswith("conc"):
            self.inp_dim = nlemb_HPs[1] + posemb_HPs[1]
        elif self.swd_inp.startswith("dual_tp") and self.swd_mode.startswith("conc"):
            self.inp_dim = nlemb_HPs[1] + tpemb_HPs[1]
        else:
            self.inp_dim = nlemb_HPs[1]

    def forward(self, nl_tensor, tp_tensor=None, pos_tensor=None, wombat_tensor=None):
        return self.inp_composition(nl_tensor, tp_tensor, pos_tensor, wombat_tensor)

    def inp_composition(self, nl_tensor, tp_tensor=None, pos_tensor=None, wombat_tensor=None):
        # nlemb: (batch, q_len, emb_size)
        # print("\t\tINSIDE SUBMODEL INPUTS: ", nl_tensor.shape, nl_tensor.device)
        nlemb = self.sembedding(nl_tensor)
        if wombat_tensor is not None:
            nlemb += wombat_tensor
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
        # print("\t\tINSIDE SUBMODEL OUTPUTS: ", en_inp.shape, en_inp.device)
        return en_inp


class Schema_Ptr(nn.Module):
    def __init__(self, hd_dim):
        super(Schema_Ptr, self).__init__()
        self.linear_copy = nn.Linear(hd_dim, 3)

    def forward(self, de_score, de_out, col_align, tab_align):
        prob = torch.softmax(de_score, -1)
        # compute probability of tokens being COL, TAB or sql
        p_dis = torch.softmax(self.linear_copy(de_out), -1)
        out_prob = torch.mul(prob, p_dis[:, :, 0].unsqueeze(-1))
        colcopy_prob = torch.mul(col_align, p_dis[:, :, 1].unsqueeze(-1))
        tabcopy_prob = torch.mul(tab_align, p_dis[:, :, 2].unsqueeze(-1))
        return torch.cat((out_prob, colcopy_prob, tabcopy_prob), dim=-1)


class GraphNN(nn.Module):
    def __init__(self, gnn_HPs):
        super(GraphNN, self).__init__()
        use_gnn, num_timesteps, num_edge_types, dropout, col_dim, tab_dim = gnn_HPs
        assert col_dim == tab_dim, print("Column emb and table emb must have the same dimension")
        self.gnn = GatedGraphConv(col_dim, num_timesteps, num_edge_types, dropout)

    def forward(self, colemb, tabemb, graph_tensor):
        graph_data_list = []
        num_tab = tabemb.size(1)
        num_col = colemb.size(1)
        num_nodes = num_tab + num_col
        graphemb = torch.cat([tabemb, colemb], dim=1)
        for gidx in range(graph_tensor.size(0)):
            graph_data = Data(graphemb[gidx])
            for eidx in range(graph_tensor[gidx].size(0)):
                edge_ = graph_tensor[gidx][eidx].transpose(0, 1).contiguous()
                graph_data[f'edge_index_type_{eidx}'] = edge_
            graph_data_list.append(graph_data)
        graph_batch = Batch.from_data_list(graph_data_list)
        gnn_output = self.gnn(graph_batch.x,
                              [graph_batch[f'edge_index_type_{eidx}'] or eidx in range(self.gnn.num_edge_types)])
        gnn_output = gnn_output.view(graph_tensor.size(0), num_nodes, -1)
        tabemb = gnn_output[:, :num_tab, :]
        colemb = gnn_output[:, num_tab:, :]
        return colemb, tabemb


class Schema_Emb(nn.Module):
    def __init__(self, schemb_HPs):
        super(Schema_Emb, self).__init__()
        colemb_HPs, tabemb_HPs, gnn_HPs, use_transformer = schemb_HPs

        self.use_transformer = use_transformer
        self.col_dim = colemb_HPs[1]
        self.tab_dim = tabemb_HPs[1]

        self.colembedding = Emb_layer(colemb_HPs)
        self.tabembedding = Emb_layer(tabemb_HPs)

        self.use_graph = gnn_HPs[0]
        if self.use_graph:
            assert self.col_dim == self.tab_dim, "Column emb and table emb must have the same dimension"
            self.gnn = GraphNN(gnn_HPs)

    def forward(self, col_tensor, tab_tensor, graph_tensor=None):
        if self.use_transformer:
            colemb = self.colembedding(col_tensor)
            tabemb = self.tabembedding(tab_tensor)
        else:
            colemb = self.colembedding(col_tensor).sum(dim=-2)
            tabemb = self.tabembedding(tab_tensor).sum(dim=-2)

        if self.use_graph:
            colemb, tabemb = self.gnn(colemb, tabemb, graph_tensor)

        return colemb, tabemb


class Schema_Att(nn.Module):
    def __init__(self, schatt_HPs):
        super(Schema_Att, self).__init__()
        sch_att, col_dim, tab_dim, hidden_dim = schatt_HPs
        self.col_dim = col_dim
        self.tab_dim = tab_dim
        self.sch_att = sch_att

        assert self.col_dim > 0
        self.col_attention = GlobalAttention(self.col_dim, hidden_dim)

        assert self.tab_dim > 0
        self.tab_attention = GlobalAttention(self.tab_dim, hidden_dim)

    def forward(self, colemb, tabemb, colmask, tabmask, enc_hn=None, dec_out=None):
        if self.sch_att == "en_hidden":
            # enc_hn: [batch, hidden_dim]
            # colemb: [batch, num_col, col_features]
            # col_context: [batch, 1, col_features]
            col_context, col_align = self.col_attention(colemb, enc_hn, colmask)
            # col_context: [batch, seq_length2, col_features]
            col_context = col_context.expand(-1, dec_out.size(1), -1)

            # tab_context: [batch, 1, tab_features]
            tab_context, tab_align = self.tab_attention(tabemb, enc_hn, tabmask)
            # tab_context: [batch, seq_length2, tab_features]
            tab_context = tab_context.expand(-1, dec_out.size(1), -1)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        elif self.sch_att == "de_hidden":
            # print("COL")
            # col_context: [batch, seq_length2, col_dim]
            col_context, col_align = self.col_attention(colemb, dec_out, colmask)
            # print("TAB")
            # tab_context: [batch, seq_length2, tab_dim]
            tab_context, tab_align = self.tab_attention(tabemb, dec_out, tabmask)
            # rnn_out = torch.cat((rnn_out, col_context, tab_context), dim=-1)
        else:
            col_context, col_align = None, None
            tab_context, tab_align = None, None
        return col_context, col_align, tab_context, tab_align


class Seq2seq(nn.Module):
    def __init__(self, semb_HPs, sch_HPs, enc_HPs, dec_HPs,
                 drop_rate=0.5, num_labels=None, enc_att=False):
        super(Seq2seq, self).__init__()
        print("\t- Add NL embedding modules:")
        self.source_emb = Source_Emb(semb_HPs)
        enc_HPs[1] = self.source_emb.inp_dim
        print("\t- Add encoding modules:")
        print("\t\t- Add encoder module")
        self.encoder = Word_Encoder(enc_HPs)
        print("\t- Add decoding modules:")
        self.decoder = Word_Decoder_v2(dec_HPs)

        schemb_HPs, schatt_HPs, use_schptr = sch_HPs
        self.sch_att = schatt_HPs[0]
        if self.sch_att != "none":
            print("\t- Add schema attention modules:")
            print("\t\t- Add col & tab embedding modules")
            self.schema_emb = Schema_Emb(schemb_HPs)
            print("\t\t- Add col & tab attention modules")
            self.schema_att = Schema_Att(schatt_HPs)

        self.enc_cnn = enc_HPs[0]
        self.ed_mode = dec_HPs[1][0]
        self.ed_bidirect = dec_HPs[1][4]

        self.enc_att = enc_att
        hidden_dim = dec_HPs[1][2]
        self.fn_dim = hidden_dim

        if self.enc_att:
            self.enc_attention = GlobalAttention(hidden_dim, hidden_dim)
            self.fn_dim += hidden_dim

        if self.sch_att != "none":
            self.fn_dim += self.schema_att.col_dim + self.schema_att.tab_dim

        self.use_schptr = use_schptr
        if self.use_schptr:
            assert self.sch_att != "none", "Schema attention: {}. Need to use either en_hidden or de_hidden".format(
                self.sch_att)
            print("\t- Add schema pointer modules:")
            print("\t\t- Add col & tab pointer modules")
            self.schema_ptr = Schema_Ptr(self.fn_dim)
            # self.linear_schcopy = mlmodels.Linear(self.fn_dim, 3)

        self.finaldrop_layer = nn.Dropout(drop_rate)

        self.num_labels = num_labels
        if num_labels > 2:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss(reduction='none')
        else:
            self.hidden2tag_layer = nn.Linear(self.fn_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()

    def forward(self, nl_tensor, nl_len_tensor, lb_tensor, teacher_force,
                col_tensor=None, tab_tensor=None, col_len_tensor=None, tab_len_tensor=None, graph_tensor=None,
                tp_tensor=None, pos_tensor=None, colmap_tensor=None, tabmap_tensor=None):
        device = nl_tensor.device
        # print("\n\t- INSIDE MODEL INPUTs: ", nl_tensor.shape, device, "\n")
        # sort lengths of input tensors in the descending mode
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = self.sort_tensors(nl_tensor, nl_len_tensor)
        if tp_tensor is not None:
            tp_tensor = self.reorder_tensor(tp_tensor, nl_ord_tensor, dim=0)
        if pos_tensor is not None:
            pos_tensor = self.reorder_tensor(pos_tensor, nl_ord_tensor, dim=0)
        # en_inp = [batch, nl_len, nl_emb]
        en_inp = self.source_emb(nl_tensor, tp_tensor, pos_tensor)
        # en_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # en_hidden = (h_n,c_n) ---> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
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

        # sort lengths of output tensors in the descending mode
        ilb_tensor = lb_tensor[:, : -1]
        lb_len_tensor = (ilb_tensor > 0).sum(dim=1)
        # olb_tensor = lb_tensor[:, 1:]
        # label_mask = olb_tensor > 0

        ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor = self.sort_tensors(ilb_tensor, lb_len_tensor)
        # reorder lengths of inputs following lengths of outputs
        en_out = self.reorder_tensor(en_out, lb_ord_tensor, dim=0)
        de_hidden = self.reorder_tensor(de_hidden, lb_ord_tensor, dim=1)
        en_hn = self.reorder_tensor(en_hn, lb_ord_tensor, dim=0)
        nl_len_tensor = self.reorder_tensor(nl_len_tensor, lb_ord_tensor, dim=0)
        en_mask = None
        if nl_len_tensor.size(0) > 1:
            en_mask = torch.arange(en_out.size(1), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:, None]
        # reorder lengths of columns and tables following lengths of outputs
        colemb, tabemb, colmask, tabmask = None, None, None, None
        colmask, tabmask = None, None
        if self.sch_att != "none":
            assert col_tensor is not None, "col_tensor should not be None"
            col_tensor = self.reorder_tensor(col_tensor, lb_ord_tensor, dim=0)
            if col_len_tensor is not None:
                col_len_tensor = self.reorder_tensor(col_len_tensor, lb_ord_tensor, dim=0)
            assert tab_tensor is not None, "tab_tensor should not be None"
            tab_tensor = self.reorder_tensor(tab_tensor, lb_ord_tensor, dim=0)
            if tab_len_tensor is not None:
                tab_len_tensor = self.reorder_tensor(tab_len_tensor, lb_ord_tensor, dim=0)
            if graph_tensor is not None:
                graph_tensor = self.reorder_tensor(graph_tensor, lb_ord_tensor, dim=0)

            if col_len_tensor.size(0) > 1:
                colmask = torch.arange(col_tensor.size(1), dtype=torch.long, device=device)[None, :] < col_len_tensor[:, None]

            if tab_len_tensor.size(0) > 1:
                tabmask = torch.arange(tab_tensor.size(1), dtype=torch.long, device=device)[None, :] < tab_len_tensor[:, None]

            colemb, tabemb = self.schema_emb(col_tensor, tab_tensor, graph_tensor)
            if self.use_schptr:
                if colmap_tensor is not None:
                    colmap_tensor = self.reorder_tensor(colmap_tensor, lb_ord_tensor, dim=0)
                if tabmap_tensor is not None:
                    tabmap_tensor = self.reorder_tensor(tabmap_tensor, lb_ord_tensor, dim=0)

        if teacher_force:
            # de_out = [batch, seq_len, hd_dim]
            # de_hidden = (h_n,c_n) ---> h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
            de_out, de_hidden = self.decoder(ilb_tensor, lb_len_tensor, de_hidden)
            enc_context, enc_align = None, None
            if self.enc_att:
                # enc_context: [batch, seq_length2, hidden_dim]
                enc_context, enc_align = self.enc_attention(en_out, de_out, en_mask)
                # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)

            col_context, col_align = None, None
            tab_context, tab_align = None, None
            if self.sch_att != "none":
                col_context, col_align, tab_context, tab_align = self.schema_att(colemb, tabemb, colmask,
                                                                                 tabmask, en_hn, de_out)
            if enc_context is not None:
                de_out = torch.cat((de_out, enc_context), dim=-1)
            if col_context is not None:
                de_out = torch.cat((de_out, col_context), dim=-1)
            if tab_context is not None:
                de_out = torch.cat((de_out, tab_context), dim=-1)
            # de_score = [batch, seq_len, num_labels]
            de_score = self.scoring(de_out)
            if self.use_schptr:
                # de_score = [batch, seq_len, num_labels + num_cols + num_tabs]
                de_score = self.schema_ptr(de_score, de_out, col_align, tab_align)
                if colmap_tensor is not None and tabmap_tensor is not None:
                    de_score = self.collapse_copy_scores(de_score, colmap_tensor, tabmap_tensor, col_tensor.size(1))

        else:
            # first input to the decoder is the <sos> token
            batch_size = ilb_tensor.shape[0]
            max_len = ilb_tensor.shape[1]
            num_outputs = self.num_labels
            if self.use_schptr:
                num_outputs = num_outputs + col_tensor.size(1) + tab_tensor.size(1)
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

                col1_context, col1_align = None, None
                tab1_context, tab1_align = None, None
                if self.sch_att != "none":
                    col1_context, col1_align, tab1_context, tab1_align = self.schema_att(colemb, tabemb, colmask,
                                                                                         tabmask, en_hn, de1_out)
                if enc1_context is not None:
                    de1_out = torch.cat((de1_out, enc1_context), dim=-1)
                if col1_context is not None:
                    de1_out = torch.cat((de1_out, col1_context), dim=-1)
                if tab1_context is not None:
                    de1_out = torch.cat((de1_out, tab1_context), dim=-1)
                # de_score = [batch, 1, num_labels]
                de1_score = self.scoring(de1_out)
                if self.use_schptr:
                    # de_score = [batch, 1, num_labels + num_cols + num_tabs]
                    de1_score = self.schema_ptr(de1_score, de1_out, col1_align, tab1_align)
                    if colmap_tensor is not None and tabmap_tensor is not None:
                        de1_score = self.collapse_copy_scores(de1_score, colmap_tensor, tabmap_tensor, col_tensor.size(1))

                de_score[:, t] = de1_score[:, 0, :]
                # TODO handle out of vocab
                output = de1_score.max(-1)[1].detach().clone()
                if self.use_schptr:
                    # mask OOV to UNK token
                    output[(output >= self.num_labels) & (output < (self.num_labels + col_tensor.size(1)))] = COL_id
                    output[output >= (self.num_labels + col_tensor.size(1))] = TAB_id
        # recover the original order of outputs to compute loss
        de_score = self.reorder_tensor(de_score, lb_recover_ord_tensor, dim=0)
        if self.use_schptr:
            if colmap_tensor is not None:
                colmap_tensor = self.reorder_tensor(colmap_tensor, lb_recover_ord_tensor, dim=0)
            if tabmap_tensor is not None:
                tabmap_tensor = self.reorder_tensor(tabmap_tensor, lb_recover_ord_tensor, dim=0)
        # total_loss = self.NLL_loss(de_score[label_mask], olb_tensor[label_mask])
        # print("\n\t- INSIDE MODEL OUTPUTs: ", total_loss.shape, de_score.shape, olb_tensor.shape, device, "\n")
        return de_score

    def collapse_copy_scores(self, de_score, colmap, tabmap, col_size=0, eps=1e-20):
        # TODO: check this function
        batch, seq_len, vocab_size = de_score.shape
        _cbatch, col_len = colmap.shape
        assert batch == _cbatch or _cbatch == 1, "Invalid batch size of colmap"
        _tbatch, tab_len = tabmap.shape
        assert batch == _tbatch or _tbatch == 1, "Invalid batch size of tabmap"
        for i in range(batch):
            de_score_i = de_score[i]
            if _cbatch == 1:
                colmap_i = colmap[0]
            else:
                colmap_i = colmap[i]
            # extract orders of overlapped columns used in vocab
            col_id = (colmap_i != UNK_id).nonzero()
            # scale up col_id with num_labels to extract real scores
            colblank = col_id.squeeze(-1) + self.num_labels
            # extract ids of used columns in tw2i
            colfill = colmap_i[colmap_i != UNK_id]
            # extract prob values of column indexes and fill to tw indexes
            de_score_i.index_add_(-1, colfill, de_score_i.index_select(-1, colblank))
            # reset all overlapped columns by a tiny number for stability
            de_score_i.index_fill_(-1, colblank, eps)

            if _tbatch == 1:
                tabmap_i = tabmap[0]
            else:
                tabmap_i = tabmap[i]
            # extract orders of overlapped tables used in vocab
            tab_id = (tabmap_i != UNK_id).nonzero()
            # scale up tab_id with num_labels and num_cols to extract real scores
            tabblank = tab_id.squeeze(-1) + self.num_labels + col_size
            # extract ids of used tables in tw2i
            tabfill = tabmap_i[tabmap_i != UNK_id]
            # extract prob values of table indexes and fill to tw indexes
            de_score_i.index_add_(-1, tabfill, de_score_i.index_select(-1, tabblank))
            # reset all overlapped tables by a tiny number for stability
            de_score_i.index_fill_(-1, tabblank, eps)
        return de_score

    def greedy_predict(self, nl_tensor, nl_len_tensor,
                       col_tensor, tab_tensor, col_len_tensor, tab_len_tensor, graph_tensor,
                       tp_tensor=None, pos_tensor=None, maxlen=500, wombat_tensor=None,
                       colmap_tensor=None, tabmap_tensor=None):
        device = nl_len_tensor.device
        # sort lengths of input tensors in the descending mode
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = self.sort_tensors(nl_tensor, nl_len_tensor)
        if wombat_tensor is not None:
            wombat_tensor = self.reorder_tensor(wombat_tensor, nl_ord_tensor, dim=0)
        if tp_tensor is not None:
            tp_tensor = self.reorder_tensor(tp_tensor, nl_ord_tensor, dim=0)
        if pos_tensor is not None:
            pos_tensor = self.reorder_tensor(pos_tensor, nl_ord_tensor, dim=0)

        en_inp = self.source_emb(nl_tensor, tp_tensor, pos_tensor, wombat_tensor)
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

        colemb, tabemb, colmask, tabmask = None, None, None, None
        if self.sch_att != "none":
            # NOTE: When inference, we only allow for single DB. Don't need to reorder col_tensor, tab_tensor
            if col_len_tensor.size(0) > 1:
                colmask = torch.arange(col_tensor.size(1), dtype=torch.long, device=device)[None, :] < col_len_tensor[:,
                                                                                                       None]
            if tab_len_tensor.size(0) > 1:
                tabmask = torch.arange(tab_tensor.size(1), dtype=torch.long, device=device)[None, :] < tab_len_tensor[:,
                                                                                                       None]

            colemb, tabemb = self.schema_emb(col_tensor, tab_tensor, graph_tensor)

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

            col_context, col_align = None, None
            tab_context, tab_align = None, None
            if self.sch_att != "none":
                col_context, col_align, tab_context, tab_align = self.schema_att(colemb, tabemb, colmask, tabmask,
                                                                                 en_hn, de_out)
            if enc_context is not None:
                de_out = torch.cat((de_out, enc_context), dim=-1)
            if col_context is not None:
                de_out = torch.cat((de_out, col_context), dim=-1)
            if tab_context is not None:
                de_out = torch.cat((de_out, tab_context), dim=-1)
            # de_score = [batch, 1, num_labels]
            de_score = self.scoring(de_out)
            if self.use_schptr:
                de_score = self.schema_ptr(de_score, de_out, col_align, tab_align)
                de_score = self.collapse_copy_scores(de_score, colmap_tensor, tabmap_tensor, col_tensor.size(1))
                log_probs = de_score.log()
            else:
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
            if self.use_schptr:
                # mask OOV to UNK token
                output[(output >= self.num_labels) & (output < (self.num_labels + col_tensor.size(1)))] = COL_id
                output[output >= (self.num_labels + col_tensor.size(1))] = TAB_id

        pred_outputs = torch.cat(pred_outputs, dim=-1)
        # acc_prob = torch.cat(acc_prob, dim=-1)
        return pred_outputs, acc_prob.exp()

    def beam_predict(self, nl_tensor, nl_len_tensor,
                     col_tensor, tab_tensor, col_len_tensor, tab_len_tensor, graph_tensor,
                     tp_tensor=None, pos_tensor=None, minlen=1, maxlen=500, bw=2, n_best=2, wombat_tensor=None,
                     colmap_tensor=None, tabmap_tensor=None):
        device = nl_len_tensor.device
        # sort lengths of input tensors in the descending mode
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = self.sort_tensors(nl_tensor, nl_len_tensor)
        # wombat_tensor = [batch, nl_len, emb_dim]
        if wombat_tensor is not None:
            wombat_tensor = self.reorder_tensor(wombat_tensor, nl_ord_tensor, dim=0)
        if tp_tensor is not None:
            tp_tensor = self.reorder_tensor(tp_tensor, nl_ord_tensor, dim=0)
        if pos_tensor is not None:
            pos_tensor = self.reorder_tensor(pos_tensor, nl_ord_tensor, dim=0)

        en_inp = self.source_emb(nl_tensor, tp_tensor, pos_tensor, wombat_tensor)
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

        colemb, tabemb, colmask, tabmask = None, None, None, None
        if self.sch_att != "none":
            # NOTE: When inference, we only allow for single DB. Don't need to reorder col_tensor, tab_tensor
            if col_len_tensor.size(0) >= 1:
                col_tensor = beam_node.fn_map_state(col_tensor, dim=0)
                col_len_tensor = beam_node.fn_map_state(col_len_tensor, dim=0)
            if tab_len_tensor.size(0) >= 1:
                tab_tensor = beam_node.fn_map_state(tab_tensor, dim=0)
                tab_len_tensor = beam_node.fn_map_state(tab_len_tensor, dim=0)

            if len(col_len_tensor) > 1:
                colmask = torch.arange(col_tensor.size(1), dtype=torch.long, device=device)[None, :] < col_len_tensor[:,
                                                                                                       None]
            if len(tab_len_tensor) > 1:
                tabmask = torch.arange(tab_tensor.size(1), dtype=torch.long, device=device)[None, :] < tab_len_tensor[:,
                                                                                                       None]

            colemb, tabemb = self.schema_emb(col_tensor, tab_tensor, graph_tensor)
        # batch_nl_len = max(nl_len_tensor)
        # batch_col_len = max(col_len_tensor)
        # batch_tab_len = max(tab_len_tensor)
        count = 0
        while True:
            count += 1
            # (2) Predict token by token
            decoder_input = beam_node.current_predictions.view(-1, 1).detach().clone()
            if self.use_schptr:
                # mask OOV to UNK token
                decoder_input[(decoder_input >= self.num_labels) &
                              (decoder_input < (self.num_labels + col_tensor.size(1)))] = COL_id
                decoder_input[decoder_input >= (self.num_labels + col_tensor.size(1))] = TAB_id
            de_out, de_hidden = self.decoder(decoder_input, None, de_hidden)
            enc_context, enc_align = None, None
            if self.enc_att:
                # enc_context: [batch, seq_length2, hidden_dim]
                # print("ENCODER:")
                enc_context, enc_align = self.enc_attention(en_out, de_out, en_mask)
                # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
            col_context, col_align = None, None
            tab_context, tab_align = None, None
            if self.sch_att != "none":
                col_context, col_align, tab_context, tab_align = self.schema_att(colemb, tabemb, colmask, tabmask,
                                                                                 en_hn, de_out)
            if enc_context is not None:
                de_out = torch.cat((de_out, enc_context), dim=-1)
            if col_context is not None:
                de_out = torch.cat((de_out, col_context), dim=-1)
            if tab_context is not None:
                de_out = torch.cat((de_out, tab_context), dim=-1)
            de_score = self.scoring(de_out)
            # (3) update current topk
            if self.use_schptr:
                de_score = self.schema_ptr(de_score, de_out, col_align, tab_align)
                de_score = self.collapse_copy_scores(de_score, colmap_tensor, tabmap_tensor, col_tensor.size(1))
                log_probs = de_score.log()
            else:
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

                colemb, tabemb, colmask, tabmask = None, None, None, None
                if self.sch_att != "none":
                    if col_len_tensor.size(0) >= 1:
                        col_tensor = col_tensor.index_select(0, select_indices)
                        col_len_tensor = col_len_tensor.index_select(0, select_indices)
                    if tab_len_tensor.size(0) >= 1:
                        tab_tensor = tab_tensor.index_select(0, select_indices)
                        tab_len_tensor = tab_len_tensor.index_select(0, select_indices)
                    if graph_tensor is not None and graph_tensor.size(0) >= 1:
                        graph_tensor = graph_tensor.index_select(0, select_indices)

                    if col_len_tensor.size(0) > 1:
                        colmask = torch.arange(col_tensor.size(1), dtype=torch.long, device=device)[None,
                                  :] < col_len_tensor[:, None]
                    if tab_len_tensor.size(0) > 1:
                        tabmask = torch.arange(tab_tensor.size(1), dtype=torch.long, device=device)[None,
                                  :] < tab_len_tensor[:, None]

                    colemb, tabemb = self.schema_emb(col_tensor, tab_tensor, graph_tensor)

        return beam_node.predictions, beam_node.scores

    def scoring(self, rnn_out):
        de_score = self.hidden2tag_layer(rnn_out)
        de_score = self.finaldrop_layer(de_score)
        return de_score

    def NLL_loss(self, label_score, label_tensor, col_size=0,
                 colalign_tensor=None, tabalign_tensor=None, eps=1e-20):
        if self.num_labels > 2:
            if self.use_schptr:
                vocab_probs = label_score.gather(1, label_tensor.unsqueeze(1)).squeeze(1)
                colcopy_ix = colalign_tensor.unsqueeze(1) + (self.num_labels - 4)
                colcopy_tok_probs = label_score.gather(1, colcopy_ix).squeeze(1)
                colcopy_tok_probs[colalign_tensor == UNK_id] = 0

                tabcopy_ix = tabalign_tensor.unsqueeze(1) + (self.num_labels + col_size - 4)
                tabcopy_tok_probs = label_score.gather(1, tabcopy_ix).squeeze(1)
                tabcopy_tok_probs[tabalign_tensor == UNK_id] = 0

                copy_tok_probs = colcopy_tok_probs + tabcopy_tok_probs + eps
                non_copy = (colalign_tensor == UNK_id) & (tabalign_tensor == UNK_id)
                # TODO: add unk mask when target2idx allow unk ids
                # non_copy = non_copy | (label_tensor != UNK_id)
                probs = torch.where(non_copy, copy_tok_probs + vocab_probs, copy_tok_probs)
                de_loss = -probs.log()
            else:
                # label_score = [B, C]; label_tensor = [B, ]
                de_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1, ))

        else:
            # label_score = [B, *]; label_tensor = [B, *]
            de_loss = self.lossF(label_score, label_tensor.float().view(-1, 1))
        return de_loss

    def inference(self, label_score, k=1, colmap=None, tabmap=None, col_size=0):
        if self.num_labels > 2:
            if self.use_schptr:
                label_prob = self.collapse_copy_scores(label_score, colmap, tabmap, col_size)
            else:
                label_prob = torch.softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred

    def logsm_inference(self, label_score, k=1, colmap=None, tabmap=None, col_size=0):
        if self.num_labels > 2:
            if self.use_schptr:
                label_prob = self.collapse_copy_scores(label_score, colmap, tabmap, col_size)
                label_prob = label_prob.log()
            else:
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

    @staticmethod
    def sort_tensors(word_tensor, seq_len_tensor):
        seq_len_tensor, seqord_tensor = seq_len_tensor.sort(0, descending=True)
        word_tensor = word_tensor[seqord_tensor]
        _, seqord_recover_tensor = seqord_tensor.sort(0, descending=False)
        return word_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, XLNetTokenizer
    from mlmodels.utils.data_utils import Schema_reader, Jsonfile, Vocab, seqPAD, PAD, SOT, SaveloadHP
    from mlmodels.modules.gated_graph_conv import GatedGraphConv
    device = torch.device("cpu")
    dtype = torch.long
    use_sql = True
    use_transformer = False
    use_pointer = False
    use_graph = False
    use_cuda = False
    db_file = "../../data/benchmark/schema/tables.json"
    filename = "../../data/benchmark/corpus/train_2.json"

    s_paras = [-1,  1]
    t_paras = [-1, 1]
    transformer_mode = "xlnet"
    if transformer_mode == "xlnet":
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased') if use_transformer else None
    else:   # "bert"
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') if use_transformer else None
    schema_reader = Schema_reader(db_file, tokenizer, transformer_mode=transformer_mode)

    vocab = Vocab(s_paras, t_paras)
    vocab.build([filename], schema_reader, use_sql=use_sql)

    nl2ids = vocab.lst2idx(vocab_words=vocab.sw2i, unk_words=True, eos=True)
    tp2ids = vocab.lst2idx(vocab_words=vocab.tp2i, unk_words=True, eos=True)
    pos2ids = vocab.lst2idx(vocab_words=vocab.pos2i, unk_words=True, eos=True)

    col2ids = vocab.hierlst2idx(vocab_words=vocab.sw2i, unk_words=True, eos=False)
    tab2ids = vocab.hierlst2idx(vocab_words=vocab.sw2i, unk_words=True, eos=False)

    col2ids = schema_reader.Trans2idx(vocab.col2i) if use_transformer else col2ids
    tab2ids = schema_reader.Trans2idx(vocab.tab2i) if use_transformer else tab2ids

    tg2ids = vocab.tgt2idx(vocab_words=vocab.tw2i, unk_words=True, sos=True, eos=True,
                           vocab_col=vocab.colname2i, vocab_tab=vocab.tabname2i)

    train_data = Jsonfile(filename, source2idx=nl2ids, tp2id=tp2ids, pos2id=pos2ids, target2idx=tg2ids,
                          use_sql=use_sql, use_pointer=use_pointer)

    data_idx = []
    batch = 8
    for d in vocab.minibatches(train_data, batch):
        data_idx.append(d)
        dbid, nl, tp, pos, target = list(zip(*d))

        nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=vocab.sw2i[PAD], nlevels=1)
        nl_tensor = Data2tensor.idx2tensor(nl_pad_ids, dtype=torch.long, device=device)
        nl_len_tensor = Data2tensor.idx2tensor(nl_lens, dtype=torch.long, device=device)
        # nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=device)
        # nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

        tp_pad_ids, tp_lens = seqPAD.pad_sequences(tp, pad_tok=vocab.tp2i[PAD], nlevels=1)
        tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=device)

        pos_pad_ids, pos_lens = seqPAD.pad_sequences(pos, pad_tok=vocab.pos2i[PAD], nlevels=1)
        pos_tensor = Data2tensor.idx2tensor(pos_pad_ids, dtype=torch.long, device=device)

        assert tp_lens == nl_lens
        col_pos, tab_pos = None, None
        if isinstance(target[0][0], tuple):
            target, col_pos, tab_pos = list(zip(*[zip(*t) for t in target]))

        lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=vocab.tw2i[PAD], nlevels=1)
        lb_tensor = Data2tensor.idx2tensor(lb_pad_ids, dtype=torch.long, device=device)
        colpos_tensor, tabpos_tensor = None, None
        if col_pos is not None:
            colpos_pad_ids, colpos_lens = seqPAD.pad_sequences(col_pos, pad_tok=PAD_id, nlevels=1)
            colpos_tensor = Data2tensor.idx2tensor(colpos_pad_ids, dtype=torch.long, device=device)
        if tab_pos is not None:
            tabpos_pad_ids, tabpos_lens = seqPAD.pad_sequences(tab_pos, pad_tok=PAD_id, nlevels=1)
            tabpos_tensor = Data2tensor.idx2tensor(tabpos_pad_ids, dtype=torch.long, device=device)

        # lb_tensors = Data2tensor.sort_labelled_tensors(lb_pad_ids, lb_lens, label=True, dtype=torch.long,
        #                                                device=device)
        # olb_tensor, ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor, _, _, _ = lb_tensors

        cols = []
        tabs = []
        edge_indexes = []
        colmaps = []
        tabmaps = []
        if len(set(dbid)) == 1:
            col, tab, edge_index = schema_reader.getts(dbid[0], col2idx=col2ids, tab2idx=tab2ids)
            cols = [col]
            tabs = [tab]
            edge_indexes = [edge_index]
            # tbcol_edge, coltb_edge, forpricol_edge, priforcol_edge, forpritab_edge, prifortab_edge = edge_index
            colmaps = [vocab.colmap[dbid[0]]]
            tabmaps = [vocab.tabmap[dbid[0]]]
        else:
            for db in dbid:
                col, tab, edge_index = schema_reader.getts(db, col2idx=col2ids, tab2idx=tab2ids)
                cols.append(col)
                tabs.append(tab)
                edge_indexes.append(edge_index)
                # tbcol_edge, coltb_edge, forpricol_edge, priforcol_edge, forpritab_edge, prifortab_edge = edge_index
                colmaps.append(vocab.colmap[db])
                tabmaps.append(vocab.tabmap[db])
        if use_transformer:
            col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=schema_reader.col2i[PAD], nlevels=1)
            col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=device)
            col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=device)

            if col_tensor.size(0) == 1:
                # # repeat for copying
                # col_tensor = col_tensor.repeat(nl_tensor.size(0), 1)
                # # expand for single view memory
                col_tensor = col_tensor.expand(nl_tensor.size(0), -1)
                col_len_tensor = col_len_tensor.expand(nl_tensor.size(0))

            tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=schema_reader.tab2i[PAD], nlevels=1)
            tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=device)
            tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=device)

            if tab_tensor.size(0) == 1:
                # # repeat for copying
                # tab_tensor = tab_tensor.repeat(nlemb.size(0), 1)
                # # expand for single view memory
                tab_tensor = tab_tensor.expand(nl_tensor.size(0), -1)
                tab_len_tensor = tab_len_tensor.expand(nl_tensor.size(0))
        else:
            col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=vocab.sw2i[PAD], nlevels=2)
            col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=device)
            col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=device)
            if col_tensor.size(0) == 1:
                # # repeat for copying
                # col_tensor = col_tensor.repeat(nl_tensor.size(0), 1, 1)
                # # expand for single view memory
                col_tensor = col_tensor.expand(nl_tensor.size(0), -1, -1)
                col_len_tensor = col_len_tensor.expand(nl_tensor.size(0))

            tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=vocab.sw2i[PAD], nlevels=2)
            tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=device)
            tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=device)

            if tab_tensor.size(0) == 1:
                # # repeat for copying
                # tab_tensor = tab_tensor.repeat(nlemb.size(0), 1, 1)
                # # expand for single view memory
                tab_tensor = tab_tensor.expand(nl_tensor.size(0), -1, -1)
                tab_len_tensor = tab_len_tensor.expand(nl_tensor.size(0))

        if use_graph:
            graph_ids, graph_len = seqPAD.pad_sequences(edge_indexes, pad_tok=vocab.sw2i[PAD], nlevels=3)
            # graph_tensor [batch, num_coltb_rel, num_col, 2]
            graph_tensor = Data2tensor.idx2tensor(graph_ids, dtype=torch.long, device=device)
            if graph_tensor.size(0) == 1:
                # # repeat for copying
                # graph_tensor = graph_tensor.repeat(nlemb.size(0), 1, 1, 1)
                # # expand for single view memory
                graph_tensor = graph_tensor.expand(nl_tensor.size(0), -1, -1, -1)
        else:
            graph_tensor = None

        colmap_pad_ids, _ = seqPAD.pad_sequences(colmaps, pad_tok=UNK_id, nlevels=1)
        colmap_tensor = Data2tensor.idx2tensor(colmap_pad_ids, dtype=torch.long, device=device)
        tabmap_pad_ids, _ = seqPAD.pad_sequences(tabmaps, pad_tok=UNK_id, nlevels=1)
        tabmap_tensor = Data2tensor.idx2tensor(tabmap_pad_ids, dtype=torch.long, device=device)

        # if use_transformer:
        #     col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=vocab.col2i[PAD], nlevels=1)
        #     col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=device)
        #     col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=device)
        #
        #     tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=vocab.tab2i[PAD], nlevels=1)
        #     tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=device)
        #     tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=device)
        #
        # else:
        #     col_pad_ids, col_lens = seqPAD.pad_sequences(cols, pad_tok=vocab.sw2i[PAD], nlevels=2)
        #     col_tensor = Data2tensor.idx2tensor(col_pad_ids, dtype=torch.long, device=device)
        #     col_len_tensor = Data2tensor.idx2tensor(col_lens, dtype=torch.long, device=device)
        #
        #     tab_pad_ids, tab_lens = seqPAD.pad_sequences(tabs, pad_tok=vocab.sw2i[PAD], nlevels=2)
        #     tab_tensor = Data2tensor.idx2tensor(tab_pad_ids, dtype=torch.long, device=device)
        #     tab_len_tensor = Data2tensor.idx2tensor(tab_lens, dtype=torch.long, device=device)
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
    module_test = "partial"

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
        enc_attention = GlobalAttention(hidden_dim, hidden_dim)

    # tables
    colemb_HPs = [len(vocab.sw2i), args.swd_dim, None, args.wd_dropout, args.wd_padding, args.snl_reqgrad]
    col_dim = args.swd_dim
    tabemb_HPs = [len(vocab.sw2i), args.swd_dim, None, args.wd_dropout, args.wd_padding, args.snl_reqgrad]
    tab_dim = args.swd_dim

    gnn_HPs = None
    if args.use_graph:
        assert col_dim == tab_dim, print("Column emb and table emb must have the same dimension")
        gnn_HPs = [args.use_graph, args.graph_timesteps, args.graph_edge_types, args.graph_dropout, col_dim, tab_dim]
    else:
        gnn_HPs = [False, args.graph_timesteps, args.graph_edge_types,
                   args.graph_dropout, col_dim, tab_dim]

    args.sch_att = "de_hidden"
    schemb_HPs = [colemb_HPs, tabemb_HPs, gnn_HPs, args.use_transformer]
    schatt_HPs = [args.sch_att, col_dim, tab_dim, args.ed_outdim]

    sch_pointer = True
    sch_HPs = [schemb_HPs, schatt_HPs, sch_pointer]
    schema_emb = Schema_Emb(schemb_HPs)
    schema_att = Schema_Att(schatt_HPs)

    hidden_dim = dec_HPs[1][2]
    fn_dim = hidden_dim
    if args.enc_att:
        fn_dim += hidden_dim
    if args.sch_att != "none":
        fn_dim += col_dim + tab_dim

    hidden2tag = nn.Linear(fn_dim, len(vocab.tw2i))
    linear_schcopy = nn.Linear(fn_dim, 3)

    schema_ptr = Schema_Ptr(fn_dim)

    if module_test == 'partial':
        # sort lengths of input tensors in the descending mode
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor = Seq2seq.sort_tensors(nl_tensor, nl_len_tensor)
        tp_tensor = tp_tensor[nl_ord_tensor]
        pos_tensor = pos_tensor[nl_ord_tensor]

        en_inp = source_emb(nl_tensor, tp_tensor, pos_tensor)

        en_out, en_hidden = encoder(en_inp, nl_len_tensor)

        ilb_tensor = lb_tensor[:, : -1]
        lb_len_tensor = (ilb_tensor > 0).sum(dim=1)
        olb_tensor = lb_tensor[:, 1:]
        if colpos_tensor is not None:
            ocolpos_tensor = colpos_tensor[:, 1:]
        if tabpos_tensor is not None:
            otabpos_tensor = tabpos_tensor[:, 1:]

        # sort lengths of output tensors in the descending mode
        ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor = Seq2seq.sort_tensors(ilb_tensor,
                                                                                               lb_len_tensor)

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

        col_tensor = col_tensor[lb_ord_tensor]
        col_len_tensor = col_len_tensor[lb_ord_tensor]

        tab_tensor = tab_tensor[lb_ord_tensor]
        tab_len_tensor = tab_len_tensor[lb_ord_tensor]

        colmask = torch.arange(max(col_len_tensor), dtype=torch.long, device=device)[None, :] < col_len_tensor[:, None]
        tabmask = torch.arange(max(tab_len_tensor), dtype=torch.long, device=device)[None, :] < tab_len_tensor[:, None]

        de_out, de_hidden = decoder(ilb_tensor, lb_len_tensor, de_hidden)

        en_mask = torch.arange(max(nl_len_tensor), dtype=torch.long, device=device)[None, :] < nl_len_tensor[:,
                                                                                               None]
        enc_context, enc_align = None, None
        if args.enc_att:
            # enc_context: [batch, seq_length2, hidden_dim]
            enc_context, enc_align = enc_attention(en_out, de_out, en_mask)
            # rnn_out = torch.cat((rnn_out, enc_context), dim=-1)

        colemb, tabemb = schema_emb(col_tensor, tab_tensor, graph_tensor)
        col_context, col_align = None, None
        tab_context, tab_align = None, None
        if args.sch_att:
            col_context, col_align, tab_context, tab_align = schema_att(colemb, tabemb, colmask, tabmask, en_hn, de_out)

        if enc_context is not None:
            de_out = torch.cat((de_out, enc_context), dim=-1)
        if col_context is not None:
            de_out = torch.cat((de_out, col_context), dim=-1)
        if tab_context is not None:
            de_out = torch.cat((de_out, tab_context), dim=-1)
        # de_score = [batch, seq_len, num_labels]
        de_score = hidden2tag(de_out)

        de_score = schema_ptr(de_score, de_out, col_align, tab_align)
    else:
        olb_tensor = lb_tensor[:, 1:]
        label_mask = olb_tensor > 0
        if colpos_tensor is not None:
            colalign_tensor = colpos_tensor[:, 1:]
            colalign_tensor = colalign_tensor[label_mask]
        if tabpos_tensor is not None:
            tabalign_tensor = tabpos_tensor[:, 1:]
            tabalign_tensor = tabalign_tensor[label_mask]
        seq2seq = Seq2seq(semb_HPs, sch_HPs, enc_HPs, dec_HPs, drop_rate=0.5,
                          num_labels=len(vocab.tw2i), enc_att=args.enc_att)
        random_force = random.random() < 1.0
        de_score = seq2seq(
                                                    nl_tensor, nl_len_tensor,
                                                    lb_tensor, random_force,
                                                    col_tensor, tab_tensor, col_len_tensor, tab_len_tensor, graph_tensor,
                                                    tp_tensor, pos_tensor)

        # total_loss = seq2seq.NLL_loss(de_score[label_mask], olb_tensor[label_mask])
        de_score = de_score[label_mask]
        olb_tensor = olb_tensor[label_mask]

        vocab_probs = de_score.gather(1, olb_tensor.unsqueeze(1)).squeeze(1)
        eps = 1e-20
        colcopy_ix = colalign_tensor.unsqueeze(1) + (len(vocab.tw2i) - 4)
        colcopy_tok_probs = de_score.gather(1, colcopy_ix).squeeze(1)
        colcopy_tok_probs[colalign_tensor == UNK_id] = 0

        tabcopy_ix = tabalign_tensor.unsqueeze(1) + (len(vocab.tw2i) + col_tensor.size(1) - 4)
        tabcopy_tok_probs = de_score.gather(1, tabcopy_ix).squeeze(1)
        tabcopy_tok_probs[tabalign_tensor == UNK_id] = 0

        copy_tok_probs = colcopy_tok_probs + tabcopy_tok_probs + eps
        non_copy = (colalign_tensor == UNK_id) & (tabalign_tensor == UNK_id)
        # non_copy = non_copy | (olb_tensor != UNK_id)
        probs = torch.where(non_copy, copy_tok_probs + vocab_probs, copy_tok_probs)
        loss = -probs.log()

        cm = torch.tensor(vocab.colmap["locate"])

        colmap = cm.squeeze(0).expand(batch, -1)
        col_id = (colmap != 3).nonzero()
        colblank = col_id[:, 1] + len(vocab.tw2i)
        colfill = colmap[colmap != 3].nonzero().squeeze(-1)
        de_score.index_add_(-1, colfill, de_score.index_select(-1, colblank))
        de_score.index_fill_(-1, colblank, 1e-10)

        tm = torch.tensor(vocab.tabmap["locate"])

        tabmap = tm.squeeze(0).expand(batch, -1)
        tab_id = (tabmap != 3).nonzero()
        tabblank = tab_id[:, 1] + len(vocab.tw2i) + col_tensor.size(1)
        tabfill = tabmap[tabmap != 3].nonzero().squeeze(-1)
        de_score.index_add_(-1, tabfill, de_score.index_select(-1, tabblank))
        de_score.index_fill_(-1, tabblank, 1e-10)

        minlen = 1
        maxlen = 1000
        bw = 2
        n_best = 2
        pass

    pass
