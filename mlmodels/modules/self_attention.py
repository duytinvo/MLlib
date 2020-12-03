# -*- coding: utf-8 -*-
"""
Created on 2020-03-03
@author: duytinvo
"""
import torch
import torch.nn as nn
import numpy as np
import math
import logging
import inspect
from enum import IntEnum

logger = logging.getLogger("tensor_shapes")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# if you want the model to continuously print tensor shapes, set to DEBUG!
logger.setLevel(1)


def getclass():
    stack = inspect.stack()
    return stack[3][0].f_locals["self"].__class__


# A helper function to check how tensor sizes change
def log_size(tsr: torch.Tensor, name: str):
    cls = getclass()
    logger.log(level=cls.level, msg=f"[{cls.__name__}] {name} size={tsr.shape}")


# Control how much debugging output we want
class TensorLoggingLevels(IntEnum):
    attention = 1
    attention_head = 2
    multihead_attention_block = 3
    enc_dec_block = 4
    enc_dec = 5


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class PositionalEmbedding(nn.Module):
    level = 1

    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        #         return self.weight[:, :x.size(1), :] # (1, Seq, Feature)
        return self.weight[:, :x.size(1), :].expand(x.size(0), -1, -1)  # (Batch, Seq, Feature)


class WordPositionEmbedding(nn.Module):
    level = 1

    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x: torch.LongTensor, mask=None) -> torch.FloatTensor:
        return self.word_embedding(x) + self.position_embedding(x)


class ScaledDotProductAttention(nn.Module):
    level = TensorLoggingLevels.attention  # Logging level:

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (Batch, Seq, Feature)
        d_k = k.size(-1)  # get the size of the key
        assert q.size(-1) == d_k

        # compute the dot product between queries and keys for
        # each batch and position in the sequence
        # (Batch, Seq, Feature) x (Batch, Feature, Seq) --> (Batch, Seq, Seq)
        attn = torch.bmm(q, k.transpose(Dim.seq, Dim.feature))  # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence
        # for each batch

        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attn = attn / math.sqrt(d_k)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        attn = torch.exp(attn)
        log_size(attn, "attention weight")  # (Batch, Seq, Seq)

        # fill attention weights with 0s where padded
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.dropout(attn)
        # (Batch, Seq, Seq) x (Batch, Seq, Feature) --> (Batch, Seq, Feature)
        output = torch.bmm(attn, v)  # (Batch, Seq, Feature)
        log_size(output, "attention output size")  # (Batch, Seq, Feature)
        return output


class AttentionHead(nn.Module):
    level = TensorLoggingLevels.attention_head
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        log_size(Q, "queries, keys, vals")
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x


class MultiHeadAttention(nn.Module):
    level = TensorLoggingLevels.multihead_attention_block

    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        # in practice, d_model == d_feature * n_heads
        assert d_model == d_feature * n_heads

        # Note that this is very inefficient:
        # I am merely implementing the heads separately because it is
        # easier to understand this way
        self.attn_heads = nn.ModuleList([AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(d_feature * n_heads, d_model)

    def forward(self, queries, keys, values, mask=None):
        log_size(queries, "Input queries")
        x = [attn(queries, keys, values, mask=mask)  # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]
        log_size(x[0], "output of single head")

        # reconcatenate
        x = torch.cat(x, dim=Dim.feature)  # (Batch, Seq, D_Feature * n_heads)
        log_size(x, "concatenated output")
        x = self.projection(x)  # (Batch, Seq, D_Model)
        log_size(x, "projected output")
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderBlock(nn.Module):
    level = TensorLoggingLevels.enc_dec_block

    def __init__(self, d_model=512, d_feature=64, d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        log_size(x, "Encoder block input")
        att = self.attn_head(x, x, x, mask=mask)
        log_size(x, "Attention output")
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm1(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        log_size(x, "Feedforward output")
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm2(pos))
        log_size(x, "Encoder size output")
        return x


class TransformerEncoder(nn.Module):
    level = TensorLoggingLevels.enc_dec

    def __init__(self, n_blocks=6, d_model=512,
                 n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.FloatTensor, mask=None):
        for encoder in self.encoders:
            x = encoder(x)
        return x


class DecoderBlock(nn.Module):
    level = TensorLoggingLevels.enc_dec_block

    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out,
                src_mask=None, tgt_mask=None):
        # Apply attention to inputs
        att = self.masked_attn_head(x, x, x, mask=src_mask)
        x = x + self.dropout(self.layer_norm1(att))
        # Apply attention to the encoder outputs and outputs of the previous layer
        att = self.attn_head(queries=x, keys=enc_out, values=enc_out, mask=tgt_mask)
        x = x + self.dropout(self.layer_norm2(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm2(pos))
        return x


class TransformerDecoder(nn.Module):
    level = TensorLoggingLevels.enc_dec

    def __init__(self, n_blocks=6, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.position_embedding = PositionalEmbedding(d_model)
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.FloatTensor,
                enc_out: torch.FloatTensor,
                src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


if __name__ == '__main__':
    posemb = PositionalEmbedding(512)
    x = torch.randint(1000, (5, 30)).to(dtype=torch.long)
    posemb(x).shape

    emb = WordPositionEmbedding(1000)
    x = torch.randint(1000, (5, 30)).to(dtype=torch.long)
    emb(x).shape

    # Double Checking
    attn = ScaledDotProductAttention()
    q = torch.rand(5, 10, 20)
    k = torch.rand(5, 10, 20)
    v = torch.rand(5, 10, 20)
    attn(q, k, v)

    # Double Checking
    attn_head = AttentionHead(20, 20)
    attn_head(q, k, v)

    # We'll supress logging from the scaled dot product attention now
    logger.setLevel(TensorLoggingLevels.attention_head)
    heads = MultiHeadAttention(20 * 8, 20, 8)
    heads(q.repeat(1, 1, 8),
          k.repeat(1, 1, 8),
          v.repeat(1, 1, 8))

    # We'll supress logging from the individual attention heads
    logger.setLevel(TensorLoggingLevels.multihead_attention_block)
    enc = EncoderBlock()
    enc(torch.rand(5, 10, 512))

    emb = WordPositionEmbedding(1000)
    encoder = TransformerEncoder()
    enc_out = encoder(emb(torch.randint(1000, (5, 30)).to(dtype=torch.long)))
    enc_out.shape

    dec = DecoderBlock()
    dec(torch.rand(5, 10, 512), enc(torch.rand(5, 10, 512)))
    decoder = TransformerDecoder()

    # We'll supress logging from the scaled dot product attention now
    logger.setLevel(TensorLoggingLevels.enc_dec_block)
    emb = WordPositionEmbedding(1000)
    encoder = TransformerEncoder()
    decoder = TransformerDecoder()

    src_ids = torch.randint(1000, (5, 30)).to(dtype=torch.long)
    tgt_ids = torch.randint(1000, (5, 30)).to(dtype=torch.long)

    x = encoder(emb(src_ids))
    decoder(emb(tgt_ids), x)