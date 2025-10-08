import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.QProj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.KProj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.VProj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.OProj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, mask=None):
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        Q = (
            self.QProj(query)
            .view(batch_size, tgt_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.KProj(key)
            .view(batch_size, src_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.VProj(value)
            .view(batch_size, src_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.bool()
            if mask.dim() == 2:
                scores = scores.masked_fill(
                    mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
            elif mask.dim() == 3:
                scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
            else:
                raise ValueError("mask must be 2 or 3 dims")

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        out = self.OProj(out)
        return out, attn


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1, device=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=device).float()
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# -----------------------------
# Feed-Forward Network
# -----------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(self.dropout(out))
        return out


# -----------------------------
# Encoder Layer
# -----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedAttention(embed_dim, num_heads, dropout)
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# -----------------------------
# Decoder Layer
# -----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadedAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadedAttention(embed_dim, num_heads, dropout)
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


# -----------------------------
# Full Transformer
# -----------------------------
class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        max_len=5000,
        dropout=0.1,
        pad_idx=0,
        device=None,
    ):
        super().__init__()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.src_tok_emb = nn.Embedding(
            src_vocab_size, embed_dim, padding_idx=pad_idx
        ).to(self.device)
        self.tgt_tok_emb = nn.Embedding(
            tgt_vocab_size, embed_dim, padding_idx=pad_idx
        ).to(self.device)
        self.positional = PositionalEncoding(
            embed_dim, max_len, dropout, device=self.device
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, ff_dim, dropout).to(self.device)
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, ff_dim, dropout).to(self.device)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim).to(self.device)
        self.output_proj = nn.Linear(embed_dim, tgt_vocab_size).to(self.device)

    def encode(self, src, src_mask=None):
        src = src.to(self.device)
        x = self.src_tok_emb(src)
        x = self.positional(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.norm(x)

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = tgt.to(self.device)
        memory = memory.to(self.device)
        x = self.tgt_tok_emb(tgt)
        x = self.positional(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        dec = self.decode(tgt, memory, tgt_mask, memory_mask)
        logits = self.output_proj(dec)
        return logits
