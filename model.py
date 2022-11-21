"""
Written by KrishPro @ KP

filename: `model.py`
"""

from typing import Callable, List
import torch.nn as nn
import numpy as np
import torch

class Embedding(nn.Module):
    """
    This class helps in converting tokens to embeddings (with added positional information and dropout)
    """
    def __init__(self, d_model:int, vocab_size: int, pad_idx:int=0, dropout_p:float=0.1, max_len:float=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.register_buffer('pos_embeddings', self.generate_sinusoids(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, tokens: torch.Tensor):
        embeddings = self.embedding(tokens)
        return self.dropout((embeddings * (self.d_model ** 0.5)) + self.pos_embeddings[:embeddings.size(1)])

    def generate_sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)



def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor = None):
    """
    Q.shape: (B, S, E)
    K.shape: (B, S, E)
    V.shape: (B, S, E)
    
    returns: (B, S, E)
    """

    if attn_mask is None: attn_mask = torch.zeros(Q.size(1), K.size(1))

    energy = torch.nan_to_num( torch.softmax((torch.bmm(Q, K.mT) / (K.size(2) ** 0.5)) + attn_mask, dim=2) )

    out = torch.bmm(energy, V)
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads: int):
        super().__init__()
        assert (d_model % n_heads) == 0, f"d_model ({d_model}) should be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Q: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)
        self.K: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)
        self.V: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)

        self.out: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        x.shape: (B, S, E)
        attn_mask.shape: (B, S, S) or (S, S)

        returns: (B, S, E)
        """
        B, S, E = x.shape
        # Note: E == d_model == n_heads*d_head

        q: torch.Tensor = self.Q(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2).reshape(B*self.n_heads, S, self.d_head)
        k: torch.Tensor = self.K(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2).reshape(B*self.n_heads, S, self.d_head)
        v: torch.Tensor = self.V(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2).reshape(B*self.n_heads, S, self.d_head)

        if (attn_mask is not None) and (attn_mask.dim() == 3): attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)

        out: torch.Tensor = self_attention(q, k, v, attn_mask=attn_mask).reshape(B, self.n_heads, S, self.d_head).transpose(1, 2).reshape(B, S, self.n_heads*self.d_head)

        out: torch.Tensor = self.out(out)
        return out

class LanguageModelLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, dropout_p:float=0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        x.shape: (B, S, E)
        
        returns: (B, S, E)
        """

        x = self.norm1(self.dropout(self.self_attn(x, attn_mask=attn_mask)) + x)

        x = self.norm2(self.dropout(self.feedforward(x)) + x)

        return x

class LanguageModel(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, vocab_size:int, n_layers:int, pad_idx:int=0, max_len:int=512, dropout_p:float=0.1) -> None:
        super().__init__()

        self.pad_idx = pad_idx
        self.embed: Callable[[torch.Tensor], torch.Tensor] = Embedding(d_model, vocab_size, pad_idx=pad_idx, dropout_p=dropout_p)

        self.layers: List[Callable[[torch.Tensor], torch.Tensor]] = nn.ModuleList([LanguageModelLayer(d_model, n_heads, dim_feedforward, dropout_p=dropout_p) for _ in range(n_layers)])

        self.output: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(d_model, vocab_size)

        self.mask = torch.empty(max_len, max_len).fill_(-np.inf).triu_(1)

    def forward(self, tokens: torch.Tensor):
        """
        tokens.shape: (B, S)

        returns: (B, S, V)
        """

        pad_mask = torch.empty(tokens.shape).masked_fill_(tokens == self.pad_idx, -np.inf)
        attn_mask = self.mask[:tokens.size(1), :tokens.size(1)]

        attn_mask = attn_mask.unsqueeze(0) + pad_mask.unsqueeze(1) + pad_mask.unsqueeze(2)

        x: torch.Tensor = self.embed(tokens)

        for layer in self.layers:
            x = layer(x, attn_mask = attn_mask)

        out = self.output(x)

        return out
