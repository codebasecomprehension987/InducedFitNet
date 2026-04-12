"""
Ligand atom encoder for InducedFitNet.

Encodes ligand atom features and 3D positions into per-atom embeddings
using a stack of message-passing / self-attention layers with pair
distance biases.

Architecture:
  1. Linear embedding of atom features + sinusoidal time embedding.
  2. N layers of:
       a. Self-attention with RBF distance bias (intra-ligand geometry).
       b. Feed-forward network with GELU activation.
  3. Output head: per-atom score vector (B, N, 3) for the R³ diffusion.
"""

from __future__ import annotations
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from inducedfitnet.models.protein_encoder import SinusoidalTimeEmbedding


class RBFDistanceBias(nn.Module):
    """Gaussian RBF bias from intra-ligand pairwise distances."""

    def __init__(self, n_rbf: int = 16, cutoff: float = 10.0, n_heads: int = 8):
        super().__init__()
        self.n_rbf  = n_rbf
        self.cutoff = cutoff
        centres = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centres", centres)
        self.proj = nn.Linear(n_rbf, n_heads, bias=False)
        self.sigma = cutoff / n_rbf

    def forward(self, pos: Tensor) -> Tensor:
        """
        Args:
            pos: (B, N, 3) atom positions.

        Returns:
            bias: (B, n_heads, N, N)
        """
        dist = torch.cdist(pos, pos)                        # (B, N, N)
        rbf  = torch.exp(
            -(dist.unsqueeze(-1) - self.centres) ** 2 / (2 * self.sigma ** 2)
        )                                                   # (B, N, N, n_rbf)
        bias = self.proj(rbf)                               # (B, N, N, n_heads)
        return rearrange(bias, "b i j h -> b h i j")


class LigandSelfAttentionLayer(nn.Module):
    """
    Single multi-head self-attention layer with RBF distance bias.

    Args:
        d_model:  Feature dimension.
        n_heads:  Attention heads.
        n_rbf:    RBF basis functions.
        dropout:  Dropout probability.
    """

    def __init__(
        self,
        d_model:  int = 128,
        n_heads:  int = 4,
        n_rbf:    int = 16,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head   = d_model // n_heads
        self.n_heads  = n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)

        self.rbf_bias = RBFDistanceBias(n_rbf, n_heads=n_heads)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)

    def forward(
        self,
        h:    Tensor,                     # (B, N, d_model)
        pos:  Tensor,                     # (B, N, 3)
        mask: Optional[Tensor] = None,    # (B, N) bool
    ) -> Tensor:
        B, N, _ = h.shape
        H, dh   = self.n_heads, self.d_head

        Q = rearrange(self.W_Q(h), "b n (h d) -> b h n d", h=H)
        K = rearrange(self.W_K(h), "b n (h d) -> b h n d", h=H)
        V = rearrange(self.W_V(h), "b n (h d) -> b h n d", h=H)

        logits = torch.einsum("bhid,bhjd->bhij", Q, K) / math.sqrt(dh)
        logits = logits + self.rbf_bias(pos)

        if mask is not None:
            inv = ~mask   # (B, N)
            logits = logits.masked_fill(inv.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out  = rearrange(torch.einsum("bhij,bhjd->bhid", attn, V), "b h n d -> b n (h d)")
        out  = self.W_O(out)

        if mask is not None:
            out = out * mask.float().unsqueeze(-1)

        return self.norm(h + out)


class LigandEncoder(nn.Module):
    """
    Stacked self-attention encoder for ligand atoms.

    Args:
        d_lig_in: Input ligand feature dimension (from featurizer).
        d_model:  Internal feature dimension.
        n_layers: Number of self-attention layers.
        n_heads:  Attention heads per layer.
        d_time:   Sinusoidal time-embedding dimension.
        dropout:  Dropout probability.
    """

    def __init__(
        self,
        d_lig_in: int = 17,
        d_model:  int = 128,
        n_layers: int = 3,
        n_heads:  int = 4,
        d_time:   int = 256,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_lig_in + d_time, d_model)
        self.time_embed = SinusoidalTimeEmbedding(d_time)

        self.layers = nn.ModuleList([
            LigandSelfAttentionLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(n_layers)
        ])

        # Score output head: predicts per-atom displacement direction
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(
        self,
        lig_feat: Tensor,             # (B, N, d_lig_in)
        x_lig:    Tensor,             # (B, N, 3)  current ligand coords
        timestep: Tensor,             # (B,)
        mask:     Optional[Tensor] = None,  # (B, N) bool
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            features:  (B, N, d_model)  per-atom embeddings
            score_x:   (B, N, 3)        ligand coordinate score
        """
        B, N, _ = lig_feat.shape
        t_emb   = self.time_embed(timestep).unsqueeze(1).expand(B, N, -1)
        h       = self.input_proj(torch.cat([lig_feat, t_emb], dim=-1))

        for layer, ffn in zip(self.layers, self.ffns):
            h = layer(h, x_lig, mask)
            h = h + ffn(h)

        score_x = self.score_head(h)   # (B, N, 3)
        if mask is not None:
            score_x = score_x * mask.float().unsqueeze(-1)

        return h, score_x
