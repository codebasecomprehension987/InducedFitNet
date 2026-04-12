"""
Cross-attention between ligand atom features and protein residue frames.

At each diffusion step, the protein backbone score network must condition
on the current ligand position, and the ligand score network must
condition on the protein frames. This module implements both directions
via a shared multi-head cross-attention mechanism.

Architecture
------------
LigandToProtein cross-attention:
    Q = protein residue features  (B, L, d_model)
    K = ligand atom features      (B, N, d_model)
    V = ligand atom features      (B, N, d_model)

ProteinToLigand cross-attention:
    Q = ligand atom features      (B, N, d_model)
    K = protein residue features  (B, L, d_model)
    V = protein residue features  (B, L, d_model)

Distance biases are computed from the current 3D coordinates and added
to the attention logits before softmax.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class StructureBiasedCrossAttention(nn.Module):
    """
    Multi-head cross-attention with 3D distance bias.

    For each query-key pair (q_i, k_j) the scalar distance
    d(q_i, k_j) is mapped through a Gaussian radial basis and added
    to the attention logit:

        A_{ij} = (Q_i · K_j) / sqrt(d_k)  +  RBF(d_{ij}) @ w_bias

    Args:
        d_model:    Feature dimension for queries and keys/values.
        n_heads:    Number of attention heads.
        n_rbf:      Number of radial basis functions.
        rbf_cutoff: Distance cutoff (Å) for RBF.
        dropout:    Attention dropout probability.
    """

    def __init__(
        self,
        d_model:    int = 256,
        n_heads:    int = 8,
        n_rbf:      int = 16,
        rbf_cutoff: float = 20.0,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_k        = d_model // n_heads
        self.rbf_cutoff = rbf_cutoff
        self.n_rbf      = n_rbf

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)

        # RBF distance bias → scalar per head
        self.rbf_proj = nn.Linear(n_rbf, n_heads, bias=False)

        # Gaussian centres (fixed, not learned)
        centres = torch.linspace(0, rbf_cutoff, n_rbf)
        self.register_buffer("rbf_centres", centres)  # (n_rbf,)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for lin in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(lin.weight)

    # ------------------------------------------------------------------
    # RBF
    # ------------------------------------------------------------------

    def _rbf(self, dist: Tensor) -> Tensor:
        """
        Gaussian RBF encoding of pairwise distances.

        Args:
            dist: (B, Q, K)  pairwise distances in Å.

        Returns:
            rbf:  (B, Q, K, n_rbf)
        """
        sigma = self.rbf_cutoff / self.n_rbf
        centres = self.rbf_centres.view(1, 1, 1, -1)          # (1,1,1,n_rbf)
        diff    = dist.unsqueeze(-1) - centres                 # (B,Q,K,n_rbf)
        return torch.exp(-(diff ** 2) / (2 * sigma ** 2))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query_feat:  Tensor,          # (B, Q, d_model)
        key_feat:    Tensor,          # (B, K, d_model)
        query_pos:   Tensor,          # (B, Q, 3)
        key_pos:     Tensor,          # (B, K, 3)
        query_mask:  Optional[Tensor] = None,  # (B, Q) bool
        key_mask:    Optional[Tensor] = None,  # (B, K) bool
    ) -> Tensor:
        """
        Args:
            query_feat, key_feat: feature tensors.
            query_pos, key_pos:   3D coordinates used for distance bias.
            query_mask, key_mask: True = valid position.

        Returns:
            out: (B, Q, d_model)  updated query features.
        """
        B, Q, _ = query_feat.shape
        _, K, _ = key_feat.shape

        # Projections
        Qp = rearrange(self.W_Q(query_feat), "b q (h d) -> b h q d", h=self.n_heads)
        Kp = rearrange(self.W_K(key_feat),   "b k (h d) -> b h k d", h=self.n_heads)
        Vp = rearrange(self.W_V(key_feat),   "b k (h d) -> b h k d", h=self.n_heads)

        # Scaled dot-product logits
        scale  = math.sqrt(self.d_k)
        logits = torch.einsum("bhqd,bhkd->bhqk", Qp, Kp) / scale   # (B,H,Q,K)

        # Distance bias
        dist   = torch.cdist(query_pos, key_pos)                    # (B, Q, K)
        rbf    = self._rbf(dist)                                     # (B, Q, K, n_rbf)
        bias   = self.rbf_proj(rbf)                                  # (B, Q, K, H)
        bias   = rearrange(bias, "b q k h -> b h q k")
        logits = logits + bias

        # Key mask: set logit to -inf for padded keys
        if key_mask is not None:
            invalid = ~key_mask  # (B, K)
            logits  = logits.masked_fill(invalid.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # Query mask: zero out padded queries after aggregation
        out = torch.einsum("bhqk,bhkd->bhqd", attn, Vp)             # (B,H,Q,d)
        out = rearrange(out, "b h q d -> b q (h d)")                 # (B,Q,d_model)
        out = self.W_O(out)

        if query_mask is not None:
            out = out * query_mask.float().unsqueeze(-1)

        return self.norm(query_feat + out)


class BidirectionalCrossAttention(nn.Module):
    """
    Performs ligand→protein AND protein→ligand cross-attention in one call.

    This is applied at every diffusion step so that:
      - protein residue features are updated with ligand information
      - ligand atom features are updated with protein frame information

    Args:
        d_model:    Shared feature dimension.
        n_heads:    Attention heads.
        n_rbf:      Radial basis functions for distance bias.
        rbf_cutoff: Cutoff radius (Å).
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        d_model:    int   = 256,
        n_heads:    int   = 8,
        n_rbf:      int   = 16,
        rbf_cutoff: float = 20.0,
        dropout:    float = 0.1,
    ):
        super().__init__()
        kwargs = dict(
            d_model    = d_model,
            n_heads    = n_heads,
            n_rbf      = n_rbf,
            rbf_cutoff = rbf_cutoff,
            dropout    = dropout,
        )
        self.lig_to_prot = StructureBiasedCrossAttention(**kwargs)
        self.prot_to_lig = StructureBiasedCrossAttention(**kwargs)

    def forward(
        self,
        prot_feat:  Tensor,           # (B, L, d_model)
        lig_feat:   Tensor,           # (B, N, d_model)
        prot_pos:   Tensor,           # (B, L, 3) Cα positions
        lig_pos:    Tensor,           # (B, N, 3) ligand atom positions
        prot_mask:  Optional[Tensor] = None,
        lig_mask:   Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            prot_feat_new: (B, L, d_model)
            lig_feat_new:  (B, N, d_model)
        """
        prot_feat_new = self.lig_to_prot(
            query_feat  = prot_feat,
            key_feat    = lig_feat,
            query_pos   = prot_pos,
            key_pos     = lig_pos,
            query_mask  = prot_mask,
            key_mask    = lig_mask,
        )
        lig_feat_new = self.prot_to_lig(
            query_feat  = lig_feat,
            key_feat    = prot_feat,
            query_pos   = lig_pos,
            key_pos     = prot_pos,
            query_mask  = lig_mask,
            key_mask    = prot_mask,
        )
        return prot_feat_new, lig_feat_new
