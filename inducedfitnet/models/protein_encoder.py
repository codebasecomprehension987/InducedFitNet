"""
Protein backbone encoder.

Encodes residue features into per-residue embeddings that are
SE(3)-equivariant with respect to the current backbone frames.

Architecture:
  1. Linear embedding of residue features + sinusoidal time embedding.
  2. Invariant Point Attention (IPA) blocks that operate on
     backbone rigid frames (R_i, t_i).
  3. Output: per-residue feature vectors (B, L, d_model) and
     predicted rotation/translation score vectors.

References:
  - Jumper et al. AlphaFold2 (2021) — IPA
  - Yim et al. FrameDiff (2023)
"""

from __future__ import annotations
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from inducedfitnet.utils.geometry import apply_se3, invert_se3


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar timestep t → d_embed sinusoidal features."""

    def __init__(self, d_embed: int = 256):
        super().__init__()
        assert d_embed % 2 == 0
        self.d_embed = d_embed
        half = d_embed // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        self.register_buffer("freqs", freqs)  # (half,)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) integer or float timesteps.

        Returns:
            emb: (B, d_embed)
        """
        t_f = t.float().unsqueeze(-1)           # (B, 1)
        args = t_f * self.freqs.unsqueeze(0)    # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class InvariantPointAttention(nn.Module):
    """
    Simplified IPA block.

    Computes invariant pair features from backbone frames and uses them
    to update residue embeddings.

    Args:
        d_model:    Residue feature dimension.
        n_heads:    Attention heads.
        n_query_points: Number of query points per head.
        n_key_points:   Number of key points per head.
    """

    def __init__(
        self,
        d_model:         int = 256,
        n_heads:         int = 8,
        n_query_points:  int = 4,
        n_key_points:    int = 4,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.nqp       = n_query_points
        self.nkp       = n_key_points

        # Standard attention projections
        self.W_Q  = nn.Linear(d_model, d_model, bias=False)
        self.W_K  = nn.Linear(d_model, d_model, bias=False)
        self.W_V  = nn.Linear(d_model, d_model, bias=False)

        # Point projections (produce 3D query/key points in local frame)
        self.W_Qp = nn.Linear(d_model, n_heads * n_query_points * 3, bias=False)
        self.W_Kp = nn.Linear(d_model, n_heads * n_key_points * 3, bias=False)
        self.W_Vp = nn.Linear(d_model, n_heads * n_key_points * 3, bias=False)

        # Output
        out_dim = n_heads * self.d_head + n_heads * n_key_points * 3
        self.W_O = nn.Linear(out_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Learned weight for point attention vs scalar attention
        self.w_C = nn.Parameter(torch.ones(n_heads) * math.log(20))

    def forward(
        self,
        s:        Tensor,           # (B, L, d_model) residue features
        R:        Tensor,           # (B, L, 3, 3) backbone rotations
        tr:       Tensor,           # (B, L, 3) Cα translations
        mask:     Optional[Tensor] = None,  # (B, L) bool
    ) -> Tensor:
        """Returns updated residue features (B, L, d_model)."""
        B, L, _ = s.shape
        H, dh    = self.n_heads, self.d_head

        # --- Scalar attention ---
        Q  = rearrange(self.W_Q(s), "b l (h d) -> b h l d", h=H)
        K  = rearrange(self.W_K(s), "b l (h d) -> b h l d", h=H)
        V  = rearrange(self.W_V(s), "b l (h d) -> b h l d", h=H)
        a_s = torch.einsum("bhid,bhjd->bhij", Q, K) / math.sqrt(dh)  # (B,H,L,L)

        # --- Point attention ---
        # Project to local-frame 3D points, transform to global
        Qp = self.W_Qp(s).reshape(B, L, H, self.nqp, 3)   # (B,L,H,nqp,3)
        Kp = self.W_Kp(s).reshape(B, L, H, self.nkp, 3)
        Vp = self.W_Vp(s).reshape(B, L, H, self.nkp, 3)

        # Transform points to global frame: x_global = R @ x_local + tr
        def local_to_global(pts, R, tr):
            # pts: (B,L,H,P,3)  R:(B,L,3,3)  tr:(B,L,3)
            R_exp  = R.unsqueeze(2).unsqueeze(3)           # (B,L,1,1,3,3)
            tr_exp = tr.unsqueeze(2).unsqueeze(3)          # (B,L,1,1,3)
            return (R_exp @ pts.unsqueeze(-1)).squeeze(-1) + tr_exp

        Qp_g = local_to_global(Qp, R, tr)   # (B,L,H,nqp,3)
        Kp_g = local_to_global(Kp, R, tr)
        Vp_g = local_to_global(Vp, R, tr)

        # Squared distances between query point i and key point j
        # Qp_g: (B,L_q,H,nqp,3) → need (B,H,L_q,L_k) after summing over points
        Qp_flat = rearrange(Qp_g, "b l h p c -> b h l (p c)")  # (B,H,L,nqp*3)
        Kp_flat = rearrange(Kp_g, "b l h p c -> b h l (p c)")
        # sum of sq distances over query/key points
        sq_dist = (
            Qp_flat.unsqueeze(3) - Kp_flat.unsqueeze(2)
        ).pow(2).reshape(B, H, L, L, self.nqp, 3).sum(dim=(-2, -1))  # (B,H,L,L) — expensive but correct

        w_C = F.softplus(self.w_C).view(1, H, 1, 1)
        a_p = -0.5 * w_C * sq_dist

        # Combined attention
        attn = F.softmax(a_s + a_p, dim=-1)   # (B,H,L,L)

        if mask is not None:
            attn = attn * mask.unsqueeze(1).unsqueeze(2).float()

        # Aggregate scalar values
        out_s  = rearrange(
            torch.einsum("bhij,bhjd->bhid", attn, V), "b h l d -> b l (h d)"
        )

        # Aggregate point values (transform back to local frame)
        Vp_agg = torch.einsum("bhij,bjhpc->bihpc", attn, Vp_g)  # (B,L,H,nkp,3)
        # Transform aggregate points to local frame of query residue
        R_inv, tr_inv = invert_se3(R, tr)
        def global_to_local(pts, R_inv, tr_inv):
            R_exp  = R_inv.unsqueeze(2).unsqueeze(3)
            tr_exp = tr_inv.unsqueeze(2).unsqueeze(3)
            return (R_exp @ (pts - tr_exp).unsqueeze(-1)).squeeze(-1)
        Vp_local = global_to_local(Vp_agg, R_inv, tr_inv)  # (B,L,H,nkp,3)
        out_p = rearrange(Vp_local, "b l h p c -> b l (h p c)")

        out = self.W_O(torch.cat([out_s, out_p], dim=-1))
        return self.norm(s + out)


class ProteinBackboneEncoder(nn.Module):
    """
    Full protein backbone encoder with stacked IPA blocks.

    Args:
        d_res_in:   Input residue feature dimension.
        d_model:    Internal feature dimension.
        n_ipa:      Number of IPA blocks.
        n_heads:    Attention heads per IPA block.
        d_time:     Time-embedding dimension.
    """

    def __init__(
        self,
        d_res_in: int = 27,
        d_model:  int = 256,
        n_ipa:    int = 4,
        n_heads:  int = 8,
        d_time:   int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_res_in + d_time, d_model)
        self.time_embed = SinusoidalTimeEmbedding(d_time)

        self.ipa_blocks = nn.ModuleList([
            InvariantPointAttention(d_model, n_heads)
            for _ in range(n_ipa)
        ])
        self.ffn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_ipa)
        ])

        # Output heads: predict rotation score (3) + translation score (3) per residue
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6),   # 3 rotation + 3 translation
        )

    def forward(
        self,
        res_feat:  Tensor,           # (B, L, d_res_in)
        R:         Tensor,           # (B, L, 3, 3)
        tr:        Tensor,           # (B, L, 3)
        timestep:  Tensor,           # (B,)
        mask:      Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            features:   (B, L, d_model)  updated residue embeddings
            score_R:    (B, L, 3)        rotation score (axis-angle)
            score_tr:   (B, L, 3)        translation score
        """
        B, L, _ = res_feat.shape
        t_emb   = self.time_embed(timestep).unsqueeze(1).expand(B, L, -1)  # (B,L,d_time)
        h       = self.input_proj(torch.cat([res_feat, t_emb], dim=-1))

        for ipa, ffn in zip(self.ipa_blocks, self.ffn_blocks):
            h = ipa(h, R, tr, mask)
            h = h + ffn(h)

        scores = self.score_head(h)                        # (B, L, 6)
        score_R  = scores[..., :3]
        score_tr = scores[..., 3:]
        return h, score_R, score_tr
