"""
JointScoreNetwork — core denoising model for InducedFitNet.

Jointly denoises:
  - Protein backbone frames  T_i = (R_i, t_i) ∈ SE(3)
  - Ligand atom positions    x_j ∈ R³

At every diffusion step, the protein backbone score network conditions
on the current ligand position, and the ligand score network conditions
on the current protein frames — mediated by a BidirectionalCrossAttention
block.

The full forward pass:
  1. Encode protein residues → prot_features (B, L, d_prot)
  2. Encode ligand atoms     → lig_features  (B, N, d_lig)
  3. Project lig to d_prot, prot to d_lig (dimension matching)
  4. BidirectionalCrossAttention × n_cross_layers
  5. Decode scores:
       score_R   (B, L, 3)  — rotation score (axis-angle, tangent at R_t)
       score_tr  (B, L, 3)  — translation score
       score_x   (B, N, 3)  — ligand coordinate score
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from inducedfitnet.models.protein_encoder import ProteinBackboneEncoder
from inducedfitnet.models.ligand_encoder import LigandEncoder
from inducedfitnet.models.cross_attention import BidirectionalCrossAttention


class JointScoreNetwork(nn.Module):
    """
    Joint protein–ligand score network.

    Args:
        cfg: DictConfig with model hyper-parameters. Expected keys:
               d_prot_in        (int, default 27)
               d_lig_in         (int, default 17)
               d_prot           (int, default 256)
               d_lig            (int, default 128)
               n_ipa            (int, default 4)
               n_lig_layers     (int, default 3)
               n_cross_layers   (int, default 4)
               n_heads_prot     (int, default 8)
               n_heads_lig      (int, default 4)
               n_heads_cross    (int, default 8)
               n_rbf            (int, default 16)
               rbf_cutoff       (float, default 20.0)
               d_time           (int, default 256)
               dropout          (float, default 0.1)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        d_prot = cfg.get("d_prot", 256)
        d_lig  = cfg.get("d_lig", 128)
        d_time = cfg.get("d_time", 256)

        # --- Sub-encoders ---
        self.prot_encoder = ProteinBackboneEncoder(
            d_res_in = cfg.get("d_prot_in", 27),
            d_model  = d_prot,
            n_ipa    = cfg.get("n_ipa", 4),
            n_heads  = cfg.get("n_heads_prot", 8),
            d_time   = d_time,
        )
        self.lig_encoder = LigandEncoder(
            d_lig_in = cfg.get("d_lig_in", 17),
            d_model  = d_lig,
            n_layers = cfg.get("n_lig_layers", 3),
            n_heads  = cfg.get("n_heads_lig", 4),
            d_time   = d_time,
            dropout  = cfg.get("dropout", 0.1),
        )

        # --- Dimension adapters for cross-attention (shared d_cross) ---
        d_cross = d_prot
        self.lig_to_cross  = nn.Linear(d_lig, d_cross)
        self.prot_to_cross = nn.Identity()   # d_prot == d_cross

        n_cross = cfg.get("n_cross_layers", 4)
        self.cross_layers = nn.ModuleList([
            BidirectionalCrossAttention(
                d_model    = d_cross,
                n_heads    = cfg.get("n_heads_cross", 8),
                n_rbf      = cfg.get("n_rbf", 16),
                rbf_cutoff = cfg.get("rbf_cutoff", 20.0),
                dropout    = cfg.get("dropout", 0.1),
            )
            for _ in range(n_cross)
        ])

        # --- Final score decoders after cross-attention ---
        # Protein: rotation + translation
        self.prot_score_head = nn.Sequential(
            nn.LayerNorm(d_cross),
            nn.Linear(d_cross, d_cross // 2),
            nn.GELU(),
            nn.Linear(d_cross // 2, 6),   # 3 rot + 3 tr
        )
        # Ligand: coordinate score
        self.lig_score_head = nn.Sequential(
            nn.LayerNorm(d_cross),
            nn.Linear(d_cross, d_cross // 2),
            nn.GELU(),
            nn.Linear(d_cross // 2, 3),
        )

        self._init_zero_scores()

    def _init_zero_scores(self):
        """Zero-initialise the final linear layers of score heads."""
        for head in [self.prot_score_head, self.lig_score_head]:
            last_lin = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_lin.weight)
            if last_lin.bias is not None:
                nn.init.zeros_(last_lin.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        R:         Tensor,               # (B, L, 3, 3) backbone rotations
        tr:        Tensor,               # (B, L, 3)    Cα translations
        x_lig:     Tensor,               # (B, N, 3)    ligand coords
        res_feat:  Tensor,               # (B, L, d_prot_in)
        lig_feat:  Tensor,               # (B, N, d_lig_in)
        timestep:  Tensor,               # (B,)
        res_mask:  Optional[Tensor] = None,  # (B, L)
        lig_mask:  Optional[Tensor] = None,  # (B, N)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            R, tr:    Current noised backbone frames.
            x_lig:    Current noised ligand coordinates.
            res_feat: Per-residue input features.
            lig_feat: Per-atom ligand features.
            timestep: Integer diffusion timesteps (B,).
            res_mask, lig_mask: Padding masks.

        Returns:
            score_R:  (B, L, 3)   rotation score (axis-angle)
            score_tr: (B, L, 3)   translation score
            score_x:  (B, N, 3)   ligand coordinate score
        """
        # 1. Encode independently
        prot_h, init_score_R, init_score_tr = self.prot_encoder(
            res_feat, R, tr, timestep, res_mask
        )
        lig_h, init_score_x = self.lig_encoder(
            lig_feat, x_lig, timestep, lig_mask
        )

        # 2. Project ligand features to shared cross-attention dimension
        prot_h_cross = self.prot_to_cross(prot_h)    # (B, L, d_cross)
        lig_h_cross  = self.lig_to_cross(lig_h)      # (B, N, d_cross)

        # 3. Bidirectional cross-attention over 3D structure
        ca_pos = tr                                    # (B, L, 3)  Cα as protein positions
        for cross in self.cross_layers:
            prot_h_cross, lig_h_cross = cross(
                prot_feat = prot_h_cross,
                lig_feat  = lig_h_cross,
                prot_pos  = ca_pos,
                lig_pos   = x_lig,
                prot_mask = res_mask,
                lig_mask  = lig_mask,
            )

        # 4. Decode scores (residual over initial estimates)
        prot_scores = self.prot_score_head(prot_h_cross)     # (B, L, 6)
        lig_score   = self.lig_score_head(lig_h_cross)       # (B, N, 3)

        score_R  = init_score_R  + prot_scores[..., :3]
        score_tr = init_score_tr + prot_scores[..., 3:]
        score_x  = init_score_x  + lig_score

        # Mask out padding
        if res_mask is not None:
            m = res_mask.float().unsqueeze(-1)
            score_R  = score_R  * m
            score_tr = score_tr * m
        if lig_mask is not None:
            score_x = score_x * lig_mask.float().unsqueeze(-1)

        return score_R, score_tr, score_x

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n = self.count_parameters()
        return (
            f"JointScoreNetwork("
            f"params={n/1e6:.2f}M, "
            f"n_cross={len(self.cross_layers)}, "
            f"n_ipa={len(self.prot_encoder.ipa_blocks)})"
        )
