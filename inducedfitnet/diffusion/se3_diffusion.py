"""
SE(3) diffusion process for the protein backbone.

Each residue is treated as a rigid frame T_i = (R_i, t_i) ∈ SE(3).
The forward process independently perturbs:
  - R_i  with isotropic Gaussian noise on SO(3)
  - t_i  with isotropic Gaussian noise on R³

The marginal distribution at time t is:
  R_i(t) = exp(ε_R) R_i(0),   ε_R ~ IGSO3(σ_R(t))
  t_i(t) = t_i(0) + ε_t,      ε_t ~ N(0, σ_t(t)² I)

References:
  - SE(3) diffusion model with application to protein backbone generation
    (Yim et al., 2023 — FrameDiff)
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
from omegaconf import DictConfig

from inducedfitnet.utils.geometry import (
    perturb_rotation,
    igso3_score,
    so3_log,
    so3_exp,
    random_rotation,
)
from inducedfitnet.diffusion.schedule import NoiseSchedule
from inducedfitnet.utils.frames import backbone_frames


class SE3Diffusion:
    """
    Marginal forward process and reverse score for SE(3)^L backbone frames.

    Args:
        cfg: DictConfig with fields:
               num_steps        (int)   : T, diffusion timesteps
               sigma_r_min/max  (float) : SO(3) noise range
               sigma_t_min/max  (float) : R³ translation noise range
               schedule_type    (str)   : "cosine" | "linear" | "sqrt"
    """

    def __init__(self, cfg: DictConfig):
        self.T = cfg.num_steps
        self.schedule = NoiseSchedule(
            T             = cfg.num_steps,
            sigma_min     = cfg.sigma_r_min,
            sigma_max     = cfg.sigma_r_max,
            schedule_type = cfg.schedule_type,
        )
        self.trans_schedule = NoiseSchedule(
            T             = cfg.num_steps,
            sigma_min     = cfg.sigma_t_min,
            sigma_max     = cfg.sigma_t_max,
            schedule_type = cfg.schedule_type,
        )

    # ------------------------------------------------------------------
    # Forward process (noising)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        R_0: Tensor,    # (B, L, 3, 3)  clean rotations
        t_0: Tensor,    # (B, L, 3)     clean translations (Cα)
        timestep: Tensor,  # (B,) integer timesteps in [0, T)
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample from q(T_t | T_0) = q(R_t | R_0) q(t_t | t_0).

        Returns:
            R_t: (B, L, 3, 3)   noised rotations
            tr_t: (B, L, 3)     noised translations
        """
        B, L = R_0.shape[:2]
        sigma_r = self.schedule.sigma(timestep)          # (B,)
        sigma_t = self.trans_schedule.sigma(timestep)    # (B,)

        # Broadcast σ over residues
        sigma_r_bl = sigma_r.view(B, 1).expand(B, L)    # (B, L)
        sigma_t_bl = sigma_t.view(B, 1).expand(B, L)

        # Perturb rotations via IGSO3
        R_t = torch.stack([
            torch.stack([
                perturb_rotation(R_0[b, l].unsqueeze(0), sigma_r_bl[b, l].item()).squeeze(0)
                for l in range(L)
            ])
            for b in range(B)
        ])  # (B, L, 3, 3)

        # Perturb translations via isotropic Gaussian
        eps_t = torch.randn_like(t_0) * sigma_t_bl.unsqueeze(-1)
        tr_t  = t_0 + eps_t

        return R_t, tr_t

    # ------------------------------------------------------------------
    # Score (used during training)
    # ------------------------------------------------------------------

    def rotation_score(
        self,
        R_t: Tensor,   # (B, L, 3, 3)  noised rotations
        R_0: Tensor,   # (B, L, 3, 3)  clean rotations
        timestep: Tensor,  # (B,)
    ) -> Tensor:
        """
        Ground-truth rotation score ∇_R log q(R_t | R_0, σ).

        Returns axis-angle vector in tangent space at R_t: (B, L, 3).
        """
        B, L = R_t.shape[:2]
        sigma_r = self.schedule.sigma(timestep)  # (B,)

        scores = []
        for b in range(B):
            row = []
            for l in range(L):
                s = igso3_score(
                    R_noisy = R_t[b, l].unsqueeze(0),
                    R_clean = R_0[b, l].unsqueeze(0),
                    sigma   = sigma_r[b].item(),
                )
                row.append(s.squeeze(0))
            scores.append(torch.stack(row))
        return torch.stack(scores)  # (B, L, 3)

    def translation_score(
        self,
        tr_t: Tensor,      # (B, L, 3)
        t_0: Tensor,       # (B, L, 3)
        timestep: Tensor,  # (B,)
    ) -> Tensor:
        """
        Ground-truth translation score ∇_t log q(tr_t | t_0, σ).

        For Gaussian: score = -(tr_t - t_0) / σ².
        Returns (B, L, 3).
        """
        sigma_t = self.trans_schedule.sigma(timestep).view(-1, 1, 1)  # (B,1,1)
        return -(tr_t - t_0) / (sigma_t ** 2 + 1e-8)

    # ------------------------------------------------------------------
    # Reverse step (used during sampling)
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        R_t: Tensor,           # (B, L, 3, 3)
        tr_t: Tensor,          # (B, L, 3)
        score_R: Tensor,       # (B, L, 3)   predicted rotation score
        score_tr: Tensor,      # (B, L, 3)   predicted translation score
        timestep: Tensor,      # (B,)
        dt: float = 1.0,
        noise_scale: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        One step of the reverse SDE / probability-flow ODE.

        Euler–Maruyama on SE(3).
        """
        B, L = R_t.shape[:2]
        sigma_r  = self.schedule.sigma(timestep)
        sigma_t  = self.trans_schedule.sigma(timestep)
        d_sigma_r  = self.schedule.d_sigma(timestep)
        d_sigma_t  = self.trans_schedule.d_sigma(timestep)

        # Translation update
        drift_tr = sigma_t.view(B,1,1) * d_sigma_t.view(B,1,1) * score_tr * dt
        noise_tr = noise_scale * (d_sigma_t * 2 * dt).sqrt().view(B,1,1) * torch.randn_like(tr_t)
        tr_new   = tr_t + drift_tr + noise_tr

        # Rotation update via exponential map
        drift_R  = sigma_r.view(B,1,1) * d_sigma_r.view(B,1,1) * score_R * dt   # (B,L,3)
        noise_R  = noise_scale * (d_sigma_r * 2 * dt).sqrt().view(B,1,1) * torch.randn_like(drift_R)
        delta_R  = (drift_R + noise_R).reshape(B * L, 3)
        R_new    = (so3_exp(delta_R) @ R_t.reshape(B * L, 3, 3)).reshape(B, L, 3, 3)

        return R_new, tr_new
