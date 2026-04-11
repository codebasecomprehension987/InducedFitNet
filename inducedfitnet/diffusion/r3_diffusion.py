"""
R³ diffusion process for ligand heavy-atom positions.

Forward:  x_t = x_0 + σ(t) ε,   ε ~ N(0, I)
Score:    ∇_x log p(x_t | x_0) = -(x_t - x_0) / σ(t)²

Centre-of-mass is removed at each step to prevent translational drift
of the whole ligand; individual atom positions diffuse within the
receptor pocket coordinate frame.
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
from omegaconf import DictConfig

from inducedfitnet.diffusion.schedule import NoiseSchedule


class R3Diffusion:
    """
    Gaussian diffusion on R³ for ligand atom coordinates.

    Args:
        cfg: DictConfig with fields:
               num_steps        (int)
               sigma_r_min/max  — reuse σ range from SE3 config,
                                   or provide ligand_sigma_min/max
               schedule_type    (str)
    """

    def __init__(self, cfg: DictConfig):
        self.T = cfg.num_steps
        sigma_min = getattr(cfg, "ligand_sigma_min", cfg.sigma_t_min)
        sigma_max = getattr(cfg, "ligand_sigma_max", cfg.sigma_t_max)
        self.schedule = NoiseSchedule(
            T             = cfg.num_steps,
            sigma_min     = sigma_min,
            sigma_max     = sigma_max,
            schedule_type = cfg.schedule_type,
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_0: Tensor,        # (B, N, 3)  clean ligand coordinates
        timestep: Tensor,   # (B,)       integer t
        mask: Tensor,       # (B, N)     bool, True = valid atom
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample x_t ~ q(x_t | x_0, t) = N(x_0, σ(t)² I).

        Noise is masked (padded atoms stay at zero).

        Returns:
            x_t  : (B, N, 3)
            eps  : (B, N, 3)  the applied noise (target for denoiser)
        """
        sigma = self.schedule.sigma(timestep).view(-1, 1, 1)  # (B,1,1)
        eps   = torch.randn_like(x_0)
        eps   = eps * mask.float().unsqueeze(-1)               # zero out padding
        x_t   = x_0 + sigma * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(
        self,
        x_t: Tensor,        # (B, N, 3)
        x_0: Tensor,        # (B, N, 3)
        timestep: Tensor,   # (B,)
    ) -> Tensor:
        """
        Analytic score ∇_x log p(x_t | x_0, σ) = -(x_t - x_0) / σ².

        Returns (B, N, 3).
        """
        sigma_sq = self.schedule.sigma(timestep).pow(2).view(-1, 1, 1) + 1e-8
        return -(x_t - x_0) / sigma_sq

    # ------------------------------------------------------------------
    # Reverse step
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        x_t: Tensor,           # (B, N, 3)
        score_x: Tensor,       # (B, N, 3)  predicted score
        timestep: Tensor,      # (B,)
        mask: Tensor,          # (B, N) bool
        dt: float = 1.0,
        noise_scale: float = 1.0,
        remove_com: bool = True,
    ) -> Tensor:
        """
        Reverse SDE step (Euler-Maruyama) for ligand coordinates.

        Optionally removes centre-of-mass drift to keep ligand anchored
        within the pocket.
        """
        sigma   = self.schedule.sigma(timestep).view(-1, 1, 1)
        d_sigma = self.schedule.d_sigma(timestep).view(-1, 1, 1)

        drift  = sigma * d_sigma * score_x * dt
        noise  = noise_scale * (2 * d_sigma * dt).clamp(min=0).sqrt() * torch.randn_like(x_t)
        noise  = noise * mask.float().unsqueeze(-1)

        x_new  = x_t + drift + noise

        # Remove centre-of-mass motion
        if remove_com:
            m = mask.float().unsqueeze(-1)                  # (B, N, 1)
            com = (x_new * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(1)
            x_new = x_new - com

        return x_new

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def sigma_at(self, timestep: Tensor) -> Tensor:
        """Return σ(t) for given integer timesteps."""
        return self.schedule.sigma(timestep)

    def prior_sample(
        self,
        n_atoms: int,
        batch_size: int,
        device: torch.device,
        pocket_center: Tensor,    # (B, 3)  approximate pocket COM
        pocket_radius: float = 10.0,
    ) -> Tensor:
        """
        Draw ligand atom positions from the prior N(pocket_center, σ_max² I).

        Clamps atoms within `pocket_radius` Å of the pocket centre.
        """
        sigma_max = self.schedule.sigma_max
        x = pocket_center.unsqueeze(1) + sigma_max * torch.randn(
            batch_size, n_atoms, 3, device=device
        )
        # Soft clamping — scale back atoms that are too far
        disp  = x - pocket_center.unsqueeze(1)
        dist  = disp.norm(dim=-1, keepdim=True)
        scale = (pocket_radius / dist.clamp(min=1e-4)).clamp(max=1.0)
        x     = pocket_center.unsqueeze(1) + disp * scale
        return x
