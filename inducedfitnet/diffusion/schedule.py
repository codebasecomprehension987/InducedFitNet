"""
Noise schedules for SE(3) and R³ diffusion.

Provides σ(t) and its derivative dσ/dt for the reverse SDE.

Supported schedules
-------------------
linear  : σ(t) = σ_min + (σ_max - σ_min) * t / T
cosine  : σ(t) = σ_min + (σ_max - σ_min) * (1 - cos(π t / T)) / 2
sqrt    : σ(t) = σ_min + (σ_max - σ_min) * sqrt(t / T)
"""

from __future__ import annotations
import math
import torch
from torch import Tensor


class NoiseSchedule:
    """
    Maps integer timestep t ∈ {0, …, T-1} → continuous noise level σ(t).

    Args:
        T:             Number of diffusion steps.
        sigma_min:     Minimum noise level (at t=0).
        sigma_max:     Maximum noise level (at t=T-1).
        schedule_type: One of "linear", "cosine", "sqrt".
    """

    def __init__(
        self,
        T: int,
        sigma_min: float,
        sigma_max: float,
        schedule_type: str = "cosine",
    ):
        self.T          = T
        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max
        self.stype      = schedule_type.lower()

        # Pre-compute σ for all steps and register as buffer
        ts = torch.arange(T, dtype=torch.float64)
        self._sigma_table  = self._compute_sigma(ts / (T - 1))   # (T,)
        self._dsigma_table = self._compute_dsigma(ts / (T - 1))  # (T,) dσ/d(t/T) / T

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sigma(self, t: Tensor) -> Tensor:
        """
        σ(t) for integer timestep tensor t of any shape.

        Returns same shape as t.
        """
        t_clamp = t.long().clamp(0, self.T - 1)
        return self._sigma_table.to(t.device)[t_clamp].float()

    def d_sigma(self, t: Tensor) -> Tensor:
        """
        dσ/dt for integer timestep tensor t.

        Computed as finite difference / analytic derivative.
        Returns same shape as t.
        """
        t_clamp = t.long().clamp(0, self.T - 1)
        return self._dsigma_table.to(t.device)[t_clamp].float()

    def continuous_sigma(self, s: Tensor) -> Tensor:
        """
        σ evaluated at continuous s ∈ [0, 1].

        Useful for ODE integration at non-integer steps.
        """
        return self._compute_sigma(s).float()

    # ------------------------------------------------------------------
    # Private: schedule functions
    # ------------------------------------------------------------------

    def _compute_sigma(self, u: Tensor) -> Tensor:
        """u ∈ [0, 1] → σ."""
        lo, hi = self.sigma_min, self.sigma_max
        if self.stype == "linear":
            return torch.tensor(lo + (hi - lo) * u, dtype=torch.float64)
        elif self.stype == "cosine":
            return torch.tensor(
                lo + (hi - lo) * (1 - torch.cos(math.pi * u)) / 2,
                dtype=torch.float64,
            )
        elif self.stype == "sqrt":
            return torch.tensor(lo + (hi - lo) * u.sqrt(), dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule type: {self.stype}")

    def _compute_dsigma(self, u: Tensor) -> Tensor:
        """dσ/du * (1/T)  — derivative wrt continuous u, scaled by 1/T."""
        lo, hi = self.sigma_min, self.sigma_max
        delta = hi - lo
        if self.stype == "linear":
            ds = torch.full_like(u, delta / self.T, dtype=torch.float64)
        elif self.stype == "cosine":
            ds = torch.tensor(
                delta * math.pi / (2 * self.T) * torch.sin(math.pi * u),
                dtype=torch.float64,
            )
        elif self.stype == "sqrt":
            ds = torch.tensor(
                delta / (2 * self.T) / u.clamp(min=1e-8).sqrt(),
                dtype=torch.float64,
            )
        else:
            raise ValueError(f"Unknown schedule type: {self.stype}")
        return ds

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NoiseSchedule(type={self.stype}, T={self.T}, "
            f"σ_min={self.sigma_min}, σ_max={self.sigma_max})"
        )
