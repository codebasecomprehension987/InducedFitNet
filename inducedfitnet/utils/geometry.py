"""
SO(3) / SE(3) mathematical utilities.

Covers:
  - Rodrigues / axis-angle ↔ rotation matrix conversions
  - Log / exp maps on SO(3)
  - Isotropic Gaussian on SO(3) (IGSO3) score and sampling
  - SE(3) frame composition helpers
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Basic SO(3) operations
# ---------------------------------------------------------------------------

def skew_symmetric(v: Tensor) -> Tensor:
    """(B, 3) → (B, 3, 3) skew-symmetric matrix."""
    B = v.shape[0]
    z = torch.zeros(B, device=v.device, dtype=v.dtype)
    row0 = torch.stack([z, -v[:, 2], v[:, 1]], dim=-1)
    row1 = torch.stack([v[:, 2], z, -v[:, 0]], dim=-1)
    row2 = torch.stack([-v[:, 1], v[:, 0], z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (B, 3, 3)


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """
    Convert axis-angle vectors to rotation matrices via Rodrigues' formula.

    Args:
        axis_angle: (*, 3)  — direction encodes axis, magnitude encodes angle.

    Returns:
        R: (*, 3, 3) rotation matrices.
    """
    shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)

    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    axis = aa / angle  # (B, 3)

    K = skew_symmetric(axis)  # (B, 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0)

    sin_a = angle.sin().unsqueeze(-1)   # (B, 1, 1)
    cos_a = angle.cos().unsqueeze(-1)   # (B, 1, 1)

    R = I + sin_a * K + (1 - cos_a) * (K @ K)
    return R.reshape(*shape, 3, 3)


def matrix_to_axis_angle(R: Tensor) -> Tensor:
    """
    Rotation matrix (*, 3, 3) → axis-angle (*, 3).

    Uses numerically stable formula based on acos of the trace.
    """
    shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    trace = R_flat.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # clamp to valid range for acos
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = cos_angle.acos()  # (B,)

    # (R - R^T) / (2 sin θ)  → skew-symmetric part gives axis
    sin_angle = angle.sin().clamp(min=1e-8)
    axis = torch.stack([
        R_flat[:, 2, 1] - R_flat[:, 1, 2],
        R_flat[:, 0, 2] - R_flat[:, 2, 0],
        R_flat[:, 1, 0] - R_flat[:, 0, 1],
    ], dim=-1) / (2 * sin_angle.unsqueeze(-1))

    aa = axis * angle.unsqueeze(-1)
    return aa.reshape(*shape, 3)


def so3_log(R: Tensor) -> Tensor:
    """SO(3) matrix → tangent vector (axis-angle). Alias for matrix_to_axis_angle."""
    return matrix_to_axis_angle(R)


def so3_exp(v: Tensor) -> Tensor:
    """Tangent vector (axis-angle) → SO(3) matrix. Alias for axis_angle_to_matrix."""
    return axis_angle_to_matrix(v)


# ---------------------------------------------------------------------------
# IGSO3 — isotropic Gaussian on SO(3)
# ---------------------------------------------------------------------------

def igso3_density(omega: Tensor, sigma: float, num_terms: int = 500) -> Tensor:
    """
    Marginal density of IGSO3(σ) at rotation angle ω (radians).

    p(ω | σ) ∝ (1 - cos ω) * Σ_{l=0}^{N} (2l+1) exp(-l(l+1)σ²/2) sin((l+½)ω) / sin(ω/2)

    Args:
        omega: (B,) rotation angles in [0, π].
        sigma: diffusion noise level.
        num_terms: truncation of the Fourier series.

    Returns:
        density: (B,) unnormalised density values.
    """
    omega = omega.clamp(1e-6, math.pi - 1e-6)
    ls = torch.arange(num_terms, device=omega.device, dtype=omega.dtype)
    # shape broadcast: (B, 1) × (1, N) → (B, N)
    coeff = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * sigma ** 2 / 2)
    angles = (ls + 0.5).unsqueeze(0) * omega.unsqueeze(-1)  # (B, N)
    series = (coeff.unsqueeze(0) * angles.sin()).sum(dim=-1)
    density = (1 - omega.cos()) * series / (2 * (omega / 2).sin())
    return density


def igso3_score(
    R_noisy: Tensor,
    R_clean: Tensor,
    sigma: float,
) -> Tensor:
    """
    Riemannian score ∇_{R} log p(R_noisy | R_clean, σ) on SO(3).

    Returns the score in the tangent space at R_noisy as an axis-angle
    vector of shape (B, 3).
    """
    # Relative rotation: R_noisy^T R_clean
    R_rel = R_noisy.transpose(-2, -1) @ R_clean
    omega_vec = so3_log(R_rel)          # (B, 3)
    omega = omega_vec.norm(dim=-1)       # (B,)

    # d/dσ log p(ω | σ) — derivative wrt angle magnitude for score magnitude
    d_log_p = -omega / (sigma ** 2)     # simple Gaussian approximation in angle
    score = d_log_p.unsqueeze(-1) * omega_vec / omega.unsqueeze(-1).clamp(1e-8)
    return score  # (B, 3)


def random_rotation(batch_size: int, device: torch.device) -> Tensor:
    """Sample uniform random rotations via QR decomposition of N(0,1) matrices."""
    M = torch.randn(batch_size, 3, 3, device=device)
    Q, R = torch.linalg.qr(M)
    # Fix sign to ensure det = +1
    sign = torch.diagonal(R, dim1=-2, dim2=-1).sign().prod(dim=-1, keepdim=True).unsqueeze(-1)
    return Q * sign


def perturb_rotation(R: Tensor, sigma: float) -> Tensor:
    """
    Add isotropic Gaussian noise on SO(3) with std σ.

    R_noisy = exp(ε) @ R,   ε ~ N(0, σ² I)  in the Lie algebra.
    """
    eps = torch.randn_like(R[..., 0])  # (*, 3)
    eps = eps * sigma
    R_eps = so3_exp(eps)
    return R_eps @ R


# ---------------------------------------------------------------------------
# SE(3) frame utilities
# ---------------------------------------------------------------------------

def compose_se3(
    R1: Tensor, t1: Tensor,
    R2: Tensor, t2: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compose two SE(3) transforms: (R1,t1) ∘ (R2,t2)."""
    R = R1 @ R2
    t = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
    return R, t


def invert_se3(R: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    """Invert an SE(3) frame."""
    R_inv = R.transpose(-2, -1)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    return R_inv, t_inv


def apply_se3(R: Tensor, t: Tensor, x: Tensor) -> Tensor:
    """Apply SE(3) transform to points x: (*, N, 3)."""
    return (R.unsqueeze(-3) @ x.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(-2)
