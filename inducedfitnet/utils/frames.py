"""
Backbone rigid-frame construction.

Each residue is represented as a local SE(3) frame built from the
N, Cα, C backbone atoms following the AlphaFold2 / Framediff convention:
  - origin  = Cα
  - x-axis  = (C - N) / ‖C - N‖  (approximately)
  - y-axis  = orthogonal component of (Cα - N)
  - z-axis  = x × y
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor


def backbone_frames(
    n_coords: Tensor,   # (B, L, 3)
    ca_coords: Tensor,  # (B, L, 3)
    c_coords: Tensor,   # (B, L, 3)
) -> Tuple[Tensor, Tensor]:
    """
    Construct local backbone SE(3) frames for each residue.

    Args:
        n_coords:  N  atom positions  (B, L, 3)
        ca_coords: Cα atom positions  (B, L, 3)
        c_coords:  C  atom positions  (B, L, 3)

    Returns:
        R:  rotation matrices  (B, L, 3, 3)
        t:  translation (Cα)   (B, L, 3)
    """
    # Vectors in the plane
    v1 = c_coords - n_coords                  # (B, L, 3)
    v2 = ca_coords - n_coords                 # (B, L, 3)

    # Gram-Schmidt orthonormalisation
    e1 = _safe_normalise(v1)                  # x-axis
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = _safe_normalise(u2)                  # y-axis
    e3 = torch.linalg.cross(e1, e2)          # z-axis

    R = torch.stack([e1, e2, e3], dim=-1)     # (B, L, 3, 3)   columns = axes
    t = ca_coords                              # origin at Cα

    return R, t


def frames_to_global(
    R_local: Tensor,   # (B, L, 3, 3)
    t_local: Tensor,   # (B, L, 3)
    x_local: Tensor,   # (B, L, N_atoms, 3)  atom positions in local frame
) -> Tensor:
    """
    Transform atom positions from residue-local frame to global coordinates.

    x_global = R_local @ x_local + t_local
    """
    x_global = (R_local.unsqueeze(-3) @ x_local.unsqueeze(-1)).squeeze(-1)
    return x_global + t_local.unsqueeze(-2)


def global_to_frames(
    R_local: Tensor,   # (B, L, 3, 3)
    t_local: Tensor,   # (B, L, 3)
    x_global: Tensor,  # (B, L, N_atoms, 3)
) -> Tensor:
    """
    Transform atom positions from global coordinates into residue-local frames.

    x_local = R_local^T @ (x_global - t_local)
    """
    diff = x_global - t_local.unsqueeze(-2)                   # (B, L, N, 3)
    R_T = R_local.transpose(-2, -1)                            # (B, L, 3, 3)
    x_local = (R_T.unsqueeze(-3) @ diff.unsqueeze(-1)).squeeze(-1)
    return x_local


def extract_backbone_coords(
    all_atom_coords: Tensor,  # (B, L, 37, 3)  full atom14/atom37 representation
    atom_mask: Tensor,        # (B, L, 37)
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Extract N (idx 0), Cα (idx 1), C (idx 2), O (idx 3) from atom37 tensor.

    Returns:
        n_coords, ca_coords, c_coords, o_coords  each (B, L, 3)
    """
    n_coords  = all_atom_coords[:, :, 0, :]
    ca_coords = all_atom_coords[:, :, 1, :]
    c_coords  = all_atom_coords[:, :, 2, :]
    o_coords  = all_atom_coords[:, :, 3, :]
    return n_coords, ca_coords, c_coords, o_coords


def torsion_angle_from_coords(
    a: Tensor, b: Tensor, c: Tensor, d: Tensor,
) -> Tensor:
    """
    Compute dihedral angle defined by four points a-b-c-d.

    Args:
        a, b, c, d: (*, 3)

    Returns:
        angle: (*,) in radians ∈ (-π, π]
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)

    m1 = torch.linalg.cross(n1, b2 / b2.norm(dim=-1, keepdim=True).clamp(1e-8))

    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_normalise(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)
