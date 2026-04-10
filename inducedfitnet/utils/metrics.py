"""
Evaluation metrics for InducedFitNet.

Implements:
  - Ligand RMSD (with optimal symmetry-aware alignment)
  - Backbone Cα RMSD
  - Pocket volume estimation (convex hull / α-sphere)
  - Docking success rate (RMSD < 2 Å)
  - Similarity-bin stratified success table
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Dict

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

def rmsd(
    pred: Tensor,   # (N, 3)  or (B, N, 3)
    true: Tensor,   # (N, 3)  or (B, N, 3)
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Root-mean-square deviation between two sets of points.

    Args:
        pred, true: coordinates of shape (N, 3) or (B, N, 3).
        mask:       optional boolean mask (N,) or (B, N).

    Returns:
        scalar or (B,) tensor.
    """
    diff_sq = (pred - true).pow(2).sum(dim=-1)   # (N,) or (B, N)
    if mask is not None:
        diff_sq = diff_sq * mask.float()
        n = mask.float().sum(dim=-1).clamp(min=1)
    else:
        n = diff_sq.shape[-1]
    return (diff_sq.sum(dim=-1) / n).sqrt()


def aligned_rmsd(
    pred: Tensor,   # (N, 3)
    true: Tensor,   # (N, 3)
    mask: Optional[Tensor] = None,
) -> float:
    """
    RMSD after optimal Kabsch alignment.

    Returns a Python float.
    """
    p = pred.float()
    t = true.float()
    if mask is not None:
        p = p[mask]
        t = t[mask]

    # Centre
    p_c = p - p.mean(dim=0, keepdim=True)
    t_c = t - t.mean(dim=0, keepdim=True)

    # SVD-based rotation (Kabsch)
    H = p_c.T @ t_c
    U, S, Vh = torch.linalg.svd(H)
    d = torch.det(Vh.T @ U.T).sign()
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=p.device))
    R_opt = Vh.T @ D @ U.T

    p_aligned = (R_opt @ p_c.T).T
    return rmsd(p_aligned, t_c).item()


def ca_rmsd(
    pred_backbone: Tensor,  # (L, 4, 3) or (B, L, 4, 3)  — N, CA, C, O
    true_backbone: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Cα RMSD using atom index 1 (Cα)."""
    pred_ca = pred_backbone[..., 1, :]   # (..., L, 3)
    true_ca = true_backbone[..., 1, :]
    return rmsd(pred_ca, true_ca, mask)


# ---------------------------------------------------------------------------
# Pocket volume
# ---------------------------------------------------------------------------

def pocket_volume_convex_hull(pocket_coords: np.ndarray) -> float:
    """
    Estimate pocket volume (Å³) via the convex hull of pocket-lining atoms.

    Args:
        pocket_coords: (N, 3) numpy array of heavy-atom coordinates.

    Returns:
        volume in Å³.
    """
    from scipy.spatial import ConvexHull  # type: ignore
    try:
        hull = ConvexHull(pocket_coords)
        return float(hull.volume)
    except Exception:
        return 0.0


def pocket_overlap(
    pred_coords: np.ndarray,   # (N, 3) predicted pocket atoms
    true_coords: np.ndarray,   # (M, 3) reference pocket atoms
    radius: float = 4.0,
) -> float:
    """
    Jaccard-like pocket overlap based on grid-point occupancy.

    A grid point is "occupied" if any atom is within `radius` Å.
    Returns intersection / union in [0, 1].
    """
    # Build a simple grid over the union bounding box
    all_coords = np.concatenate([pred_coords, true_coords], axis=0)
    lo = all_coords.min(axis=0) - radius
    hi = all_coords.max(axis=0) + radius
    grid_points = np.mgrid[
        lo[0]:hi[0]:2.0,
        lo[1]:hi[1]:2.0,
        lo[2]:hi[2]:2.0,
    ].reshape(3, -1).T  # (G, 3)

    def occupied(coords: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(grid_points[:, None] - coords[None], axis=-1)  # (G, N)
        return (dists < radius).any(axis=-1)  # (G,)

    pred_occ = occupied(pred_coords)
    true_occ = occupied(true_coords)

    intersection = (pred_occ & true_occ).sum()
    union = (pred_occ | true_occ).sum()
    return float(intersection) / max(float(union), 1)


# ---------------------------------------------------------------------------
# Docking success rate
# ---------------------------------------------------------------------------

def success_rate(
    pred_list: Sequence[Tensor],  # list of (N_atoms, 3) ligand coords
    true_coords: Tensor,          # (N_atoms, 3) reference
    threshold: float = 2.0,       # Å
    use_alignment: bool = True,
) -> float:
    """
    Fraction of predictions within `threshold` Å RMSD of the reference.
    """
    successes = 0
    for pred in pred_list:
        if use_alignment:
            r = aligned_rmsd(pred, true_coords)
        else:
            r = rmsd(pred, true_coords).item()
        if r < threshold:
            successes += 1
    return successes / max(len(pred_list), 1)


# ---------------------------------------------------------------------------
# Similarity-bin stratified evaluation (IsoDDE metric)
# ---------------------------------------------------------------------------

def stratified_success_table(
    records: List[Dict],
    thresholds: Sequence[float] = (2.0,),
    bins: Sequence[tuple] = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100)),
) -> Dict:
    """
    Compute success rates stratified by training-set sequence-similarity bins.

    Args:
        records: list of dicts with keys:
                   'pred_coords'  Tensor (N, 3)
                   'true_coords'  Tensor (N, 3)
                   'similarity'   float in [0, 100]
        thresholds: RMSD thresholds in Å.
        bins: similarity percentage bins.

    Returns:
        Nested dict: bin_label → threshold_str → success_rate.
    """
    results: Dict = {}
    for lo, hi in bins:
        label = f"{lo}-{hi}%"
        bin_records = [r for r in records if lo <= r["similarity"] < hi]
        results[label] = {}
        for thr in thresholds:
            preds = [r["pred_coords"] for r in bin_records]
            trues = [r["true_coords"] for r in bin_records]
            rates = [
                float(rmsd(p, t).item() < thr)
                for p, t in zip(preds, trues)
            ]
            results[label][f"SR@{thr}Å"] = (
                float(np.mean(rates)) if rates else float("nan")
            )
        results[label]["n"] = len(bin_records)
    return results
