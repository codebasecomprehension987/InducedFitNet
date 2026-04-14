"""
Shared pytest fixtures for InducedFitNet tests.
"""

import torch
import pytest
from omegaconf import OmegaConf

from inducedfitnet.data.complex import ProteinLigandComplex


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_complex():
    """Minimal synthetic ProteinLigandComplex for fast unit tests."""
    L, N = 12, 8
    return ProteinLigandComplex(
        backbone_coords  = torch.randn(L, 4, 3),
        residue_features = torch.randn(L, 27),
        residue_mask     = torch.ones(L, dtype=torch.bool),
        ligand_coords    = torch.randn(N, 3),
        ligand_features  = torch.randn(N, 17),
        ligand_mask      = torch.ones(N, dtype=torch.bool),
        pdb_id           = "TEST",
        similarity       = 15.0,   # hard bin: 0-20%
    )


@pytest.fixture
def small_diff_cfg():
    return OmegaConf.create({
        "num_steps":         50,
        "schedule_type":     "cosine",
        "sigma_r_min":       0.02,
        "sigma_r_max":       1.5,
        "sigma_t_min":       0.1,
        "sigma_t_max":       10.0,
        "ligand_sigma_min":  0.1,
        "ligand_sigma_max":  5.0,
    })


@pytest.fixture
def small_model_cfg():
    return OmegaConf.create({
        "d_prot_in":      27,
        "d_lig_in":       17,
        "d_prot":         64,
        "d_lig":          32,
        "d_time":         64,
        "n_ipa":          2,
        "n_lig_layers":   2,
        "n_cross_layers": 2,
        "n_heads_prot":   4,
        "n_heads_lig":    2,
        "n_heads_cross":  4,
        "n_rbf":          8,
        "rbf_cutoff":     15.0,
        "dropout":        0.0,
    })
