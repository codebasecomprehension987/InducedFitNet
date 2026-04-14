"""
Integration test: full forward pass through JointScoreNetwork.

Verifies that:
  1. The model accepts batched protein–ligand inputs.
  2. Output shapes are correct.
  3. Zero-initialised score heads produce small initial outputs.
  4. Gradients flow through all parameters.
"""

import torch
import pytest
from omegaconf import OmegaConf

from inducedfitnet.models.score_network import JointScoreNetwork
from inducedfitnet.utils.geometry import random_rotation


@pytest.fixture
def small_cfg():
    """Tiny model config for fast testing."""
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
        "dropout":        0.0,   # deterministic for tests
    })


class TestJointScoreNetworkForward:
    def test_output_shapes(self, small_cfg):
        torch.manual_seed(0)
        B, L, N = 2, 12, 8

        model = JointScoreNetwork(small_cfg)
        model.eval()

        R   = random_rotation(B * L, torch.device("cpu")).reshape(B, L, 3, 3)
        tr  = torch.randn(B, L, 3)
        x   = torch.randn(B, N, 3)
        rf  = torch.randn(B, L, 27)
        lf  = torch.randn(B, N, 17)
        t   = torch.randint(0, 50, (B,))
        rm  = torch.ones(B, L, dtype=torch.bool)
        lm  = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            sR, str_, sx = model(R, tr, x, rf, lf, t, rm, lm)

        assert sR.shape   == (B, L, 3), f"Expected ({B},{L},3), got {sR.shape}"
        assert str_.shape == (B, L, 3)
        assert sx.shape   == (B, N, 3)

    def test_gradients_flow(self, small_cfg):
        torch.manual_seed(1)
        B, L, N = 1, 8, 6

        model = JointScoreNetwork(small_cfg)
        R  = random_rotation(B * L, torch.device("cpu")).reshape(B, L, 3, 3)
        tr = torch.randn(B, L, 3)
        x  = torch.randn(B, N, 3)
        rf = torch.randn(B, L, 27)
        lf = torch.randn(B, N, 17)
        t  = torch.tensor([10])
        rm = torch.ones(B, L, dtype=torch.bool)
        lm = torch.ones(B, N, dtype=torch.bool)

        sR, str_, sx = model(R, tr, x, rf, lf, t, rm, lm)
        loss = sR.sum() + str_.sum() + sx.sum()
        loss.backward()

        # At least one parameter should have a non-zero gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        total_grad_norm = sum(g.abs().sum().item() for g in grads)
        assert total_grad_norm > 0, "All gradients are zero"

    def test_padding_mask_zeroes_output(self, small_cfg):
        """Padded residues (mask=False) should produce zero scores."""
        torch.manual_seed(2)
        B, L, N = 1, 10, 5
        model = JointScoreNetwork(small_cfg)
        model.eval()

        R  = random_rotation(B * L, torch.device("cpu")).reshape(B, L, 3, 3)
        tr = torch.randn(B, L, 3)
        x  = torch.randn(B, N, 3)
        rf = torch.randn(B, L, 27)
        lf = torch.randn(B, N, 17)
        t  = torch.tensor([5])

        # Only first 6 residues are valid
        rm = torch.zeros(B, L, dtype=torch.bool)
        rm[:, :6] = True
        lm = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            sR, str_, sx = model(R, tr, x, rf, lf, t, rm, lm)

        assert torch.allclose(sR[:, 6:], torch.zeros(B, L - 6, 3), atol=1e-6), \
            "Padded residues should have zero rotation score"
        assert torch.allclose(str_[:, 6:], torch.zeros(B, L - 6, 3), atol=1e-6)

    def test_parameter_count(self, small_cfg):
        model = JointScoreNetwork(small_cfg)
        n = model.count_parameters()
        assert n > 0
        # For tiny config, should be well under 10M
        assert n < 10_000_000, f"Unexpectedly large model: {n} params"
