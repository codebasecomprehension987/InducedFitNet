"""
Unit tests for SE(3) and R³ diffusion processes.
"""

import torch
import pytest
from omegaconf import OmegaConf

from inducedfitnet.diffusion.schedule import NoiseSchedule
from inducedfitnet.diffusion.se3_diffusion import SE3Diffusion
from inducedfitnet.diffusion.r3_diffusion import R3Diffusion
from inducedfitnet.utils.geometry import random_rotation


@pytest.fixture
def diff_cfg():
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


class TestNoiseSchedule:
    def test_monotone_cosine(self):
        sched = NoiseSchedule(100, 0.01, 1.0, "cosine")
        sigmas = sched.sigma(torch.arange(100))
        # Should be monotonically increasing
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs >= 0).all(), "Cosine schedule should be non-decreasing"

    def test_bounds(self):
        sched = NoiseSchedule(100, 0.01, 1.0, "linear")
        assert abs(sched.sigma(torch.tensor([0])).item() - 0.01) < 1e-4
        assert abs(sched.sigma(torch.tensor([99])).item() - 1.0) < 1e-4

    def test_all_schedules(self):
        for stype in ["linear", "cosine", "sqrt"]:
            sched = NoiseSchedule(50, 0.01, 1.0, stype)
            sig   = sched.sigma(torch.arange(50))
            assert sig.shape == (50,)
            assert (sig > 0).all()


class TestSE3Diffusion:
    def test_q_sample_shapes(self, diff_cfg):
        diff = SE3Diffusion(diff_cfg)
        B, L = 2, 10
        R_0  = random_rotation(B * L, torch.device("cpu")).reshape(B, L, 3, 3)
        tr_0 = torch.randn(B, L, 3)
        t    = torch.randint(0, diff_cfg.num_steps, (B,))
        R_t, tr_t = diff.q_sample(R_0, tr_0, t)
        assert R_t.shape  == (B, L, 3, 3)
        assert tr_t.shape == (B, L, 3)

    def test_q_sample_rotation_validity(self, diff_cfg):
        diff = SE3Diffusion(diff_cfg)
        B, L = 2, 5
        R_0  = random_rotation(B * L, torch.device("cpu")).reshape(B, L, 3, 3)
        tr_0 = torch.randn(B, L, 3)
        t    = torch.tensor([10, 20])
        R_t, _ = diff.q_sample(R_0, tr_0, t)
        # R_t should still be rotation matrices
        I_approx = R_t.reshape(B * L, 3, 3).transpose(-2, -1) @ R_t.reshape(B * L, 3, 3)
        I = torch.eye(3).unsqueeze(0).expand(B * L, -1, -1)
        assert torch.allclose(I_approx, I, atol=1e-4)

    def test_rotation_score_shape(self, diff_cfg):
        diff = SE3Diffusion(diff_cfg)
        B, L = 2, 8
        R_0  = random_rotation(B * L, torch.device("cpu")).reshape(B, L, 3, 3)
        tr_0 = torch.randn(B, L, 3)
        t    = torch.randint(0, diff_cfg.num_steps, (B,))
        R_t, _ = diff.q_sample(R_0, tr_0, t)
        score  = diff.rotation_score(R_t, R_0, t)
        assert score.shape == (B, L, 3)

    def test_translation_score_sign(self, diff_cfg):
        """Score should point from x_t back towards x_0."""
        diff = SE3Diffusion(diff_cfg)
        B, L = 1, 3
        R_0  = random_rotation(B * L, torch.device("cpu")).reshape(B, L, 3, 3)
        tr_0 = torch.zeros(B, L, 3)
        t    = torch.tensor([40])
        _, tr_t = diff.q_sample(R_0, tr_0, t)
        score = diff.translation_score(tr_t, tr_0, t)
        # Score should have opposite sign to (tr_t - tr_0) ≈ noise
        inner = (score * (tr_t - tr_0)).sum()
        assert inner.item() < 0, "Translation score should point back toward origin"


class TestR3Diffusion:
    def test_q_sample_shapes(self, diff_cfg):
        diff = R3Diffusion(diff_cfg)
        B, N = 3, 15
        x_0  = torch.randn(B, N, 3)
        mask = torch.ones(B, N, dtype=torch.bool)
        t    = torch.randint(0, diff_cfg.num_steps, (B,))
        x_t, eps = diff.q_sample(x_0, t, mask)
        assert x_t.shape  == (B, N, 3)
        assert eps.shape  == (B, N, 3)

    def test_masked_atoms_unchanged(self, diff_cfg):
        diff = R3Diffusion(diff_cfg)
        B, N = 2, 10
        x_0  = torch.randn(B, N, 3)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :5] = True
        t    = torch.tensor([30, 30])
        x_t, eps = diff.q_sample(x_0, t, mask)
        # Masked-out (padding) atoms: eps should be 0
        assert torch.allclose(eps[:, 5:], torch.zeros(B, 5, 3))

    def test_score_shape(self, diff_cfg):
        diff = R3Diffusion(diff_cfg)
        B, N = 2, 12
        x_0  = torch.randn(B, N, 3)
        mask = torch.ones(B, N, dtype=torch.bool)
        t    = torch.tensor([10, 20])
        x_t, _ = diff.q_sample(x_0, t, mask)
        score   = diff.score(x_t, x_0, t)
        assert score.shape == (B, N, 3)

    def test_prior_sample(self, diff_cfg):
        diff = R3Diffusion(diff_cfg)
        B, N = 4, 20
        center = torch.zeros(B, 3)
        x = diff.prior_sample(N, B, torch.device("cpu"), center, pocket_radius=10.0)
        assert x.shape == (B, N, 3)
        # All atoms within or near the pocket radius (soft clamped)
        dist = x.norm(dim=-1)
        assert (dist < 15.0).all(), "Prior atoms should be near pocket centre"
