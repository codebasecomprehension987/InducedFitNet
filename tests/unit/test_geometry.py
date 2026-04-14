"""
Unit tests for SO(3) / SE(3) geometry utilities.
"""

import math
import torch
import pytest

from inducedfitnet.utils.geometry import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    so3_log,
    so3_exp,
    random_rotation,
    perturb_rotation,
    compose_se3,
    invert_se3,
    apply_se3,
)


class TestAxisAngleConversions:
    def test_identity_zero_angle(self):
        aa = torch.zeros(4, 3)
        R  = axis_angle_to_matrix(aa)
        I  = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        # Near-zero axis-angle → identity (Rodrigues: sin(0)K + (1-cos(0))K² = 0)
        assert torch.allclose(R, I, atol=1e-5)

    def test_round_trip(self):
        torch.manual_seed(0)
        aa_orig = torch.randn(16, 3) * 0.5   # small angles for numerical stability
        R       = axis_angle_to_matrix(aa_orig)
        aa_rec  = matrix_to_axis_angle(R)
        R_rec   = axis_angle_to_matrix(aa_rec)
        assert torch.allclose(R, R_rec, atol=1e-5)

    def test_rotation_matrix_properties(self):
        torch.manual_seed(1)
        aa = torch.randn(8, 3) * 0.8
        R  = axis_angle_to_matrix(aa)
        # Orthogonality: R^T R ≈ I
        I_approx = R.transpose(-2, -1) @ R
        I = torch.eye(3).unsqueeze(0).expand(8, -1, -1)
        assert torch.allclose(I_approx, I, atol=1e-5)
        # Determinant ≈ +1
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(8), atol=1e-5)

    def test_specific_90deg_rotation(self):
        """90° rotation around z-axis."""
        aa = torch.tensor([[0.0, 0.0, math.pi / 2]])
        R  = axis_angle_to_matrix(aa).squeeze(0)
        # Expected: [[0,-1,0],[1,0,0],[0,0,1]]
        expected = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        assert torch.allclose(R, expected, atol=1e-6)


class TestSO3LogExp:
    def test_log_exp_inverse(self):
        torch.manual_seed(2)
        v = torch.randn(12, 3) * 0.4
        R = so3_exp(v)
        v2 = so3_log(R)
        R2 = so3_exp(v2)
        assert torch.allclose(R, R2, atol=1e-5)


class TestRandomRotation:
    def test_is_rotation_matrix(self):
        R = random_rotation(32, torch.device("cpu"))
        I_approx = R.transpose(-2, -1) @ R
        I = torch.eye(3).unsqueeze(0).expand(32, -1, -1)
        assert torch.allclose(I_approx, I, atol=1e-5)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(32), atol=1e-5)

    def test_uniform_distribution(self):
        """Check that traces are distributed as expected for uniform SO(3)."""
        R   = random_rotation(10000, torch.device("cpu"))
        tr  = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (N,)
        # E[trace] for Haar measure = 0 (not exactly, but close for large N)
        assert abs(tr.mean().item()) < 0.1, "Mean trace should be near 0 for uniform SO(3)"


class TestPerturbRotation:
    def test_stays_rotation(self):
        R = random_rotation(8, torch.device("cpu"))
        R_p = perturb_rotation(R, sigma=0.3)
        I_approx = R_p.transpose(-2, -1) @ R_p
        I = torch.eye(3).unsqueeze(0).expand(8, -1, -1)
        assert torch.allclose(I_approx, I, atol=1e-5)

    def test_zero_noise_identity(self):
        R   = random_rotation(4, torch.device("cpu"))
        R_p = perturb_rotation(R, sigma=0.0)
        assert torch.allclose(R, R_p, atol=1e-5)


class TestSE3:
    def test_compose_invert(self):
        torch.manual_seed(3)
        R1 = random_rotation(4, torch.device("cpu"))
        t1 = torch.randn(4, 3)
        R2, t2 = invert_se3(R1, t1)
        R_comp, t_comp = compose_se3(R1, t1, R2, t2)
        I = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        z = torch.zeros(4, 3)
        assert torch.allclose(R_comp, I, atol=1e-5)
        assert torch.allclose(t_comp, z, atol=1e-5)

    def test_apply_se3(self):
        R = torch.eye(3).unsqueeze(0)    # identity rotation
        t = torch.tensor([[1.0, 2.0, 3.0]])
        x = torch.zeros(1, 5, 3)
        out = apply_se3(R, t, x)
        expected = t.unsqueeze(1).expand(1, 5, 3)
        assert torch.allclose(out, expected, atol=1e-6)
