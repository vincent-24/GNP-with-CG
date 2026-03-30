"""Tests for neural network architectures."""

import pytest
import torch
import numpy as np


class TestResGCN:
    """Tests for ResGCN network."""

    def test_resgcn_forward_pass(self, small_1d_laplacian_torch, device):
        """ResGCN should produce output of correct shape."""
        from GNP.nn import ResGCN

        A = small_1d_laplacian_torch
        n = A.shape[0]

        net = ResGCN(
            A=A,
            num_layers=4,
            embed=16,
            hidden=32,
            drop_rate=0.0
        ).to(device)

        # Single vector input
        r = torch.randn(n, dtype=torch.float64, device=device)
        z = net(r)

        assert z.shape == r.shape, f"Output shape mismatch: {z.shape} vs {r.shape}"
        assert z.dtype == r.dtype

    def test_resgcn_batch_input(self, small_1d_laplacian_torch, device):
        """ResGCN should handle batched input."""
        from GNP.nn import ResGCN

        A = small_1d_laplacian_torch
        n = A.shape[0]
        batch_size = 4

        net = ResGCN(
            A=A,
            num_layers=4,
            embed=16,
            hidden=32,
            drop_rate=0.0
        ).to(device)

        # Batched input (n, batch_size)
        r = torch.randn(n, batch_size, dtype=torch.float64, device=device)
        z = net(r)

        assert z.shape == r.shape


class TestLinearMGGNN:
    """Tests for LinearMGGNN network - the main architecture for PCG compatibility."""

    def test_linear_mggnn_forward_pass(self, small_1d_laplacian_torch, device):
        """LinearMGGNN should produce output of correct shape."""
        from GNP.nn import LinearMGGNN

        A = small_1d_laplacian_torch
        n = A.shape[0]

        net = LinearMGGNN(
            A=A,
            num_levels=3,
            embed=16,
            hidden=32,
            drop_rate=0.0,
            num_vcycles=1,
            smoother_K=2,
            coarsest_K=3
        ).to(device)

        r = torch.randn(n, dtype=torch.float64, device=device)
        z = net(r)

        assert z.shape == r.shape

    def test_linear_mggnn_linearity(self, small_1d_laplacian_torch, device):
        """LinearMGGNN MUST be linear: M(ax + by) = aM(x) + bM(y).

        This is CRITICAL for standard PCG to work correctly.
        Non-linear preconditioners require Flexible CG.
        """
        from GNP.nn import LinearMGGNN

        A = small_1d_laplacian_torch
        n = A.shape[0]

        net = LinearMGGNN(
            A=A,
            num_levels=3,
            embed=16,
            hidden=32,
            drop_rate=0.0,
            num_vcycles=1,
            smoother_K=2,
            coarsest_K=3,
            share_smoothers=True  # Important for symmetry
        ).to(device)
        net.eval()  # Disable dropout

        # Random test vectors
        torch.manual_seed(42)
        x = torch.randn(n, dtype=torch.float64, device=device)
        y = torch.randn(n, dtype=torch.float64, device=device)
        a = 2.5
        b = -1.3

        with torch.no_grad():
            # Compute M(ax + by)
            lhs = net(a * x + b * y)

            # Compute aM(x) + bM(y)
            Mx = net(x)
            My = net(y)
            rhs = a * Mx + b * My

        # Check linearity
        error = torch.norm(lhs - rhs) / torch.norm(rhs)
        assert error < 1e-10, \
            f"LinearMGGNN is NOT linear! Error: {error:.2e}. " \
            "This breaks standard PCG convergence guarantees."

    def test_linear_mggnn_homogeneity(self, small_1d_laplacian_torch, device):
        """LinearMGGNN should satisfy M(cx) = cM(x) for any scalar c."""
        from GNP.nn import LinearMGGNN

        A = small_1d_laplacian_torch
        n = A.shape[0]

        net = LinearMGGNN(
            A=A,
            num_levels=3,
            embed=16,
            hidden=32,
            drop_rate=0.0,
            num_vcycles=1,
            smoother_K=2,
            coarsest_K=3
        ).to(device)
        net.eval()

        torch.manual_seed(123)
        x = torch.randn(n, dtype=torch.float64, device=device)

        for c in [0.5, 2.0, -1.0, 100.0]:
            with torch.no_grad():
                lhs = net(c * x)
                rhs = c * net(x)

            error = torch.norm(lhs - rhs) / (torch.norm(rhs) + 1e-12)
            assert error < 1e-10, f"Homogeneity failed for c={c}: error={error:.2e}"


class TestNetworkAliases:
    """Test that MGGNN and UNetGCN are aliases for LinearMGGNN."""

    def test_aliases_are_same_class(self):
        """MGGNN and UNetGCN should be aliases for LinearMGGNN."""
        from GNP.nn import LinearMGGNN, MGGNN, UNetGCN

        assert MGGNN is LinearMGGNN
        assert UNetGCN is LinearMGGNN


class TestCastToFloat64:
    """Test the cast_module_to_float64 utility."""

    def test_cast_to_float64(self):
        """cast_module_to_float64 should convert all params to float64."""
        from GNP.nn.layers import cast_module_to_float64
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

        # Initially float32
        assert model[0].weight.dtype == torch.float32

        cast_module_to_float64(model)

        # After casting, should be float64
        assert model[0].weight.dtype == torch.float64
        assert model[2].weight.dtype == torch.float64
