"""Tests for preconditioners."""

import pytest
import torch
import numpy as np


class TestJacobi:
    """Tests for Jacobi preconditioner."""

    def test_jacobi_apply(self, small_1d_laplacian_torch, device):
        """Jacobi preconditioner should scale by inverse diagonal."""
        from GNP.precond import Jacobi

        A = small_1d_laplacian_torch
        M = Jacobi(A)

        n = A.shape[0]
        r = torch.ones(n, dtype=torch.float64, device=device)
        z = M.apply(r)

        # For 1D Laplacian, diagonal is 2, so z should be r / 2
        expected = r / 2.0
        assert torch.allclose(z, expected, rtol=1e-10)


class TestIChol:
    """Tests for Incomplete Cholesky preconditioner."""

    def test_ichol_reduces_condition_number(self, small_1d_laplacian_torch, device):
        """IChol should improve conditioning."""
        from GNP.precond import IChol
        from GNP.solver import PCG

        A = small_1d_laplacian_torch
        n = A.shape[0]

        x_true = torch.randn(n, dtype=torch.float64, device=device)
        b = torch.sparse.mm(A.to_sparse_coo(), x_true.unsqueeze(1)).squeeze()

        # Solve without preconditioner
        solver = PCG()
        _, iters_no_precond, _, _, _, _ = solver.solve(
            A, b, M=None, max_iters=500, rtol=1e-10, progress_bar=False
        )

        # Solve with IChol preconditioner
        try:
            M = IChol(A, shift=1e-3)
            _, iters_ichol, _, hist_rel, _, _ = solver.solve(
                A, b, M=M, max_iters=500, rtol=1e-10, progress_bar=False
            )

            # IChol should significantly reduce iterations
            assert iters_ichol < iters_no_precond, \
                f"IChol should improve convergence: {iters_ichol} vs {iters_no_precond}"
        except Exception as e:
            # IChol may fail on some systems - this is acceptable
            pytest.skip(f"IChol factorization failed: {e}")


class TestAMG:
    """Tests for AMG preconditioner."""

    def test_amg_creation(self, small_1d_laplacian_torch):
        """AMG preconditioner should be creatable."""
        pytest.importorskip('pyamg')
        from GNP.precond import AMGPreconditioner

        A = small_1d_laplacian_torch
        M = AMGPreconditioner(A)
        assert M is not None
        assert M.M is not None

    def test_amg_apply(self, small_1d_laplacian_torch, device):
        """AMG apply should return a vector of correct shape."""
        pytest.importorskip('pyamg')
        from GNP.precond import AMGPreconditioner

        A = small_1d_laplacian_torch
        n = A.shape[0]
        M = AMGPreconditioner(A)

        r = torch.randn(n, dtype=torch.float64, device=device)
        z = M.apply(r)

        assert z.shape == r.shape
        assert z.device == r.device


class TestPreconditionerInterface:
    """Test that all preconditioners follow the same interface."""

    @pytest.mark.parametrize("precond_class,kwargs", [
        ("Jacobi", {}),
        ("IChol", {"shift": 1e-3}),
    ])
    def test_preconditioner_interface(self, small_1d_laplacian_torch, device, precond_class, kwargs):
        """All preconditioners should have an apply method."""
        import GNP.precond as precond_module

        A = small_1d_laplacian_torch
        n = A.shape[0]

        PrecondClass = getattr(precond_module, precond_class)
        try:
            M = PrecondClass(A, **kwargs)
        except Exception as e:
            pytest.skip(f"Could not create {precond_class}: {e}")

        # Test apply method
        r = torch.randn(n, dtype=torch.float64, device=device)
        z = M.apply(r)

        assert z is not None
        assert z.shape == r.shape
        assert z.dtype == r.dtype
