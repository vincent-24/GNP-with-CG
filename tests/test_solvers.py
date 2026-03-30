"""Tests for iterative solvers."""

import pytest
import torch
import numpy as np


class TestPCG:
    """Tests for Preconditioned Conjugate Gradient solver."""

    def test_pcg_converges_on_1d_laplacian(self, small_1d_laplacian_torch, device):
        """PCG should converge on a simple SPD problem."""
        from GNP.solver import PCG

        A = small_1d_laplacian_torch
        n = A.shape[0]

        # Create a known solution and compute b = A @ x_true
        x_true = torch.ones(n, dtype=torch.float64, device=device)
        b = torch.sparse.mm(A.to_sparse_coo(), x_true.unsqueeze(1)).squeeze()

        solver = PCG()
        x, iters, hist_abs, hist_rel, hist_time, ortho_map = solver.solve(
            A, b, M=None, max_iters=200, rtol=1e-8, progress_bar=False
        )

        # Check convergence
        assert hist_rel[-1] < 1e-8, f"PCG did not converge: final rel_res = {hist_rel[-1]}"
        assert iters < 200, f"PCG took too many iterations: {iters}"

        # Check solution accuracy
        error = torch.norm(x - x_true) / torch.norm(x_true)
        assert error < 1e-6, f"Solution error too large: {error}"

    def test_pcg_with_jacobi_preconditioner(self, small_1d_laplacian_torch, device):
        """PCG with Jacobi preconditioner should converge faster."""
        from GNP.solver import PCG
        from GNP.precond import Jacobi

        A = small_1d_laplacian_torch
        n = A.shape[0]

        x_true = torch.randn(n, dtype=torch.float64, device=device)
        b = torch.sparse.mm(A.to_sparse_coo(), x_true.unsqueeze(1)).squeeze()

        # Solve without preconditioner
        solver = PCG()
        _, iters_no_precond, _, _, _, _ = solver.solve(
            A, b, M=None, max_iters=500, rtol=1e-8, progress_bar=False
        )

        # Solve with Jacobi preconditioner
        M = Jacobi(A)
        _, iters_jacobi, _, hist_rel, _, _ = solver.solve(
            A, b, M=M, max_iters=500, rtol=1e-8, progress_bar=False
        )

        assert hist_rel[-1] < 1e-8, "PCG with Jacobi did not converge"
        # Jacobi should help (or at least not hurt much) for 1D Laplacian
        assert iters_jacobi <= iters_no_precond + 5


class TestFCG:
    """Tests for Flexible Conjugate Gradient solver."""

    def test_fcg_converges_on_1d_laplacian(self, small_1d_laplacian_torch, device):
        """FCG should converge on a simple SPD problem."""
        from GNP.solver import FCG

        A = small_1d_laplacian_torch
        n = A.shape[0]

        x_true = torch.ones(n, dtype=torch.float64, device=device)
        b = torch.sparse.mm(A.to_sparse_coo(), x_true.unsqueeze(1)).squeeze()

        solver = FCG(truncation_k=None)  # Full orthogonalization
        x, iters, hist_abs, hist_rel, hist_time, ortho_map = solver.solve(
            A, b, M=None, max_iters=200, rtol=1e-8, progress_bar=False
        )

        assert hist_rel[-1] < 1e-8, f"FCG did not converge: final rel_res = {hist_rel[-1]}"

        error = torch.norm(x - x_true) / torch.norm(x_true)
        assert error < 1e-6, f"Solution error too large: {error}"

    def test_fcg_handles_truncation(self, small_1d_laplacian_torch, device):
        """FCG with truncation should still converge."""
        from GNP.solver import FCG

        A = small_1d_laplacian_torch
        n = A.shape[0]

        x_true = torch.ones(n, dtype=torch.float64, device=device)
        b = torch.sparse.mm(A.to_sparse_coo(), x_true.unsqueeze(1)).squeeze()

        solver = FCG(truncation_k=10)  # Truncated orthogonalization
        x, iters, _, hist_rel, _, _ = solver.solve(
            A, b, M=None, max_iters=300, rtol=1e-8, progress_bar=False
        )

        assert hist_rel[-1] < 1e-8, f"FCG with truncation did not converge"


class TestGMRES:
    """Tests for GMRES solver."""

    def test_gmres_converges(self, small_1d_laplacian_torch, device):
        """GMRES should converge on SPD problem."""
        from GNP.solver import GMRES

        A = small_1d_laplacian_torch
        n = A.shape[0]

        x_true = torch.ones(n, dtype=torch.float64, device=device)
        b = torch.sparse.mm(A.to_sparse_coo(), x_true.unsqueeze(1)).squeeze()

        solver = GMRES()
        x, iters, hist_abs, hist_rel, hist_time, ortho_map = solver.solve(
            A, b, M=None, restart=20, max_iters=200, rtol=1e-8, progress_bar=False
        )

        assert hist_rel[-1] < 1e-8, f"GMRES did not converge: final rel_res = {hist_rel[-1]}"


class TestBreakdownDetection:
    """Test numerical breakdown detection in solvers."""

    def test_pcg_handles_near_zero_denominator(self, device):
        """PCG should not crash on nearly singular systems."""
        from GNP.solver import PCG
        import numpy as np
        from scipy import sparse as sp

        # Create a nearly singular matrix (small eigenvalue)
        n = 50
        A_np = sp.diags([2.0] * n, 0, format='csr') + sp.diags([-1.0] * (n-1), 1, format='csr') + sp.diags([-1.0] * (n-1), -1, format='csr')
        A_np = A_np.astype(np.float64)

        A_csc = sp.csc_matrix(A_np)
        A = torch.sparse_csc_tensor(
            torch.from_numpy(A_csc.indptr.astype(np.int64)),
            torch.from_numpy(A_csc.indices.astype(np.int64)),
            torch.from_numpy(A_csc.data),
            size=A_csc.shape,
            dtype=torch.float64
        ).to(device)

        b = torch.randn(n, dtype=torch.float64, device=device)

        solver = PCG()
        # Should not raise an exception
        x, iters, _, _, _, _ = solver.solve(
            A, b, M=None, max_iters=100, rtol=1e-8, progress_bar=False
        )
        assert x is not None
