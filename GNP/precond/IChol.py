import torch
import numpy as np
from scipy import sparse

from .base import BasePreconditioner

class IChol(BasePreconditioner):
    def __init__(self, A, drop_tol=1e-3, shift=0.0):
        """
        Incomplete Cholesky Preconditioner tracking MATLAB's 'ict' implementation.

        GPU-native: stores L and LT as torch.sparse_csr_tensor on device so that
        forward/backward substitutions run entirely on GPU via
        torch.linalg.solve_triangular.

        Fallback: if the NATURAL-ordering ILU factorisation fails, we fall back
        to a guaranteed-SPD **Jacobi (diagonal) preconditioner** instead of a
        pivoted ILU (which would produce an asymmetric, invalid preconditioner
        that crashes PCG).

        Args:
            A: Sparse CSC system matrix.
            drop_tol: ILU drop tolerance.
            shift: Optional diagonal shift added as shift * I for stability.
        """
        self.device = A.device
        self.dtype = A.dtype
        self._is_jacobi = False

        if A.layout != torch.sparse_csc:
            raise Exception('To use IChol, A must be sparse csc')

        A_cpu = A.to('cpu')
        n = A_cpu.shape[0]
        spA = sparse.csc_matrix(
            (A_cpu.values().numpy(),
             A_cpu.row_indices().numpy(),
             A_cpu.ccol_indices().numpy()),
            shape=(n, n)
        )

        # Backward-compatible diagonal stabilization used by config/tests.
        shift = 0.0 if shift is None else float(shift)

        if shift != 0.0:
            spA = spA + shift * sparse.eye(n, format='csc', dtype=spA.dtype)

        # ---------------------------------------------------------
        # 1. Exact replication of MATLAB's opts.diagcomp
        # MATLAB: opts.diagcomp = max(sum(abs(A),2) ./ diag(A)) - 2;
        # ---------------------------------------------------------
        A_abs_row_sum = np.array(np.abs(spA).sum(axis=1)).flatten()
        A_diag = spA.diagonal()
        safe_diag = np.where(np.abs(A_diag) < 1e-12, 1e-12, np.abs(A_diag))
        alpha = np.max(A_abs_row_sum / safe_diag) - 2.0

        if alpha > 0:
            diag_matrix = sparse.diags(A_diag)
            spA = spA + (alpha * diag_matrix)

        # ---------------------------------------------------------
        # 2. Factorization (Mimicking 'ict')
        # ---------------------------------------------------------
        try:
            ilu = sparse.linalg.spilu(
                spA,
                drop_tol=drop_tol,
                fill_factor=10.0,
                diag_pivot_thresh=0.0,
                permc_spec='NATURAL'
            )
        except Exception as e:
            # Pivoted ILU produces U != D L^T  => asymmetric preconditioner
            # that is invalid for PCG.  Fall back to Jacobi (diagonal) instead.
            print(f"IChol Warning: NATURAL ordering failed ({e}). "
                  f"Falling back to Jacobi (diagonal) preconditioner.")
            diag_vals = spA.diagonal().copy()
            diag_vals = np.maximum(np.abs(diag_vals), 1e-12)
            self._jacobi_inv = torch.from_numpy(1.0 / diag_vals).to(
                dtype=self.dtype, device=self.device
            )
            self._is_jacobi = True
            return

        L_sp = ilu.L.tocsr()
        U_sp = ilu.U.tocsr()
        diag_U = U_sp.diagonal()

        if np.any(diag_U < 0):
            print("IChol Warning: Negative diagonals found in U. "
                  "Preconditioner is not strictly SPD.")

        diag_U = np.maximum(np.abs(diag_U), 1e-12)
        sqrt_diag = np.sqrt(diag_U)

        # L_ichol = L * diag(sqrt(|diag(U)|))  so that  L_ichol @ L_ichol^T ≈ A
        L_final = L_sp.multiply(sqrt_diag.reshape(1, -1)).tocsr()
        LT_final = L_final.T.tocsr()

        # ---------------------------------------------------------
        # 3. Convert SciPy CSR -> torch.sparse_csr_tensor on device
        #    so that apply() never leaves the GPU.
        # ---------------------------------------------------------
        self.L = self._scipy_csr_to_torch(L_final, n)
        self.LT = self._scipy_csr_to_torch(LT_final, n)

        # Dense triangular copies for torch.linalg.solve_triangular.
        # For matrices small enough to fit in GPU memory this is the
        # fastest path on A100.  For very large n we would need a
        # sparse triangular solve (cuSPARSE), but PyTorch doesn't
        # expose that yet — the dense path is still faster than the
        # CPU round-trip for matrices up to ~30k.
        self.L_dense = self.L.to_dense().to(dtype=self.dtype, device=self.device)
        self.LT_dense = self.LT.to_dense().to(dtype=self.dtype, device=self.device)

    def _scipy_csr_to_torch(self, M_csr, n):
        """Convert a SciPy CSR matrix to a torch.sparse_csr_tensor on self.device."""
        crow = torch.from_numpy(M_csr.indptr.astype(np.int64))
        col  = torch.from_numpy(M_csr.indices.astype(np.int64))
        vals = torch.from_numpy(M_csr.data.astype(np.float64))
        T = torch.sparse_csr_tensor(crow, col, vals, size=(n, n), dtype=self.dtype)
        return T.to(self.device)

    def apply(self, r):
        """
        Applies the preconditioner M^{-1} r = (L L^T)^{-1} r.

        All computation stays on the GPU device — no CPU round-trips.
        """
        if self._is_jacobi:
            return self._jacobi_inv * r

        # r may be 1-D; solve_triangular needs a 2-D column vector
        r_col = r.unsqueeze(-1) if r.dim() == 1 else r

        # Forward substitution:  L y = r
        y = torch.linalg.solve_triangular(self.L_dense, r_col, upper=False, unitriangular=False)
        # Backward substitution: L^T z = y
        z = torch.linalg.solve_triangular(self.LT_dense, y, upper=True, unitriangular=False)

        return z.squeeze(-1) if r.dim() == 1 else z
