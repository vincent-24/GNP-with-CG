"""
ICholSparseTensorNet: IChol-guided sparse-tensor architecture.

This module keeps the strong non-parametric structure of incomplete Cholesky
while exposing a tiny trainable calibration layer:

    z = gain * (alpha * z_ichol + (1 - alpha) * r)

where z_ichol = M_ichol^{-1} r is computed from frozen IChol factors.

The heavy lifting (factorization + triangular solves) is non-parametric and
matrix-structured; only alpha/gain are learned scalars so this can be trained
through the existing GNP pipeline without changing solver contracts.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

from GNP.precond.IChol import IChol


class ICholSparseTensorNet(nn.Module):
    """
    IChol-inspired network with a frozen sparse-tensor core.

    Args:
        A: System matrix.
        num_layers/embed/hidden/drop_rate: Unused, kept for API compatibility.
        drop_tol: IChol drop tolerance.
        shift: Diagonal shift for IChol stability.
        init_alpha: Initial blend between IChol output and identity path.
        init_gain: Initial global gain.
        min_alpha: Lower bound for alpha to keep IChol contribution dominant.
        gain_delta: Gain is constrained to [1-gain_delta, 1+gain_delta].
        dtype: Exposed for API compatibility; computations use float64.
    """

    def __init__(
        self,
        A,
        num_layers: int = 4,
        embed: int = 32,
        hidden: int = 64,
        drop_rate: float = 0.0,
        drop_tol: float = 1e-3,
        shift: float = 1e-3,
        init_alpha: float = 0.95,
        init_gain: float = 1.0,
        min_alpha: float = 0.9,
        gain_delta: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.dtype = torch.float64
        self.drop_tol = float(drop_tol)
        self.shift = float(shift)
        self.min_alpha = float(min(max(min_alpha, 0.0), 0.999))
        self.gain_delta = float(max(gain_delta, 1e-6))

        A_csc = self._to_sparse_csc(A, dtype=self.dtype)
        self.n = A_csc.shape[0]

        ichol = IChol(A_csc, drop_tol=self.drop_tol, shift=self.shift)
        self._is_jacobi = bool(getattr(ichol, '_is_jacobi', False))

        device = A_csc.device

        # Store factor tensors as non-persistent buffers so checkpoints remain
        # lightweight and only include learned calibration scalars.
        if self._is_jacobi:
            self.register_buffer(
                'jacobi_inv',
                ichol._jacobi_inv.detach().clone().to(dtype=self.dtype, device=device),
                persistent=False,
            )
            self.register_buffer(
                'L_dense',
                torch.empty(0, dtype=self.dtype, device=device),
                persistent=False,
            )
            self.register_buffer(
                'LT_dense',
                torch.empty(0, dtype=self.dtype, device=device),
                persistent=False,
            )
        else:
            self.register_buffer(
                'L_dense',
                ichol.L_dense.detach().clone().to(dtype=self.dtype, device=device),
                persistent=False,
            )
            self.register_buffer(
                'LT_dense',
                ichol.LT_dense.detach().clone().to(dtype=self.dtype, device=device),
                persistent=False,
            )
            self.register_buffer(
                'jacobi_inv',
                torch.empty(0, dtype=self.dtype, device=device),
                persistent=False,
            )

        init_alpha = float(min(max(init_alpha, self.min_alpha + 1e-4), 1.0 - 1e-4))
        alpha_scaled = (init_alpha - self.min_alpha) / (1.0 - self.min_alpha)
        alpha_scaled = float(min(max(alpha_scaled, 1e-4), 1.0 - 1e-4))
        mix_logit = math.log(alpha_scaled / (1.0 - alpha_scaled))

        init_gain = float(min(max(init_gain, 1.0 - self.gain_delta + 1e-6), 1.0 + self.gain_delta - 1e-6))
        gain_scaled = (init_gain - 1.0) / self.gain_delta
        gain_scaled = float(min(max(gain_scaled, -0.999), 0.999))
        gain_raw = 0.5 * math.log((1.0 + gain_scaled) / (1.0 - gain_scaled))

        self.mix_logit = nn.Parameter(torch.tensor(mix_logit, dtype=self.dtype))
        self.gain_raw = nn.Parameter(torch.tensor(gain_raw, dtype=self.dtype))

        print('[ICholSparseTensorNet] Created with frozen IChol sparse-tensor core')
        print(f'  n={self.n}, drop_tol={self.drop_tol}, shift={self.shift}')
        print(f'  Fallback mode: {"Jacobi" if self._is_jacobi else "IChol triangular solves"}')
        print(f'  Alpha range: [{self.min_alpha:.3f}, 1.000], Gain range: [{1.0-self.gain_delta:.3f}, {1.0+self.gain_delta:.3f}]')
        print(f'  Trainable scalars: {sum(p.numel() for p in self.parameters())}')

    def _to_sparse_csc(self, A, dtype: torch.dtype) -> torch.Tensor:
        """Convert matrix-like input to torch sparse CSC tensor."""
        if torch.is_tensor(A):
            A = A.to(dtype)

            if A.layout == torch.sparse_csc:
                return A

            if A.layout in (torch.sparse_coo, torch.sparse_csr):
                return A.to_sparse_csc()

            if A.layout == torch.strided:
                return A.to_sparse_csc()

            raise TypeError(f'Unsupported torch layout for A: {A.layout}')

        if sparse.issparse(A):
            A_csc = A.tocsc()
            ccol = torch.from_numpy(A_csc.indptr.astype(np.int64))
            row = torch.from_numpy(A_csc.indices.astype(np.int64))
            vals = torch.from_numpy(A_csc.data).to(dtype)
            return torch.sparse_csc_tensor(ccol, row, vals, size=A_csc.shape, dtype=dtype)

        A_dense = torch.tensor(A, dtype=dtype)
        return A_dense.to_sparse_csc()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply calibrated IChol action to residual(s)."""
        squeezed = r.dim() == 1

        if squeezed:
            r = r.unsqueeze(1)
        elif r.dim() != 2:
            raise ValueError(f'Expected 1D or 2D input, got shape {tuple(r.shape)}')

        r = r.to(self.dtype)

        if self._is_jacobi:
            z_ichol = self.jacobi_inv.unsqueeze(1) * r
        else:
            y = torch.linalg.solve_triangular(
                self.L_dense,
                r,
                upper=False,
                unitriangular=False,
            )
            z_ichol = torch.linalg.solve_triangular(
                self.LT_dense,
                y,
                upper=True,
                unitriangular=False,
            )

        alpha = self.min_alpha + (1.0 - self.min_alpha) * torch.sigmoid(self.mix_logit)
        gain = 1.0 + self.gain_delta * torch.tanh(self.gain_raw)
        z = gain * (alpha * z_ichol + (1.0 - alpha) * r)

        if squeezed:
            z = z.squeeze(1)

        return z

    def get_aux_losses(self) -> dict:
        """Compatibility hook for training infrastructure."""
        return {}
