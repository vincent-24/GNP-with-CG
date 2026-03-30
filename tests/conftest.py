"""Pytest fixtures for GNP tests."""

import pytest
import torch
import numpy as np
from scipy import sparse


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_1d_laplacian():
    """Create a small 1D Laplacian matrix for testing (n=100).

    Returns scipy sparse CSR matrix.
    """
    n = 100
    diagonals = [
        -1 * np.ones(n - 1),
        2 * np.ones(n),
        -1 * np.ones(n - 1)
    ]
    A = sparse.diags(diagonals, [-1, 0, 1], format='csr')
    return A.astype(np.float64)


@pytest.fixture
def small_1d_laplacian_torch(small_1d_laplacian, device):
    """Convert 1D Laplacian to torch sparse CSC tensor."""
    A_csc = sparse.csc_matrix(small_1d_laplacian)
    A_torch = torch.sparse_csc_tensor(
        torch.from_numpy(A_csc.indptr.astype(np.int64)),
        torch.from_numpy(A_csc.indices.astype(np.int64)),
        torch.from_numpy(A_csc.data),
        size=A_csc.shape,
        dtype=torch.float64
    ).to(device)
    return A_torch


@pytest.fixture
def medium_1d_laplacian():
    """Create a medium 1D Laplacian matrix for testing (n=500)."""
    n = 500
    diagonals = [
        -1 * np.ones(n - 1),
        2 * np.ones(n),
        -1 * np.ones(n - 1)
    ]
    A = sparse.diags(diagonals, [-1, 0, 1], format='csr')
    return A.astype(np.float64)


@pytest.fixture
def medium_1d_laplacian_torch(medium_1d_laplacian, device):
    """Convert medium 1D Laplacian to torch sparse CSC tensor."""
    A_csc = sparse.csc_matrix(medium_1d_laplacian)
    A_torch = torch.sparse_csc_tensor(
        torch.from_numpy(A_csc.indptr.astype(np.int64)),
        torch.from_numpy(A_csc.indices.astype(np.int64)),
        torch.from_numpy(A_csc.data),
        size=A_csc.shape,
        dtype=torch.float64
    ).to(device)
    return A_torch


@pytest.fixture
def random_rhs(small_1d_laplacian_torch):
    """Create a random right-hand side vector."""
    n = small_1d_laplacian_torch.shape[0]
    torch.manual_seed(42)
    b = torch.randn(n, dtype=torch.float64, device=small_1d_laplacian_torch.device)
    return b
