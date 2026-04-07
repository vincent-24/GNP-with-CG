"""Spectral loss functions for neural preconditioner training.

These losses measure the quality of a preconditioner M^{-1} for a SPD system A
and are used as training objectives for the neural network.

Both functions accept:
    A       - sparse system matrix (n, n) on device
    M_inv   - callable (n, k) -> (n, k) that applies M^{-1}
    n       - system dimension
"""
import math
import torch

def _spmm(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse-or-dense matrix-vector product."""
    if A.is_sparse or A.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc):
        return torch.sparse.mm(A, x)

    return torch.matmul(A, x)

def spectral_radius_loss(A: torch.Tensor, M_inv: callable, n: int, num_vectors: int = 32, power_iters: int = 10) -> torch.Tensor:
    """Estimate rho(I - M^{-1}A) via power iteration.

    This is the per-iteration convergence rate of stationary Richardson
    iteration.  Minimising it also reduces kappa(M^{-1}A) (valid but
    conservative proxy for PCG).

    Only the **last** power iteration builds a computational graph;
    all preceding iterations run under ``torch.no_grad()`` so that
    VRAM usage equals a single forward pass regardless of *power_iters*.

    Returns
        rho : torch.Tensor  (scalar, differentiable)
    """
    v = torch.randn(n, num_vectors, dtype=A.dtype, device=A.device)
    v = v / torch.linalg.norm(v, dim=0, keepdim=True).clamp(min=1e-12)

    with torch.no_grad():
        for _ in range(power_iters - 1):
            Av = _spmm(A, v)
            Ev = v - M_inv(Av)
            v = Ev / torch.linalg.norm(Ev, dim=0, keepdim=True).clamp(min=1e-12)

    # Final iteration WITH gradients
    Av = _spmm(A, v)
    Ev = v - M_inv(Av)

    rho_estimates = torch.linalg.norm(Ev, dim=0)  # (num_vectors,)

    return rho_estimates.mean()

def condition_number_loss( A: torch.Tensor, M_inv: callable, n: int, num_vectors: int = 32, power_iters: int = 10) -> torch.Tensor:
    """Estimate log kappa(M^{-1}A) via two-phase power iteration.

    Phase 1: power iteration on M^{-1}A  -> lambda_max  (Rayleigh quotient).
    Phase 2: power iteration on (lam_hat I - M^{-1}A)  -> lambda_min
             (the dominant eigenvector of the shifted operator is the
              eigenvector of M^{-1}A for its smallest eigenvalue).

    The loss is log(kappa) = log(lambda_max) - log(lambda_min), which is
    scale-invariant: it penalises eigenvalue *spread*, not distance from 1.
    This is the directly relevant quantity for PCG convergence:
        ||e_k||_A <= 2 ((sqrt(kappa)-1)/(sqrt(kappa)+1))^k ||e_0||_A

    Cost: ~2x the M^{-1} evaluations of ``spectral_radius_loss``.

    Returns
        log_kappa : torch.Tensor  (scalar, differentiable)
    """
    dtype, device = A.dtype, A.device

    # === Phase 1: lambda_max via power iteration on B = M^{-1}A ===
    v = torch.randn(n, num_vectors, dtype=dtype, device=device)
    v = v / torch.linalg.norm(v, dim=0, keepdim=True).clamp(min=1e-12)

    with torch.no_grad():
        for _ in range(power_iters - 1):
            v = M_inv(_spmm(A, v))
            v = v / torch.linalg.norm(v, dim=0, keepdim=True).clamp(min=1e-12)

    # Final iteration WITH gradients — Rayleigh quotient
    Bv = M_inv(_spmm(A, v))
    lam_max_vec = torch.sum(v * Bv, dim=0)               # (num_vectors,)
    lam_max = lam_max_vec.max()

    # === Phase 2: lambda_min via shifted power iteration ===
    lam_hat = lam_max.detach() * 1.1                       # 10 % safety margin

    w = torch.randn(n, num_vectors, dtype=dtype, device=device)
    w = w / torch.linalg.norm(w, dim=0, keepdim=True).clamp(min=1e-12)

    with torch.no_grad():
        for _ in range(power_iters - 1):
            Cw = lam_hat * w - M_inv(_spmm(A, w))
            w = Cw / torch.linalg.norm(Cw, dim=0, keepdim=True).clamp(min=1e-12)

    # Final iteration WITH gradients — Rayleigh quotient of M^{-1}A at w
    Bw = M_inv(_spmm(A, w))
    lam_min_vec = torch.sum(w * Bw, dim=0)                # (num_vectors,)
    lam_min = lam_min_vec.min()

    # Safety clamps
    lam_min = lam_min.clamp(min=1e-15)
    lam_max = lam_max.clamp(min=lam_min.detach() + 1e-15)

    return torch.log(lam_max) - torch.log(lam_min)
