"""
LinearMGGNN: Linear Multigrid Graph Neural Network for PCG-compatible Preconditioning

This architecture is strictly LINEAR, making it compatible with Preconditioned
Conjugate Gradient (PCG) which requires:
    M^{-1}(alpha*x + beta*y) = alpha*M^{-1}(x) + beta*M^{-1}(y)

Key features:
- Polynomial smoothers with learnable scalar coefficients (no MLPs, no activations)
- Proper V-cycle structure with adjacent-level-only communication
- All operations are sparse matrix-vector products or scalar operations

Reference: Classical multigrid theory + GNP framework
"""

import math
import torch
import torch.nn as nn

from GNP.utils import build_multigrid_hierarchy, scale_A_by_spectral_radius


class PolynomialSmoother(nn.Module):
    """
    Linear polynomial smoother: z = sum_{k=0}^K theta_k * A^k * r

    This is provably LINEAR in r:
        smoother(alpha*r1 + beta*r2) = sum_k theta_k * A^k * (alpha*r1 + beta*r2)
                                     = alpha * sum_k theta_k * A^k * r1 + beta * sum_k theta_k * A^k * r2
                                     = alpha * smoother(r1) + beta * smoother(r2)

    Args:
        K: Polynomial degree (number of powers of A to use)
        init_style: Initialization style ('damped_jacobi' or 'chebyshev')
    """

    def __init__(self, K: int = 3, init_style: str = 'damped_jacobi'):
        super().__init__()
        self.K = K
        # Learnable scalar coefficients theta_0, theta_1, ..., theta_K
        self.theta = nn.Parameter(torch.zeros(K + 1, dtype=torch.float64))
        self._init_coefficients(init_style)

    def _init_coefficients(self, style: str):
        """Initialize smoother coefficients to approximate classical smoothers."""
        with torch.no_grad():
            if style == 'damped_jacobi':
                # Approximate damped Jacobi: z ~ omega * D^{-1} * r
                # For scaled A, this approximates to theta_1 * A * r contribution
                self.theta[0] = 0.0       # No constant term
                self.theta[1] = 0.6667    # omega ~ 2/3 (classical damping)
                for k in range(2, self.K + 1):
                    self.theta[k] = 0.1 / k  # Small higher-order corrections
            elif style == 'chebyshev':
                # Chebyshev-like initialization for coarse solver
                for k in range(self.K + 1):
                    self.theta[k] = 1.0 / (k + 1)
            else:
                # Identity-like: just pass through
                self.theta[0] = 1.0

    def forward(self, r: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Apply polynomial smoother: z = sum_{k=0}^K theta_k * A^k * r

        Args:
            r: Residual vector (n,) or (n, batch_size)
            A: System matrix (sparse COO, n x n)

        Returns:
            z: Smoothed approximation (n,) or (n, batch_size)
        """
        squeezed = r.dim() == 1
        if squeezed:
            r = r.unsqueeze(1)

        n, batch_size = r.shape

        # z = sum_{k=0}^K theta_k * A^k * r
        z = self.theta[0] * r  # k=0 term: theta_0 * I * r = theta_0 * r

        A_k_r = r  # A^0 * r = r
        for k in range(1, self.K + 1):
            # A^k * r = A * (A^{k-1} * r)
            A_k_r = torch.sparse.mm(A, A_k_r)
            z = z + self.theta[k] * A_k_r

        if squeezed:
            z = z.squeeze(1)

        return z


class LinearVCycle(nn.Module):
    """
    A single linear V-cycle with adjacent-level-only communication.

    V-cycle structure:
        Level 0:  Pre-smooth -> Restrict residual --->
        Level 1:  Pre-smooth -> Restrict residual --->
        ...
        Level L:  Coarse solve (polynomial smoother)
        ...
        Level 1:  <--- Prolong correction -> Post-smooth
        Level 0:  <--- Prolong correction -> Post-smooth

    All operations are LINEAR (sparse matmul + scalar operations).

    Args:
        hierarchy: MultigridHierarchy from build_multigrid_hierarchy
        level: Current level index (0 = finest)
        smoother_K: Polynomial degree for smoothers
        coarsest_K: Polynomial degree for coarsest level solver
        share_smoothers: If True, pre and post smoothers share weights (ensures symmetry)
    """

    def __init__(
        self,
        hierarchy,
        level: int,
        smoother_K: int = 3,
        coarsest_K: int = 5,
        share_smoothers: bool = True,
    ):
        super().__init__()
        self.level = level
        self.num_levels = hierarchy.num_levels
        self.share_smoothers = share_smoothers

        # Get level data
        level_data = hierarchy.levels[level]
        self.n_nodes = level_data.n_nodes

        # Store scaled A for this level (scale by spectral radius for stability)
        A_scaled = scale_A_by_spectral_radius(level_data.A)
        self.register_buffer('A', A_scaled.to(torch.float64))

        # Store original A for residual computation (unscaled)
        self.register_buffer('A_orig', level_data.A.to(torch.float64))

        # Store transfer operators if not at coarsest level
        if level_data.R is not None:
            self.register_buffer('R', level_data.R.to(torch.float64))
        else:
            self.R = None

        if level_data.P is not None:
            self.register_buffer('P', level_data.P.to(torch.float64))
        else:
            self.P = None

        # Pre-smoother
        self.pre_smoother = PolynomialSmoother(K=smoother_K, init_style='damped_jacobi')

        # Post-smoother (share with pre-smoother for symmetry, or separate)
        if share_smoothers:
            self.post_smoother = self.pre_smoother
        else:
            self.post_smoother = PolynomialSmoother(K=smoother_K, init_style='damped_jacobi')

        # Recursive coarse-level V-cycle (if not at coarsest level)
        if level < hierarchy.num_levels - 1:
            self.coarse_vcycle = LinearVCycle(
                hierarchy,
                level=level + 1,
                smoother_K=smoother_K,
                coarsest_K=coarsest_K,
                share_smoothers=share_smoothers,
            )
            self.coarse_solver = None
        else:
            # At coarsest level: use polynomial smoother as approximate direct solve
            self.coarse_vcycle = None
            self.coarse_solver = PolynomialSmoother(K=coarsest_K, init_style='chebyshev')

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply one V-cycle starting at this level.

        The V-cycle computes z ~ A^{-1} @ r through:
        1. Pre-smoothing
        2. Restrict residual to coarse level
        3. Solve/recurse on coarse level
        4. Prolong correction back
        5. Post-smoothing

        Args:
            r: Residual vector (n_level,) or (n_level, batch_size)

        Returns:
            z: Approximate solution to A @ z = r
        """
        squeezed = r.dim() == 1
        if squeezed:
            r = r.unsqueeze(1)

        r = r.to(torch.float64)

        # 1. Pre-smoothing: z = S_pre(r, A)
        z = self.pre_smoother(r, self.A)

        # 2. Compute residual after pre-smoothing: r_smooth = r - A @ z
        Az = torch.sparse.mm(self.A_orig, z)
        r_smooth = r - Az

        if self.coarse_vcycle is not None and self.R is not None:
            # 3. Restrict residual to coarse level: r_c = R @ r_smooth
            r_coarse = torch.sparse.mm(self.R, r_smooth)

            # 4. Recursive solve on coarse level
            z_coarse = self.coarse_vcycle(r_coarse)

            # 5. Prolong correction back: z += P @ z_c
            Pz_coarse = torch.sparse.mm(self.P, z_coarse)
            z = z + Pz_coarse

            # Recompute residual after coarse correction
            Az = torch.sparse.mm(self.A_orig, z)
            r_smooth = r - Az
        elif self.coarse_solver is not None:
            # At coarsest level: apply polynomial smoother
            z_coarse_correction = self.coarse_solver(r_smooth, self.A)
            z = z + z_coarse_correction

            # Recompute residual
            Az = torch.sparse.mm(self.A_orig, z)
            r_smooth = r - Az

        # 6. Post-smoothing: z += S_post(r_smooth, A)
        z_post = self.post_smoother(r_smooth, self.A)
        z = z + z_post

        if squeezed:
            z = z.squeeze(1)

        return z


class LinearMGGNN(nn.Module):
    """
    Linear Multigrid Graph Neural Network for PCG-compatible preconditioning.

    This network is STRICTLY LINEAR, satisfying:
        M^{-1}(alpha*x + beta*y) = alpha*M^{-1}(x) + beta*M^{-1}(y)

    This makes it compatible with PCG which requires a linear preconditioner.

    Architecture:
        - Polynomial smoothers with learnable scalar coefficients
        - V-cycle structure with adjacent-level communication only
        - Optional stacking of multiple V-cycles

    Training objective: minimize rho(I - M^{-1}A) via spectral radius estimation.

    Args:
        A: System matrix (torch sparse tensor or scipy sparse)
        num_layers: Unused (API compatibility with other networks)
        embed: Unused (API compatibility)
        hidden: Unused (API compatibility)
        drop_rate: Unused (API compatibility)
        num_levels: Number of multigrid levels (None = auto)
        num_vcycles: Number of V-cycles to stack
        smoother_K: Polynomial degree for pre/post smoothers
        coarsest_K: Polynomial degree for coarsest level solver
        share_smoothers: Share pre/post smoother weights (ensures symmetry)
        dtype: Torch dtype for computations
    """

    def __init__(
        self,
        A,
        num_layers: int = 4,       # Unused, API compatibility
        embed: int = 32,           # Unused, API compatibility
        hidden: int = 64,          # Unused, API compatibility
        drop_rate: float = 0.0,    # Unused, API compatibility
        num_levels: int = None,
        num_vcycles: int = 2,
        smoother_K: int = 3,
        coarsest_K: int = 5,
        share_smoothers: bool = True,
        dtype: torch.dtype = torch.float32,  # Note: internally uses float64
    ):
        super().__init__()
        self.dtype = torch.float64  # Always use float64 for numerical stability
        self.num_vcycles = num_vcycles

        # Determine device from input matrix
        if torch.is_tensor(A):
            device = A.device
        else:
            device = torch.device('cpu')

        # Auto-determine number of levels
        n = A.shape[0]
        if num_levels is None:
            num_levels = min(8, max(2, int(math.ceil(math.log2(n))) - 3))

        # Build multigrid hierarchy using existing infrastructure
        self.hierarchy = build_multigrid_hierarchy(
            A,
            num_levels=num_levels,
            coarsening_ratio=0.5,
            min_nodes=max(10, n // (2 ** num_levels)),
            dtype=torch.float64,
            device=device,
        )
        self.num_levels = self.hierarchy.num_levels

        # Store level sizes for logging
        self.level_sizes = [level.n_nodes for level in self.hierarchy.levels]

        # Store finest level A for multi-vcycle residual computation
        A_fine = self.hierarchy.levels[0].A
        self.register_buffer('A_fine', A_fine.to(torch.float64))

        # Build V-cycle(s)
        self.vcycles = nn.ModuleList([
            LinearVCycle(
                self.hierarchy,
                level=0,  # Start at finest level
                smoother_K=smoother_K,
                coarsest_K=coarsest_K,
                share_smoothers=share_smoothers,
            )
            for _ in range(num_vcycles)
        ])

        # Learnable output scale (linear operation)
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        # Auxiliary losses storage (API compatibility)
        self._aux_losses = {}

        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())

        print(f"[LinearMGGNN] Created with:")
        print(f"  Levels: {self.num_levels} | Sizes: {self.level_sizes}")
        print(f"  V-cycles: {num_vcycles} | Smoother K: {smoother_K} | Coarsest K: {coarsest_K}")
        print(f"  Share smoothers: {share_smoothers}")
        print(f"  Total parameters: {num_params}")

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: z = M^{-1}(r)

        Applies num_vcycles V-cycles. Each V-cycle is linear,
        so composition is also linear.

        Args:
            r: Residual vector (n,) or (n, batch_size)

        Returns:
            z: Preconditioned residual
        """
        squeezed = r.dim() == 1
        if squeezed:
            r = r.unsqueeze(1)

        r = r.to(self.dtype)

        # Apply V-cycles
        if self.num_vcycles == 1:
            # Single V-cycle: just apply it
            z = self.vcycles[0](r)
        else:
            # Multiple V-cycles: use residual correction form
            z = torch.zeros_like(r)
            current_r = r

            for vcycle in self.vcycles:
                # Each V-cycle computes a correction to the current residual
                correction = vcycle(current_r)
                z = z + correction

                # Update residual for next V-cycle
                Az = torch.sparse.mm(self.A_fine, z)
                current_r = r - Az

        # Apply learnable output scaling (still linear!)
        z = self.output_scale * z

        if squeezed:
            z = z.squeeze(1)

        return z

    def get_aux_losses(self) -> dict:
        """Return auxiliary losses for compatibility with training infrastructure."""
        return self._aux_losses


# Alias for backwards compatibility / config reference
MGGNN = LinearMGGNN
UNetGCN = LinearMGGNN
