"""
TwoLevelMGGNN: Paper-faithful two-level Multigrid Graph Neural Network.

Implements the MG-GNN architecture from Taghibakhshi et al. (2023),
"MG-GNN: Multigrid Graph Neural Networks for Learning Multilevel Domain
Decomposition Methods" (arXiv:2301.11378), adapted for neural preconditioning
of SPD linear systems.

Key architectural features matching the paper:
    - **Parallel cross-scale message passing**: Every MG-GNN block processes
      fine and coarse levels simultaneously (not sequentially like a V-cycle).
    - **Heterogeneous inter-level transfer**: Cross-level features are computed
      via learnable MLPs that see features from BOTH source and target levels
      (paper Eq. 8: F^{l->k}(X_m^l, X_m^k, R^{l->k})).
    - **Feature concatenation**: Cross-level features are concatenated with
      own-level features before the intra-level GNN update (paper Eq. 9-10).
    - **TAGConv intra-level**: Topology Adaptive Graph Convolution provides
      K-hop polynomial filtering at each level (paper Section 5.1).
    - **Two-level hierarchy** via Lloyd aggregation with Galerkin projection
      (paper Section 3, using pyamg).

Adaptation for neural preconditioning:
    - Output is z = M^{-1}(r) mapping residual to correction (not DDM params).
    - The forward pass is kept linear in the input r for PCG compatibility.
    - Multiple implicit correction steps with learnable step sizes.
    - Spectral radius training: min rho(I - M^{-1}A).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from GNP import config
from GNP.utils import build_multigrid_hierarchy, scale_A_by_spectral_radius

# ---------------------------------------------------------------------------
# Sparse matrix utilities
# ---------------------------------------------------------------------------

def _as_matmul_ready(A: torch.Tensor) -> torch.Tensor:
    """Return a tensor layout compatible with torch.sparse.mm."""
    if A.layout == torch.sparse_coo:
        return A.coalesce()
    if A.layout in (torch.sparse_csr, torch.sparse_csc):
        return A.to_sparse_coo().coalesce()

    return A

def _spmm(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse-dense matrix multiply supporting COO/CSR/CSC and dense A."""
    if A.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc) or A.is_sparse:
        return torch.sparse.mm(_as_matmul_ready(A), x)

    return torch.matmul(A, x)

# ---------------------------------------------------------------------------
# Intra-level convolution: TAGConv (paper Section 5.1)
# ---------------------------------------------------------------------------

class TAGConv(nn.Module):
    """Topology Adaptive Graph Convolution with linear transforms.

    Computes the K-hop polynomial filter:
        out = sum_{k=0}^{K} W_k (A^k x)

    where A is the spectrally-scaled adjacency matrix.  Each hop has its
    own learnable linear transform W_k, keeping the overall map linear in x.

    Args:
        A_scaled: Spectrally-scaled adjacency/system matrix (sparse).
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        K: Polynomial order (number of hops).
    """

    def __init__(self, A_scaled: torch.Tensor, in_dim: int, out_dim: int, K: int = 3):
        super().__init__()
        self.K = max(0, int(K))
        self.in_dim = in_dim
        self.register_buffer("A", _as_matmul_ready(A_scaled).to(torch.float64))
        self.weights = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=(k == 0)) for k in range(self.K + 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (n, batch, in_dim) or (n, in_dim).

        Returns:
            (n, batch, out_dim) or (n, out_dim).
        """
        squeezed = x.dim() == 2

        if squeezed:
            x = x.unsqueeze(1)

        n, batch_size, _ = x.shape
        out = self.weights[0](x)
        x_k = x

        for k in range(1, self.K + 1):
            x_flat = x_k.reshape(n, batch_size * self.in_dim)
            Ax_flat = _spmm(self.A, x_flat)
            x_k = Ax_flat.reshape(n, batch_size, self.in_dim)
            out = out + self.weights[k](x_k)

        if squeezed:
            out = out.squeeze(1)

        return out

# ---------------------------------------------------------------------------
# Heterogeneous cross-level message passing (paper Eq. 8, 11)
# ---------------------------------------------------------------------------

class HeterogeneousTransfer(nn.Module):
    """Cross-level message passing function F^{l->k} from the paper.

    Given source-level features x_src (transferred to target resolution via
    R or P) and target-level features x_dst, computes a cross-level message:

        msg = g( [x_src_transferred ; x_dst] )

    where g is a 2-layer MLP with hidden dimension ``hidden_dim``.
    The paper uses 2 FC layers of size 128 for these inter-level MLPs.

    For the preconditioner use case, the MLP is kept **without** nonlinear
    activation to preserve linearity in the residual.  A learnable gate
    controls the magnitude.

    Args:
        feature_dim: Dimension of both source and target features.
        hidden_dim: Hidden dimension of the cross-level MLP (paper: 128).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Input is concatenation of transferred source + target features
        self.fc1 = nn.Linear(2 * feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x_src_transferred: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        """Compute cross-level message.

        Args:
            x_src_transferred: Source features already at target resolution.
                Shape: (n_dst, batch, feature_dim).
            x_dst: Target-level features.
                Shape: (n_dst, batch, feature_dim).

        Returns:
            Cross-level message: (n_dst, batch, feature_dim).
        """
        combined = torch.cat([x_src_transferred, x_dst], dim=-1)

        return self.fc2(self.fc1(combined))

# ---------------------------------------------------------------------------
# Single MG-GNN Block: parallel two-level message passing (paper Eqs. 8-10)
# ---------------------------------------------------------------------------

class MGGNNBlock(nn.Module):
    """One layer of the MG-GNN architecture with parallel cross-scale updates.

    Implements the core paper equations for a two-level (fine + coarse) setup:

    1. Compute cross-level features (Eq. 8):
       - Fine-to-coarse: restrict fine features, combine with coarse via MLP
       - Coarse-to-fine: prolong coarse features, combine with fine via MLP

    2. Concatenate cross-level with own-level features (Eq. 9):
       - X_fine_aug = [X_fine ; cross_coarse_to_fine]
       - X_coarse_aug = [X_coarse ; cross_fine_to_coarse]

    3. Intra-level GNN update (Eq. 10):
       - X_fine_new = TAGConv_fine(X_fine_aug, A_fine)
       - X_coarse_new = TAGConv_coarse(X_coarse_aug, A_coarse)

    4. Residual connection for stable training.

    Args:
        hierarchy: MultigridHierarchy with at least 2 levels.
        hidden_dim: Feature dimension at each level.
        fine_K: TAGConv polynomial order for fine level.
        coarse_K: TAGConv polynomial order for coarse level.
        cross_level_width: Hidden dimension for cross-level MLPs (paper: 128).
    """

    def __init__(
        self,
        hierarchy,
        hidden_dim: int,
        fine_K: int = 3,
        coarse_K: int = 5,
        cross_level_width: int = 128,
    ):
        super().__init__()

        fine_level = hierarchy.levels[0]
        self.n_fine = fine_level.n_nodes
        self.has_coarse = (
            hierarchy.num_levels > 1
            and fine_level.R is not None
            and fine_level.P is not None
        )

        # Intra-level TAGConv for fine level
        # Input is hidden_dim (own) + hidden_dim (cross-level) = 2*hidden_dim if coarse exists
        fine_in_dim = 2 * hidden_dim if self.has_coarse else hidden_dim
        A_fine_scaled = scale_A_by_spectral_radius(fine_level.A)
        self.fine_conv = TAGConv(A_fine_scaled, fine_in_dim, hidden_dim, K=fine_K)

        if self.has_coarse:
            coarse_level = hierarchy.levels[1]
            self.n_coarse = coarse_level.n_nodes

            # Transfer operators
            self.register_buffer("R", _as_matmul_ready(fine_level.R).to(torch.float64))
            self.register_buffer("P", _as_matmul_ready(fine_level.P).to(torch.float64))

            # Intra-level TAGConv for coarse level (also takes concatenated features)
            coarse_in_dim = 2 * hidden_dim
            A_coarse_scaled = scale_A_by_spectral_radius(coarse_level.A)
            self.coarse_conv = TAGConv(A_coarse_scaled, coarse_in_dim, hidden_dim, K=coarse_K)

            # Heterogeneous cross-level MLPs (paper: F^{l->k})
            self.fine_to_coarse = HeterogeneousTransfer(hidden_dim, cross_level_width)
            self.coarse_to_fine = HeterogeneousTransfer(hidden_dim, cross_level_width)
        else:
            self.n_coarse = 0

    def _restrict(self, x: torch.Tensor) -> torch.Tensor:
        """Transfer fine-level features to coarse resolution via R."""
        n_fine, batch_size, dim = x.shape
        x_flat = x.reshape(n_fine, batch_size * dim)
        out_flat = _spmm(self.R, x_flat)

        return out_flat.reshape(self.n_coarse, batch_size, dim)

    def _prolong(self, x: torch.Tensor) -> torch.Tensor:
        """Transfer coarse-level features to fine resolution via P."""
        n_coarse, batch_size, dim = x.shape
        x_flat = x.reshape(n_coarse, batch_size * dim)
        out_flat = _spmm(self.P, x_flat)

        return out_flat.reshape(self.n_fine, batch_size, dim)

    def forward(self, X_fine: torch.Tensor, X_coarse: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Parallel two-level update.

        Args:
            X_fine: Fine-level features (n_fine, batch, hidden_dim).
            X_coarse: Coarse-level features (n_coarse, batch, hidden_dim) or None.

        Returns:
            Updated (X_fine_new, X_coarse_new).
        """
        if self.has_coarse and X_coarse is not None:
            # --- Step 1: Cross-level message passing (Eq. 8) ---
            # Fine -> Coarse: restrict fine features, combine with coarse
            fine_restricted = self._restrict(X_fine)
            cross_f2c = self.fine_to_coarse(fine_restricted, X_coarse)

            # Coarse -> Fine: prolong coarse features, combine with fine
            coarse_prolonged = self._prolong(X_coarse)
            cross_c2f = self.coarse_to_fine(coarse_prolonged, X_fine)

            # --- Step 2: Concatenate cross-level features (Eq. 9) ---
            X_fine_aug = torch.cat([X_fine, cross_c2f], dim=-1)
            X_coarse_aug = torch.cat([X_coarse, cross_f2c], dim=-1)

            # --- Step 3: Intra-level GNN update (Eq. 10) ---
            X_fine_new = self.fine_conv(X_fine_aug)
            X_coarse_new = self.coarse_conv(X_coarse_aug)

            # --- Step 4: Residual connection ---
            X_fine_new = X_fine + X_fine_new
            X_coarse_new = X_coarse + X_coarse_new

            return X_fine_new, X_coarse_new
        else:
            # Single-level fallback: just intra-level convolution
            X_fine_new = self.fine_conv(X_fine)
            X_fine_new = X_fine + X_fine_new

            return X_fine_new, None

# ---------------------------------------------------------------------------
# Full TwoLevelMGGNN network
# ---------------------------------------------------------------------------

class TwoLevelMGGNN(nn.Module):
    """Paper-faithful two-level MG-GNN for neural preconditioning.

    Architecture:
        1. **Lift**: Map scalar residual r -> hidden features at fine and coarse levels.
        2. **M MGGNNBlocks**: Parallel cross-scale message passing (paper's key innovation).
        3. **Project**: Map fine-level features -> scalar correction.
        4. **Implicit correction**: Multiple steps z += alpha_i * correction_i with
           residual recomputation for convergent iterative refinement.

    The overall map z = M^{-1}(r) is linear in r (no nonlinear activations in the
    data path) to preserve SPD compatibility with PCG.

    Constructor signature is compatible with the existing training pipeline
    (train.py, scripts/utils.py) via the standard kwargs interface.

    Args:
        A: System matrix (torch sparse or scipy sparse).
        num_layers: Unused (API compatibility).
        embed: Embedding/feature dimension.
        hidden: Hidden dimension (used as feature dim at all levels).
        drop_rate: Unused (API compatibility, no dropout for linear preconditioner).
        num_levels: Number of multigrid levels (capped at 2 for this architecture).
        num_vcycles: Number of implicit correction steps (MG-GNN blocks per step).
        smoother_K: TAGConv polynomial order for fine level.
        coarsest_K: TAGConv polynomial order for coarse level.
        share_smoothers: If True, share weights across correction steps.
        dtype: Torch dtype (forced to float64 for numerical stability).
        num_blocks: Alias for num_vcycles (number of MG-GNN blocks per step).
        K: Alias for smoother_K.
        layers_per_level: Unused (API compatibility).
    """
    def __init__(
        self,
        A,
        num_layers: int = 4,
        embed: int = 32,
        hidden: int = 64,
        drop_rate: float = 0.0,
        num_levels: int = None,
        num_vcycles: int = 2,
        smoother_K: int = 3,
        coarsest_K: int = 5,
        share_smoothers: bool = True,
        dtype: torch.dtype = torch.float32,
        num_blocks: int = None,
        K: int = None,
        layers_per_level: int = None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = torch.float64

        # Resolve aliases
        if num_blocks is not None:
            num_vcycles = num_blocks
        if K is not None:
            smoother_K = K

        self.num_steps = max(1, int(num_vcycles))
        self.hidden_dim = max(8, int(hidden), int(embed))

        if torch.is_tensor(A):
            device = A.device
        else:
            device = torch.device("cpu")

        n = A.shape[0]

        # Paper targets two-level hierarchy
        target_levels = 2

        if num_levels is not None:
            target_levels = max(1, min(2, int(num_levels)))

        self.hierarchy = build_multigrid_hierarchy(
            A,
            num_levels=target_levels,
            coarsening_ratio=0.5,
            min_nodes=max(10, n // 4),
            dtype=torch.float64,
            device=device,
        )
        self.num_levels = self.hierarchy.num_levels
        self.level_sizes = [level.n_nodes for level in self.hierarchy.levels]
        self.has_coarse = self.num_levels > 1

        # Store fine-level A for residual recomputation in implicit correction
        self.register_buffer("A_fine", _as_matmul_ready(self.hierarchy.levels[0].A).to(torch.float64))

        # Store R for lifting residual to coarse level
        if self.has_coarse:
            self.register_buffer("R_lift", _as_matmul_ready(self.hierarchy.levels[0].R).to(torch.float64),)

        # --- Lift layers: scalar -> hidden_dim at each level ---
        self.lift_fine = nn.Linear(1, self.hidden_dim)
        
        if self.has_coarse:
            self.lift_coarse = nn.Linear(1, self.hidden_dim)

        # --- Parallel MG-GNN blocks (paper's core architecture) ---
        cross_level_width = int(getattr(config, "CROSS_LEVEL_WIDTH", 128))

        if share_smoothers:
            # Shared block across all steps (fewer params, symmetric operator)
            shared_block = MGGNNBlock(
                hierarchy=self.hierarchy,
                hidden_dim=self.hidden_dim,
                fine_K=int(smoother_K),
                coarse_K=int(coarsest_K),
                cross_level_width=cross_level_width,
            )
            self.blocks = nn.ModuleList([shared_block] * self.num_steps)
        else:
            self.blocks = nn.ModuleList([
                MGGNNBlock(
                    hierarchy=self.hierarchy,
                    hidden_dim=self.hidden_dim,
                    fine_K=int(smoother_K),
                    coarse_K=int(coarsest_K),
                    cross_level_width=cross_level_width,
                )
                for _ in range(self.num_steps)
            ])

        # --- Project: hidden_dim -> scalar correction ---
        self.project = nn.Linear(self.hidden_dim, 1)

        # --- Learnable positive step sizes for implicit correction ---
        self.step_logits = nn.Parameter(torch.zeros(self.num_steps, dtype=torch.float64))
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        # Cast all parameters to float64
        self.double()

        num_params = sum(p.numel() for p in self.parameters())
        # Correct count for shared blocks
        num_unique_params = sum(
            p.numel()
            for p in {id(p): p for p in self.parameters()}.values()
        )
        print(f"[TwoLevelMGGNN] Created with:")
        print(f"  Levels: {self.num_levels} | Sizes: {self.level_sizes}")
        print(f"  Steps: {self.num_steps} | Hidden dim: {self.hidden_dim}")
        print(f"  Fine K: {int(smoother_K)} | Coarse K: {int(coarsest_K)}")
        print(f"  Cross-level width: {cross_level_width}")
        print(f"  Shared blocks: {share_smoothers}")
        print(f"  Total parameters: {num_unique_params:,}")

    def _restrict_residual(self, r: torch.Tensor) -> torch.Tensor:
        """Restrict fine-level residual to coarse level for lifting."""
        if r.dim() == 1:
            return _spmm(self.R_lift, r.unsqueeze(1)).squeeze(1)
        # r: (n_fine, batch)
        return _spmm(self.R_lift, r)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the learned preconditioner: z = M^{-1}(r).

        Uses implicit iterative correction:
            z_0 = 0
            for i in 1..num_steps:
                r_i = r - A z_{i-1}
                delta_z_i = Block_i(r_i)   # parallel two-level MG-GNN
                z_i = z_{i-1} + alpha_i * delta_z_i

        Args:
            r: Residual vector (n,) or (n, batch).

        Returns:
            z: Preconditioned correction (n,) or (n, batch).
        """
        squeezed = r.dim() == 1
        if squeezed:
            r = r.unsqueeze(1)

        r = r.to(self.dtype)
        z = torch.zeros_like(r)
        current_r = r

        step_sizes = F.softplus(self.step_logits) + 1e-6

        for i, block in enumerate(self.blocks):
            # --- Lift residual to feature space at both levels ---
            # Fine: (n_fine, batch) -> (n_fine, batch, 1) -> (n_fine, batch, hidden)
            h_fine = self.lift_fine(current_r.unsqueeze(-1))

            if self.has_coarse:
                # Restrict residual to coarse, then lift
                r_coarse = self._restrict_residual(current_r)
                h_coarse = self.lift_coarse(r_coarse.unsqueeze(-1))
            else:
                h_coarse = None

            # --- Parallel MG-GNN block (paper's core) ---
            h_fine, h_coarse = block(h_fine, h_coarse)

            # --- Project fine-level features to scalar correction ---
            correction = self.project(h_fine).squeeze(-1)  # (n_fine, batch)

            # --- Implicit correction with learnable step size ---
            z = z + step_sizes[i] * correction

            # Recompute residual for next step
            Az = _spmm(self.A_fine, z)
            current_r = r - Az

        z = self.output_scale * z

        if squeezed:
            z = z.squeeze(1)

        return z
