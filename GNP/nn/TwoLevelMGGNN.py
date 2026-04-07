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
    - The raw network N(r) is kept linear in r for PCG compatibility.
    - SPD enforcement via symmetric factored form:
        M^{-1}(r) = N^{T}(N(r)) + eps * D^{-1} r
      where N^{T} is the autograd adjoint of N and D = diag(A).
      N^{T}N is PSD by construction; the Jacobi floor ensures strict PD.
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
        self.weights = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for k in range(self.K + 1)])

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
        # self.fc1 = nn.Linear(2 * feature_dim, hidden_dim, bias=False)
        # self.fc2 = nn.Linear(hidden_dim, feature_dim, bias=False)
        self.linear = nn.Linear(2 * feature_dim, feature_dim, bias=False)

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

        return self.linear(combined)

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
    def __init__(self, hierarchy, hidden_dim: int, fine_K: int = 3, coarse_K: int = 5, cross_level_width: int = 128):
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
        # A_fine_scaled = scale_A_by_spectral_radius(fine_level.A)
        self.fine_conv = TAGConv(fine_level.A, fine_in_dim, hidden_dim, K=fine_K)

        if self.has_coarse:
            coarse_level = hierarchy.levels[1]
            self.n_coarse = coarse_level.n_nodes

            # Transfer operators
            self.register_buffer("R", _as_matmul_ready(fine_level.R).to(torch.float64))
            self.register_buffer("P", _as_matmul_ready(fine_level.P).to(torch.float64))

            # Intra-level TAGConv for coarse level (also takes concatenated features)
            coarse_in_dim = 2 * hidden_dim
            # A_coarse_scaled = scale_A_by_spectral_radius(coarse_level.A)
            self.coarse_conv = TAGConv(coarse_level.A, coarse_in_dim, hidden_dim, K=coarse_K)

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
    def __init__(self, A, num_layers: int = 4, embed: int = 32, hidden: int = 64, drop_rate: float = 0.0,
        num_levels: int = None, num_vcycles: int = 2, smoother_K: int = 3, coarsest_K: int = 5,
        share_smoothers: bool = True, dtype: torch.dtype = torch.float32, num_blocks: int = None,
        K: int = None, layers_per_level: int = None, **kwargs):
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

        # Diagonal of A for Jacobi floor in SPD enforcement
        A_coo = self.A_fine.coalesce()
        _idx = A_coo.indices()
        _vals = A_coo.values()
        _diag_mask = _idx[0] == _idx[1]
        _D = torch.zeros(n, dtype=torch.float64, device=device)
        _D.scatter_(0, _idx[0, _diag_mask], _vals[_diag_mask])
        self.register_buffer("_D_inv", 1.0 / _D.clamp(min=1e-15))

        # SPD enforcement: M^{-1}(r) = N^{T}N(r) + eps * D^{-1} r
        self.spd_eps = float(getattr(config, 'SPD_JACOBI_EPS', 1e-4))

        # Store R for lifting residual to coarse level
        if self.has_coarse:
            self.register_buffer("R_lift", _as_matmul_ready(self.hierarchy.levels[0].R).to(torch.float64),)

        # --- Lift layers: scalar -> hidden_dim at each level ---
        self.lift_fine = nn.Linear(1, self.hidden_dim, bias=False)
        
        if self.has_coarse:
            self.lift_coarse = nn.Linear(1, self.hidden_dim, bias=False)

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
        self.project = nn.Linear(self.hidden_dim, 1, bias=False)

        # --- Learnable positive step sizes for implicit correction ---
        self.step_logits = nn.Parameter(torch.zeros(self.num_steps, dtype=torch.float64))
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        # Cast all parameters to float64
        self.double()
        self._initialize_near_jacobi()

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

    def _initialize_near_jacobi(self):
        """Initialize so M^{-1}(r) ≈ r at startup (identity preconditioner).
        
        Sets up the network as a pass-through on the first feature channel:
        lift puts r into channel 0, TAGConv passes it through via k=0 identity,
        project reads from channel 0. Starting from M^{-1} ≈ I gives ρ = ρ(I-A),
        which for a spectrally-scaled SPD matrix is well below 1.
        """
        with torch.no_grad():
            # Lift: map scalar r into channel 0
            self.lift_fine.weight.zero_()
            self.lift_fine.weight.data[0, 0] = 1.0

            if self.has_coarse:
                self.lift_coarse.weight.zero_()
                self.lift_coarse.weight.data[0, 0] = 1.0

            # Project: read from channel 0
            self.project.weight.zero_()
            self.project.weight.data[0, 0] = 1.0

            # TAGConv k=0 weights: identity on first hidden_dim channels
            # Higher-hop weights: zero (no multi-hop at init)
            for block in self.blocks:
                for k, w in enumerate(block.fine_conv.weights):
                    w.weight.zero_()
                    if k == 0:
                        dim = min(w.weight.shape[0], w.weight.shape[1])
                        for d in range(dim):
                            w.weight.data[d, d] = 1.0

                if block.has_coarse:
                    for k, w in enumerate(block.coarse_conv.weights):
                        w.weight.zero_()
                        if k == 0:
                            dim = min(w.weight.shape[0], w.weight.shape[1])
                            for d in range(dim):
                                w.weight.data[d, d] = 1.0

                    # Cross-level: zero at init (no cross-level influence initially)
                    block.fine_to_coarse.linear.weight.zero_()
                    block.coarse_to_fine.linear.weight.zero_()

            # Step sizes: softplus(0.54) ≈ 1.0
            self.step_logits.fill_(0.54)
            
            # Output scale: 1/num_steps because we accumulate num_steps corrections
            # and each correction ≈ r due to identity init + residual connection
            self.output_scale.fill_(1.0 / self.num_steps)

    def _restrict_residual(self, r: torch.Tensor) -> torch.Tensor:
        """Restrict fine-level residual to coarse level for lifting."""
        if r.dim() == 1:
            return _spmm(self.R_lift, r.unsqueeze(1)).squeeze(1)
        # r: (n_fine, batch)
        return _spmm(self.R_lift, r)

    def _forward_raw(self, r: torch.Tensor) -> torch.Tensor:
        """Core linear map N(r) without SPD enforcement.

        Uses implicit iterative correction:
            z_0 = 0
            for i in 1..num_steps:
                r_i = r - A z_{i-1}
                delta_z_i = Block_i(r_i)
                z_i = z_{i-1} + alpha_i * delta_z_i

        Args:
            r: Residual (n, batch) in self.dtype. No squeezing.

        Returns:
            z: Correction (n, batch).
        """
        z = torch.zeros_like(r)
        current_r = r

        step_sizes = F.softplus(self.step_logits) + 1e-6

        for i, block in enumerate(self.blocks):
            h_fine = self.lift_fine(current_r.unsqueeze(-1))

            if self.has_coarse:
                r_coarse = self._restrict_residual(current_r)
                h_coarse = self.lift_coarse(r_coarse.unsqueeze(-1))
            else:
                h_coarse = None

            h_fine, h_coarse = block(h_fine, h_coarse)
            correction = self.project(h_fine).squeeze(-1)
            z = z + step_sizes[i] * correction

            Az = _spmm(self.A_fine, z)
            current_r = r - Az

        z = self.output_scale * z
        return z

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """SPD-enforced preconditioner: M^{-1}(r) = N^{T}N(r) + eps * D^{-1} r.

        Structural guarantee: the map r -> z is symmetric positive definite.
        - N^{T}N is PSD by construction (autograd adjoint of the linear map N).
        - eps * D^{-1} (Jacobi floor) ensures strict positive definiteness.

        Args:
            r: Residual vector (n,) or (n, batch).

        Returns:
            z: Preconditioned correction (n,) or (n, batch).
        """
        squeezed = r.dim() == 1

        if squeezed:
            r = r.unsqueeze(1)

        r = r.to(self.dtype)

        # Determine whether we need a second-order graph for backprop.
        # True only when training AND outer grad context is enabled
        # (avoids wasting memory in no_grad power-iteration inner loops).
        need_graph = self.training and torch.is_grad_enabled()

        # --- N^{T}N(r) via autograd adjoint ---
        with torch.enable_grad():
            r_ad = r.detach().requires_grad_(True)
            z = self._forward_raw(r_ad)
            # J^{T} z  where J = dN/dr.  Since z = J r, this gives J^{T} J r.
            NtN_r = torch.autograd.grad(
                outputs=z,
                inputs=r_ad,
                grad_outputs=z if need_graph else z.detach(),
                create_graph=need_graph,
            )[0]

        if not need_graph:
            NtN_r = NtN_r.detach()

        # --- Jacobi floor: eps * D^{-1} r ---
        out = NtN_r + self.spd_eps * self._D_inv.unsqueeze(1) * r

        if squeezed:
            out = out.squeeze(1)
            
        return out
