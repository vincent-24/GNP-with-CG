"""
MG-GNN: Multigrid Graph Neural Network for Preconditioning

Implements parallel cross-scale message passing across a multigrid hierarchy.
The architecture achieves a global receptive field in O(n) time without the
oversmoothing degradation seen in deep U-Nets.

Key components:
    - MGBlock: Single multigrid block with parallel intra-level and inter-level passing
    - MGGNN: Complete network with lift, M blocks, and projection

Reference: "Learning Interface Conditions in Domain Decomposition Solvers"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from GNP.utils import scale_A_by_spectral_radius, build_multigrid_hierarchy
from GNP import config
from .layers import MLP, scatter_add


class TAGConv(nn.Module):
    """
    Topology Adaptive Graph Convolution (TAG).

    Computes K-hop neighborhood aggregation with learnable weights:
        out = sum_{k=0}^{K} W_k @ (A^k @ x)

    This provides polynomial filtering of the graph signal with K-hop receptive field.

    Args:
        edge_index: Edge indices for message passing (2, E)
        n_nodes: Number of nodes
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        K: Number of hops (polynomial order)
    """

    def __init__(
        self,
        A_scaled: torch.Tensor,
        in_dim: int,
        out_dim: int,
        K: int = 3,
    ):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Store the scaled adjacency matrix for efficient sparse matmul
        self.register_buffer('A', A_scaled)

        # Learnable weights for each hop
        self.weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=(k == 0))
            for k in range(K + 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (n, batch_size, in_dim) or (n, in_dim)

        Returns:
            (n, batch_size, out_dim) or (n, out_dim)
        """
        squeezed = x.dim() == 2
        if squeezed:
            x = x.unsqueeze(1)  # (n, 1, in_dim)

        n, batch_size, _ = x.shape
        out = self.weights[0](x)  # k=0: identity transform

        # Iteratively compute A^k @ x using sparse matmul
        x_k = x
        for k in range(1, self.K + 1):
            # x_k = A @ x_{k-1}
            # Reshape for sparse matmul: (n, batch_size * in_dim)
            x_flat = x_k.view(n, batch_size * self.in_dim)
            Ax_flat = torch.sparse.mm(self.A, x_flat)
            x_k = Ax_flat.view(n, batch_size, self.in_dim)

            out = out + self.weights[k](x_k)

        if squeezed:
            out = out.squeeze(1)

        return out


class InterLevelMLP(nn.Module):
    """
    MLP for transforming features during inter-level transfer.

    Used for both restriction (fine -> coarse) and prolongation (coarse -> fine).
    The paper uses 2 FC layers of size 128 for cross-level communication.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MGBlock(nn.Module):
    """
    Single Multigrid Graph Neural Network Block.

    Performs parallel intra-level and inter-level message passing:
        X_{m+1}^{(l)} = TAGConv(X_m^{(l)}, A^{(l)}) + sum_{k != l} F_{k->l}(X_m^{(k)})

    Where F_{k->l} is the heterogeneous message passing function:
        - Down-sampling (k < l): Fine features are pooled using R
        - Up-sampling (k > l): Coarse features are broadcast using P = R^T
    """

    def __init__(
        self,
        hierarchy,  # MultigridHierarchy
        hidden_dim: int,
        K: int = 3,
        use_attention: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_levels = hierarchy.num_levels
        self.hidden_dim = hidden_dim

        # Store transfer operators as buffers
        for l in range(hierarchy.num_levels):
            level = hierarchy.levels[l]
            if level.R is not None:
                self.register_buffer(f'R_{l}', level.R)
            if level.P is not None:
                self.register_buffer(f'P_{l}', level.P)

        # Intra-level TAGConv for each level
        self.intra_convs = nn.ModuleList()
        for l, level in enumerate(hierarchy.levels):
            A_scaled = scale_A_by_spectral_radius(level.A).to(hierarchy.dtype)
            self.intra_convs.append(
                TAGConv(A_scaled, hidden_dim, hidden_dim, K=K)
            )

        # Inter-level MLPs: one for each direction at each level pair
        # down_mlps[l]: from level l to level l+1 (restriction)
        # up_mlps[l]: from level l+1 to level l (prolongation)
        self.down_mlps = nn.ModuleList()
        self.up_mlps = nn.ModuleList()
        cross_level_width = getattr(config, 'CROSS_LEVEL_WIDTH', 128)

        for l in range(hierarchy.num_levels - 1):
            self.down_mlps.append(InterLevelMLP(hidden_dim, hidden_dim, cross_level_width))
            self.up_mlps.append(InterLevelMLP(hidden_dim, hidden_dim, cross_level_width))

        # Layer normalization for stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(hierarchy.num_levels)
        ])

    def _restrict(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Apply restriction: fine -> coarse using R."""
        R = getattr(self, f'R_{level}')
        # x: (n_fine, batch, hidden) -> (n_coarse, batch, hidden)
        n_fine, batch_size, d = x.shape
        x_flat = x.view(n_fine, batch_size * d)
        Rx = torch.sparse.mm(R, x_flat)
        n_coarse = Rx.shape[0]
        return Rx.view(n_coarse, batch_size, d)

    def _prolong(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Apply prolongation: coarse -> fine using P = R^T."""
        P = getattr(self, f'P_{level}')
        # x: (n_coarse, batch, hidden) -> (n_fine, batch, hidden)
        n_coarse, batch_size, d = x.shape
        x_flat = x.view(n_coarse, batch_size * d)
        Px = torch.sparse.mm(P, x_flat)
        n_fine = Px.shape[0]
        return Px.view(n_fine, batch_size, d)

    def forward(self, X: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass performing parallel multi-scale message passing.

        Args:
            X: List of feature tensors, one per level
               X[l] has shape (n_l, batch_size, hidden_dim)

        Returns:
            List of updated feature tensors
        """
        X_new = []

        for l in range(self.num_levels):
            # 1. Intra-level message passing via TAGConv
            h_intra = self.intra_convs[l](X[l])

            # 2. Inter-level message passing
            h_inter = torch.zeros_like(h_intra)

            # Receive from finer levels (k < l): restrict then transform
            for k in range(l):
                # Cascade restriction from level k to level l
                h_k = X[k]
                for m in range(k, l):
                    h_k = self._restrict(h_k, m)
                h_inter = h_inter + self.down_mlps[l - 1](h_k)

            # Receive from coarser levels (k > l): prolong then transform
            for k in range(l + 1, self.num_levels):
                # Cascade prolongation from level k to level l
                h_k = X[k]
                for m in range(k - 1, l - 1, -1):
                    h_k = self._prolong(h_k, m)
                h_inter = h_inter + self.up_mlps[l](h_k)

            # 3. Combine and normalize with residual connection
            h_out = h_intra + h_inter + X[l]  # Residual
            h_out = self.layer_norms[l](h_out)
            h_out = F.relu(h_out)

            X_new.append(h_out)

        return X_new


class MGGNN(nn.Module):
    """
    Multigrid Graph Neural Network for Neural Preconditioning.

    Architecture:
        1. Lift: Map input (n, 1) -> (n, hidden_dim) and restrict up hierarchy
        2. M MGBlocks: Parallel cross-scale message passing
        3. Project: Map (n, hidden_dim) -> (n, 1) for preconditioner output

    The network learns to approximate M^{-1} such that the error propagation
    operator I - M^{-1}A has minimal spectral radius.

    Args:
        A: System matrix (torch sparse or scipy sparse)
        num_layers: Unused (for API compatibility with ResGCN)
        embed: Embedding dimension
        hidden: Hidden dimension (same as embed for MGGNN)
        drop_rate: Dropout rate
        num_levels: Number of multigrid levels (None = auto)
        num_blocks: Number of stacked MGBlocks
        K: TAGConv polynomial order (K-hop neighborhood)
        dtype: Torch dtype
    """

    def __init__(
        self,
        A,
        num_layers: int = 4,  # Unused, kept for API compatibility
        embed: int = 32,
        hidden: int = 64,
        drop_rate: float = 0.0,
        num_levels: int = None,
        num_blocks: int = 4,
        K: int = 3,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.hidden_dim = hidden
        self.num_blocks = num_blocks
        self.drop_rate = drop_rate

        # Determine device from input matrix
        if torch.is_tensor(A):
            device = A.device
        else:
            device = torch.device('cpu')

        # Auto-determine number of levels
        n = A.shape[0]
        if num_levels is None:
            num_levels = min(8, max(2, int(math.ceil(math.log2(n))) - 3))
        self.num_levels = num_levels

        # Build multigrid hierarchy
        self.hierarchy = build_multigrid_hierarchy(
            A,
            num_levels=num_levels,
            coarsening_ratio=0.5,
            min_nodes=max(10, n // (2 ** num_levels)),
            dtype=dtype,
            device=device,
        )
        self.num_levels = self.hierarchy.num_levels  # May be less due to min_nodes

        # Store level sizes for convenience
        self.level_sizes = [level.n_nodes for level in self.hierarchy.levels]

        # Lift MLP: map (n, 1) -> (n, hidden_dim)
        self.lift = MLP(1, hidden, num_layers=3, hidden=hidden, drop_rate=drop_rate)

        # Stack of MGBlocks
        self.blocks = nn.ModuleList([
            MGBlock(
                self.hierarchy,
                hidden_dim=hidden,
                K=K,
                use_attention=getattr(config, 'USE_ATTENTION', False),
                num_heads=getattr(config, 'NUM_ATTENTION_HEADS', 4),
            )
            for _ in range(num_blocks)
        ])

        # Dropout between blocks
        self.dropout = nn.Dropout(drop_rate)

        # Project MLP: map (n, hidden_dim) -> (n, 1)
        self.project = MLP(hidden, 1, num_layers=3, hidden=hidden, drop_rate=drop_rate, is_output_layer=True)

        # Store scaled adjacency for computing residual connections
        self.register_buffer('A_scaled', scale_A_by_spectral_radius(
            self.hierarchy.levels[0].A if torch.is_tensor(A) else
            self.hierarchy.levels[0].A
        ).to(dtype))

        # Cast to dtype
        if dtype == torch.float64:
            self._cast_to_float64()

        # Auxiliary losses storage (for MGGNN v2 features)
        self._aux_losses = {}

    def _cast_to_float64(self):
        """Cast all modules to float64."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.BatchNorm1d, nn.LayerNorm)):
                module.weight.data = module.weight.data.double()
                if module.bias is not None:
                    module.bias.data = module.bias.data.double()
            if isinstance(module, nn.BatchNorm1d):
                if module.running_mean is not None:
                    module.running_mean = module.running_mean.double()
                if module.running_var is not None:
                    module.running_var = module.running_var.double()

    def _restrict_to_hierarchy(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Restrict features from finest level to all coarser levels.

        Args:
            x: Features at finest level (n_0, batch_size, hidden_dim)

        Returns:
            List of features at all levels
        """
        X = [x]
        x_current = x

        for l in range(self.num_levels - 1):
            R = self.hierarchy.levels[l].R
            n_fine, batch_size, d = x_current.shape
            x_flat = x_current.view(n_fine, batch_size * d)
            Rx = torch.sparse.mm(R, x_flat)
            n_coarse = Rx.shape[0]
            x_current = Rx.view(n_coarse, batch_size, d)
            X.append(x_current)

        return X

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing z = M_theta(r).

        Args:
            r: Residual vector (n, batch_size) or (n,)

        Returns:
            z: Preconditioned residual (n, batch_size) or (n,)
        """
        squeezed = r.dim() == 1
        if squeezed:
            r = r.unsqueeze(1)  # (n, 1)

        n, batch_size = r.shape
        r = r.to(self.dtype)

        # Add feature dimension: (n, batch_size) -> (n, batch_size, 1)
        r = r.unsqueeze(-1)

        # 1. Lift: (n, batch_size, 1) -> (n, batch_size, hidden_dim)
        X_0 = self.lift(r)

        # 2. Restrict to all levels to initialize hierarchy
        X = self._restrict_to_hierarchy(X_0)

        # 3. Apply M MGBlocks
        for block in self.blocks:
            X = block(X)
            # Apply dropout between blocks
            X = [self.dropout(x) for x in X]

        # 4. Project finest level: (n, batch_size, hidden_dim) -> (n, batch_size, 1)
        z = self.project(X[0])

        # Remove feature dimension: (n, batch_size, 1) -> (n, batch_size)
        z = z.squeeze(-1)

        if squeezed:
            z = z.squeeze(1)

        return z

    def get_aux_losses(self) -> dict:
        """Return auxiliary losses for regularization (MGGNN v2 features)."""
        return self._aux_losses


# Alias for backwards compatibility
UNetGCN = MGGNN
