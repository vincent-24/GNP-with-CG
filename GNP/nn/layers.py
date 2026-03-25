import torch
from torch import nn
import torch.nn.functional as F


def scatter_add(
    src: torch.Tensor,
    index: torch.LongTensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Pure-PyTorch replacement for ``torch_scatter.scatter_add``."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    idx = index.view([-1 if i == dim else 1 for i in range(src.dim())]).expand_as(src)
    out = torch.zeros(
        [dim_size if i == dim else s for i, s in enumerate(src.shape)],
        dtype=src.dtype,
        device=src.device,
    )
    out.scatter_add_(dim, idx, src)
    return out


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden, drop_rate, use_batchnorm=False, is_output_layer=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.is_output_layer = is_output_layer
        self.lin = nn.ModuleList()
        self.lin.append(nn.Linear(in_dim, hidden))

        for i in range(1, num_layers-1):
            self.lin.append(nn.Linear(hidden, hidden))

        self.lin.append(nn.Linear(hidden, out_dim))

        if use_batchnorm:
            self.batchnorm = nn.ModuleList()

            for i in range(0, num_layers-1):
                self.batchnorm.append(nn.BatchNorm1d(hidden))

            if not is_output_layer:
                self.batchnorm.append(nn.BatchNorm1d(out_dim))

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, R):
        assert len(R.shape) >= 2
        for i in range(self.num_layers):
            R = self.lin[i](R)
            if i != self.num_layers-1 or not self.is_output_layer:
                if self.use_batchnorm:
                    shape = R.shape
                    R = R.view(-1, shape[-1])
                    R = self.batchnorm[i](R)
                    R = R.view(shape)
                R = self.dropout(F.relu(R))
        return R

class GCNConv(nn.Module):
    def __init__(self, AA, in_dim, out_dim):
        super().__init__()
        self.register_buffer('AA', AA)  
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, R):
        assert len(R.shape) == 3
        n, batch_size, in_dim = R.shape
        if in_dim > self.out_dim:
            R = self.fc(R)
            R = R.view(n, batch_size * self.out_dim)
            R = self.AA @ R
            R = R.view(n, batch_size, self.out_dim)
        else:
            R = R.view(n, batch_size * in_dim)
            R = self.AA @ R
            R = R.view(n, batch_size, in_dim)
            R = self.fc(R)
        return R


# ---------------------------------------------------------------------------
# GATv2Conv — Multi-Head Graph Attention (Phase 3: Anisotropic Message Passing)
# ---------------------------------------------------------------------------

class GATv2Conv(nn.Module):
    r"""GATv2 multi-head attention convolution for graph-structured data.

    Implements the GATv2 attention mechanism::

        e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}_l \mathbf{h}_i \| \mathbf{W}_r \mathbf{h}_j])
        \alpha_{ij} = \text{softmax}_j(e_{ij})
        \mathbf{h}'_i = \|_{k=1}^{H} \sigma\bigl(\sum_j \alpha_{ij}^k \mathbf{W}_r^k \mathbf{h}_j\bigr)

    Uses scatter-based sparse aggregation (no dense attention matrix).

    Parameters
    ----------
    edge_index : ``(2, E)`` long tensor of source/target edges
    in_dim     : input feature dimension
    out_dim    : output feature dimension (must be divisible by ``heads``)
    heads      : number of attention heads
    n_nodes    : number of nodes (for scatter target size)
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        n_nodes: int = 0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_k = out_dim // heads
        self.n_nodes = n_nodes
        self.register_buffer('edge_index', edge_index)

        self.W_l = nn.Linear(in_dim, heads * self.d_k, bias=False)
        self.W_r = nn.Linear(in_dim, heads * self.d_k, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, self.d_k))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))  # (1, H, d_k)

        self.proj = nn.Linear(heads * self.d_k, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index_override: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: ``(n, batch_size, in_dim)`` node features
            edge_index_override: optional ``(2, E)`` to use instead of stored edges
        Returns:
            ``(n, batch_size, out_dim)``
        """
        ei = edge_index_override if edge_index_override is not None else self.edge_index
        src, dst = ei[0], ei[1]  # message: src → dst (dst aggregates)
        n, B, _ = x.shape

        # Project
        Wl_x = self.W_l(x)  # (n, B, H*d_k)
        Wr_x = self.W_r(x)  # (n, B, H*d_k)

        Wl_src = Wl_x[src]  # (E, B, H*d_k)
        Wr_dst = Wr_x[dst]  # (E, B, H*d_k)

        E = src.shape[0]

        # Reshape for multi-head: (E, B, H, d_k)
        Wl_src = Wl_src.view(E, B, self.heads, self.d_k)
        Wr_dst = Wr_dst.view(E, B, self.heads, self.d_k)

        # Attention logits: LeakyReLU(a^T [Wl h_i + Wr h_j])
        e = self.leaky_relu(Wl_src + Wr_dst)  # (E, B, H, d_k)
        e = (e * self.attn.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (E, B, H)

        # Sparse softmax via scatter
        e_max = scatter_max(e, dst, dim=0, dim_size=n)
        e = e - e_max[dst]
        alpha = e.exp()  # (E, B, H)
        alpha_sum = scatter_add(alpha, dst, dim=0, dim_size=n)  # (n, B, H)
        alpha = alpha / (alpha_sum[dst] + 1e-12)  # (E, B, H)

        # Weighted message: alpha * Wr h_src
        Wr_src = Wr_x[src].view(E, B, self.heads, self.d_k)  # (E, B, H, d_k)
        msg = alpha.unsqueeze(-1) * Wr_src  # (E, B, H, d_k)

        # Aggregate
        msg_flat = msg.reshape(E, B, self.heads * self.d_k)
        out = scatter_add(msg_flat, dst, dim=0, dim_size=n)  # (n, B, H*d_k)

        return self.proj(out)  # (n, B, out_dim)


def scatter_max(src: torch.Tensor, index: torch.LongTensor, dim: int = 0, dim_size: int | None = None) -> torch.Tensor:
    """Compute scatter max along ``dim``, returning the max values per index bucket."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    # Expand index to match src shape
    expand_shape = list(src.shape)
    expand_shape[dim] = -1
    idx = index.view([-1 if i == dim else 1 for i in range(src.dim())]).expand_as(src)

    out = torch.full(
        [dim_size if i == dim else s for i, s in enumerate(src.shape)],
        float('-inf'), dtype=src.dtype, device=src.device,
    )
    out.scatter_reduce_(dim, idx, src, reduce='amax')
    return out