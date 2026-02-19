import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from __future__ import annotations
from typing import List, Optional, Tuple
from scipy import sparse as sp
from GNP.utils import scale_A_by_spectral_radius
from .layers import MLP, GCNConv

DENSE_THRESHOLD = 2_000

def _torch_to_scipy(A: torch.Tensor) -> sp.csc_matrix:
    A_cpu = A.detach().cpu()

    if A_cpu.layout == torch.sparse_csc:
        return sp.csc_matrix((A_cpu.values().numpy(), A_cpu.row_indices().numpy(), A_cpu.ccol_indices().numpy()), shape=tuple(A_cpu.shape))

    if A_cpu.layout == torch.sparse_coo:
        A_cpu = A_cpu.coalesce()
        idx = A_cpu.indices().numpy()

        return sp.coo_matrix((A_cpu.values().numpy(), (idx[0], idx[1])), shape=tuple(A_cpu.shape)).tocsc()

    return sp.csc_matrix(A_cpu.numpy())

def _scipy_to_torch_sparse(S: sp.spmatrix, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    S_coo = sp.coo_matrix(S, dtype=np.float64)
    indices = torch.from_numpy(np.vstack((S_coo.row, S_coo.col)).astype(np.int64))
    values = torch.from_numpy(S_coo.data.copy()).to(dtype)
    out = torch.sparse_coo_tensor(indices, values, size=S_coo.shape, dtype=dtype, device=device)

    return out.coalesce()

def _scipy_to_torch_dense(S: sp.spmatrix, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(S.toarray().astype(np.float64)).to(dtype).to(device)

def _to_torch(S: sp.spmatrix, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if S.shape[0] <= DENSE_THRESHOLD:
        return _scipy_to_torch_dense(S, dtype, device)
    return _scipy_to_torch_sparse(S, dtype, device)

def _scale_scipy_csc(A: sp.spmatrix) -> sp.spmatrix:
    absA = sp.csc_matrix(abs(A))
    n = absA.shape[0]
    ones = np.ones((n, 1))
    row_sum = np.asarray(absA @ ones).ravel()
    col_sum = np.asarray(ones.T @ absA).ravel()
    gamma = min(float(row_sum.max()), float(col_sum.max()))

    if gamma < 1e-30:
        gamma = 1.0

    return A * (1.0 / gamma)

def _sparse_heavy_edge_matching(A_scipy: sp.spmatrix) -> List[List[int]]:
    A_csr = sp.csr_matrix(abs(A_scipy))
    n = A_csr.shape[0]
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data
    matched = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []

    for i in range(n):
        if matched[i]:
            continue

        row_start, row_end = indptr[i], indptr[i + 1]
        best_j = -1
        best_w = 0.0

        for idx in range(row_start, row_end):
            j = indices[idx]
            w = data[idx]

            if j != i and not matched[j] and w > best_w:
                best_j = j
                best_w = w

        if best_j >= 0:
            clusters.append([i, best_j])
            matched[i] = True
            matched[best_j] = True
        else:
            clusters.append([i])
            matched[i] = True

    return clusters

def _build_sparse_transfer_ops(clusters: List[List[int]], n_fine: int) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    n_coarse = len(clusters)
    rows, cols, vals = [], [], []

    for c, nodes in enumerate(clusters):
        w = 1.0 / math.sqrt(len(nodes))

        for node in nodes:
            rows.append(c)
            cols.append(node)
            vals.append(w)

    R = sp.coo_matrix((np.array(vals, dtype=np.float64), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))), shape=(n_coarse, n_fine)).tocsc()
    P = R.T.tocsc()

    return R, P

def build_multigrid_hierarchy(A: torch.Tensor, num_levels: int, dtype: torch.dtype = torch.float64,) -> Tuple[
        List[torch.Tensor], 
        List[torch.Tensor],
        List[torch.Tensor], 
        List[int]
    ]:
    device = A.device if hasattr(A, 'device') else torch.device('cpu')
    A_sp = _torch_to_scipy(A)
    sp_adjs: List[sp.spmatrix] = [_scale_scipy_csc(A_sp)]
    sp_restrict: List[sp.spmatrix] = []
    sp_prolong: List[sp.spmatrix] = []
    level_sizes: List[int] = [A_sp.shape[0]]
    A_current = A_sp

    for _ in range(1, num_levels):
        n = A_current.shape[0]

        if n <= 2:
            break

        clusters = _sparse_heavy_edge_matching(A_current)
        n_coarse = len(clusters)

        if n_coarse >= n:               
            break

        R, P = _build_sparse_transfer_ops(clusters, n)
        A_coarse = R @ A_current @ P
        A_coarse = 0.5 * (A_coarse + A_coarse.T)
        A_coarse = sp.csc_matrix(A_coarse)

        sp_restrict.append(R)
        sp_prolong.append(P)
        sp_adjs.append(_scale_scipy_csc(A_coarse))
        level_sizes.append(n_coarse)

        A_current = A_coarse

    scaled_adjs  = [_to_torch(S, dtype, device) for S in sp_adjs]
    restrict_ops = [_to_torch(S, dtype, device) for S in sp_restrict]
    prolong_ops  = [_to_torch(S, dtype, device) for S in sp_prolong]

    return scaled_adjs, restrict_ops, prolong_ops, level_sizes

class _GCNBlock(nn.Module):
    def __init__(self, AA: torch.Tensor, embed: int, num_layers: int, drop_rate: float) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.gconvs  = nn.ModuleList()
        self.skips   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.dropout = nn.Dropout(drop_rate)

        for _ in range(num_layers):
            self.gconvs.append(GCNConv(AA, embed, embed))
            self.skips.append(nn.Linear(embed, embed))
            self.bns.append(nn.BatchNorm1d(embed))

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        n, batch_size, embed = R.shape

        for i in range(self.num_layers):
            R = self.gconvs[i](R) + self.skips[i](R)
            R = R.view(n * batch_size, -1)
            R = self.bns[i](R)
            R = R.view(n, batch_size, -1)
            R = self.dropout(F.relu(R))

        return R

class UNetGCN(nn.Module):
    def __init__(
        self,
        A: torch.Tensor,
        num_levels: Optional[int] = None,
        layers_per_level: int = 1,
        embed: int = 16,
        hidden: int = 32,
        drop_rate: float = 0.0,
        scale_input: bool = True,
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale_input = scale_input
        self.embed = embed

        n = A.shape[0]

        if num_levels is None:
            num_levels = min(8, max(2, int(math.ceil(math.log2(max(n, 4))))))

        scaled_adjs, restrict_ops, prolong_ops, level_sizes = \
            build_multigrid_hierarchy(A, num_levels, dtype)

        actual_levels = len(scaled_adjs)
        self.num_levels = actual_levels
        self.level_sizes = level_sizes
        self.num_transfers = len(restrict_ops)

        print(f"[UNetGCN] {actual_levels} levels, sizes = {level_sizes}, "
              f"layers_per_level = {layers_per_level}")

        for i, R in enumerate(restrict_ops):
            self.register_buffer(f'R_{i}', R)
        for i, P in enumerate(prolong_ops):
            self.register_buffer(f'P_{i}', P)

        self.enc_mlp = MLP(1, embed, 2, hidden, drop_rate)
        self.dec_mlp = MLP(embed, 1, 2, hidden, drop_rate, is_output_layer=True)
        self.enc_blocks = nn.ModuleList()

        for lvl in range(actual_levels - 1):
            self.enc_blocks.append(_GCNBlock(scaled_adjs[lvl], embed, layers_per_level, drop_rate))

        self.bottleneck = _GCNBlock(scaled_adjs[-1], embed, layers_per_level, drop_rate)
        self.dec_blocks = nn.ModuleList()

        for lvl in range(actual_levels - 2, -1, -1):
            self.dec_blocks.append(_GCNBlock(scaled_adjs[lvl], embed, layers_per_level, drop_rate))

        if dtype == torch.float64:
            self._cast_to_float64()

    def _cast_to_float64(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
                module.weight.data = module.weight.data.double()
                if module.bias is not None:
                    module.bias.data = module.bias.data.double()
            if isinstance(module, nn.BatchNorm1d):
                if module.running_mean is not None:
                    module.running_mean = module.running_mean.double()
                if module.running_var is not None:
                    module.running_var = module.running_var.double()

    def _transfer(self, features: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        n_in, batch_size, embed = features.shape
        F_flat = features.reshape(n_in, batch_size * embed)
        F_out = op @ F_flat                                
        n_out = F_out.shape[0]
        return F_out.reshape(n_out, batch_size, embed)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        n, batch_size = r.shape
        r = r.to(self.dtype)

        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / math.sqrt(n)
            scaling = torch.where(scaling < 1e-12, torch.ones_like(scaling), scaling)
            r = r / scaling

        r = r.view(n, batch_size, 1)
        R = self.enc_mlp(r)                 
        skips: List[torch.Tensor] = []

        for lvl in range(self.num_levels - 1):
            R = self.enc_blocks[lvl](R)
            skips.append(R)                   
            R_op = getattr(self, f'R_{lvl}')  
            R = self._transfer(R, R_op)         

        R = self.bottleneck(R)

        for i, lvl in enumerate(range(self.num_levels - 2, -1, -1)):
            P_op = getattr(self, f'P_{lvl}')   
            R = self._transfer(R, P_op)      
            R = R + skips[lvl]                 
            R = self.dec_blocks[i](R)

        z = self.dec_mlp(R).view(n, batch_size)

        if self.scale_input:
            z = z * scaling

        return z
