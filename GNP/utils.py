import os
import mat73
import torch
import numpy as np
import scipy.io as sio
from scipy import sparse
from ssgetpy import fetch
from dataclasses import dataclass
from typing import List, Optional

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False

# -----------------------------------------------------------------------------
# Multigrid Hierarchy for MG-GNN
# -----------------------------------------------------------------------------

@dataclass
class MultigridLevel:
    """Stores data for a single level in the multigrid hierarchy."""
    A: torch.Tensor                # Adjacency/system matrix at this level (sparse)
    edge_index: torch.Tensor       # Edge index for PyG-style message passing (2, E)
    n_nodes: int                   # Number of nodes at this level
    R: Optional[torch.Tensor]      # Restriction operator to next coarser level (sparse)
    P: Optional[torch.Tensor]      # Prolongation operator from next coarser level (sparse)

@dataclass
class MultigridHierarchy:
    """Complete multigrid hierarchy with all levels and transfer operators."""
    levels: List[MultigridLevel]
    num_levels: int
    device: torch.device
    dtype: torch.dtype

    def to(self, device: torch.device):
        """Move all tensors in hierarchy to specified device."""
        for level in self.levels:
            level.A = level.A.to(device)
            level.edge_index = level.edge_index.to(device)
            if level.R is not None:
                level.R = level.R.to(device)
            if level.P is not None:
                level.P = level.P.to(device)
        self.device = device
        return self

def _scipy_to_torch_sparse(A_scipy, dtype=torch.float64, device='cpu'):
    """Convert scipy sparse matrix to torch sparse COO tensor."""
    A_coo = A_scipy.tocoo()
    indices = torch.from_numpy(np.vstack((A_coo.row, A_coo.col))).long()
    values = torch.from_numpy(A_coo.data).to(dtype)
    shape = torch.Size(A_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()

def _torch_to_scipy_sparse(A_torch):
    """Convert torch sparse tensor to scipy sparse CSR matrix. Handles both COO and CSC layouts."""
    A_cpu = A_torch.cpu()

    if A_cpu.layout == torch.sparse_csc:
        ccol_indices = A_cpu.ccol_indices().numpy()
        row_indices = A_cpu.row_indices().numpy()
        values = A_cpu.values().numpy()
        return sparse.csc_matrix((values, row_indices, ccol_indices), shape=A_cpu.shape).tocsr()
    elif A_cpu.layout == torch.sparse_csr:
        crow_indices = A_cpu.crow_indices().numpy()
        col_indices = A_cpu.col_indices().numpy()
        values = A_cpu.values().numpy()
        return sparse.csr_matrix((values, col_indices, crow_indices), shape=A_cpu.shape)
    else:
        A_coo = A_cpu.coalesce()
        indices = A_coo.indices().numpy()
        values = A_coo.values().numpy()
        shape = A_coo.shape
        return sparse.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()

def _extract_edge_index(A_sparse):
    """Extract edge_index (2, E) from sparse adjacency matrix for message passing."""
    if torch.is_tensor(A_sparse):
        A_cpu = A_sparse.cpu()
        if A_cpu.layout == torch.sparse_csc:
            A_coo = A_cpu.to_sparse_coo().coalesce()
        elif A_cpu.layout == torch.sparse_csr:
            A_coo = A_cpu.to_sparse_coo().coalesce()
        else:
            A_coo = A_cpu.coalesce()
        return A_coo.indices()
    else:
        A_coo = A_sparse.tocoo()
        return torch.from_numpy(np.vstack((A_coo.row, A_coo.col))).long()

def _lloyd_aggregation(A_scipy, num_aggregates):
    """
    Compute Lloyd aggregation to partition graph nodes into clusters.

    Uses pyamg's Lloyd aggregation algorithm which performs k-means style
    clustering on the graph structure.

    Args:
        A_scipy: Scipy sparse adjacency matrix
        num_aggregates: Target number of clusters/aggregates

    Returns:
        R: Restriction operator (num_aggregates, n) as scipy sparse matrix
    """
    if not HAS_PYAMG:
        raise ImportError("pyamg is required for Lloyd aggregation. Install via: pip install pyamg")

    n = A_scipy.shape[0]

    # Ensure symmetric for aggregation (use pattern only)
    A_pattern = (A_scipy + A_scipy.T).tocsr()
    A_pattern.data[:] = 1.0  # Use uniform weights

    # Lloyd aggregation takes a ratio parameter (fraction of nodes to be aggregate centers)
    # ratio = num_aggregates / n, but must be > 0 and <= 1
    ratio = max(0.01, min(1.0, num_aggregates / n))

    # Lloyd aggregation returns AggOp: (n, num_aggregates) mapping fine to coarse
    AggOp, _ = pyamg.aggregation.lloyd_aggregation(A_pattern, ratio=ratio, maxiter=10)

    # R is the transpose: (num_aggregates, n)
    R = AggOp.T.tocsr()

    # Normalize R so each row sums to 1 (averaging instead of summation)
    row_sums = np.array(R.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    R = sparse.diags(1.0 / row_sums) @ R

    return R

def build_multigrid_hierarchy(
    A,
    num_levels: int = 2,
    coarsening_ratio: float = 0.5,
    min_nodes: int = 10,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu'),
) -> MultigridHierarchy:
    """
    Build a multigrid hierarchy for the MG-GNN architecture.

    Constructs coarse grids via Lloyd aggregation and Galerkin projection.
    The restriction operator R pools fine-grid features to coarse grid,
    and prolongation P = R^T broadcasts coarse features to fine grid.

    Args:
        A: System matrix - can be torch sparse tensor, scipy sparse, or dense
        num_levels: Number of multigrid levels (including finest level)
        coarsening_ratio: Target ratio of coarse nodes to fine nodes per level
        min_nodes: Minimum nodes at coarsest level (stops coarsening early if reached)
        dtype: Torch dtype for all tensors
        device: Target device for final hierarchy

    Returns:
        MultigridHierarchy containing all levels with A, R, P, and edge_index
    """
    # Convert input to scipy sparse for aggregation
    if torch.is_tensor(A):
        # Check for any sparse layout (COO, CSR, CSC)
        is_sparse = A.is_sparse or A.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
        if is_sparse:
            A_scipy = _torch_to_scipy_sparse(A.to(torch.float64))
        else:
            A_scipy = sparse.csr_matrix(A.cpu().numpy())
    elif sparse.issparse(A):
        A_scipy = A.tocsr()
    else:
        A_scipy = sparse.csr_matrix(A)

    levels = []
    A_current = A_scipy

    for level_idx in range(num_levels):
        n_nodes = A_current.shape[0]

        # Convert current level matrix to torch
        A_torch = _scipy_to_torch_sparse(A_current, dtype=dtype, device='cpu')
        edge_index = _extract_edge_index(A_torch)

        # Compute transfer operators (except for coarsest level)
        R_torch = None
        P_torch = None

        if level_idx < num_levels - 1:
            # Target number of coarse nodes
            n_coarse = max(min_nodes, int(n_nodes * coarsening_ratio))

            # Stop coarsening if we've reached minimum size
            if n_coarse >= n_nodes - 1:
                # This becomes the coarsest level
                levels.append(MultigridLevel(
                    A=A_torch,
                    edge_index=edge_index,
                    n_nodes=n_nodes,
                    R=None,
                    P=None
                ))
                break

            # Compute restriction via Lloyd aggregation
            R_scipy = _lloyd_aggregation(A_current, n_coarse)
            P_scipy = R_scipy.T.tocsr()  # Prolongation is transpose of restriction

            # Galerkin projection: A_coarse = R @ A @ P
            A_coarse = R_scipy @ A_current @ P_scipy

            # Convert operators to torch
            R_torch = _scipy_to_torch_sparse(R_scipy, dtype=dtype, device='cpu')
            P_torch = _scipy_to_torch_sparse(P_scipy, dtype=dtype, device='cpu')

            A_current = A_coarse

        levels.append(MultigridLevel(
            A=A_torch,
            edge_index=edge_index,
            n_nodes=n_nodes,
            R=R_torch,
            P=P_torch
        ))

    hierarchy = MultigridHierarchy(
        levels=levels,
        num_levels=len(levels),
        device=torch.device('cpu'),
        dtype=dtype
    )

    # Move to target device
    if device != torch.device('cpu'):
        hierarchy.to(device)

    return hierarchy

#-----------------------------------------------------------------------------
# Load problem of SuiteSparse.
# problem must be in the form group/name.
# Return torch.sparse_csc_tensor in torch.float64 precision in device.
# For the python interface of SuiteSparse, see https://github.com/drdarshan/ssgetpy
def load_suitesparse(location, problem, device):
    matrix = fetch(problem, format='MAT', dry_run=True)
    
    if len(matrix) != 0:
        location = os.path.abspath(os.path.expanduser(location))
        group, _ = problem.split('/', 1)
        mat_path = os.path.join(location, problem + '.mat')

        def _download_mat():
            fetch(problem, format='MAT', location=os.path.join(location, group))[0]

        # Ensure a MAT file exists locally before attempting to parse it.
        _download_mat()

        for attempt in range(2):
            try:
                try:
                    P = sio.loadmat(mat_path)
                    A = P['Problem']['A'][0][0]
                except NotImplementedError:
                    P = mat73.loadmat(mat_path)
                    A = P['Problem']['A']

                del P
                A = torch.sparse_csc_tensor(A.indptr, A.indices, A.data, A.shape, dtype=torch.float64).to(device)
                return A
            except OSError as exc:
                if attempt == 0:
                    if os.path.exists(mat_path):
                        os.remove(mat_path)
                    _download_mat()
                    continue

                raise OSError(
                    f'Failed to read SuiteSparse MAT file after re-download: {mat_path}'
                ) from exc
    else:
        raise Exception(f'Unsupported problem {problem}!')
 
#-----------------------------------------------------------------------------
# Scale A by an estimated spectral radius according to the Gershgorin
# circle theorem.
def scale_A_by_spectral_radius(A):
    if A.layout == torch.sparse_csc:
        absA = torch.absolute(A)
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=A.dtype, device=A.device)
        col_sum = torch.ones(1, m, dtype=A.dtype, device=A.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = A * (1. / gamma.item())
    elif A.layout == torch.sparse_coo:
        A_coo = A.coalesce()
        m, n = A_coo.shape
        absA = torch.sparse_coo_tensor(
            A_coo.indices(),
            torch.abs(A_coo.values()),
            A_coo.shape,
            device=A_coo.device,
            dtype=A_coo.dtype
        ).coalesce()
        row_sum = torch.sparse.mm(absA, torch.ones(n, 1, dtype=A.dtype, device=A.device))
        col_sum = torch.sparse.mm(absA.t(), torch.ones(m, 1, dtype=A.dtype, device=A.device))
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = torch.sparse_coo_tensor(
            A_coo.indices(),
            A_coo.values() / gamma.item(),
            A_coo.shape,
            device=A_coo.device,
            dtype=A_coo.dtype
        ).coalesce()
    elif A.layout == torch.strided:
        absA = torch.absolute(A)
        row_sum = torch.sum(absA, dim=1)
        col_sum = torch.sum(absA, dim=0)
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = A / gamma
    else:
        raise NotImplementedError('A must be either torch.sparse_csc_tensor, torch.sparse_coo_tensor, or torch.tensor')
    
    return outA

#-----------------------------------------------------------------------------
# Extract the diagonal of A.
def extract_diagonal(A):
    if A.layout == torch.sparse_csc:
        n = A.shape[0]
        D = torch.zeros(n, device=A.device, dtype=A.dtype)
        A = A.to_sparse_coo().coalesce()
        indices = A.indices()
        mask = indices[0] == indices[1]
        diagonal_values = A.values()[mask]
        diagonal_indices = indices[0][mask]
        D = D.scatter_add(dim=0, index=diagonal_indices, src=diagonal_values)
    elif A.layout == torch.strided:
        D = torch.diagonal(A)
    else:
        raise NotImplementedError('A must be either torch.sparse_csc_tensor or torch.tensor')
    
    return D
