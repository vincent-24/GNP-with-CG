"""
Graph Hierarchy generation for MG-GNN using Lloyd Aggregation.

This module provides utilities for creating multi-level graph hierarchies
from sparse matrices, which are essential for the Multigrid GNN architecture.
"""

import torch
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class GraphLevel:
    """Represents a single level in the graph hierarchy."""
    edge_index: torch.Tensor  # [2, num_edges] - edges at this level
    edge_weight: torch.Tensor  # [num_edges] - edge weights
    num_nodes: int
    node_positions: Optional[torch.Tensor] = None  # [num_nodes, dim] for geometric info


@dataclass  
class GraphHierarchy:
    """
    Multi-level graph hierarchy for MG-GNN.
    
    Attributes:
        levels: List of GraphLevel objects (0 = finest, L-1 = coarsest)
        restriction_matrices: List of sparse restriction matrices R^{l->l+1}
        interpolation_matrices: List of sparse interpolation matrices P^{l+1->l}
        coarse_to_fine_edges: Edge connections between consecutive levels
    """
    levels: List[GraphLevel]
    restriction_matrices: List[torch.Tensor]
    interpolation_matrices: List[torch.Tensor]
    coarse_to_fine_edges: List[torch.Tensor]  # [2, num_inter_edges] for each level pair
    
    @property
    def num_levels(self) -> int:
        return len(self.levels)
    
    def get_num_nodes(self, level: int) -> int:
        return self.levels[level].num_nodes


def lloyd_aggregation(
    A: sparse.spmatrix,
    num_clusters: int,
    max_iters: int = 100,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, sparse.csr_matrix]:
    """
    Perform Lloyd Aggregation (K-Means on graph) to cluster fine nodes into coarse nodes.
    
    This algorithm iteratively:
    1. Assigns each node to the nearest cluster center (based on graph distance)
    2. Updates cluster centers to be the node with minimum total distance to all members
    
    Args:
        A: Sparse adjacency matrix (or system matrix) of the fine graph
        num_clusters: Number of coarse nodes (clusters) to create
        max_iters: Maximum number of Lloyd iterations
        seed: Random seed for reproducibility
        
    Returns:
        partition: Array of length n_fine, where partition[i] is the cluster ID for node i
        R: Restriction matrix of shape (n_coarse, n_fine) mapping fine to coarse
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = A.shape[0]
    num_clusters = min(num_clusters, n)
    
    # Convert to CSR for efficient row access
    A_csr = csr_matrix(A)
    
    # Make symmetric for aggregation (use |A| + |A^T|)
    A_sym = (np.abs(A_csr) + np.abs(A_csr.T)) / 2
    
    # Initialize: random seed nodes as initial cluster centers
    centers = np.random.choice(n, size=num_clusters, replace=False)
    partition = np.zeros(n, dtype=np.int32)
    
    # Use BFS-based distance for assignment (graph distance)
    for iteration in range(max_iters):
        old_partition = partition.copy()
        
        # Step 1: Assign each node to nearest center using BFS waves
        partition = _assign_to_nearest_center(A_sym, centers)
        
        # Step 2: Update centers to be the "medoid" of each cluster
        new_centers = []
        for c in range(num_clusters):
            cluster_nodes = np.where(partition == c)[0]
            if len(cluster_nodes) == 0:
                # Empty cluster: pick a random unassigned node or keep old center
                new_centers.append(centers[c])
            else:
                # Pick the node with highest connectivity within cluster
                medoid = _find_cluster_medoid(A_sym, cluster_nodes)
                new_centers.append(medoid)
        
        centers = np.array(new_centers)
        
        # Check for convergence
        if np.array_equal(partition, old_partition):
            break
    
    # Build restriction matrix R: R[c, f] = 1 if fine node f belongs to coarse node c
    R = _build_restriction_matrix(partition, num_clusters)
    
    return partition, R


def _assign_to_nearest_center(A_sym: csr_matrix, centers: np.ndarray) -> np.ndarray:
    """Assign each node to the nearest cluster center using BFS."""
    n = A_sym.shape[0]
    partition = np.full(n, -1, dtype=np.int32)
    distance = np.full(n, np.inf)
    
    # Initialize BFS from all centers simultaneously
    from collections import deque
    queue = deque()
    
    for c_idx, center in enumerate(centers):
        partition[center] = c_idx
        distance[center] = 0
        queue.append((center, c_idx, 0))
    
    while queue:
        node, cluster, dist = queue.popleft()
        
        # Get neighbors
        row_start = A_sym.indptr[node]
        row_end = A_sym.indptr[node + 1]
        neighbors = A_sym.indices[row_start:row_end]
        
        for neighbor in neighbors:
            new_dist = dist + 1
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                partition[neighbor] = cluster
                queue.append((neighbor, cluster, new_dist))
    
    # Handle any disconnected nodes
    unassigned = np.where(partition == -1)[0]
    if len(unassigned) > 0:
        # Assign to random clusters
        partition[unassigned] = np.random.randint(0, len(centers), size=len(unassigned))
    
    return partition


def _find_cluster_medoid(A_sym: csr_matrix, cluster_nodes: np.ndarray) -> int:
    """Find the node with maximum total edge weight to other cluster members."""
    if len(cluster_nodes) == 1:
        return cluster_nodes[0]
    
    # Extract submatrix for this cluster
    cluster_set = set(cluster_nodes)
    best_node = cluster_nodes[0]
    best_score = -1
    
    for node in cluster_nodes:
        row_start = A_sym.indptr[node]
        row_end = A_sym.indptr[node + 1]
        neighbors = A_sym.indices[row_start:row_end]
        weights = A_sym.data[row_start:row_end]
        
        # Sum weights to neighbors within cluster
        score = sum(w for n, w in zip(neighbors, weights) if n in cluster_set)
        
        if score > best_score:
            best_score = score
            best_node = node
    
    return best_node


def _build_restriction_matrix(partition: np.ndarray, num_clusters: int) -> csr_matrix:
    """
    Build the restriction matrix R from partition.
    R[c, f] = 1/|cluster_c| if node f belongs to cluster c, else 0.
    
    This gives a normalized restriction where coarse values are averages of fine values.
    """
    n_fine = len(partition)
    
    # Count nodes per cluster for normalization
    cluster_sizes = np.bincount(partition, minlength=num_clusters).astype(np.float64)
    cluster_sizes[cluster_sizes == 0] = 1  # Avoid division by zero
    
    # Build COO format
    row_indices = partition
    col_indices = np.arange(n_fine)
    values = 1.0 / cluster_sizes[partition]
    
    R = coo_matrix((values, (row_indices, col_indices)), 
                   shape=(num_clusters, n_fine)).tocsr()
    
    return R


def build_graph_hierarchy(
    A: torch.Tensor,
    num_levels: int = 2,
    coarsening_ratio: int = 8,
    seed: Optional[int] = None,
    device: str = 'cpu'
) -> GraphHierarchy:
    """
    Build a multi-level graph hierarchy from system matrix A.
    
    Args:
        A: System matrix as torch sparse tensor
        num_levels: Number of levels in hierarchy (including finest)
        coarsening_ratio: Ratio n_l / n_{l+1} between consecutive levels
        seed: Random seed for reproducibility
        device: Target device for tensors
        
    Returns:
        GraphHierarchy object containing all levels and inter-level operators
    """
    n = A.shape[0]
    
    # Convert torch tensor to scipy sparse - handle different layouts
    # Note: CSC/CSR tensors return False for is_sparse, so check layout directly
    if A.layout == torch.sparse_csc:
        # CSC format - must use csc_matrix then convert to csr
        A_cpu = A.cpu()
        A_scipy = csc_matrix(
            (A_cpu.values().numpy(),
             A_cpu.row_indices().numpy(),
             A_cpu.ccol_indices().numpy()),
            shape=(n, n)
        ).tocsr()
    elif A.layout == torch.sparse_csr:
        # CSR format
        A_cpu = A.cpu()
        A_scipy = csr_matrix(
            (A_cpu.values().numpy(),
             A_cpu.col_indices().numpy(),
             A_cpu.crow_indices().numpy()),
            shape=(n, n)
        )
    elif A.is_sparse:
        # COO format (A.is_sparse returns True for COO only)
        A_coo = A.coalesce()
        indices = A_coo.indices().cpu().numpy()
        values = A_coo.values().cpu().numpy()
        A_scipy = coo_matrix((values, (indices[0], indices[1])), shape=(n, n)).tocsr()
    else:
        # Dense tensor
        A_scipy = csr_matrix(A.cpu().numpy())
    
    levels = []
    restriction_matrices = []
    interpolation_matrices = []
    coarse_to_fine_edges = []
    
    current_A = A_scipy
    current_n = current_A.shape[0]
    
    for level in range(num_levels):
        # Create GraphLevel for current level
        edge_index, edge_weight = _matrix_to_edge_index(current_A, device)
        
        graph_level = GraphLevel(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=current_n
        )
        levels.append(graph_level)
        
        # If not the coarsest level, create coarsening operators
        if level < num_levels - 1:
            # Determine number of coarse nodes
            n_coarse = max(current_n // coarsening_ratio, 1)
            
            # Perform Lloyd aggregation
            partition, R_scipy = lloyd_aggregation(
                current_A, 
                n_coarse, 
                seed=seed
            )
            
            # Convert R to torch
            R_torch = _scipy_to_torch_sparse(R_scipy, device)
            restriction_matrices.append(R_torch)
            
            # Interpolation P = R^T (simple transpose for now)
            P_scipy = R_scipy.T.tocsr()
            P_torch = _scipy_to_torch_sparse(P_scipy, device)
            interpolation_matrices.append(P_torch)
            
            # Build coarse-to-fine edge connections
            c2f_edges = _build_inter_level_edges(partition, device)
            coarse_to_fine_edges.append(c2f_edges)
            
            # Coarsen the matrix for next level: A_coarse = R @ A @ R^T
            current_A = R_scipy @ current_A @ R_scipy.T
            current_n = current_A.shape[0]
    
    return GraphHierarchy(
        levels=levels,
        restriction_matrices=restriction_matrices,
        interpolation_matrices=interpolation_matrices,
        coarse_to_fine_edges=coarse_to_fine_edges
    )


def _matrix_to_edge_index(A: csr_matrix, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sparse matrix to edge_index format."""
    A_coo = A.tocoo()
    
    # Remove self-loops and keep only edges where row <= col for undirected
    row = A_coo.row
    col = A_coo.col
    data = A_coo.data
    
    # Keep all edges (including self-loops for the GNN)
    edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(data, dtype=torch.float32, device=device)
    
    return edge_index, edge_weight


def _scipy_to_torch_sparse(A: csr_matrix, device: str) -> torch.Tensor:
    """Convert scipy sparse matrix to torch sparse tensor."""
    A_coo = A.tocoo()
    indices = torch.tensor(np.stack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    shape = torch.Size(A_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _build_inter_level_edges(partition: np.ndarray, device: str) -> torch.Tensor:
    """
    Build edge connections between fine and coarse levels.
    
    Returns edge_index [2, num_edges] where:
    - edge_index[0] = coarse node indices
    - edge_index[1] = fine node indices
    """
    n_fine = len(partition)
    coarse_indices = partition
    fine_indices = np.arange(n_fine)
    
    edge_index = torch.tensor(
        np.stack([coarse_indices, fine_indices]), 
        dtype=torch.long, 
        device=device
    )
    
    return edge_index


def generate_subdomains(
    A: sparse.spmatrix,
    num_subdomains: int,
    overlap: int = 1,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[sparse.csr_matrix]]:
    """
    Generate overlapping subdomains for domain decomposition.
    
    This partitions the graph into subdomains with specified overlap,
    which is needed for the ORAS (Optimized Restricted Additive Schwarz) component.
    
    Args:
        A: System matrix
        num_subdomains: Number of subdomains to create
        overlap: Number of layers of overlap (δ in the paper)
        seed: Random seed
        
    Returns:
        interior_nodes: List of arrays, interior nodes for each subdomain (R_tilde)
        extended_nodes: List of arrays, nodes including overlap for each subdomain (R)
        local_matrices: List of local subdomain matrices A_i
    """
    A_csr = csr_matrix(A)
    n = A_csr.shape[0]
    
    # First partition without overlap using Lloyd
    partition, _ = lloyd_aggregation(A_csr, num_subdomains, seed=seed)
    
    interior_nodes = []
    extended_nodes = []
    local_matrices = []
    
    for s in range(num_subdomains):
        # Interior nodes (no overlap)
        interior = np.where(partition == s)[0]
        interior_nodes.append(interior)
        
        # Extend with overlap layers
        extended = set(interior)
        frontier = set(interior)
        
        for _ in range(overlap):
            new_frontier = set()
            for node in frontier:
                row_start = A_csr.indptr[node]
                row_end = A_csr.indptr[node + 1]
                neighbors = A_csr.indices[row_start:row_end]
                new_frontier.update(neighbors)
            frontier = new_frontier - extended
            extended.update(frontier)
        
        extended = np.array(sorted(extended))
        extended_nodes.append(extended)
        
        # Extract local matrix A_i (submatrix for extended nodes)
        A_local = A_csr[extended][:, extended]
        local_matrices.append(A_local)
    
    return interior_nodes, extended_nodes, local_matrices


def get_boundary_edges(
    A: sparse.spmatrix,
    interior_nodes: List[np.ndarray],
    extended_nodes: List[np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Identify boundary edges for each subdomain.
    
    Boundary edges are those connecting interior nodes to non-interior 
    nodes within the extended subdomain. These are where the Robin 
    boundary conditions (L_i weights) will be applied.
    
    Returns:
        List of (row_local, col_local) index pairs for boundary edges in each subdomain
    """
    A_csr = csr_matrix(A)
    boundary_edges = []
    
    for interior, extended in zip(interior_nodes, extended_nodes):
        interior_set = set(interior)
        extended_set = set(extended)
        
        # Map global indices to local indices
        global_to_local = {g: l for l, g in enumerate(extended)}
        
        boundary_rows = []
        boundary_cols = []
        
        for node in extended:
            is_interior = node in interior_set
            row_start = A_csr.indptr[node]
            row_end = A_csr.indptr[node + 1]
            neighbors = A_csr.indices[row_start:row_end]
            
            for neighbor in neighbors:
                if neighbor in extended_set:
                    neighbor_is_interior = neighbor in interior_set
                    # Boundary edge: connects interior to non-interior (or vice versa)
                    if is_interior != neighbor_is_interior:
                        boundary_rows.append(global_to_local[node])
                        boundary_cols.append(global_to_local[neighbor])
        
        boundary_edges.append((np.array(boundary_rows), np.array(boundary_cols)))
    
    return boundary_edges
