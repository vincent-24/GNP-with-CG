import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, diags
from scipy.sparse.linalg import splu, spsolve
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from GNP.data.graph_hierarchy import (
    build_graph_hierarchy, 
    generate_subdomains,
    get_boundary_edges,
    GraphHierarchy
)
from GNP.nn.MGGNN import MGGNN

@dataclass
class SubdomainData:
    """Pre-computed data for a single subdomain."""
    interior_nodes: np.ndarray      # R̃_i indices (no overlap)
    extended_nodes: np.ndarray      # R_i indices (with overlap)
    local_matrix: sparse.csr_matrix # A_i
    boundary_mask: np.ndarray       # Boolean mask for boundary edges in local matrix
    lu_factor: object               # Pre-factorized (A_i + L_i)


class MGGNNPreconditioner:
    def __init__(
        self,
        levels: int = 2,
        coarsening_ratio: int = 8,
        num_subdomains: Optional[int] = None,
        overlap: int = 1,
        hidden_dim: int = 64,
        num_mg_layers: int = 4,
        checkpoint_path: Optional[str] = None,
        auto_load_checkpoint: bool = True,
        problem_name: Optional[str] = None,
        seed: int = 42
    ):
        self.levels = levels
        self.coarsening_ratio = coarsening_ratio
        self.num_subdomains = num_subdomains
        self.overlap = overlap
        self.hidden_dim = hidden_dim
        self.num_mg_layers = num_mg_layers
        self.checkpoint_path = checkpoint_path
        self.auto_load_checkpoint = auto_load_checkpoint
        self.problem_name = problem_name
        self.seed = seed
        
        self.device = None
        self.dtype = None
        self.n = None
        self.A_scipy = None
        self.A_torch = None
        
        self.hierarchy = None
        self.network = None
        
        self.P = None           # Interpolation operator (torch sparse)
        self.P_scipy = None     # Interpolation operator (scipy)
        self.A_coarse = None    # Coarse grid operator P^T A P
        self.A_coarse_lu = None # LU factorization of A_coarse
        
        self.subdomains: List[SubdomainData] = []
        
        self._is_setup = False
    
    def setup(self, A: torch.Tensor):
        print("[MG-GNN] Setting up preconditioner...")
        
        self.device = A.device
        self.dtype = A.dtype
        self.n = A.shape[0]
        
        self._prepare_matrix(A)
        
        if self.num_subdomains is None:
            self.num_subdomains = max(4, min(int(np.sqrt(self.n)), 64))
        
        print(f"[MG-GNN] Matrix size: {self.n}, Subdomains: {self.num_subdomains}")
        
        print("[MG-GNN] Building graph hierarchy...")
        self.hierarchy = build_graph_hierarchy(
            A,
            num_levels=self.levels,
            coarsening_ratio=self.coarsening_ratio,
            seed=self.seed,
            device=str(self.device)
        )
        
        n_coarse = self.hierarchy.levels[-1].num_nodes
        print(f"[MG-GNN] Hierarchy: {self.n} → {n_coarse} nodes ({self.levels} levels)")
        
        print("[MG-GNN] Generating subdomains...")
        interior_nodes, extended_nodes, local_matrices = generate_subdomains(
            self.A_scipy,
            self.num_subdomains,
            overlap=self.overlap,
            seed=self.seed
        )
        print("[MG-GNN] Running neural network...")
        self._setup_network()
        
        with torch.no_grad():
            predictions = self.network(self.hierarchy)
        
        print("[MG-GNN] Assembling interpolation operator P...")
        self._assemble_P_operator(predictions['P_weights'])
        
        print("[MG-GNN] Computing coarse grid operator...")
        self._compute_coarse_operator()
        
        print("[MG-GNN] Setting up subdomain solvers...")
        boundary_edges = get_boundary_edges(
            self.A_scipy, interior_nodes, extended_nodes
        )
        
        self._setup_subdomains(
            interior_nodes, 
            extended_nodes, 
            local_matrices,
            boundary_edges,
            predictions['L_weights']
        )
        
        self._is_setup = True
        print(f"[MG-GNN] Setup complete. Coarse size: {n_coarse}")
    
    def _prepare_matrix(self, A: torch.Tensor):
        if A.layout == torch.sparse_csc:
            A_cpu = A.cpu()
            self.A_scipy = csc_matrix(
                (A_cpu.values().numpy(), 
                 A_cpu.row_indices().numpy(), 
                 A_cpu.ccol_indices().numpy()),
                shape=(self.n, self.n)
            ).tocsr()
            A_coo_scipy = self.A_scipy.tocoo()
            indices = torch.tensor(
                np.stack([A_coo_scipy.row, A_coo_scipy.col]), 
                dtype=torch.long, device=self.device
            )
            values = torch.tensor(A_coo_scipy.data, dtype=self.dtype, device=self.device)
            self.A_torch = torch.sparse_coo_tensor(
                indices, values, (self.n, self.n)
            ).coalesce()
        elif A.layout == torch.sparse_csr:
            A_cpu = A.cpu()
            self.A_scipy = csr_matrix(
                (A_cpu.values().numpy(),
                 A_cpu.col_indices().numpy(),
                 A_cpu.crow_indices().numpy()),
                shape=(self.n, self.n)
            )
            A_coo_scipy = self.A_scipy.tocoo()
            indices = torch.tensor(
                np.stack([A_coo_scipy.row, A_coo_scipy.col]),
                dtype=torch.long, device=self.device
            )
            values = torch.tensor(A_coo_scipy.data, dtype=self.dtype, device=self.device)
            self.A_torch = torch.sparse_coo_tensor(
                indices, values, (self.n, self.n)
            ).coalesce()
        elif A.is_sparse:
            A_coo = A.coalesce()
            indices = A_coo.indices().cpu().numpy()
            values = A_coo.values().cpu().numpy()
            self.A_scipy = coo_matrix(
                (values, (indices[0], indices[1])), 
                shape=(self.n, self.n)
            ).tocsr()
            self.A_torch = A_coo
        else:
            self.A_scipy = csr_matrix(A.cpu().numpy())
            A_np = A.cpu().numpy()
            A_sp = coo_matrix(A_np)
            indices = torch.tensor(np.stack([A_sp.row, A_sp.col]), dtype=torch.long)
            values = torch.tensor(A_sp.data, dtype=self.dtype)
            self.A_torch = torch.sparse_coo_tensor(
                indices, values, (self.n, self.n), device=self.device
            ).coalesce()
    
    def _setup_network(self):
        import os
        from pathlib import Path
        
        self.network = MGGNN(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_mg_layers,
            num_levels=self.levels,
            K=3,
            dropout=0.0,  
            dtype=torch.float32
        )
        
        checkpoint_to_load = self.checkpoint_path
        
        if checkpoint_to_load is None and self.auto_load_checkpoint:
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / "data" / "mggnn_data"
            if model_dir.exists():
                if self.problem_name is not None:
                    problem_safe = self.problem_name.replace("/", "_").replace("\\", "_")
                    specific_ckpt = model_dir / f"mggnn_{problem_safe}.pt"
                    if specific_ckpt.exists():
                        checkpoint_to_load = str(specific_ckpt)
                        print(f"[MG-GNN] Found checkpoint for {self.problem_name}: {checkpoint_to_load}")
                
                if checkpoint_to_load is None:
                    candidates = list(model_dir.glob("mggnn_*.pt"))
                    if candidates:
                        checkpoint_to_load = str(max(candidates, key=os.path.getmtime))
                        print(f"[MG-GNN] WARNING: No checkpoint for problem '{self.problem_name}', using most recent: {checkpoint_to_load}")
        
        if checkpoint_to_load is not None:
            print(f"[MG-GNN] Loading weights from {checkpoint_to_load}")
            state_dict = torch.load(checkpoint_to_load, map_location='cpu', weights_only=True)
            self.network.load_state_dict(state_dict)
        else:
            print("[MG-GNN] WARNING: No checkpoint loaded, using random weights!")
            print("[MG-GNN] Run 'python scripts/train_mggnn.py' to train the model first.")
        
        self.network.eval()
        self.network.to(self.device)
    
    def _assemble_P_operator(self, P_weights_list: List[torch.Tensor]):
        
        P_weights = P_weights_list[0]  
        c2f_edges = self.hierarchy.coarse_to_fine_edges[0]
        
        coarse_idx = c2f_edges[0].cpu().numpy()
        fine_idx = c2f_edges[1].cpu().numpy()
        
        weights_logits = P_weights.detach().cpu()
        weights = torch.tanh(weights_logits).numpy()
        
        n_fine = self.hierarchy.levels[0].num_nodes
        n_coarse = self.hierarchy.levels[1].num_nodes
        
        weight_sums = np.zeros(n_fine)
        np.add.at(weight_sums, fine_idx, np.abs(weights))
        weight_sums[weight_sums == 0] = 1.0
        normalized_weights = weights / weight_sums[fine_idx]
        
        self.P_scipy = coo_matrix(
            (normalized_weights, (fine_idx, coarse_idx)),
            shape=(n_fine, n_coarse)
        ).tocsr()
        
        P_coo = self.P_scipy.tocoo()
        indices = torch.tensor(np.stack([P_coo.row, P_coo.col]), dtype=torch.long)
        values = torch.tensor(P_coo.data, dtype=self.dtype)
        self.P = torch.sparse_coo_tensor(
            indices, values, self.P_scipy.shape, device=self.device
        ).coalesce()
    
    def _compute_coarse_operator(self):
        self.A_coarse = self.P_scipy.T @ self.A_scipy @ self.P_scipy
        self.A_coarse = self.A_coarse.tocsc()
        
        n_coarse = self.A_coarse.shape[0]
        
        if n_coarse <= 1000:
            try:
                self.A_coarse_lu = splu(self.A_coarse)
            except Exception as e:
                print(f"[MG-GNN] Warning: LU factorization failed ({e}), using dense inverse")
                A_dense = self.A_coarse.toarray()
                self.A_coarse_inv = np.linalg.inv(A_dense)
                self.A_coarse_lu = None
        else:
            self.A_coarse_lu = splu(self.A_coarse)
    
    def _setup_subdomains(
        self,
        interior_nodes: List[np.ndarray],
        extended_nodes: List[np.ndarray],
        local_matrices: List[sparse.csr_matrix],
        boundary_edges: List[Tuple[np.ndarray, np.ndarray]],
        L_weights: torch.Tensor
    ):
        L_logits = L_weights.detach().cpu()
        L_weights_np = (0.1 * torch.tanh(L_logits)).numpy()
        
        fine_edge_index = self.hierarchy.levels[0].edge_index.cpu().numpy()
        edge_to_weight = {}
        for idx, (r, c) in enumerate(zip(fine_edge_index[0], fine_edge_index[1])):
            edge_to_weight[(r, c)] = L_weights_np[idx]
        
        self.subdomains = []
        
        for s_idx, (interior, extended, A_local, (bnd_rows, bnd_cols)) in enumerate(
            zip(interior_nodes, extended_nodes, local_matrices, boundary_edges)
        ):
            global_to_local = {g: l for l, g in enumerate(extended)}
            
            n_local = A_local.shape[0]
            
            L_data = []
            L_rows = []
            L_cols = []
            
            for r_local, c_local in zip(bnd_rows, bnd_cols):
                r_global = extended[r_local]
                c_global = extended[c_local]
                
                weight = edge_to_weight.get((r_global, c_global), 0.0)
                if weight > 0:
                    L_data.append(weight)
                    L_rows.append(r_local)
                    L_cols.append(c_local)
            
            if len(L_data) > 0:
                L_i = coo_matrix(
                    (L_data, (L_rows, L_cols)), 
                    shape=(n_local, n_local)
                ).tocsr()
            else:
                L_i = csr_matrix((n_local, n_local))
            
            boundary_mask = np.zeros(n_local, dtype=bool)
            interior_local = [global_to_local[g] for g in interior]
            boundary_mask[:] = True
            boundary_mask[interior_local] = False
            
            A_plus_L = A_local + L_i
            A_plus_L = A_plus_L.tocsc()
            
            try:
                lu_factor = splu(A_plus_L)
            except Exception as e:
                print(f"[MG-GNN] Warning: Subdomain {s_idx} factorization failed, adding regularization")
                reg = 1e-6 * sparse.eye(n_local, format='csc')
                A_plus_L = A_plus_L + reg
                lu_factor = splu(A_plus_L)
            
            subdomain = SubdomainData(
                interior_nodes=interior,
                extended_nodes=extended,
                local_matrix=A_local,
                boundary_mask=boundary_mask,
                lu_factor=lu_factor
            )
            self.subdomains.append(subdomain)
    
    def apply(self, r: torch.Tensor) -> torch.Tensor:
        if not self._is_setup:
            raise RuntimeError("Preconditioner not set up. Call setup(A) first.")
        
        r_np = r.detach().cpu().numpy().astype(np.float64)
        x_f = self._apply_oras(r_np)
        r_new = r_np - self.A_scipy @ x_f
        x_c = self._apply_coarse_correction(r_new)
        z_np = x_f + x_c
        z = torch.from_numpy(z_np).to(dtype=self.dtype, device=self.device)
        
        return z
    
    def _apply_oras(self, r: np.ndarray) -> np.ndarray:
        x = np.zeros(self.n, dtype=np.float64)
        
        for subdomain in self.subdomains:
            r_local = r[subdomain.extended_nodes]
            
            z_local = subdomain.lu_factor.solve(r_local)
            
            interior_local_idx = np.array([
                np.where(subdomain.extended_nodes == g)[0][0]
                for g in subdomain.interior_nodes
            ])
            
            x[subdomain.interior_nodes] += z_local[interior_local_idx]
        
        return x
    
    def _apply_coarse_correction(self, r: np.ndarray) -> np.ndarray:
        r_c = self.P_scipy.T @ r
        
        if self.A_coarse_lu is not None:
            e_c = self.A_coarse_lu.solve(r_c)
        else:
            e_c = self.A_coarse_inv @ r_c
        
        x_c = self.P_scipy @ e_c
        
        return x_c

class MGGNNPreconditionerAdaptive(MGGNNPreconditioner):
    def __init__(self, update_frequency: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.update_frequency = update_frequency
        self.iteration_count = 0
    
    def apply(self, r: torch.Tensor) -> torch.Tensor:
        self.iteration_count += 1
        
        if self.iteration_count % self.update_frequency == 0:
            self._update_operators(r)
        
        return super().apply(r)
    
    def _update_operators(self, r: torch.Tensor):
        if hasattr(self.network, 'forward_with_residual'):
            with torch.no_grad():
                predictions = self.network.forward_with_residual(
                    self.hierarchy, 
                    r.cpu()
                )
            
            self._update_L_weights(predictions['L_weights'])
    
    def _update_L_weights(self, L_weights: torch.Tensor):
        pass  
