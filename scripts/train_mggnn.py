import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.sparse import csr_matrix, csc_matrix, diags
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyamg
from GNP.data.graph_hierarchy import build_graph_hierarchy, GraphHierarchy
from GNP.nn.MGGNN import MGGNN

DTYPE = torch.float64
LR_PHASE_1 = 1e-2      # Aggressive learning for imitation
LR_PHASE_2 = 1e-5      # Very gentle learning for spectral tuning (reduced for stability)
EPOCHS_PHASE_1 = 500   # Imitation learning epochs
EPOCHS_PHASE_2 = 200   # Spectral optimization epochs
GAMMA = 0.01           # Trace regularization weight
HIDDEN_DIM = 64
NUM_MG_LAYERS = 4
NUM_LEVELS = 2
COARSENING_RATIO = 8
OUTPUT_DIR = project_root / "data" / "mggnn_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_matrix_from_path(path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, csr_matrix]:
    print(f"[Data] Loading matrix from {path}")
    
    mat_data = loadmat(str(path))
    
    if 'Problem' in mat_data:
        A_scipy = mat_data['Problem']['A'][0, 0]
    else:
        for key, value in mat_data.items():
            if not key.startswith('_') and hasattr(value, 'tocsr'):
                A_scipy = value
                break
        else:
            raise ValueError(f"Could not find sparse matrix in {path}")
    
    A_scipy = csr_matrix(A_scipy).astype(np.float64)
    n = A_scipy.shape[0]
    print(f"[Data] Matrix size: {n} x {n}, nnz: {A_scipy.nnz}")
    
    diag = A_scipy.diagonal()
    diag = np.where(np.abs(diag) > 1e-12, diag, 1.0)
    D_inv = diags(1.0 / diag)
    A_scaled = D_inv @ A_scipy
    
    max_diag = np.max(np.abs(diag))
    min_diag = np.min(np.abs(diag[np.abs(diag) > 1e-12]))
    print(f"[Data] Diagonal range: [{min_diag:.2e}, {max_diag:.2e}]")
    
    A_dense = torch.tensor(A_scaled.toarray(), dtype=DTYPE, device=device)
    
    A_csc = csc_matrix(A_scaled)
    ccol_indices = torch.tensor(A_csc.indptr, dtype=torch.int64)
    row_indices = torch.tensor(A_csc.indices, dtype=torch.int64)
    values = torch.tensor(A_csc.data, dtype=DTYPE)
    A_sparse = torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=(n, n))
    
    return A_dense, A_sparse, csr_matrix(A_scaled)


def load_matrix_from_tensor(A_csc: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, csr_matrix]:
    n = A_csc.shape[0]
    print(f"[Data] Using provided matrix: {n} x {n}")
    
    A_cpu = A_csc.cpu()
    A_scipy_csc = csc_matrix(
        (A_cpu.values().numpy(),
         A_cpu.row_indices().numpy(),
         A_cpu.ccol_indices().numpy()),
        shape=(n, n)
    )
    A_scipy = A_scipy_csc.tocsr()
    
    if n > 5000:
        print(f"[Data] WARNING: Matrix size {n} may cause OOM with dense training")
    
    A_dense = torch.tensor(A_scipy.toarray(), dtype=DTYPE, device=device)
    
    return A_dense, A_csc, A_scipy

def get_amg_ground_truth(
    A_scipy: csr_matrix, 
    hierarchy: GraphHierarchy,
    device: torch.device
) -> torch.Tensor:
    print("[AMG] Building Ruge-Stuben solver for ground truth...")
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            ml = pyamg.ruge_stuben_solver(A_scipy, max_coarse=50)
        except Exception as e:
            print(f"[AMG] Ruge-Stuben failed: {e}, trying Smoothed Aggregation...")
            ml = pyamg.smoothed_aggregation_solver(A_scipy, max_coarse=50)
    
    P_amg = ml.levels[0].P  # scipy sparse matrix
    P_amg_csr = csr_matrix(P_amg)
    
    print(f"[AMG] AMG P matrix: {P_amg_csr.shape}, nnz: {P_amg_csr.nnz}")
    print(f"[AMG] Our hierarchy: {hierarchy.levels[0].num_nodes} -> {hierarchy.levels[1].num_nodes}")
    
    c2f_edges = hierarchy.coarse_to_fine_edges[0]
    coarse_idx = c2f_edges[0].cpu().numpy()
    fine_idx = c2f_edges[1].cpu().numpy()
    num_edges = len(fine_idx)
    target_weights = np.zeros(num_edges, dtype=np.float64)
    n_fine = hierarchy.levels[0].num_nodes
    n_coarse_amg = P_amg_csr.shape[1]
    n_coarse_ours = hierarchy.levels[1].num_nodes
    
    print(f"[AMG] Coarse nodes - AMG: {n_coarse_amg}, Ours: {n_coarse_ours}")
    
    matched = 0
    for i, (c_our, f) in enumerate(zip(coarse_idx, fine_idx)):
        if f < P_amg_csr.shape[0]:
            row_start = P_amg_csr.indptr[f]
            row_end = P_amg_csr.indptr[f + 1]
            amg_coarse_nodes = P_amg_csr.indices[row_start:row_end]
            amg_weights = P_amg_csr.data[row_start:row_end]
            
            if len(amg_weights) > 0:
                max_idx = np.argmax(np.abs(amg_weights))
                target_weights[i] = amg_weights[max_idx]
                matched += 1
    
    print(f"[AMG] Matched {matched}/{num_edges} edges with AMG weights")
    
    weight_sums = np.zeros(n_fine)
    np.add.at(weight_sums, fine_idx, np.abs(target_weights))
    
    zero_mask = weight_sums < 1e-12
    if np.any(zero_mask):
        num_zero = np.sum(zero_mask)
        print(f"[AMG] WARNING: {num_zero} fine nodes have no AMG connections, using uniform weights")
        edge_counts = np.zeros(n_fine)
        np.add.at(edge_counts, fine_idx, 1.0)
        for i in range(num_edges):
            f = fine_idx[i]
            if zero_mask[f] and edge_counts[f] > 0:
                target_weights[i] = 1.0 / edge_counts[f]
        weight_sums = np.zeros(n_fine)
        np.add.at(weight_sums, fine_idx, np.abs(target_weights))
    
    weight_sums = np.maximum(weight_sums, 1e-12)
    target_weights = target_weights / weight_sums[fine_idx]
    
    if np.any(np.isnan(target_weights)):
        print(f"[AMG] WARNING: NaN detected in target weights, replacing with zeros")
        target_weights = np.nan_to_num(target_weights, nan=0.0)
    
    target_P_weights = torch.tensor(target_weights, dtype=DTYPE, device=device)
    print(f"[Init] Target P weights: {target_P_weights.shape}, range: [{target_P_weights.min():.4f}, {target_P_weights.max():.4f}]")
    
    return target_P_weights

def assemble_soft_oras(
    A: torch.Tensor,
    P_weights: torch.Tensor,
    L_weights: torch.Tensor,
    hierarchy: GraphHierarchy,
    device: torch.device,
    num_subdomains: int = 16,
    overlap: int = 1
) -> Dict[str, torch.Tensor]:
    n = A.shape[0]
    n_fine = hierarchy.levels[0].num_nodes
    n_coarse = hierarchy.levels[1].num_nodes
    
    c2f_edges = hierarchy.coarse_to_fine_edges[0]
    coarse_idx = c2f_edges[0]  # [num_edges]
    fine_idx = c2f_edges[1]    # [num_edges]
    
    P_weights_activated = torch.tanh(P_weights)
    P_weights_abs = torch.abs(P_weights_activated) + 1e-12
    weight_sums = torch.zeros(n_fine, dtype=DTYPE, device=device)
    weight_sums.scatter_add_(0, fine_idx, P_weights_abs)
    weight_sums = torch.clamp(weight_sums, min=1e-12)
    P_weights_normalized = P_weights_activated / weight_sums[fine_idx]
    
    P = torch.zeros(n_fine, n_coarse, dtype=DTYPE, device=device)
    P[fine_idx, coarse_idx] = P_weights_normalized
    
    A_c = P.T @ A @ P
    A_c = A_c + 1e-8 * torch.eye(n_coarse, dtype=DTYPE, device=device)
    
    diag_A = torch.diag(A)
    diag_A = torch.where(torch.abs(diag_A) > 1e-12, diag_A, torch.ones_like(diag_A))
    D_inv = torch.diag(1.0 / diag_A)
    
    edge_index = hierarchy.levels[0].edge_index
    row_idx = edge_index[0]
    col_idx = edge_index[1]
    
    L_correction = torch.zeros(n_fine, n_fine, dtype=DTYPE, device=device)
    
    L_scaled = 0.1 * torch.tanh(L_weights)  # Keep bounded
    L_correction[row_idx, col_idx] = L_scaled
    L_correction = 0.5 * (L_correction + L_correction.T)  # Symmetrize
    
    M_fine = D_inv + 0.01 * L_correction
    
    return {
        'P': P,
        'A_c': A_c,
        'M_fine': M_fine,
        'n_coarse': n_coarse
    }


def apply_two_level_oras(
    r: torch.Tensor,
    A: torch.Tensor,
    oras_components: Dict[str, torch.Tensor]
) -> torch.Tensor:
    P = oras_components['P']
    A_c = oras_components['A_c']
    M_fine = oras_components['M_fine']
    
    if r.dim() == 1:
        r = r.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    z_fine = M_fine @ r
    residual = r - A @ z_fine
    r_coarse = P.T @ residual
    
    try:
        e_coarse = torch.linalg.solve(A_c, r_coarse)
    except RuntimeError:
        e_coarse = torch.linalg.lstsq(A_c, r_coarse).solution
    
    z_coarse = P @ e_coarse
    z = z_fine + z_coarse
    
    if squeeze_output:
        z = z.squeeze(1)
    
    return z

def imitation_loss(
    pred_P_weights: torch.Tensor,
    target_P_weights: torch.Tensor
) -> torch.Tensor:
    return torch.mean((pred_P_weights - target_P_weights) ** 2)

def spectral_loss(
    A: torch.Tensor,
    oras_components: Dict[str, torch.Tensor],
    device: torch.device,
    num_samples: int = 32,
    gamma: float = GAMMA
) -> Tuple[torch.Tensor, float]:
    n = A.shape[0]
    
    X = torch.randn(n, num_samples, dtype=DTYPE, device=device)
    X = X / torch.norm(X, dim=0, keepdim=True)  # Normalize
    AX = A @ X
    M_inv_AX = apply_two_level_oras(AX, A, oras_components)
    X_new = X - M_inv_AX
    norms_old = torch.norm(X, dim=0)
    norms_new = torch.norm(X_new, dim=0)
    rho_samples = norms_new / (norms_old + 1e-12)
    rho_mean = torch.mean(rho_samples)
    rho_max = torch.max(rho_samples)
    loss = 0.7 * rho_mean + 0.3 * rho_max
    P = oras_components['P']
    trace_reg = gamma * torch.trace(P.T @ P) / P.shape[1]
    loss = loss + trace_reg
    
    return loss, rho_mean.item()

def train_mggnn(
    problem_name: str,
    device: torch.device,
    A_csc: Optional[torch.Tensor] = None,
    epochs_p1: int = EPOCHS_PHASE_1,
    epochs_p2: int = EPOCHS_PHASE_2,
    lr_p1: float = LR_PHASE_1,
    lr_p2: float = LR_PHASE_2,
    hidden_dim: int = HIDDEN_DIM,
    verbose: bool = True
) -> str:
    if verbose:
        print("=" * 70)
        print("MG-GNN Hybrid Training (Warm Start + Spectral Optimization)")
        print("=" * 70)
        print(f"Problem: {problem_name}")
        print(f"Device: {device}")
        print()
    
    problem_safe = problem_name.replace("/", "_").replace("\\", "_")
    checkpoint_path = OUTPUT_DIR / f"mggnn_{problem_safe}.pt"
    
    if A_csc is not None:
        A_dense, A_sparse, A_scipy = load_matrix_from_tensor(A_csc, device)
    else:
        data_path = project_root / "data" / f"{problem_name}.mat"
        if not data_path.exists():
            data_path = project_root / "data" / "SuiteSparse" / f"{problem_name}.mat"
        A_dense, A_sparse, A_scipy = load_matrix_from_path(data_path, device)
    
    n = A_dense.shape[0]
    if verbose:
        print(f"[Init] Matrix loaded: {n} x {n}")
    
    if n > 5000:
        print(f"[WARNING] Matrix size {n} may cause memory issues with dense training")
        print(f"[WARNING] Consider using sparse training for large matrices")
    
    if verbose:
        print("\n[Init] Building graph hierarchy...")
    hierarchy = build_graph_hierarchy(
        A_sparse.to(device),
        num_levels=NUM_LEVELS,
        coarsening_ratio=COARSENING_RATIO,
        seed=42,
        device=str(device)
    )
    if verbose:
        print(f"[Init] Hierarchy: {hierarchy.levels[0].num_nodes} -> {hierarchy.levels[1].num_nodes} nodes")
    
    if verbose:
        print("\n[Init] Extracting AMG ground truth...")
    target_P_weights = get_amg_ground_truth(A_scipy, hierarchy, device)
    if verbose:
        print(f"[Init] Target P weights: {target_P_weights.shape}, "
              f"range: [{target_P_weights.min():.4f}, {target_P_weights.max():.4f}]")
    
    if verbose:
        print("\n[Init] Creating MG-GNN model...")
    model = MGGNN(
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=NUM_MG_LAYERS,
        num_levels=NUM_LEVELS,
        K=3,
        dropout=0.1,
        dtype=torch.float32  # Model in float32, convert outputs
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[Init] Model parameters: {num_params:,}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: Imitation Learning (Cloning AMG)")
        print("=" * 70)
    
    optimizer_p1 = optim.Adam(model.parameters(), lr=lr_p1)
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, epochs_p1)
    
    best_loss_p1 = float('inf')
    phase1_ckpt = OUTPUT_DIR / f"mggnn_{problem_safe}_phase1.pt"
    
    for epoch in range(epochs_p1):
        model.train()
        optimizer_p1.zero_grad()
        
        predictions = model(hierarchy)
        pred_P_weights_raw = predictions['P_weights'][0].to(DTYPE)  # Level 0->1
        
        pred_P_weights = torch.tanh(pred_P_weights_raw)
        loss = imitation_loss(pred_P_weights, target_P_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_p1.step()
        scheduler_p1.step()
        
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            with torch.no_grad():
                pred_np = pred_P_weights.cpu().numpy()
                target_np = target_P_weights.cpu().numpy()
                correlation = np.corrcoef(pred_np, target_np)[0, 1]
            
            print(f"  Epoch {epoch+1:4d}/{epochs_p1} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Corr: {correlation:.4f} | "
                  f"LR: {scheduler_p1.get_last_lr()[0]:.2e}")
        
        if loss.item() < best_loss_p1:
            best_loss_p1 = loss.item()
            torch.save(model.state_dict(), phase1_ckpt)
    
    if verbose:
        print(f"\n[Phase 1] Complete. Best loss: {best_loss_p1:.6f}")
    
    model.load_state_dict(torch.load(phase1_ckpt))
    
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: Spectral Optimization (Fine-Tuning)")
        print("=" * 70)
    
    with torch.no_grad():
        model.eval()
        baseline_preds = model(hierarchy)
        baseline_P = baseline_preds['P_weights'][0].to(DTYPE)
        baseline_L = baseline_preds['L_weights'].to(DTYPE)
        baseline_oras = assemble_soft_oras(A_dense, baseline_P, baseline_L, hierarchy, device)
        _, baseline_rho = spectral_loss(A_dense, baseline_oras, device, num_samples=50)
    
    if verbose:
        print(f"[Phase 2 Start] Baseline AMG Clone ρ: {baseline_rho:.4f}")
        print(f"[Phase 2] Only saving model if ρ improves over baseline")
    
    optimizer_p2 = optim.Adam(model.parameters(), lr=lr_p2)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, epochs_p2)
    
    best_rho = baseline_rho
    best_loss_p2 = float('inf')
    improved_over_baseline = False
    
    for epoch in range(epochs_p2):
        model.train()
        optimizer_p2.zero_grad()
        
        predictions = model(hierarchy)
        pred_P_weights = predictions['P_weights'][0].to(DTYPE)
        pred_L_weights = predictions['L_weights'].to(DTYPE)
        
        oras_components = assemble_soft_oras(
            A_dense,
            pred_P_weights, 
            pred_L_weights,
            hierarchy,
            device
        )
        
        loss, rho_est = spectral_loss(A_dense, oras_components, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer_p2.step()
        scheduler_p2.step()
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            status = "  [BEAT BASELINE]" if rho_est < baseline_rho else ""
            print(f"  Epoch {epoch+1:4d}/{epochs_p2} | "
                  f"Loss: {loss.item():.6f} | "
                  f"ρ_est: {rho_est:.4f} | "
                  f"LR: {scheduler_p2.get_last_lr()[0]:.2e}{status}")
        
        if rho_est < best_rho:
            best_rho = rho_est
            best_loss_p2 = loss.item()
            torch.save(model.state_dict(), checkpoint_path)
            if rho_est < baseline_rho and not improved_over_baseline:
                improved_over_baseline = True
                if verbose:
                    print(f"  [NEW BEST] Epoch {epoch+1}: ρ = {best_rho:.4f} (improved over AMG baseline!)")
    
    if verbose:
        if improved_over_baseline:
            print(f"\n[Phase 2] Complete. Best ρ: {best_rho:.4f} (improved over baseline {baseline_rho:.4f})")
        else:
            print(f"\n[Phase 2] Complete. Could not improve over baseline ρ: {baseline_rho:.4f}")
            print(f"[Phase 2] Using Phase 1 (AMG clone) model as final checkpoint")
    
    if phase1_ckpt.exists():
        phase1_ckpt.unlink()
    
    if verbose:
        print("\n" + "=" * 70)
        print("FINAL VALIDATION")
        print("=" * 70)
    
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    with torch.no_grad():
        predictions = model(hierarchy)
        pred_P_weights = predictions['P_weights'][0].to(DTYPE)
        pred_L_weights = predictions['L_weights'].to(DTYPE)
        
        oras_components = assemble_soft_oras(
            A_dense, pred_P_weights, pred_L_weights, hierarchy, device
        )
        
        _, rho_final = spectral_loss(A_dense, oras_components, device, num_samples=100)
        
        if verbose:
            print(f"\n[Validation] Final estimated ρ: {rho_final:.4f}")
            print(f"[Validation] Expected iterations for 1e-6 tol: ~{int(-6 / np.log10(max(rho_final, 0.01)))}")
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"Training complete! Model saved to: {checkpoint_path}")
        print("=" * 70)
    
    return str(checkpoint_path)

def train(args):
    """Standalone training entry point."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_mggnn(
        problem_name=args.problem,
        device=device,
        epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2,
        lr_p1=args.lr_p1,
        lr_p2=args.lr_p2,
        hidden_dim=args.hidden_dim,
        verbose=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MG-GNN with hybrid strategy")
    parser.add_argument("--problem", type=str, default="Boeing/msc01050")
    parser.add_argument("--epochs-p1", type=int, default=EPOCHS_PHASE_1)
    parser.add_argument("--epochs-p2", type=int, default=EPOCHS_PHASE_2)
    parser.add_argument("--lr-p1", type=float, default=LR_PHASE_1)
    parser.add_argument("--lr-p2", type=float, default=LR_PHASE_2)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    args = parser.parse_args()
    
    train(args)
