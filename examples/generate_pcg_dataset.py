"""
PCG Trajectory Harvester for Offline Training.

This script harvests real PCG trajectories (residuals and errors) from synthetic
problems where we know the ground truth solution. The collected data can be used
to train the GNN preconditioner to predict error from residual (r_i -> e_i).

Usage:
    python generate_pcg_dataset.py --problem HB/bcsstk16 --num_runs 50 --output data/pcg_harvested.pt
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GNP.solver import PCG
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse
from GNP.problems import gen_x_randn
from GNP import config

def parse_args():
    parser = argparse.ArgumentParser(description='Harvest PCG trajectories for offline training')
    parser.add_argument('--problem', type=str, default='HB/bcsstk16')
    parser.add_argument('--location', type=str, default=config.SUITE_SPARSE_PATH, help='Path to SuiteSparse data directory')
    parser.add_argument('--num-runs', type=int, default=50, help='Number of random problems to generate')
    parser.add_argument('--max-iters', type=int, default=200, help='Maximum PCG iterations per run')
    parser.add_argument('--rtol', type=float, default=1e-10, help='Relative tolerance for PCG convergence')
    parser.add_argument('--output', type=str, default='data/pcg_harvested.pt', help='Output path for harvested dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-every', type=int, default=1, help='Sample every N-th iteration (1 = all iterations)')
    return parser.parse_args()

def harvest_single_run(solver, A, x_true, max_iters, rtol):
    """
    Run PCG on a single synthetic problem and harvest the trajectory.
    
    Args:
        solver: PCG solver instance
        A: System matrix (sparse)
        x_true: Ground truth solution
        max_iters: Maximum iterations
        rtol: Relative tolerance
        
    Returns:
        residuals: List of residual tensors (r_i) on same device as input
        errors: List of error tensors (e_i = x_true - x_i) on same device as input
    """
    with torch.no_grad():
        b = A @ x_true
        errors = []
        
        _, _, _, _, _, _, trajectory = solver.solve(A, b, progress_bar=False, return_trajectory=True)
        history_r, history_x = trajectory

        for x_i in history_x:
            e_i = x_true - x_i
            errors.append(e_i)
    
    return history_r, errors


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device(args.device)

    A = load_suitesparse(args.location, args.problem, device)
    A = scale_A_by_spectral_radius(A)
    n = A.shape[0]

    solver = PCG()
    all_residuals = []
    all_errors = []
    total_samples = 0

    print(f"Device: {device}")
    print(f"\nLoading problem: {args.problem}")
    print(f"Matrix dimension: n={n}, nnz={A._nnz()}")
    print(f"\nHarvesting trajectories from {args.num_runs} random problems...")
    print(f"Max iterations per run: {args.max_iters}")
    print(f"Convergence tolerance: {args.rtol}")
    
    for run_idx in tqdm(range(args.num_runs), desc="Harvesting"):
        x_true = gen_x_randn(n).to(device)
        residuals, errors = harvest_single_run(solver, A, x_true, args.max_iters, args.rtol)
        
        if args.skip_every > 1:
            residuals = residuals[::args.skip_every]
            errors = errors[::args.skip_every]
        
        for r, e in zip(residuals, errors):
            all_residuals.append(r.cpu())
            all_errors.append(e.cpu())
        
        total_samples += len(residuals)
    
    print(f"\nTotal samples collected: {total_samples}")
    
    dataset_r = torch.stack(all_residuals, dim=0)  # Shape: (N_samples, n)
    dataset_e = torch.stack(all_errors, dim=0)      # Shape: (N_samples, n)
    print(f"Dataset residuals shape: {dataset_r.shape}")
    print(f"Dataset errors shape: {dataset_e.shape}")
    r_norms = torch.linalg.norm(dataset_r, dim=1)
    e_norms = torch.linalg.norm(dataset_e, dim=1)
    print(f"\nResidual norms - mean: {r_norms.mean():.4e}, std: {r_norms.std():.4e}")
    print(f"Error norms - mean: {e_norms.mean():.4e}, std: {e_norms.std():.4e}")
    
    output_path = os.path.abspath(args.output)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    
    dataset = {
        'r': dataset_r,
        'e': dataset_e,
        'metadata': {
            'problem': args.problem,
            'n': n,
            'num_runs': args.num_runs,
            'max_iters': args.max_iters,
            'rtol': args.rtol,
            'total_samples': total_samples,
            'skip_every': args.skip_every,
            'seed': config.SEED
        }
    }
    
    torch.save(dataset, output_path)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    main()
