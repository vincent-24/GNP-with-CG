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
import math
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
    parser.add_argument('--problem', type=str, default=config.PROBLEM_PATH)
    parser.add_argument('--location', type=str, default=config.SUITE_SPARSE_PATH, help='Path to SuiteSparse data directory')
    parser.add_argument('--num-runs', type=int, default=config.HARVEST_NUM_RUNS, help='Number of random problems to generate')
    parser.add_argument('--max-iters', type=int, default=config.HARVEST_MAX_ITERS, help='Maximum PCG iterations per run')
    parser.add_argument('--rtol', type=float, default=config.HARVEST_RTOL, help='Relative tolerance for PCG convergence')
    parser.add_argument('--output', type=str, default=None, help='Output path (auto-generated if not specified)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-every', type=int, default=1, help='Sample every N-th iteration (1 = all iterations, ignored if log-sampling is enabled)')
    parser.add_argument('--log-sampling', action='store_true', default=config.LOG_SAMPLING, help='Use logarithmic iteration sampling (e.g., 1, 2, 4, 8, 16...)')
    parser.add_argument('--log-sampling-base', type=float, default=config.LOG_SAMPLING_BASE, help='Base for logarithmic sampling (default: 2.0)')
    args = parser.parse_args()
    
    # Auto-generate output path to match train.py expectations
    if args.output is None:
        problem_name = args.problem.replace('/', '_')
        args.output = os.path.join(config.OFFLINE_DATASET_DIR, f'pcg_harvested_{problem_name}.pt')
    
    return args

def get_log_sample_indices(total_iters, base=2.0):
    """
    Generate logarithmically-spaced iteration indices.
    
    Returns indices like [0, 1, 2, 4, 8, 16, ...] up to total_iters-1.
    Always includes iteration 0 (initial) and samples more densely at early iterations.
    
    Args:
        total_iters: Total number of iterations available
        base: Base for logarithmic spacing (default 2.0 gives powers of 2)
        
    Returns:
        List of iteration indices to sample
    """
    if total_iters <= 0:
        return []
    
    indices = set([0])  # Always include first iteration
    
    # Add powers of base: base^0=1, base^1, base^2, ...
    power = 0
    while True:
        idx = int(base ** power)
        if idx >= total_iters:
            break
        indices.add(idx)
        power += 1
    
    # Always include last iteration if we have samples
    if total_iters > 0:
        indices.add(total_iters - 1)
    
    return sorted(indices)


def harvest_single_run(solver, A, x_true, max_iters, rtol):
    """
    Run PCG on a single synthetic problem and harvest error vectors.
    
    Storage Optimization: Only error vectors are collected.
    Residuals can be reconstructed on-the-fly via r = A @ e.
    
    Args:
        solver: PCG solver instance
        A: System matrix (sparse)
        x_true: Ground truth solution
        max_iters: Maximum iterations
        rtol: Relative tolerance
        
    Returns:
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
    
    return errors


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device(args.device)

    A = load_suitesparse(args.location, args.problem, device)
    # A = scale_A_by_spectral_radius(A)
    n = A.shape[0]

    solver = PCG()
    all_errors = []
    total_samples = 0

    print(f"Device: {device}")
    print(f"\nLoading problem: {args.problem}")
    print(f"Matrix dimension: n={n}, nnz={A._nnz()}")
    print(f"\nHarvesting trajectories from {args.num_runs} random problems...")
    print(f"Max iterations per run: {args.max_iters}")
    print(f"Convergence tolerance: {args.rtol}")
    print(f"Storage optimization: Only saving error vectors (r computed on-the-fly)")
    
    if args.log_sampling:
        print(f"Log sampling enabled with base {args.log_sampling_base}")
        print(f"  (sampling iterations like 0, 1, 2, 4, 8, 16, ... to bias towards early iterations)")
    elif args.skip_every > 1:
        print(f"Skip every: {args.skip_every}")
    
    for run_idx in tqdm(range(args.num_runs), desc="Harvesting"):
        # x_true = gen_x_randn(n).to(device)
        x_true = gen_x_randn(n).to(device).to(torch.float64)
        errors = harvest_single_run(solver, A, x_true, args.max_iters, args.rtol)
        
        # Apply sampling strategy
        if args.log_sampling:
            # Logarithmic sampling: 0, 1, 2, 4, 8, 16, ...
            sample_indices = get_log_sample_indices(len(errors), args.log_sampling_base)
            errors = [errors[i] for i in sample_indices]
        elif args.skip_every > 1:
            # Linear skip sampling
            errors = errors[::args.skip_every]
        
        for e in errors:
            all_errors.append(e.cpu())
        
        total_samples += len(errors)
    
    print(f"\nTotal samples collected: {total_samples}")
    
    dataset_e = torch.stack(all_errors, dim=0)      # Shape: (N_samples, n)
    print(f"Dataset errors shape: {dataset_e.shape}")
    e_norms = torch.linalg.norm(dataset_e, dim=1)
    print(f"Error norms - mean: {e_norms.mean():.4e}, std: {e_norms.std():.4e}")
    
    output_path = os.path.abspath(args.output)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    
    # Storage optimization: only save errors, residuals computed on-the-fly as r = A @ e
    dataset = {
        'e': dataset_e,
        'metadata': {
            'problem': args.problem,
            'n': n,
            'num_runs': args.num_runs,
            'max_iters': args.max_iters,
            'rtol': args.rtol,
            'total_samples': total_samples,
            'skip_every': args.skip_every,
            'log_sampling': args.log_sampling,
            'log_sampling_base': args.log_sampling_base if args.log_sampling else None,
            'seed': config.SEED
        }
    }
    
    torch.save(dataset, output_path)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == '__main__':
    main()