import os
import time
import torch
import argparse
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

from GNP.problems import *
from GNP.solver import GMRES, FCG 
from GNP.precond import *
# Ensure these imports are correct based on your previous fixes
from GNP.nn import ResGCN, SplitResGCN
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse

def main():
    parser = argparse.ArgumentParser(description='Compare FCG and FGMRES with Caching')
    parser.add_argument('--location', type=str, default='~/data/SuiteSparse/ssget/mat')
    parser.add_argument('--problem', type=str, default='HB/bcsstk17') 
    parser.add_argument('--out_path', type=str, default='./dump/')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--force_retrain', action='store_true', help='Ignore saved models and retrain')
    args = parser.parse_args()

    # --- SETUP ---
    configs = [
        {'name': 'FGMRES', 'solver_cls': GMRES, 'use_lanczos': False, 'net_cls': ResGCN},
        {'name': 'FCG',    'solver_cls': FCG,   'use_lanczos': True,  'net_cls': SplitResGCN}
    ]

    restart = 5 
    max_iters = 100
    m = 40
    batch_size = 16
    epochs = 2000
    lr = 5e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load & Normalize Matrix
    print(f'Loading {args.problem}...')
    try:
        A = load_suitesparse(args.location, args.problem, device)
    except Exception:
        print(f"Could not load {args.problem}, falling back to Synthetic SPD Laplacian.")
        A = gen_1d_laplacian(1000).to(device)
        
    A = scale_A_by_spectral_radius(A)
    n = A.shape[0]
    print(f'Matrix n={n}, nnz={A._nnz()}')

    x_gt = gen_x_all_ones(n).to(device)
    b = A @ x_gt
    
    args.out_path = os.path.abspath(os.path.expanduser(args.out_path))
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    # Base prefix for the problem
    base_prefix = os.path.join(args.out_path, f"{args.problem.replace('/', '_')}")

    results = {}

    # --- MAIN LOOP ---
    for config in configs:
        name = config['name']
        print(f"\n{'='*40}")
        print(f"Configuration: {name}")
        print(f"{'='*40}")

        # 1. Baseline Solve (No Precond) - Always run (fast)
        solver = config['solver_cls']()
        print(f"Solving {name} (No Preconditioner)...")
        _, _, _, no_pre_res, no_pre_time = solver.solve(
            A, b, M=None, restart=restart, max_iters=max_iters, progress_bar=True
        )

        # 2. Prepare Preconditioner
        print(f"Preparing {name} Preconditioner ({config['net_cls'].__name__})...")
        net = config['net_cls'](A, num_layers=8, embed=16, hidden=32, drop_rate=0.0).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        
        # Define Checkpoint Path
        # e.g., dump/HB_bcsstk17_FGMRES_best.pt
        model_prefix = f"{base_prefix}_{name}_"
        model_path = f"{model_prefix}best.pt"
        
        M = GNP(A, 'x_mix', m, net, device, use_lanczos=config['use_lanczos'])

        # CHECK FOR CACHED MODEL
        if os.path.exists(model_path) and not args.force_retrain:
            print(f"Found cached model: {model_path}")
            print("Loading weights and skipping training...")
            net.load_state_dict(torch.load(model_path, map_location=device))
            hist_loss = [] # Loss history is lost, but that's fine for solver comparison
        else:
            print(f"No cached model found (or forced retrain). Training...")
            hist_loss, _, _, _ = M.train(
                batch_size, 1, epochs, optimizer, num_workers=args.num_workers, 
                checkpoint_prefix_with_path=model_prefix, # Save for next time
                progress_bar=True
            )

        # 3. Solve with Preconditioner
        print(f"Solving {name} (With GNP)...")
        try:
            _, _, _, gnp_res, gnp_time = solver.solve(
                A, b, M=M, restart=restart, max_iters=max_iters, progress_bar=True
            )
        except Exception as e:
            print(f"Solver failed: {e}")
            gnp_res = []
            gnp_time = []

        results[name] = {
            'no_pre_res': no_pre_res,
            'no_pre_time': no_pre_time,
            'gnp_res': gnp_res,
            'gnp_time': gnp_time,
            'train_loss': hist_loss
        }

    # --- PLOTTING ---
    print("\nGenerating Comparison Plots...")
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        if len(res['gnp_res']) > 0:
            plt.semilogy(res['gnp_res'], linewidth=2, label=f'{name} (GNP)')
        plt.semilogy(res['no_pre_res'], linestyle='--', alpha=0.5, label=f'{name} (No Precond)')
            
    plt.title(f'{args.problem}: Solver Convergence Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Relative Residual')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.savefig(f"{base_prefix}_comparison_iters.png")
    print(f"Saved {base_prefix}_comparison_iters.png")

    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        if len(res['gnp_res']) > 0:
            plt.semilogy(res['gnp_time'], res['gnp_res'], linewidth=2, label=f'{name} (GNP)')
            
    plt.title(f'{args.problem}: Time-to-Solution Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Residual')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.savefig(f"{base_prefix}_comparison_time.png")
    print(f"Saved {base_prefix}_comparison_time.png")

if __name__ == '__main__':
    main()