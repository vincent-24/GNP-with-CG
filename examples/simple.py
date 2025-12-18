import os
import time
import torch
import argparse
import warnings
import random
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from GNP.problems import *
from GNP.solver import GMRES, FCG 
from GNP.precond import *
from GNP.nn import ResGCN, SplitResGCN
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse

def get_timestamp_dir(base_dump_path):
    """Creates a directory based on today's date (MM-DD-YYYY)."""
    date_str = datetime.now().strftime("%m-%d-%Y")
    path = os.path.join(base_dump_path, date_str)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def generate_run_id():
    return str(random.randint(10000, 99999))

def resolve_checkpoint_path(dump_root, filename):
    """
    Resolves the full path of a checkpoint. 
    Assumes filename is in dump_root if not an absolute path.
    """
    if os.path.isabs(filename):
        return filename
    return os.path.join(dump_root, filename)

def main():
    parser = argparse.ArgumentParser(description='Compare FCG and FGMRES with Checkpoint Management')
    parser.add_argument('--location', type=str, default='~/data/SuiteSparse/ssget/mat')
    parser.add_argument('--problem', type=str, default='HB/bcsstk17') 
    parser.add_argument('--dump_root', type=str, default='./dump/', help='Root directory for checkpoints')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fcg-ckpt', type=str, default=None, help='ckpt.pt')
    parser.add_argument('--fgmres-ckpt', type=str, default=None, help='ckpt.pt')
    args = parser.parse_args()

    # directory setup
    args.dump_root = os.path.abspath(os.path.expanduser(args.dump_root))
    Path(args.dump_root).mkdir(parents=True, exist_ok=True)
    plot_dir = get_timestamp_dir(args.dump_root)
    print(f"plots will be saved to: {plot_dir}")
    print(f"checkpoints will be saved/loaded from: {args.dump_root}")

    # problem setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nLoading {args.problem}...')
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
    
    params = {
        'restart': 10,
        'max_iters': 100,
        'm': 40,
        'batch_size': 16,
        'epochs': 2000,
        'lr': 1e-3
    }

    configs = [
        {
            'name': 'FGMRES', 
            'solver_cls': GMRES, 
            'use_lanczos': False, 
            'net_cls': ResGCN,
            'user_ckpt': args.fgmres_ckpt
        },
        {
            'name': 'FCG',    
            'solver_cls': FCG,   
            'use_lanczos': True,  
            'net_cls': SplitResGCN,
            'user_ckpt': args.fcg_ckpt
        }
    ]

    results = {}
    run_ids = {} 

    # main loop
    for config in configs:
        name = config['name']
        print(f"\n{'='*40}")
        print(f"Configuration: {name}")
        print(f"{'='*40}")

        # no precond
        solver = config['solver_cls']()
        print(f"Solving {name} (No Preconditioner)...")
        solve_kwargs = {
            'restart': params['restart'], 
            'max_iters': params['max_iters'], 
            'progress_bar': True
        }
        # truncation for FCG
        if name == 'FCG': solve_kwargs['truncation_k'] = 5

        _, _, _, no_pre_res, no_pre_time = solver.solve(A, b, M=None, **solve_kwargs)

        # prep precond load/train
        print(f"Preparing {name} Preconditioner ({config['net_cls'].__name__})...")
        net = config['net_cls'](A, num_layers=8, embed=16, hidden=32, drop_rate=0.0).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])
        
        M = GNP(A, 'x_mix', params['m'], net, device, use_lanczos=config['use_lanczos'])

        # checkpoint logic
        if config['user_ckpt']:
            # load user checkpoint
            ckpt_path = resolve_checkpoint_path(args.dump_root, config['user_ckpt'])
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
            print(f"Loading checkpoint: {config['user_ckpt']}")
            net.load_state_dict(torch.load(ckpt_path, map_location=device))
            
            try:
                run_id = Path(config['user_ckpt']).stem.split('_')[-1]
            except:
                run_id = "loaded"
            run_ids[name] = run_id
            hist_loss = []
            
        else:
            # train without checkpoint
            run_id = generate_run_id()
            run_ids[name] = run_id
            
            clean_problem_name = args.problem.replace('/', '_')
            ckpt_filename = f"{clean_problem_name}_{name}_{run_id}.pt"
            ckpt_path = os.path.join(args.dump_root, ckpt_filename)
            
            temp_prefix = os.path.join(args.dump_root, f"TEMP_{name}_{run_id}_")
            
            print(f"Training new model (Run ID: {run_id})...")
            hist_loss, _, _, trained_file = M.train(
                params['batch_size'], 1, params['epochs'], optimizer, 
                num_workers=args.num_workers, 
                checkpoint_prefix_with_path=temp_prefix,
                progress_bar=True
            )
            
            if trained_file and os.path.exists(trained_file):
                print(f"Saving checkpoint to: {ckpt_filename}")
                os.rename(trained_file, ckpt_path)
            else:
                print("Warning: No checkpoint file returned from training.")

        # solve with precond
        print(f"Solving {name} (With GNP)...")
        try:
            _, _, _, gnp_res, gnp_time = solver.solve(A, b, M=M, **solve_kwargs)
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

    # plotting
    print("\nGenerating Comparison Plots...")
    
    suffix = f"_fcg{run_ids.get('FCG', 'nan')}_fgmres{run_ids.get('FGMRES', 'nan')}"
    base_prefix = os.path.join(plot_dir, f"{args.problem.replace('/', '_')}_comparison")
    
    # convergence plot
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        if len(res['gnp_res']) > 0:
            plt.semilogy(res['gnp_res'], linewidth=2, label=f'{name} (GNP) #{run_ids[name]}')
        plt.semilogy(res['no_pre_res'], linestyle='--', alpha=0.5, label=f'{name} (No Precond)')
            
    plt.title(f'Convergence Comparison ({args.problem})')
    plt.xlabel('Iterations')
    plt.ylabel('Relative Residual')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    out_file = f"{base_prefix}_iters{suffix}.png"
    plt.savefig(out_file)
    print(f"Saved plot: {out_file}")

    # time plot
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        if len(res['gnp_res']) > 0:
            plt.semilogy(res['gnp_time'], res['gnp_res'], linewidth=2, label=f'{name} (GNP) #{run_ids[name]}')
            
    plt.title(f'Time-to-Solution Comparison ({args.problem})')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Residual')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    out_file = f"{base_prefix}_time{suffix}.png"
    plt.savefig(out_file)
    print(f"Saved plot: {out_file}")

if __name__ == '__main__':
    main()