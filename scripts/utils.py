"""
Utility functions for experiment setup, data loading, plotting, and preconditioner factory.
"""
import os
import sys

# Ensure GNP package is importable from any directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from GNP.problems import gen_x_all_ones
from GNP.precond import GNP
from GNP.precond.ILU import ILU
from GNP.precond.IChol import IChol
from GNP.precond.AMGPreconditioner import AMGPreconditioner
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse
from GNP import config
from GNP.factory import get_solver_info, get_network_class

def get_timestamp_str():
    return datetime.now().strftime("%m-%d-%Y")

def get_timestamp_dir(base_dump_path):
    path = os.path.join(base_dump_path, get_timestamp_str())
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path, 'configs')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    return path

def generate_run_id():
    return str(random.randint(10000, 99999))

def setup_experiment(args):
    run_id = generate_run_id()
    
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    args.dump_root = os.path.abspath(os.path.expanduser(args.dump_root))
    Path(args.dump_root).mkdir(parents=True, exist_ok=True)
    plot_dir = get_timestamp_dir(args.dump_root)
    print(f"\nOutput directory: {plot_dir}")
    print(f"Run ID: {run_id}")
    
    return plot_dir, run_id

def load_problem(args, device):
    """
    Load matrix A, scale it, generate b, and return (A, A_csc, b, x_gt).
    A_csc is created if any classical preconditioner is in EXPERIMENTS.
    """
    print(f'\nLoading {args.problem}...')
    A = load_suitesparse(args.location, args.problem, device)
    A = scale_A_by_spectral_radius(A)
    n = A.shape[0]
    print(f'Matrix n={n}, nnz={A._nnz()}')
    
    classical_preconds = {'ILU', 'IChol', 'AMG'}
    needs_csc = any(
        exp.get('precond') in classical_preconds 
        for exp in config.EXPERIMENTS
    )
    
    A_csc = None
    if needs_csc:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Sparse CSC tensor support is in beta')
            A_csc = A.to_sparse_csc()
        enabled = [exp['name'] for exp in config.EXPERIMENTS if exp.get('precond') in classical_preconds]
        print(f"Classical preconditioners enabled: {', '.join(enabled)}")
    
    x_gt = gen_x_all_ones(n).to(device)
    b = A @ x_gt
    
    return A, A_csc, b, x_gt

def get_preconditioner(precond_type, A, A_csc, device, args, master_ckpt_path=None, **kwargs):
    solver_name = kwargs.pop('solver_name', config.BASELINE_SOLVER)
    
    if precond_type is None:
        return None
    
    if precond_type == 'GNP':
        if master_ckpt_path is None:
            raise ValueError("master_ckpt_path required for GNP preconditioner")
        
        _, cfg = get_solver_info(solver_name)
        
        if args.network_override:
            net_cls = get_network_class(args.network_override)
        else:
            net_cls = get_network_class(cfg['default_net'])
        
        current_m = config.LANCZOS_M if cfg['use_lanczos'] else config.ARNOLDI_M
        
        net_kwargs = {
            'A': A, 
            'num_layers': config.NUM_LAYERS, 
            'embed': config.EMBED_DIM,
            'hidden': config.HIDDEN_DIM, 
            'drop_rate': config.DROP_RATE
        }
        if net_cls.__name__ == 'SplitResGCN':
            net_kwargs['tie_weights'] = args.tie_weights
        
        net = net_cls(**net_kwargs).to(device)
        net.load_state_dict(torch.load(master_ckpt_path, map_location=device))
        
        return GNP(A, 'x_mix', current_m, net, device, use_lanczos=cfg['use_lanczos'])
    
    elif precond_type == 'IChol':
        return IChol(A_csc, **kwargs)
    
    elif precond_type == 'ILU':
        return ILU(A_csc, ilu_factors_file=None, save_ilu_factors=False, **kwargs)
    
    elif precond_type == 'AMG':
        return AMGPreconditioner(A_csc, **kwargs)
    
    else:
        raise ValueError(f"Unknown preconditioner type: {precond_type}")

def plot_results(results, args, plot_dir, run_id):
    print("\nGenerating Comparison Plots...")
    
    base_prefix = os.path.join(plot_dir, f"{args.problem.replace('/', '_')}_ID{run_id}_comparison")
    style_map = {exp['name']: exp['style'] for exp in config.EXPERIMENTS}
    
    plt.figure(figsize=(12, 6))
    
    for name, res in results.items():
        if len(res.get('res_history', [])) > 0:
            style = style_map.get(name, {'color': 'gray', 'linestyle': '-', 'linewidth': 2})
            plt.semilogy(
                res['res_history'], 
                color=style.get('color', 'gray'),
                linestyle='-', 
                linewidth=style.get('linewidth', 2),
                label=name
            )
    
    plt.title(f'Convergence Comparison ({args.problem})')
    plt.xlabel('Iterations')
    plt.ylabel('Relative Residual')
    plt.xlim(0, config.MAX_ITERS)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_prefix}_iters.png", bbox_inches='tight')
    plt.figure(figsize=(12, 6))
    
    for name, res in results.items():
        if len(res.get('res_history', [])) > 0 and len(res.get('time_history', [])) > 0:
            style = style_map.get(name, {'color': 'gray', 'linestyle': '-', 'linewidth': 2})
            plt.semilogy(
                res['time_history'], 
                res['res_history'],
                color=style.get('color', 'gray'),
                linestyle='-',  
                linewidth=style.get('linewidth', 2),
                label=name
            )
    
    plt.title(f'Time-to-Solution Comparison ({args.problem})')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Residual')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_prefix}_time.png", bbox_inches='tight')
    
    for name, res in results.items():
        ortho_map = res.get('ortho_map')
        if ortho_map is not None:
            plt.figure(figsize=(8, 6))
            im = plt.imshow(ortho_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar(im)
            
            if 'GMRES' in name or 'FGMRES' in name:
                plt.title(r"Euclidean Orthogonality: $|v_i^T v_j|$ - " + name)
            else:
                plt.title(r"A-Orthogonality: $|d_i^T A d_j|$ - " + name)
            
            plt.xlabel(r"Iteration $j$")
            plt.ylabel(r"Iteration $i$")
            plt.tight_layout()
            filename = f"{args.problem.replace('/', '_')}_ID{run_id}_{name.replace(' ', '_').replace('(', '').replace(')', '')}_heatmap.png"
            plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
            print(f"Heatmap saved: {filename}")
    
    print(f"Plots saved to {base_prefix}_*.png")

def plot_learning_curve(train_loss, val_loss, args, plot_dir, run_id, best_epoch=None):
    if not train_loss or not val_loss:
        print("Warning: Empty loss history, skipping learning curve plot.")
        return
    
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='#2563eb', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_loss, label='Validation Loss', color='#ea580c', linewidth=2, linestyle='-', marker='s', markersize=4)
    
    if best_epoch is not None and 1 <= best_epoch <= len(val_loss):
        best_val = val_loss[best_epoch - 1]
        plt.axvline(x=best_epoch, color='#16a34a', linestyle=':', linewidth=1.5, alpha=0.7)
        plt.scatter([best_epoch], [best_val], color='#16a34a', s=100, zorder=5, 
                    marker='*', label=f'Best Val (epoch {best_epoch})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Physics Loss', fontsize=12)
    plt.title(f'GNP Training Learning Curve ({args.problem})', fontsize=14)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='-', alpha=0.2)
    plt.legend(loc='upper right', fontsize=10)
    plt.xticks(epochs)
    plt.xlim(0.5, len(train_loss) + 0.5)
    plt.tight_layout()

    filename = f"{args.problem.replace('/', '_')}_ID{run_id}_learning_curve.png"
    save_path = os.path.join(plot_dir, filename)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curve saved: {save_path}")