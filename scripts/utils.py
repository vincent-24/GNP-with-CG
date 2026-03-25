import os
import sys

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
from GNP.problems import gen_x_all_ones, gen_x_randn, gen_x_sinusoid, gen_x_alternating, gen_x_ramp
from GNP.precond import GNP
from GNP.precond.ILU import ILU
from GNP.precond.IChol import IChol
from GNP.precond.AMGPreconditioner import AMGPreconditioner
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse
from GNP import config
from GNP.factory import get_solver_info, get_network_class

def get_timestamp_str():
    return datetime.now().strftime("%m-%d-%Y")

def get_month_str():
    return datetime.now().strftime("%m-%Y")

def get_matrix_output_dir(base_dump_path, problem):
    """Create hierarchical output directory: base/MM-YYYY/MM-DD-YYYY/Category/MatrixName/"""
    month_str = get_month_str()
    date_str = get_timestamp_str()

    # Parse problem name (e.g., "Boeing/msc01050" -> category="Boeing", name="msc01050")
    if '/' in problem:
        category, name = problem.split('/', 1)
    else:
        category = "misc"
        name = problem

    path = os.path.join(base_dump_path, month_str, date_str, category, name)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_timestamp_dir(base_dump_path):
    """Legacy function - creates date-based directory without matrix hierarchy."""
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

    # Use new hierarchical directory structure: dump/MM-YYYY/MM-DD-YYYY/Category/MatrixName/
    plot_dir = get_matrix_output_dir(args.dump_root, args.problem)

    print(f"Output directory: {plot_dir}")
    print(f"Run ID: {run_id}")

    return plot_dir, run_id

def load_problem(args, device):
    print(f'\nLoading {args.problem}')
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
    
    # x_gt = gen_x_all_ones(n).to(device)     # <- x_gt used in the GNP paper
    x_gt = gen_x_randn(n).to(device).to(torch.float64)
    # x_gt = gen_x_ramp(n).to(device)
    # x_gt = gen_x_sinusoid(n).to(device)
    # x_gt = gen_x_alternating(n).to(device)
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
        
        net_kwargs = {
            'A': A, 
            'num_layers': config.NUM_LAYERS, 
            'embed': config.EMBED_DIM,
            'hidden': config.HIDDEN_DIM, 
            'drop_rate': config.DROP_RATE
        }

        if net_cls.__name__ == 'SplitResGCN':
            net_kwargs['tie_weights'] = args.tie_weights
        elif net_cls.__name__ == 'UNetGCN':
            net_kwargs['num_levels'] = config.NUM_LEVELS
            net_kwargs['layers_per_level'] = config.LAYERS_PER_LEVEL
        elif net_cls.__name__ == 'MGGNN':
            net_kwargs['num_levels'] = config.NUM_LEVELS
            net_kwargs['num_blocks'] = config.NUM_BLOCKS
            net_kwargs['K'] = config.TAGCONV_K
        elif net_cls.__name__ == 'FNO':
            net_kwargs['modes'] = config.FNO_MODES
            net_kwargs['grid_size'] = config.FNO_GRID_SIZE
        
        net = net_cls(**net_kwargs).to(device)
        net.load_state_dict(torch.load(master_ckpt_path, map_location=device))
        
        return GNP(A, net, device)
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

def plot_learning_curve(train_loss, val_loss, args, plot_dir, run_id, best_epoch=None, ylabel='Physics Loss'):
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
        plt.scatter(
            [best_epoch],
            [best_val],
            color='#16a34a',
            s=100,
            zorder=5,
            marker='*',
            label=f'Best Val (epoch {best_epoch})'
        )

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
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

def _exponential_moving_average(values, alpha=0.05):
    ema = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        ema.append(s)
    return ema

def _simple_moving_average(values, window=50):
    import numpy as _np
    out = _np.empty(len(values))
    cumsum = _np.cumsum(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0)) / (i - lo + 1)
    return out.tolist()

def plot_stepwise_learning_curve(step_data, args, plot_dir, run_id, best_epoch=None, sma_window=50, ema_alpha=None):
    step_losses = step_data.get('step_losses', [])
    val_steps   = step_data.get('val_steps', [])
    val_losses  = step_data.get('val_losses', [])
    batches_per_epoch = step_data.get('batches_per_epoch', 1)

    if not step_losses:
        print("Warning: No step-level loss data, skipping step-wise plot.")
        return

    total_steps = len(step_losses)

    if ema_alpha is not None:
        smoothed = _exponential_moving_average(step_losses, alpha=ema_alpha)
        smooth_label = f'EMA (α={ema_alpha})'
    else:
        smoothed = _simple_moving_average(step_losses, window=sma_window)
        smooth_label = f'SMA (w={sma_window})'

    steps = list(range(1, total_steps + 1))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(steps, step_losses, color='#2563eb', alpha=0.20, linewidth=0.5, label='Raw batch loss')
    ax.plot(steps, smoothed, color='#2563eb', alpha=1.0, linewidth=1.8, label=smooth_label)

    if val_steps and val_losses:
        ax.scatter(
            val_steps, 
            val_losses, 
            color='#ea580c', 
            s=60, 
            zorder=5, 
            marker='s', 
            edgecolors='black', 
            linewidths=0.5, 
            label='Validation loss'
        )
        ax.plot(val_steps, val_losses, color='#ea580c', linewidth=1.0, alpha=0.5, linestyle='--')

    if best_epoch is not None and best_epoch >= 1 and best_epoch <= len(val_losses):
        best_step = val_steps[best_epoch - 1]
        best_val = val_losses[best_epoch - 1]
        ax.scatter(
            [best_step], 
            [best_val], 
            color='#16a34a', 
            s=140, 
            zorder=6, 
            marker='*', 
            edgecolors='black', 
            linewidths=0.5, 
            label=f'Best val (epoch {best_epoch})'
        )

    ax.set_yscale('log')
    ax2 = ax.twiny()
    epoch_ticks = [i * batches_per_epoch for i in range(1, total_steps // batches_per_epoch + 1)]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(epoch_ticks)
    ax2.set_xticklabels([str(i) for i in range(1, len(epoch_ticks) + 1)], fontsize=8)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax.set_xlabel('Global Step (Optimizer Updates)', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title(f'Step-wise Learning Curve – {args.problem}', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', linestyle='-', alpha=0.15)
    fig.tight_layout()
    filename = f"{args.problem.replace('/', '_')}_ID{run_id}_stepwise_learning_curve.png"
    save_path = os.path.join(plot_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    import json as _json
    json_path = os.path.join(plot_dir, f"{args.problem.replace('/', '_')}_ID{run_id}_step_losses.json")
    with open(json_path, 'w') as f:
        _json.dump(step_data, f)

    print(f"Step-wise learning curve saved: {save_path}")
    print(f"Step loss data saved: {json_path}")