import os
import time
import torch
import numpy as np
import argparse
import random
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from GNP.problems import *
from GNP.precond import *
from GNP.precond.ILU import ILU
from GNP.precond.AMGPreconditioner import AMGPreconditioner
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse
from GNP import config
from GNP.factory import get_solver_and_network, get_network_class
from GNP.solver import PCG
from tqdm import tqdm

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

def harvest_pcg_dataset(A, problem, output_path, device, num_runs=50, max_iters=200, rtol=1e-10):
    """
    Harvest PCG trajectories for offline training.
    
    Runs unpreconditioned PCG on synthetic problems where we know the solution,
    collecting (residual, error) pairs at each iteration.
    
    Args:
        A: System matrix (sparse, already scaled)
        problem: Problem name string (for metadata)
        output_path: Where to save the dataset
        device: torch device
        num_runs: Number of random PCG runs
        max_iters: Max iterations per run
        rtol: Convergence tolerance
        
    Returns:
        output_path: Path to saved dataset
    """
    n = A.shape[0]
    solver = PCG()
    
    all_residuals = []
    all_errors = []
    total_samples = 0
    
    print(f"Harvesting from {num_runs} random problems (max {max_iters} iters each)...")
    
    for run_idx in tqdm(range(num_runs), desc="Harvesting"):
        x_true = gen_x_randn(n).to(device)
        
        with torch.no_grad():
            b = A @ x_true
            
            # Run PCG with trajectory harvesting (no preconditioner)
            _, _, _, _, _, _, trajectory = solver.solve(A, b, progress_bar=False, return_trajectory=True)
            
            history_r, history_x = trajectory
            
            for r_i, x_i in zip(history_r, history_x):
                e_i = x_true - x_i
                all_residuals.append(r_i.cpu())
                all_errors.append(e_i.cpu())
            
            total_samples += len(history_r)
    
    dataset_r = torch.stack(all_residuals, dim=0)
    dataset_e = torch.stack(all_errors, dim=0)
    
    print(f"Total samples collected: {total_samples}")
    print(f"Dataset shape: {dataset_r.shape}")
    
    dataset = {
        'r': dataset_r,
        'e': dataset_e,
        'metadata': {
            'problem': problem,
            'n': n,
            'num_runs': num_runs,
            'max_iters': max_iters,
            'rtol': rtol,
            'total_samples': total_samples,
            'seed': config.SEED
        }
    }
    
    torch.save(dataset, output_path)
    print(f"Dataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path

def setup_experiment(args):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    args.dump_root = os.path.abspath(os.path.expanduser(args.dump_root))
    Path(args.dump_root).mkdir(parents=True, exist_ok=True)
    plot_dir = get_timestamp_dir(args.dump_root)
    print(f"Output directory: {plot_dir}")
    
    return plot_dir

def load_problem(args, device):
    """Load matrix A, scale it, generate b, and return (A, A_csc, b, x_gt)."""
    print(f'\nLoading {args.problem}...')
    A = load_suitesparse(args.location, args.problem, device)
    A = scale_A_by_spectral_radius(A)
    n = A.shape[0]
    print(f'Matrix n={n}, nnz={A._nnz()}')
    
    A_csc = None
    if args.classical:
        A_csc = A.to_sparse_csc()
        print("Classical preconditioner comparison enabled (ILU, AMG)")
    
    x_gt = gen_x_all_ones(n).to(device)
    b = A @ x_gt
    
    return A, A_csc, b, x_gt

# --mode train
def train_routine(args, A, selected_solvers, device, plot_dir):
    """Handle the entire TRAIN MODE logic."""
    print(f"\n{'='*50}")
    print("MODE: TRAIN - Training master model")
    print(f"{'='*50}")
    
    train_solver_name = selected_solvers[0]
    solver_cls, net_cls, cfg = get_solver_and_network(train_solver_name)
    
    if args.network_override:
        net_cls = get_network_class(args.network_override)
        print(f"Network override: {args.network_override}")
    
    current_m = config.LANCZOS_M if cfg['use_lanczos'] else config.ARNOLDI_M
    print(f"Training with solver config: {train_solver_name}")
    print(f"Network: {net_cls.__name__}")
    print(f"Krylov Size m={current_m} ({'Lanczos' if cfg['use_lanczos'] else 'Arnoldi'})")
    
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
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    M = GNP(A, 'x_mix', current_m, net, device, use_lanczos=cfg['use_lanczos'])
    
    run_id = generate_run_id()
    temp_prefix = os.path.join(plot_dir, 'checkpoints', f"TEMP_master_{run_id}_")
    
    dataset_path = None
    if config.TRAIN_OFFLINE:
        problem_name = args.problem.split('/')[-1]
        dataset_filename = f"pcg_harvested_{problem_name}.pt"
        dataset_dir = os.path.abspath(config.OFFLINE_DATASET_DIR)
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, dataset_filename)
        
        if not os.path.exists(dataset_path):
            print(f"\n{'='*50}")
            print("AUTO-HARVESTING: Dataset not found, generating...")
            print(f"{'='*50}")
            dataset_path = harvest_pcg_dataset(
                A, args.problem, dataset_path, device,
                num_runs=config.HARVEST_NUM_RUNS,
                max_iters=config.HARVEST_MAX_ITERS,
                rtol=config.HARVEST_RTOL
            )
        
        print(f"Training new model (Run ID: {run_id}) [OFFLINE MODE]...")
        print(f"  Dataset: {dataset_path}")
    else:
        print(f"Training new model (Run ID: {run_id}) [STREAMING MODE]...")
    
    hist_loss, best_loss, best_epoch, trained_file = M.train(
        config.BATCH_SIZE, 1, config.EPOCHS, optimizer, 
        num_workers=args.num_workers, 
        checkpoint_prefix_with_path=temp_prefix,
        progress_bar=True,
        dataset_path=dataset_path
    )
    
    if args.checkpoint_path:
        master_ckpt_path = os.path.abspath(args.checkpoint_path)
        Path(os.path.dirname(master_ckpt_path)).mkdir(parents=True, exist_ok=True)
    else:
        problem_name = args.problem.split('/')[-1]
        ckpt_filename = f"master_{problem_name}.pt"
        master_ckpt_path = os.path.join(plot_dir, 'checkpoints', ckpt_filename)
    
    torch.save(net.state_dict(), master_ckpt_path)
    print(f"\nMaster checkpoint saved to {master_ckpt_path}")
    
    config_filename = os.path.basename(master_ckpt_path).replace('.pt', '_config.json')
    config_path = os.path.join(plot_dir, 'configs', config_filename)
    run_config = {
        'problem': args.problem,
        'train_solver': train_solver_name,
        'network': net_cls.__name__,
        'm': current_m,
        'layers': config.NUM_LAYERS,
        'embed': config.EMBED_DIM,
        'hidden': config.HIDDEN_DIM,
        'epochs': config.EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'date': get_timestamp_str()
    }
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=4)
    print(f"Config saved to {config_path}")
    
    if trained_file and os.path.exists(trained_file):
        os.remove(trained_file)
    
    print("\nTraining complete.")

# --mode eval
def eval_routine(args, A, A_csc, b, selected_solvers, device, master_ckpt_path):
    """Handle the entire EVAL MODE solver loop. Returns results dict."""
    print(f"\n{'='*50}")
    print("MODE: EVAL - Evaluating with master checkpoint")
    print(f"{'='*50}")
    print(f"Loading master checkpoint: {master_ckpt_path}")
    print(f"Baseline solver for comparisons: {config.BASELINE_SOLVER}")
    
    results = {}
    default_solve_kwargs = {
        'rtol': 1e-6,
        'max_iters': 1000,
        'restart': 80
    }

    for name in selected_solvers:
        solver_cls, net_cls, cfg = get_solver_and_network(name)
        
        if args.network_override:
            net_cls = get_network_class(args.network_override)

        print(f"\n{'='*40}")
        print(f"Configuration: {name}")
        print(f"{'='*40}")
        current_m = config.LANCZOS_M if cfg['use_lanczos'] else config.ARNOLDI_M
        print(f"Krylov Size m={current_m} ({'Lanczos' if cfg['use_lanczos'] else 'Arnoldi'})")
        print(f"Network: {net_cls.__name__}")
        solver = solver_cls()

        current_kwargs = default_solve_kwargs.copy()
        if "FCG" in name or "CG" in name: 
            current_kwargs.pop('restart', None)

        # --- Unpreconditioned Run (only for baseline solver) ---
        no_pre_res, no_pre_time = [], []
        if name == config.BASELINE_SOLVER:
            print(f"Solving {name} (No Preconditioner)...")
            result = solver.solve(A, b, M=None, **current_kwargs)
            # Handle both 5 and 6 return values
            if len(result) == 6:
                _, _, _, no_pre_res, no_pre_time, _ = result
            else:
                _, _, _, no_pre_res, no_pre_time = result

        # --- GNP Preconditioner Run ---
        print(f"Preparing {name} Preconditioner ({net_cls.__name__})...")
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
        
        print(f"Loading weights from master checkpoint...")
        net.load_state_dict(torch.load(master_ckpt_path, map_location=device))
        
        M = GNP(A, 'x_mix', current_m, net, device, use_lanczos=cfg['use_lanczos'])

        print(f"Solving {name} (With GNP)...")
        ortho_map = None
        try:
            result = solver.solve(A, b, M=M, **current_kwargs)
            # GMRES returns 5 values, CG-based solvers return 6 
            if len(result) == 6:
                _, _, _, gnp_res, gnp_time, ortho_map = result
            else:
                _, _, _, gnp_res, gnp_time = result
        except Exception as e:
            print(f"Solver failed: {e}")
            gnp_res, gnp_time = [], []

        # --- Classical Benchmarks (only for baseline solver) ---
        ilu_res, ilu_time = [], []
        amg_res, amg_time = [], []
        
        if args.classical and name == config.BASELINE_SOLVER:
            print(f"Solving {name} (With ILU)...")
            try:
                M_ilu = ILU(A_csc, ilu_factors_file=None, save_ilu_factors=False)
                result = solver.solve(A, b, M=M_ilu, **current_kwargs)
                if len(result) == 6:
                    _, _, _, ilu_res, ilu_time, _ = result
                else:
                    _, _, _, ilu_res, ilu_time = result
            except Exception as e:
                print(f"ILU solver failed: {e}")
                ilu_res, ilu_time = [], []
            
            print(f"Solving {name} (With AMG)...")
            try:
                M_amg = AMGPreconditioner(A_csc)
                result = solver.solve(A, b, M=M_amg, **current_kwargs)
                if len(result) == 6:
                    _, _, _, amg_res, amg_time, _ = result
                else:
                    _, _, _, amg_res, amg_time = result
            except Exception as e:
                print(f"AMG solver failed: {e}")
                amg_res, amg_time = [], []

        results[name] = {
            'no_pre_res': no_pre_res, 'no_pre_time': no_pre_time,
            'gnp_res': gnp_res, 'gnp_time': gnp_time,
            'ilu_res': ilu_res, 'ilu_time': ilu_time,
            'amg_res': amg_res, 'amg_time': amg_time,
            'ortho_map': ortho_map
        }

    return results

def plot_results(results, args, plot_dir):
    """Generate and save comparison plots."""
    print("\nGenerating Comparison Plots...")
    
    base_prefix = os.path.join(plot_dir, f"{args.problem.replace('/', '_')}_comparison")
    color_map = {name: plt.cm.tab10(i) for i, name in enumerate(results.keys())}
    
    # --- 1. Convergence (Iterations) ---
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        color = color_map[name]
        
        if len(res['gnp_res']) > 0:
            plt.semilogy(res['gnp_res'], linewidth=2, color=color, label=f'{name} (GNP)')
        
        # Only plot unpreconditioned/classical for baseline solver
        if name == config.BASELINE_SOLVER:
            if len(res['no_pre_res']) > 0:
                plt.semilogy(res['no_pre_res'], linestyle='--', alpha=0.6, color=color, label=f'{name} (No Precond)')
            if len(res.get('ilu_res', [])) > 0:
                plt.semilogy(res['ilu_res'], linestyle='-.', linewidth=1.5, color=color, label=f'{name} (ILU)')
            if len(res.get('amg_res', [])) > 0:
                plt.semilogy(res['amg_res'], linestyle=':', linewidth=2, color=color, label=f'{name} (AMG)')
    
    plt.title(f'Convergence Comparison ({args.problem})')
    plt.xlabel('Iterations')
    plt.ylabel('Relative Residual')
    plt.xlim(0, config.MAX_ITERS)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_prefix}_iters.png", bbox_inches='tight')
    
    # --- 2. Time-to-Solution ---
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        color = color_map[name]
        
        # Always plot GNP
        if len(res['gnp_res']) > 0:
            plt.semilogy(res['gnp_time'], res['gnp_res'], linewidth=2, color=color, label=f'{name} (GNP)')
        
        # Only plot classical for baseline solver
        if name == config.BASELINE_SOLVER:
            if len(res.get('ilu_res', [])) > 0:
                plt.semilogy(res['ilu_time'], res['ilu_res'], linestyle='-.', linewidth=1.5, color=color, label=f'{name} (ILU)')
            if len(res.get('amg_res', [])) > 0:
                plt.semilogy(res['amg_time'], res['amg_res'], linestyle=':', linewidth=2, color=color, label=f'{name} (AMG)')
    
    plt.title(f'Time-to-Solution Comparison ({args.problem})')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Residual')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_prefix}_time.png", bbox_inches='tight')
    
    # --- 3. Orthogonality Heatmaps ---
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
            filename = f"{args.problem.replace('/', '_')}_{name}_heatmap.png"
            plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
            print(f"Heatmap saved: {filename}")
    
    print(f"Plots saved to {base_prefix}_*.png")

def main():
    parser = argparse.ArgumentParser(description='GNP Solver Comparison')
    available_solvers = list(config.SOLVER_REGISTRY.keys())
    parser.add_argument('--solvers', nargs='+', default=config.SOLVERS, help=f'Solvers to run: {available_solvers} or "all"')
    parser.add_argument('--location', type=str, default=config.SUITE_SPARSE_PATH)
    parser.add_argument('--problem', type=str, default=config.PROBLEM_PATH) 
    parser.add_argument('--dump_root', type=str, default=config.DEFAULT_DUMP_PATH)
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default=config.MODE)
    parser.add_argument('--network-override', type=str, default=config.NETWORK_OVERRIDE)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--tie-weights', action='store_true', dest='tie_weights', default=config.TIE_WEIGHTS, help='Use weight tying in SplitResGCN')
    parser.add_argument('--classical', action='store_true', default=config.CLASSICAL, help='Compare against ILU and AMG preconditioners')
    args = parser.parse_args()
    
    if 'all' in args.solvers:
        selected_solvers = available_solvers
    else:
        selected_solvers = args.solvers
    
    for s in selected_solvers:
        if s not in config.SOLVER_REGISTRY:
            raise ValueError(f"Solver '{s}' not found in registry.")
    
    print(f"Running in {args.mode.upper()} mode. Network: {args.network_override}, Tie Weights: {args.tie_weights}")
    
    plot_dir = setup_experiment(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A, A_csc, b, x_gt = load_problem(args, device)
    
    if args.mode == 'train':
        train_routine(args, A, selected_solvers, device, plot_dir)
    else:
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required in eval mode.")
        
        master_ckpt_path = os.path.abspath(args.checkpoint_path)

        if not os.path.exists(master_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {master_ckpt_path}")
        
        results = eval_routine(args, A, A_csc, b, selected_solvers, device, master_ckpt_path)
        plot_results(results, args, plot_dir)
    
    print("\nDone.")

if __name__ == '__main__':
    main()