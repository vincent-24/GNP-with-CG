#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import threading

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from GNP import config
from scripts.utils import setup_experiment, load_problem
from scripts.train import train_routine
from scripts.test import eval_routine

def get_device(args):
    """Determine compute device."""
    if args.device:
        return torch.device(args.device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_mggnn_async(problem_name: str, A_csc: torch.Tensor, device: torch.device):
    """Train MG-GNN in a separate thread (runs concurrently with GNP training)."""
    try:
        from scripts.train_mggnn import train_mggnn
        print("\n[MG-GNN] Starting MG-GNN training (concurrent with GNP)...")
        checkpoint_path = train_mggnn(
            problem_name=problem_name,
            device=device,
            A_csc=A_csc,
            epochs_p1=300,  # Reduced for concurrent training
            epochs_p2=100,
            verbose=True
        )
        print(f"\n[MG-GNN] Training complete: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"\n[MG-GNN] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=config.MODE, choices=['train', 'eval', 'both'], help=f'Run mode (default: {config.MODE})')
    parser.add_argument('--problem', type=str, default=config.PROBLEM_PATH, help=f'SuiteSparse matrix name (default: {config.PROBLEM_PATH})')
    parser.add_argument('--location', type=str, default=os.path.join(_PROJECT_ROOT, config.SUITE_SPARSE_PATH, 'SuiteSparse', 'ssget'), help='Path to SuiteSparse data')
    parser.add_argument('--dump_root', type=str, default=os.path.join(_PROJECT_ROOT, config.DEFAULT_DUMP_PATH.lstrip('./')), help=f'Root directory for outputs (default: {config.DEFAULT_DUMP_PATH})')
    parser.add_argument('--data_root', type=str, default=os.path.join(_PROJECT_ROOT, config.OFFLINE_DATASET_DIR.lstrip('./')), help=f'Directory for PCG harvested datasets (default: {config.OFFLINE_DATASET_DIR})')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to existing checkpoint for evaluation')
    parser.add_argument('--solver', type=str, default=config.BASELINE_SOLVER, choices=list(config.SOLVER_CONFIG.keys()), help=f'Solver for training data harvesting (default: {config.BASELINE_SOLVER})')
    parser.add_argument('--network_override', type=str, default=config.NETWORK_OVERRIDE, choices=['ResGCN', 'SplitResGCN', None], help=f'Override network architecture (default: {config.NETWORK_OVERRIDE})')
    parser.add_argument('--tie_weights', action='store_true', default=config.TIE_WEIGHTS, help=f'Tie weights in SplitResGCN (default: {config.TIE_WEIGHTS})')
    parser.add_argument('--harvest_dataset', type=str, default=config.HARVEST_DATASET_PATH, help='Path to pre-harvested dataset (.pt file). If None, auto-generates new dataset.')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu/mps). Auto-detected if not specified.')
    parser.add_argument('--train_mggnn', action='store_true', default=False, help='Also train MG-GNN preconditioner (runs concurrently with GNP)')
    args = parser.parse_args()
    device = get_device(args)
    
    print(f"GNP Experiment Runner:")
    print(f"Mode: {args.mode}")
    print(f"Problem: {args.problem}")
    print(f"Device: {device}")
    print(f"Train MG-GNN: {args.train_mggnn}")
    print(f"{'='*60}")
    
    plot_dir, run_id = setup_experiment(args)   # sets seeds and creates output dirs
    A, A_csc, b, x_gt = load_problem(args, device)
    master_ckpt_path = args.checkpoint
    
    # MG-GNN training thread (if enabled)
    mggnn_thread = None
    if args.train_mggnn and args.mode in ('train', 'both'):
        # Start MG-GNN training in a separate thread
        mggnn_thread = threading.Thread(
            target=train_mggnn_async,
            args=(args.problem, A_csc, device),
            daemon=False
        )
        mggnn_thread.start()
    
    # Train GNP (main thread)
    if args.mode in ('train', 'both'):
        print("\n\n\nTRAINING (GNP)")
        print(f"{'='*60}")
        master_ckpt_path = train_routine(A, b, x_gt, device, args, plot_dir, run_id)
    
    # Wait for MG-GNN training to complete before evaluation
    if mggnn_thread is not None:
        print("\n[Main] Waiting for MG-GNN training to complete...")
        mggnn_thread.join()
        print("[Main] MG-GNN training finished.")
    
    # Eval
    if args.mode in ('eval', 'both'):
        if master_ckpt_path is None:
            print("\nWARNING: No checkpoint specified for evaluation.")
            print("GNP experiments will fail. Other experiments will still run.")
        
        print("\n\n\nEVALUATION")
        print(f"{'='*60}")
        eval_routine(A, A_csc, b, x_gt, device, args, plot_dir, master_ckpt_path, run_id)
    
    print("\nExperiment completed.")