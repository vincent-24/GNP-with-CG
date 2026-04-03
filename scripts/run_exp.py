#!/usr/bin/env python3
import os
import sys
import argparse
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from GNP import config
from scripts.utils import setup_experiment, load_problem
from scripts.train import train_routine
from scripts.test import eval_routine

def _normalize_path(path):
    return os.path.abspath(os.path.expanduser(path))

def _redirect_slurm_path(path, env_var, default_path, label):
    resolved = _normalize_path(path)
    fallback = _normalize_path(os.getenv(env_var, default_path))

    if os.getenv('SLURM_JOB_ID') and resolved.startswith('/u/') and not fallback.startswith('/u/'):
        print(f"[Path Guard] {label}: redirecting {resolved} -> {fallback}")
        
        return fallback

    return resolved

def get_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=config.MODE, choices=['train', 'eval', 'both'])
    parser.add_argument('--problem', type=str, default=config.PROBLEM_PATH, help='SuiteSparse matrix name')
    parser.add_argument('--location', type=str, default=config.SUITE_SPARSE_PATH, help='Path to SuiteSparse data')
    parser.add_argument('--dump_root', type=str, default=config.DEFAULT_DUMP_PATH, help='Results output directory')
    parser.add_argument('--data_root', type=str, default=config.OFFLINE_DATASET_DIR, help='Directory for CG harvested datasets')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to existing checkpoint for evaluation')
    parser.add_argument('--solver', type=str, default=config.BASELINE_SOLVER, choices=list(config.SOLVER_CONFIG.keys()), help='Solver for training data harvesting')
    parser.add_argument('--network_override', type=str, default=config.NETWORK_OVERRIDE, choices=['TwoLevelMGGNN'], help='Neural network architecture')
    parser.add_argument('--harvest_dataset', type=str, default=config.HARVEST_DATASET_PATH, help='Path to pre-harvested dataset (.pt file). Auto-generates if not specified.')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu/mps). Auto-detected if not specified.')
    parser.add_argument('--ckpt_root', type=str, default=None, help='Root directory for checkpoints.')
    parser.add_argument('--flat_hierarchy', action='store_true', help='Use flat directory hierarchy (Category/Matrix) without date nesting')
    args = parser.parse_args()

    args.location = _redirect_slurm_path(args.location, 'SUITESPARSE_PATH', config.SUITE_SPARSE_PATH, 'SuiteSparse path')
    args.dump_root = _redirect_slurm_path(args.dump_root, 'GNP_DUMP_PATH', config.DEFAULT_DUMP_PATH, 'Dump root')
    args.data_root = _redirect_slurm_path(args.data_root, 'GNP_OFFLINE_DATASET_DIR', config.OFFLINE_DATASET_DIR, 'Offline dataset root')

    args.location = _normalize_path(args.location)
    args.dump_root = _normalize_path(args.dump_root)
    args.data_root = _normalize_path(args.data_root)

    if args.ckpt_root:
        args.ckpt_root = _normalize_path(args.ckpt_root)
        os.makedirs(args.ckpt_root, exist_ok=True)

    os.makedirs(args.dump_root, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    device = get_device(args)
    
    print(f"\nEXPERIMENT RUNNER")
    print(f"\tMode: {args.mode}")
    print(f"\tProblem: {args.problem}")
    print(f"\tDevice: {device}")
    print(f"\tSuiteSparse path: {args.location}")
    print(f"\tDump path: {args.dump_root}")
    print(f"\tData path: {args.data_root}")
    
    plot_dir, ckpt_dir, run_id = setup_experiment(args)
    A, A_csc, b, x_gt = load_problem(args, device)
    master_ckpt_path = args.checkpoint

    if args.mode in ('train', 'both'):
        print("\n\n\nTRAINING")
        print(f"{'='*60}")
        master_ckpt_path = train_routine(A, b, x_gt, device, args, plot_dir, run_id, ckpt_dir=ckpt_dir)
    
    if args.mode in ('eval', 'both'):
        if master_ckpt_path is None:
            print("\nWARNING: No checkpoint specified for evaluation.")
            print("GNP experiments will fail. Other experiments will still run.")
        
        print("\n\n\nEVALUATION")
        print(f"{'='*60}")
        eval_routine(A, A_csc, b, x_gt, device, args, plot_dir, master_ckpt_path, run_id)
    
    print("\nExperiment completed.")