"""
Training routines for GNP neural preconditioner.
Implements standard deep learning best practices:
- Train/Validation split (90/10)
- Proper epoch-based training (1 epoch = 1 full pass through training data)
- Model checkpointing based on validation loss
"""
import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# Ensure GNP package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from GNP.precond.GNP import GNP, OfflineDataset
from GNP import config
from GNP.factory import get_solver_info, get_network_class
from GNP.solver.PCG import PCG
from GNP.problems import gen_x_randn
from scripts.utils import plot_learning_curve

def harvest_pcg_dataset(A, b, device, args, run_id):
    """
    Harvest PCG trajectories to generate training data.
    Runs unpreconditioned PCG on random RHS vectors and collects (r, e) pairs.
    
    Memory-efficient: saves incrementally to avoid OOM on large matrices.
    """
    print(f"\n[Harvest] Generating training data from {config.HARVEST_NUM_RUNS} CG runs.")
    
    dataset_dir = config.OFFLINE_DATASET_DIR
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filename = f'pcg_harvested_{args.problem.replace("/", "_")}_ID{run_id}.pt'
    path = os.path.join(dataset_dir, filename)
    
    solver = PCG()
    n = A.shape[0]
    
    # Memory-efficient: save in chunks to avoid OOM on large matrices
    CHUNK_SIZE = 50  # Save every 50 PCG runs
    all_r_chunks = []
    all_e_chunks = []
    chunk_r = []
    chunk_e = []
    total_samples = 0
    
    harvest_miniters = max(1, config.HARVEST_NUM_RUNS // 100)
    
    for i in tqdm(range(config.HARVEST_NUM_RUNS), desc="Harvesting", miniters=harvest_miniters, mininterval=0):
        x_true = gen_x_randn(n).to(device).to(A.dtype)
        b_sample = A @ x_true
        
        result = solver.solve(
            A, b_sample, M=None, 
            max_iters=config.HARVEST_MAX_ITERS, 
            rtol=config.HARVEST_RTOL, 
            progress_bar=False,
            return_trajectory=True
        )
        
        trajectory = result[-1] 
        history_r, history_x = trajectory
        
        for r_k, x_k in zip(history_r, history_x):
            e_k = x_true - x_k
            chunk_r.append(r_k.cpu())  
            chunk_e.append(e_k.cpu())
        
        # Save chunk to disk periodically to free memory
        if (i + 1) % CHUNK_SIZE == 0 and chunk_r:
            all_r_chunks.append(torch.stack(chunk_r))
            all_e_chunks.append(torch.stack(chunk_e))
            total_samples += len(chunk_r)
            chunk_r = []
            chunk_e = []
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if chunk_r:
        all_r_chunks.append(torch.stack(chunk_r))
        all_e_chunks.append(torch.stack(chunk_e))
        total_samples += len(chunk_r)
    
    dataset = {
        'r': torch.cat(all_r_chunks, dim=0),
        'e': torch.cat(all_e_chunks, dim=0),
        'metadata': {
            'problem': args.problem,
            'num_runs': config.HARVEST_NUM_RUNS,
            'total_samples': total_samples
        }
    }
    torch.save(dataset, path)
    print(f"[Harvest] Dataset saved to {path} ({total_samples} samples)")
    return path

def train_routine(A, b, x_gt, device, args, plot_dir, run_id):
    """
    Train the GNP neural preconditioner with proper train/val split.
    
    Implements:
    - 90/10 train/validation split
    - Epoch-based training (1 epoch = full pass through train set)
    - Checkpointing based on validation loss (not train loss)
    """
    solver_cls, cfg = get_solver_info(args.solver)
    net_cls = get_network_class(args.network_override) if args.network_override else get_network_class(cfg['default_net'])
    current_m = config.LANCZOS_M if cfg['use_lanczos'] else config.ARNOLDI_M
    
    net_kwargs = {
        'A': A,
        'num_layers': config.NUM_LAYERS,
        'embed': config.EMBED_DIM,
        'hidden': config.HIDDEN_DIM,
        'drop_rate': config.DROP_RATE,
    }
    if net_cls.__name__ == 'SplitResGCN':
        net_kwargs['tie_weights'] = args.tie_weights
    
    net = net_cls(**net_kwargs).to(device)
    
    print(f"\nNetwork: {net_cls.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    gnp = GNP(A, 'x_mix', current_m, net, device, use_lanczos=cfg['use_lanczos'])
    # optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    optimizer = torch.optim.Adam(
        list(net.parameters()) + [gnp.loss_params],
        lr=config.LEARNING_RATE, 
        weight_decay=1e-4
    )
    
    # Use command-line arg if provided, otherwise fall back to config, otherwise auto-generate
    harvest_path = getattr(args, 'harvest_dataset', None) or config.HARVEST_DATASET_PATH
    if harvest_path is not None:
        dataset_path = harvest_path
        print(f"Using specified dataset: {dataset_path}")
    else:
        print(f"Generating new dataset with run ID {run_id}...")
        dataset_path = harvest_pcg_dataset(A, b, device, args, run_id)
    
    full_dataset = OfflineDataset(dataset_path)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(config.SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"\nData Split:")
    print(f"  Training:   {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    master_ckpt_path = os.path.join(plot_dir, 'checkpoints', f'master_{args.problem.replace("/", "_")}_ID{run_id}.pt')
    
    print(f"\nTraining for {config.EPOCHS} epochs...")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    hist_train_loss, hist_val_loss, best_val_loss, best_epoch = gnp.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        optimizer=optimizer,
        scheduler=None,
        checkpoint_path=master_ckpt_path,
        progress_bar=True
    )
    
    cfg_path = os.path.join(plot_dir, 'configs', f'master_{args.problem.replace("/", "_")}_ID{run_id}_config.json')
    train_cfg = {
        'run_id': run_id,
        'problem': args.problem,
        'device': str(device),
        'seed': config.SEED,
        'solver': args.solver,
        'tolerance': config.TOLERANCE,
        'max_iters': config.MAX_ITERS,
        'fgmres_restart': config.RESTART,
        'lanczos_m': config.LANCZOS_M if cfg['use_lanczos'] else None,
        'arnoldi_m': config.ARNOLDI_M if not cfg['use_lanczos'] else None,
        
        'nn_architecture': {
            'network': net_cls.__name__,
            'num_layers': config.NUM_LAYERS,
            'embed_dim': config.EMBED_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'drop_rate': config.DROP_RATE,
            'tie_weights': args.tie_weights if net_cls.__name__ == 'SplitResGCN' else None,
        },
        'nn_training': {
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_train_loss': hist_train_loss[-1] if hist_train_loss else None,
            'final_val_loss': hist_val_loss[-1] if hist_val_loss else None,
        },
        'harvest': {
            'dataset_path': dataset_path,
            'num_runs': config.HARVEST_NUM_RUNS,
            'max_iters': config.HARVEST_MAX_ITERS,
            'rtol': config.HARVEST_RTOL,
            'train_offline': config.TRAIN_OFFLINE,
            'offline_dataset_dir': config.OFFLINE_DATASET_DIR,
        },

        'checkpoint_path': master_ckpt_path,
    }
    with open(cfg_path, 'w') as f:
        json.dump(train_cfg, f, indent=2)
    
    plot_learning_curve(hist_train_loss, hist_val_loss, args, plot_dir, run_id, best_epoch=best_epoch)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val Loss: {best_val_loss:.4e} (epoch {best_epoch})")
    print(f"  Checkpoint: {master_ckpt_path}")
    print(f"{'='*60}")
    
    return master_ckpt_path