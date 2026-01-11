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

def harvest_pcg_dataset(A, b, device, args):
    """
    Harvest PCG trajectories to generate training data.
    Runs unpreconditioned PCG on random RHS vectors and collects (r, e) pairs.
    """
    print(f"\n[Harvest] Generating training data from {config.HARVEST_NUM_RUNS} CG runs.")
    
    dataset_dir = config.OFFLINE_DATASET_DIR
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filename = f'pcg_harvested_{args.problem.replace("/", "_")}.pt'
    path = os.path.join(dataset_dir, filename)
    
    solver = PCG()
    all_r = []
    all_e = []
    n = A.shape[0]
    
    for i in tqdm(range(config.HARVEST_NUM_RUNS), desc="Harvesting"):
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
            all_r.append(r_k.cpu())  
            all_e.append(e_k.cpu())
            
    dataset = {
        'r': torch.stack(all_r),
        'e': torch.stack(all_e),
        'metadata': {
            'problem': args.problem,
            'num_runs': config.HARVEST_NUM_RUNS,
            'total_samples': len(all_r)
        }
    }
    torch.save(dataset, path)
    print(f"[Harvest] Dataset saved to {path} ({len(all_r)} samples)")
    return path


def train_routine(A, b, x_gt, device, args, plot_dir):
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
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    dataset_name = f'pcg_harvested_{args.problem.replace("/", "_")}.pt'
    dataset_path = os.path.join(args.data_root, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found. Harvesting...")
        dataset_path = harvest_pcg_dataset(A, b, device, args)
    else:
        print(f"Using existing dataset: {dataset_path}")
    
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
    
    master_ckpt_path = os.path.join(plot_dir, 'checkpoints', f'master_{args.problem.replace("/", "_")}.pt')
    
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
    
    cfg_path = os.path.join(plot_dir, 'configs', f'master_{args.problem.replace("/", "_")}_config.json')
    train_cfg = {
        'problem': args.problem,
        'solver': args.solver,
        'network': net_cls.__name__,
        'epochs': config.EPOCHS,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_train_loss': hist_train_loss[-1] if hist_train_loss else None,
        'final_val_loss': hist_val_loss[-1] if hist_val_loss else None,
    }
    with open(cfg_path, 'w') as f:
        json.dump(train_cfg, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val Loss: {best_val_loss:.4e} (epoch {best_epoch})")
    print(f"  Checkpoint: {master_ckpt_path}")
    print(f"{'='*60}")
    
    return master_ckpt_path