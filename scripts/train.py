import os
import gc
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

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
    ratio = config.RANDOM_RATIO
    print(f"\n[Harvest] Generating training data  "
          f"(random_ratio={ratio:.2f}, {config.HARVEST_NUM_RUNS} PCG runs)")
    
    dataset_dir = config.OFFLINE_DATASET_DIR
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    ratio_tag = f'r{ratio:.2f}'
    filename = f'{ratio_tag}_harvested_{args.problem.replace("/", "_")}_ID{run_id}.pt'
    path = os.path.join(dataset_dir, filename)
    n = A.shape[0]
    CHUNK_SIZE = config.HARVEST_CHUNK_SIZE
    all_e_chunks: list[torch.Tensor] = []
    total_pcg = 0

    if ratio < 1.0:
        solver = PCG()
        chunk_e: list[torch.Tensor] = []
        harvest_miniters = max(1, config.HARVEST_NUM_RUNS // 100)

        for i in tqdm(range(config.HARVEST_NUM_RUNS), desc="Harvesting PCG", miniters=harvest_miniters, mininterval=0):
            x_true = gen_x_randn(n).to(device).to(A.dtype)
            b_sample = A @ x_true

            result = solver.solve(
                A, b_sample, M=None,
                max_iters=config.HARVEST_MAX_ITERS,
                rtol=config.HARVEST_RTOL,
                progress_bar=False,
                return_trajectory=True,
            )

            trajectory = result[-1]
            _, history_x = trajectory

            if config.LOG_SAMPLING:
                num_iters = len(history_x)
                sample_indices = {0}
                power = 0

                while True:
                    idx = int(config.LOG_SAMPLING_BASE ** power)

                    if idx >= num_iters:
                        break

                    sample_indices.add(idx)
                    power += 1

                if num_iters > 0:
                    sample_indices.add(num_iters - 1)

                for idx in sorted(sample_indices):
                    chunk_e.append((x_true - history_x[idx]).cpu())
            else:
                for x_k in history_x:
                    chunk_e.append((x_true - x_k).cpu())

            if (i + 1) % CHUNK_SIZE == 0 and chunk_e:
                all_e_chunks.append(torch.stack(chunk_e))
                total_pcg += len(chunk_e)
                chunk_e = []
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if chunk_e:
            all_e_chunks.append(torch.stack(chunk_e))
            total_pcg += len(chunk_e)

    total_random = 0

    if ratio > 0.0:
        if ratio == 1.0:
            est_per_run = 20
            num_random = config.HARVEST_NUM_RUNS * est_per_run
        else:
            num_random = int(total_pcg * ratio / (1.0 - ratio))

        if num_random > 0:
            print(f"[Harvest] Generating {num_random} white-noise vectors")
            noise = torch.randn(num_random, n, dtype=torch.float64)
            all_e_chunks.append(noise)
            total_random = num_random

    total_samples = total_pcg + total_random
    dataset_e = torch.cat(all_e_chunks, dim=0)
    rng = torch.Generator().manual_seed(config.SEED)
    perm = torch.randperm(total_samples, generator=rng)
    dataset_e = dataset_e[perm]

    dataset = {
        'e': dataset_e,
        'metadata': {
            'random_ratio': ratio,
            'problem': args.problem,
            'num_runs': config.HARVEST_NUM_RUNS,
            'num_pcg_samples': total_pcg,
            'num_random_samples': total_random,
            'total_samples': total_samples,
            'log_sampling': config.LOG_SAMPLING,
            'log_sampling_base': config.LOG_SAMPLING_BASE if config.LOG_SAMPLING else None,
        }
    }

    torch.save(dataset, path)
    print(f"[Harvest] Dataset saved to {path}  "
          f"({total_samples} samples: {total_pcg} PCG + {total_random} random)")

    return path

def train_routine(A, b, x_gt, device, args, plot_dir, run_id):
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
    elif net_cls.__name__ == 'UNetGCN':
        net_kwargs['num_levels'] = config.NUM_LEVELS
        net_kwargs['layers_per_level'] = config.LAYERS_PER_LEVEL
    
    net = net_cls(**net_kwargs).to(device)
    
    print(f"\nNetwork: {net_cls.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    gnp = GNP(A, 'x_mix', current_m, net, device, use_lanczos=cfg['use_lanczos'])
    optimizer = torch.optim.Adam(list(net.parameters()) + [gnp.loss_params], lr=config.LEARNING_RATE, weight_decay=1e-4)
    harvest_path = getattr(args, 'harvest_dataset', None) or config.HARVEST_DATASET_PATH

    if harvest_path is not None:
        dataset_path = harvest_path
        print(f"Using specified dataset: {dataset_path}")
    else:
        print(f"Generating new dataset with run ID {run_id}...")
        dataset_path = harvest_pcg_dataset(A, b, device, args, run_id)
    
    full_dataset = OfflineDataset(dataset_path, A=A)
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
    
    hist_train_loss, hist_val_loss, best_val_loss, best_epoch, step_data = gnp.train(
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
            'random_ratio': config.RANDOM_RATIO,
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

    from scripts.utils import plot_stepwise_learning_curve
    plot_stepwise_learning_curve(step_data, args, plot_dir, run_id, best_epoch=best_epoch)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Val Loss: {best_val_loss:.4e} (epoch {best_epoch})")
    print(f"  Checkpoint: {master_ckpt_path}")
    print(f"{'='*60}")
    
    return master_ckpt_path