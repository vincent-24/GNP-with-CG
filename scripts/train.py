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


def _use_spectral_training(network_name: str) -> bool:
    """
    Determine whether to use unsupervised spectral radius training.

    Returns True for MGGNN/UNetGCN/LinearMGGNN if RANDOM_RATIO >= 1.0 (pure noise mode)
    or if explicitly configured. Supervised training is default otherwise.
    """
    # Networks that support unsupervised spectral training
    spectral_networks = {'MGGNN', 'UNetGCN', 'LinearMGGNN'}

    if network_name not in spectral_networks:
        return False

    # Use spectral training if configured for 100% random vectors (no PCG trajectories)
    # This is the unsupervised mode described in the MGGNN architecture
    return config.RANDOM_RATIO >= 1.0

def harvest_pcg_dataset(A, b, device, args, run_id):
    ratio = config.RANDOM_RATIO
    print(f"\n[Harvest] Generating training data  "
          f"(random_ratio={ratio:.2f}, {config.HARVEST_NUM_RUNS} PCG runs)")
    
    dataset_dir = getattr(args, 'data_root', None) or config.OFFLINE_DATASET_DIR
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))

    if os.getenv('SLURM_JOB_ID') and dataset_dir.startswith('/u/'):
        fallback_dir = os.path.abspath(os.path.expanduser(os.getenv('GNP_OFFLINE_DATASET_DIR', config.OFFLINE_DATASET_DIR)))
        if not fallback_dir.startswith('/u/'):
            print(f"[Path Guard] Offline dataset dir: redirecting {dataset_dir} -> {fallback_dir}")
            dataset_dir = fallback_dir

    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    ratio_tag = f'r{ratio:.2f}'
    filename = f'{ratio_tag}_harvested_{args.problem.replace("/", "_")}_ID{run_id}.pt'
    path = os.path.join(dataset_dir, filename)
    n = A.shape[0]
    CHUNK_SIZE = config.HARVEST_CHUNK_SIZE
    all_e_chunks: list[torch.Tensor] = []
    total_pcg = 0
    total_random = 0

    if ratio < 1.0:
        solver = PCG()
        chunk_e: list[torch.Tensor] = []
        harvest_miniters = max(1, config.HARVEST_NUM_RUNS // 100)

        for i in tqdm(range(config.HARVEST_NUM_RUNS), desc="Harvesting PCG", miniters=harvest_miniters, mininterval=0):
            x_true = gen_x_randn(n).to(device).to(A.dtype)
            b_sample = A @ x_true

            result = solver.solve(
                A, 
                b_sample, 
                M=None,     # <- no precond during harvesting
                max_iters=config.HARVEST_MAX_ITERS,
                rtol=config.HARVEST_RTOL,
                progress_bar=False,
                return_trajectory=True
            )

            history_x = result[-1]

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
    solver_class, cfg = get_solver_info(args.solver)
    network_class = get_network_class(args.network_override) if args.network_override else get_network_class(cfg['default_net'])

    net_kwargs = {
        'A': A,
        'num_layers': config.NUM_LAYERS,
        'embed': config.EMBED_DIM,
        'hidden': config.HIDDEN_DIM,
        'drop_rate': config.DROP_RATE,
    }

    if network_class.__name__ == 'SplitResGCN':
        net_kwargs['tie_weights'] = args.tie_weights
    elif network_class.__name__ == 'UNetGCN':
        net_kwargs['num_levels'] = config.NUM_LEVELS
        net_kwargs['layers_per_level'] = config.LAYERS_PER_LEVEL
    elif network_class.__name__ in ('MGGNN', 'LinearMGGNN'):
        net_kwargs['num_levels'] = config.NUM_LEVELS
        net_kwargs['num_vcycles'] = config.LINEAR_MGGNN_NUM_VCYCLES
        net_kwargs['smoother_K'] = config.LINEAR_MGGNN_SMOOTHER_K
        net_kwargs['coarsest_K'] = config.LINEAR_MGGNN_COARSEST_K
        net_kwargs['share_smoothers'] = config.LINEAR_MGGNN_SHARE_SMOOTHERS

    net = network_class(**net_kwargs).to(device)

    print(f"\nNetwork: {network_class.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")

    gnp = GNP(A, net, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Store checkpoint and config directly in matrix-specific directory
    master_ckpt_path = os.path.join(plot_dir, f'master_{args.problem.replace("/", "_")}_ID{run_id}.pt')
    cfg_path = os.path.join(plot_dir, f'master_{args.problem.replace("/", "_")}_ID{run_id}_config.json')

    # Determine training mode
    use_spectral = _use_spectral_training(network_class.__name__)

    if use_spectral:
        # ===== UNSUPERVISED SPECTRAL RADIUS TRAINING =====
        # No dataset needed - train directly on matrix A
        print(f"\n[Spectral Training] Using unsupervised spectral radius minimization")
        print(f"  No PCG trajectories needed - training directly on matrix A")
        print(f"  Probe vectors: {config.SPECTRAL_NUM_VECTORS}")
        print(f"  Power iterations: {config.SPECTRAL_POWER_ITERS}")
        print(f"  Steps per epoch: {config.SPECTRAL_STEPS_PER_EPOCH}")

        hist_train_loss, best_loss, best_epoch = gnp.train_spectral(
            epochs=config.EPOCHS,
            optimizer=optimizer,
            scheduler=None,
            checkpoint_path=master_ckpt_path,
            num_vectors=config.SPECTRAL_NUM_VECTORS,
            power_iters=config.SPECTRAL_POWER_ITERS,
            steps_per_epoch=config.SPECTRAL_STEPS_PER_EPOCH,
            progress_bar=True
        )

        # Build config for spectral training
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
                'network': network_class.__name__,
                'num_layers': config.NUM_LAYERS,
                'embed_dim': config.EMBED_DIM,
                'hidden_dim': config.HIDDEN_DIM,
                'drop_rate': config.DROP_RATE,
                'num_levels': config.NUM_LEVELS if network_class.__name__ in ['MGGNN', 'UNetGCN'] else None,
                'num_blocks': config.NUM_BLOCKS if network_class.__name__ == 'MGGNN' else None,
                'tagconv_k': config.TAGCONV_K if network_class.__name__ == 'MGGNN' else None,
            },
            'nn_training': {
                'mode': 'spectral',
                'epochs': config.EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'spectral_num_vectors': config.SPECTRAL_NUM_VECTORS,
                'spectral_power_iters': config.SPECTRAL_POWER_ITERS,
                'spectral_steps_per_epoch': config.SPECTRAL_STEPS_PER_EPOCH,
                'best_spectral_radius': best_loss,
                'best_epoch': best_epoch,
                'final_spectral_radius': hist_train_loss[-1] if hist_train_loss else None,
            },

            'checkpoint_path': master_ckpt_path,
        }

        with open(cfg_path, 'w') as f:
            json.dump(train_cfg, f, indent=2)

        # Plot spectral training curve (reuse learning curve plotter)
        plot_learning_curve(hist_train_loss, hist_train_loss, args, plot_dir, run_id,
                           best_epoch=best_epoch, ylabel='Spectral Radius ρ(I - M⁻¹A)')

        print(f"\n{'='*60}")
        print(f"Spectral Training Complete!")
        print(f"  Best Spectral Radius: {best_loss:.6f} (epoch {best_epoch})")
        print(f"  Checkpoint: {master_ckpt_path}")
        print(f"{'='*60}")

    else:
        # ===== SUPERVISED TRAJECTORY TRAINING =====
        harvest_path = getattr(args, 'harvest_dataset', None) or config.HARVEST_DATASET_PATH

        if harvest_path is not None:
            dataset_path = harvest_path
            print(f"Using specified dataset: {dataset_path}")
        else:
            print(f"Generating new dataset with run ID {run_id}...")
            dataset_path = harvest_pcg_dataset(A, b, device, args, run_id)

        full_dataset = OfflineDataset(dataset_path, A=A)
        train_size = int(config.TRAIN_VAL_SPLIT * len(full_dataset))
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
                'network': network_class.__name__,
                'num_layers': config.NUM_LAYERS,
                'embed_dim': config.EMBED_DIM,
                'hidden_dim': config.HIDDEN_DIM,
                'drop_rate': config.DROP_RATE,
                'tie_weights': args.tie_weights if network_class.__name__ == 'SplitResGCN' else None,
            },
            'nn_training': {
                'mode': 'supervised',
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