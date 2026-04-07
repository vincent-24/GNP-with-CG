import os
import sys
import json
import torch
from torch.utils.data import DataLoader, random_split

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from GNP.precond.GNP import GNP, OfflineDataset
from GNP import config
from GNP.factory import get_solver_info, get_network_class
from scripts.utils import plot_learning_curve
from scripts.harvest import harvest_pcg_dataset

# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def _build_network(A, device, args):
    """Instantiate the neural network and GNP wrapper from config.

    Returns (net, gnp, network_class) so callers can log the class name.
    """
    _, cfg = get_solver_info(args.solver)
    network_class = (get_network_class(args.network_override) if args.network_override else get_network_class(cfg['default_net']))
    net_kwargs = {
        'A': A,
        'num_layers': config.NUM_LAYERS,
        'embed': config.EMBED_DIM,
        'hidden': config.HIDDEN_DIM,
        'drop_rate': config.DROP_RATE,
        'num_levels': 2,
        'num_vcycles': config.TWO_LEVEL_NUM_STEPS,
        'smoother_K': config.TWO_LEVEL_FINE_K,
        'coarsest_K': config.TWO_LEVEL_COARSE_K,
        'share_smoothers': config.TWO_LEVEL_SHARE_BLOCKS,
    }

    net = network_class(**net_kwargs).to(device)

    print(f"\tNetwork: {network_class.__name__}")
    print(f"\tParameters: {sum(p.numel() for p in net.parameters()):,}")

    gnp = GNP(A, net, device)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    return net, gnp, optimizer, network_class

def _base_train_config(run_id, args, device, network_class):
    """Build the config dict entries shared by both training modes."""
    _, cfg = get_solver_info(args.solver)

    return {
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
            'num_levels': 2,
            'num_steps': config.TWO_LEVEL_NUM_STEPS,
            'fine_K': config.TWO_LEVEL_FINE_K,
            'coarse_K': config.TWO_LEVEL_COARSE_K,
            'share_blocks': config.TWO_LEVEL_SHARE_BLOCKS,
        },
    }

# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------

def _train_spectral(gnp, optimizer, args, plot_dir, run_id, ckpt_path, cfg_path, device, network_class):
    """Unsupervised training: minimise a spectral loss (rho or kappa)."""
    loss_type = getattr(config, 'SPECTRAL_LOSS_TYPE', 'rho')

    print(f"\nSPECTRAL TRAINING")
    print(f"\tLoss type: {loss_type}")
    print(f"\tProbe vectors: {config.SPECTRAL_NUM_VECTORS}")
    print(f"\tPower iterations: {config.SPECTRAL_POWER_ITERS}")
    print(f"\tSteps per epoch: {config.SPECTRAL_STEPS_PER_EPOCH}")

    hist_train_loss, best_loss, best_epoch = gnp.train_spectral(
        epochs=config.EPOCHS,
        optimizer=optimizer,
        scheduler=None,
        checkpoint_path=ckpt_path,
        num_vectors=config.SPECTRAL_NUM_VECTORS,
        power_iters=config.SPECTRAL_POWER_ITERS,
        steps_per_epoch=config.SPECTRAL_STEPS_PER_EPOCH,
        progress_bar=True,
    )

    # Save config
    train_cfg = _base_train_config(run_id, args, device, network_class)
    train_cfg['nn_training'] = {
        'mode': 'spectral',
        'loss_type': loss_type,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'spectral_num_vectors': config.SPECTRAL_NUM_VECTORS,
        'spectral_power_iters': config.SPECTRAL_POWER_ITERS,
        'spectral_steps_per_epoch': config.SPECTRAL_STEPS_PER_EPOCH,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_loss': hist_train_loss[-1] if hist_train_loss else None,
    }
    train_cfg['checkpoint_path'] = ckpt_path

    with open(cfg_path, 'w') as f:
        json.dump(train_cfg, f, indent=2)

    ylabel = 'log κ(M⁻¹A)' if loss_type == 'kappa' else 'ρ(I - M⁻¹A)'
    plot_learning_curve(
        hist_train_loss, hist_train_loss, args, plot_dir, run_id,
        best_epoch=best_epoch, ylabel=ylabel,
    )

    print(f"\n{'='*60}")
    print(f"Spectral Training Complete")
    print(f"\tBest {ylabel}: {best_loss:.6f} (epoch {best_epoch})")
    print(f"\tCheckpoint: {ckpt_path}")
    print(f"{'='*60}")

def _train_supervised(gnp, optimizer, A, device, args, plot_dir, run_id, ckpt_path, cfg_path, network_class):
    """Supervised training on harvested PCG error vectors."""
    # Resolve or generate dataset
    harvest_path = getattr(args, 'harvest_dataset', None) or config.HARVEST_DATASET_PATH

    if harvest_path is not None:
        dataset_path = harvest_path
        print(f"Using specified dataset: {dataset_path}")
    else:
        print(f"Generating new dataset with run ID {run_id}...")
        dataset_path = harvest_pcg_dataset(A, device, args, run_id)

    full_dataset = OfflineDataset(dataset_path, A=A)

    # Train / validation split
    train_size = int(config.TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(config.SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator,)

    print(f"\nDATA SPLIT")
    print(f"\tTraining:   {len(train_dataset)} samples")
    print(f"\tValidation: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"\nTraining for {config.EPOCHS} epochs...")
    print(f"\tBatch size: {config.BATCH_SIZE}")
    print(f"\tBatches per epoch: {len(train_loader)}")

    hist_train_loss, hist_val_loss, best_val_loss, best_epoch, step_data = gnp.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        optimizer=optimizer,
        scheduler=None,
        checkpoint_path=ckpt_path,
        progress_bar=True,
    )

    # Save config
    train_cfg = _base_train_config(run_id, args, device, network_class)
    train_cfg['nn_training'] = {
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
    }
    train_cfg['harvest'] = {
        'dataset_path': dataset_path,
        'random_ratio': config.RANDOM_RATIO,
        'num_runs': config.HARVEST_NUM_RUNS,
        'max_iters': config.HARVEST_MAX_ITERS,
        'rtol': config.HARVEST_RTOL,
        'train_offline': config.TRAIN_OFFLINE,
        'offline_dataset_dir': config.OFFLINE_DATASET_DIR,
    }
    train_cfg['checkpoint_path'] = ckpt_path

    with open(cfg_path, 'w') as f:
        json.dump(train_cfg, f, indent=2)

    plot_learning_curve(hist_train_loss, hist_val_loss, args, plot_dir, run_id, best_epoch=best_epoch)

    from scripts.utils import plot_stepwise_learning_curve
    plot_stepwise_learning_curve(step_data, args, plot_dir, run_id, best_epoch=best_epoch)

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"\tBest Val Loss: {best_val_loss:.4e} (epoch {best_epoch})")
    print(f"\tCheckpoint: {ckpt_path}")
    print(f"{'='*60}")

# ---------------------------------------------------------------------------
# public train routine
# ---------------------------------------------------------------------------

def train_routine(A, b, x_gt, device, args, plot_dir, run_id, ckpt_dir=None):
    """Top-level training dispatcher called by run_exp.py."""
    _, gnp, optimizer, network_class = _build_network(A, device, args)

    if ckpt_dir is None:
        ckpt_dir = plot_dir

    problem_tag = args.problem.replace("/", "_")
    ckpt_path = os.path.join(ckpt_dir, f'master_{problem_tag}_ID{run_id}.pt')
    cfg_path = os.path.join(ckpt_dir, f'master_{problem_tag}_ID{run_id}_config.json')

    if config.TRAIN_OFFLINE:
        _train_supervised(gnp, optimizer, A, device, args, plot_dir, run_id, ckpt_path, cfg_path, network_class,)
    else:
        _train_spectral(gnp, optimizer, args, plot_dir, run_id, ckpt_path, cfg_path, device, network_class,)

    return ckpt_path
