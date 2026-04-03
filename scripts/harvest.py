import os
import gc
import sys
import torch
from pathlib import Path
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from GNP import config
from GNP.solver.PCG import PCG
from GNP.problems import gen_x_randn

def _resolve_dataset_dir(args):
    """Resolve the dataset directory, applying SLURM path guards and flat hierarchy."""
    dataset_dir = getattr(args, 'data_root', None) or config.OFFLINE_DATASET_DIR
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))

    if os.getenv('SLURM_JOB_ID') and dataset_dir.startswith('/u/'):
        fallback = os.path.abspath(os.path.expanduser(os.getenv('GNP_OFFLINE_DATASET_DIR', config.OFFLINE_DATASET_DIR)))

        if not fallback.startswith('/u/'):
            print(f"[Path Guard] Offline dataset dir: redirecting {dataset_dir} -> {fallback}")
            dataset_dir = fallback

    Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    if getattr(args, 'flat_hierarchy', False) and '/' in args.problem:
        category = args.problem.split('/', 1)[0]
        dataset_dir = os.path.join(dataset_dir, category)
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    return dataset_dir

def _sample_trajectory(history_x, x_true):
    """Extract error vectors from a PCG trajectory, optionally log-sampled."""
    if not config.LOG_SAMPLING:
        return [(x_true - x_k).cpu() for x_k in history_x]

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

    return [(x_true - history_x[idx]).cpu() for idx in sorted(sample_indices)]

def _harvest_pcg_errors(A, device):
    """Run unpreconditioned PCG solves and collect error vectors in chunks.

    Returns a list of stacked tensor chunks and the total sample count.
    """
    n = A.shape[0]
    solver = PCG()
    chunk_size = config.HARVEST_CHUNK_SIZE
    miniters = max(1, config.HARVEST_NUM_RUNS // 100)

    all_chunks: list[torch.Tensor] = []
    current_chunk: list[torch.Tensor] = []
    total = 0

    for i in tqdm(range(config.HARVEST_NUM_RUNS), desc="Harvesting PCG", miniters=miniters, mininterval=0):
        x_true = gen_x_randn(n).to(device).to(A.dtype)
        b_sample = A @ x_true
        result = solver.solve(
            A, b_sample,
            M=None,
            max_iters=config.HARVEST_MAX_ITERS,
            rtol=config.HARVEST_RTOL,
            progress_bar=False,
            return_trajectory=True,
        )
        history_x = result[-1]
        current_chunk.extend(_sample_trajectory(history_x, x_true))

        if (i + 1) % chunk_size == 0 and current_chunk:
            all_chunks.append(torch.stack(current_chunk))
            total += len(current_chunk)
            current_chunk = []
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if current_chunk:
        all_chunks.append(torch.stack(current_chunk))
        total += len(current_chunk)

    return all_chunks, total

def _generate_random_errors(n, num_pcg_samples, ratio):
    """Generate white-noise error vectors to mix with PCG-harvested samples."""
    if ratio == 1.0:
        num_random = config.HARVEST_NUM_RUNS * 20
    else:
        num_random = int(num_pcg_samples * ratio / (1.0 - ratio))

    if num_random <= 0:
        return None, 0

    print(f"[Harvest] Generating {num_random} white-noise vectors")
    noise = torch.randn(num_random, n, dtype=torch.float64)

    return noise, num_random

def harvest_pcg_dataset(A, device, args, run_id):
    """Harvest a training dataset of error vectors from PCG trajectories.

    Combines PCG-derived errors with optional white-noise vectors, shuffles,
    and saves to disk as a .pt file.
    """
    ratio = config.RANDOM_RATIO
    print(f"\n[Harvest] Generating training data  "
          f"(random_ratio={ratio:.2f}, {config.HARVEST_NUM_RUNS} PCG runs)")

    dataset_dir = _resolve_dataset_dir(args)
    ratio_tag = f'r{ratio:.2f}'
    filename = f'{ratio_tag}_harvested_{args.problem.replace("/", "_")}_ID{run_id}.pt'
    path = os.path.join(dataset_dir, filename)

    n = A.shape[0]
    all_chunks: list[torch.Tensor] = []
    total_pcg = 0
    total_random = 0

    # PCG trajectory harvesting
    if ratio < 1.0:
        pcg_chunks, total_pcg = _harvest_pcg_errors(A, device)
        all_chunks.extend(pcg_chunks)

    # Random noise augmentation
    if ratio > 0.0:
        noise, total_random = _generate_random_errors(n, total_pcg, ratio)

        if noise is not None:
            all_chunks.append(noise)

    # Shuffle and save
    total_samples = total_pcg + total_random
    dataset_e = torch.cat(all_chunks, dim=0)
    rng = torch.Generator().manual_seed(config.SEED)
    dataset_e = dataset_e[torch.randperm(total_samples, generator=rng)]
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
