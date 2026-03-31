"""
scripts/generate_pcg_dataset.py

Generates a mixed training dataset of PCG error vectors and white-noise
vectors for the GNP neural preconditioner.

The key parameter is --random-ratio (config.RANDOM_RATIO):
    0.0 -> pure PCG residuals (smooth / low-frequency)
    0.5 -> half PCG, half white noise  (recommended)
    1.0 -> pure white noise (high-frequency only)

Memory-efficient: harvests in chunks and stacks once at the end.
I/O-efficient: stores only error vectors in float32 on disk; residuals
are reconstructed on-the-fly during training via r = A @ e.
"""
import gc
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GNP.solver import PCG
from GNP.utils import scale_A_by_spectral_radius, load_suitesparse
from GNP.problems import gen_x_randn
from GNP import config

def parse_args():
    parser = argparse.ArgumentParser(description='Harvest CG trajectories + white-noise vectors for offline GNP training')
    parser.add_argument('--random-ratio', type=float, default=config.RANDOM_RATIO,
                        help='Fraction of white noise '
                             '(0.0 = pure CG, 0.5 = 50/50, 1.0 = pure noise)')
    parser.add_argument('--problem', type=str, default=config.PROBLEM_PATH)
    parser.add_argument('--location', type=str, default=config.SUITE_SPARSE_PATH)
    parser.add_argument('--num-runs', type=int, default=config.HARVEST_NUM_RUNS)
    parser.add_argument('--max-iters', type=int, default=config.HARVEST_MAX_ITERS)
    parser.add_argument('--rtol', type=float, default=config.HARVEST_RTOL)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-every', type=int, default=1)
    parser.add_argument('--log-sampling', action='store_true', default=config.LOG_SAMPLING)
    parser.add_argument('--log-sampling-base', type=float, default=config.LOG_SAMPLING_BASE)
    parser.add_argument('--fp16-storage', action='store_true',
                        help='Save dataset in float16 to halve disk usage '
                             '(training still runs in float64 after upcast)')

    args = parser.parse_args()

    if not 0.0 <= args.random_ratio <= 1.0:
        parser.error(f'--random-ratio must be in [0, 1], got {args.random_ratio}')

    if args.output is None:
        problem_name = args.problem.replace('/', '_')
        ratio_tag = f'r{args.random_ratio:.2f}'
        args.output = os.path.join(config.OFFLINE_DATASET_DIR, f'{ratio_tag}_harvested_{problem_name}.pt')

    return args

def get_log_sample_indices(total_iters: int, base: float = 2.0) -> list[int]:
    if total_iters <= 0:
        return []

    indices = {0}
    power = 0

    while True:
        idx = int(base ** power)

        if idx >= total_iters:
            break

        indices.add(idx)
        power += 1

    indices.add(total_iters - 1)

    return sorted(indices)

@torch.no_grad()
def harvest_single_run(solver, A, x_true, max_iters, rtol):
    b = A @ x_true
    _, _, _, _, _, _, trajectory = solver.solve(A, b, progress_bar=False, return_trajectory=True)
    _, history_x = trajectory

    return [x_true - x_k for x_k in history_x]

def harvest_pcg_errors(A, n, device, args) -> list[torch.Tensor]:
    print(f"\n[PCG] Harvesting smooth error vectors")
    print(f"  Runs: {args.num_runs}, Max Iters: {args.max_iters}")

    solver = PCG()
    samples: list[torch.Tensor] = []
    chunk_size = config.HARVEST_CHUNK_SIZE

    for i in tqdm(range(args.num_runs), desc="Harvesting PCG"):
        x_true = gen_x_randn(n).to(device).to(torch.float64)
        errors = harvest_single_run(solver, A, x_true, args.max_iters, args.rtol)

        if args.log_sampling:
            keep = get_log_sample_indices(len(errors), args.log_sampling_base)
            errors = [errors[j] for j in keep]
        elif args.skip_every > 1:
            errors = errors[::args.skip_every]

        samples.extend(e.cpu() for e in errors)

        if (i + 1) % chunk_size == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return samples

def generate_random_errors(n: int, num_samples: int) -> list[torch.Tensor]:
    print(f"\n[Random] Generating {num_samples} white-noise vectors")
    noise = torch.randn(num_samples, n, dtype=torch.float64)

    return list(noise.unbind(0))

def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device(args.device)

    print(f"\nLoading problem: {args.problem}")
    A = load_suitesparse(args.location, args.problem, device)
    n = A.shape[0]
    print(f"Matrix dimension: n={n}, nnz={A._nnz()}")

    ratio = args.random_ratio
    print(f"\nRandom ratio: {ratio:.2f}  "
          f"({'pure PCG' if ratio == 0 else 'pure noise' if ratio == 1 else f'{1-ratio:.0%} PCG + {ratio:.0%} noise'})")

    pcg_errors: list[torch.Tensor] = []

    if ratio < 1.0:
        pcg_errors = harvest_pcg_errors(A, n, device, args)
        print(f"  PCG samples collected: {len(pcg_errors)}")

    if ratio == 0.0:
        num_random = 0
    elif ratio == 1.0:
        est_samples_per_run = 20
        num_random = args.num_runs * est_samples_per_run
    else:
        num_random = int(len(pcg_errors) * ratio / (1.0 - ratio))

    rand_errors: list[torch.Tensor] = []

    if num_random > 0:
        rand_errors = generate_random_errors(n, num_random)
        print(f"  Random samples generated: {len(rand_errors)}")

    all_errors = pcg_errors + rand_errors
    total_samples = len(all_errors)

    print(f"\nTotal samples: {total_samples}  "
          f"(PCG: {len(pcg_errors)}, Random: {len(rand_errors)})")

    rng = torch.Generator().manual_seed(config.SEED)
    perm = torch.randperm(total_samples, generator=rng)
    dataset_e = torch.stack(all_errors, dim=0)[perm]

    del all_errors, pcg_errors, rand_errors
    gc.collect()
    storage_dtype = torch.float16 if args.fp16_storage else dataset_e.dtype

    if args.fp16_storage:
        mb_before = dataset_e.nelement() * dataset_e.element_size() / (1024 ** 2)
        dataset_e = dataset_e.to(torch.float16)
        mb_after = dataset_e.nelement() * dataset_e.element_size() / (1024 ** 2)
        print(f"  Storage downcast: float64 -> float16  ({mb_before:.1f} MB -> {mb_after:.1f} MB)")

    output_path = os.path.abspath(args.output)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    dataset = {
        'e': dataset_e,
        'metadata': {
            'random_ratio': ratio,
            'problem': args.problem,
            'n': n,
            'num_pcg_samples': total_samples - num_random,
            'num_random_samples': num_random,
            'total_samples': total_samples,
            'num_runs': args.num_runs,
            'log_sampling': args.log_sampling,
            'storage_dtype': str(storage_dtype),
            'seed': config.SEED,
        }
    }

    torch.save(dataset, output_path)
    file_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"\nDataset saved to: {output_path}  ({file_mb:.1f} MB)")

if __name__ == '__main__':
    main()