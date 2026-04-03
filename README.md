# Implicit Muligrid Graph Neural Preconditioner for Conjugate Gradient

A neural network preconditioner for the Preconditioned Conjugate Gradient (PCG) method, targeting extremely ill-conditioned symmetric positive definite (SPD) linear systems. Built on the **Graph Neural Preconditioner (GNP)** framework of [Li et al. (2024)](https://arxiv.org/abs/2406.00809), which learns a preconditioner $M^{-1}$ by training a GNN to approximate the error propagation operator. The GNN backbone is a **two-level Multigrid Graph Neural Network (MG-GNN)** based on the parallel cross-scale message passing architecture of [Taghibakhshi et al. (2023)](https://arxiv.org/abs/2301.11378).

## Architecture

The preconditioner learns a map $z = M^{-1}(r)$ from residual to correction that is **linear in $r$** (no nonlinear activations in the data path), preserving SPD compatibility with PCG.

**TwoLevelMGGNN** implements:

1. **Lift**: Scalar residual $r \to$ hidden features at fine and coarse multigrid levels.
2. **Parallel MG-GNN blocks** (paper Eqs. 8-10):
   - Heterogeneous cross-level transfer via learnable MLPs (fine-to-coarse and coarse-to-fine).
   - Feature concatenation of own-level and cross-level messages.
   - Intra-level TAGConv (Topology Adaptive Graph Convolution) polynomial filtering.
   - Residual connections.
3. **Project**: Fine-level features $\to$ scalar correction.
4. **Implicit iterative correction**: Multiple steps $z_i = z_{i-1} + \alpha_i \cdot \Delta z_i$ with learnable positive step sizes and residual recomputation.

The two-level multigrid hierarchy is constructed via **Lloyd aggregation** (pyamg) with **Galerkin projection** ($A_c = R A P$, where $P = R^T$).

## Training

Two training modes are supported:

- **Spectral radius minimization** (default): Unsupervised training that minimizes $\rho(I - M^{-1}A)$ via power iteration with random probe vectors. No harvested data needed.
- **Supervised**: Train on error vectors harvested from unpreconditioned PCG trajectories, with optional white-noise augmentation.

## Project Structure

```
GNP/
    config.py           # Hyperparameters and experiment configuration
    factory.py          # Solver and network class resolution
    problems.py         # Test matrix and RHS generators
    utils.py            # Multigrid hierarchy construction, SuiteSparse I/O, sparse utilities
    nn/
        TwoLevelMGGNN.py    # MG-GNN architecture (TAGConv, cross-level transfer, implicit correction)
    precond/
        base.py             # BasePreconditioner / CPUPreconditioner ABCs
        GNP.py              # Neural preconditioner wrapper (training loops, spectral loss, inference)
        AMGPreconditioner.py # Algebraic Multigrid baseline (pyamg)
        IChol.py            # Incomplete Cholesky baseline (GPU-native triangular solves)
        Jacobi.py           # Diagonal preconditioner baseline
    solver/
        base.py             # IterativeSolver base (residual history, orthogonality tracking)
        PCG.py              # Preconditioned Conjugate Gradient
        GMRES.py            # Flexible GMRES with Arnoldi process
        Lanczos.py          # Lanczos iteration with full reorthogonalization
scripts/
    run_exp.py          # Experiment entry point (train + eval pipeline)
    train.py            # Training dispatcher (spectral / supervised)
    test.py             # Evaluation routine with baseline comparisons
    harvest.py          # PCG trajectory harvesting for supervised datasets
    utils.py            # Experiment setup, plotting, preconditioner loading
```

## Usage

### Run an experiment

```bash
python scripts/run_exp.py \
    --problem HB/bcsstk19 \
    --mode both \
    --solver PCG
```

This will train the TwoLevelMGGNN preconditioner on the specified SuiteSparse matrix and evaluate it against baselines (unpreconditioned CG, IChol, AMG).

### Key configuration

Edit `GNP/config.py` or pass CLI arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIDDEN_DIM` | 32 | Feature dimension at all MG-GNN levels |
| `TWO_LEVEL_NUM_STEPS` | 4 | Number of implicit correction steps |
| `TWO_LEVEL_FINE_K` | 3 | TAGConv polynomial order (fine level) |
| `TWO_LEVEL_COARSE_K` | 5 | TAGConv polynomial order (coarse level) |
| `TRAIN_OFFLINE` | False | False = spectral training, True = supervised |
| `EPOCHS` | 100 | Training epochs |
| `SPECTRAL_NUM_VECTORS` | 32 | Probe vectors per spectral loss step |
| `SPECTRAL_POWER_ITERS` | 10 | Power iteration depth for $\rho$ estimation |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `WORK_ROOT` | Base directory for data/checkpoints/dumps |
| `SUITESPARSE_PATH` | Local SuiteSparse matrix cache |
| `DUMP_PATH` | Results output directory |

## Installation

```bash
pip install -e .
```

For AMG baselines:
```bash
pip install -e ".[amg]"
```

Requires Python >= 3.10, PyTorch >= 2.0.

## References

- Li, J., Huang, Y., Yin, H., & Zhang, L. (2024). *Graph Neural Preconditioners for Iterative Solutions of Sparse Linear Systems*. [arXiv:2406.00809](https://arxiv.org/abs/2406.00809)
- Taghibakhshi, A., Ogden, S., Peng, Y., Siddiqui, K., MacLachlan, S., & Bhowmick, S. (2023). *MG-GNN: Multigrid Graph Neural Networks for Learning Multilevel Domain Decomposition Methods*. ICML 2023. [arXiv:2301.11378](https://arxiv.org/abs/2301.11378)

## License

MIT
