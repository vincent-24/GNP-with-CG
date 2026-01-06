"""
Solver and Training hyperparameters.
"""
SEED = 42

# Solver Configurations
RESTART = 10
MAX_ITERS = 100
TOLERANCE = 1e-8

# FCG Specific Settings
TRUNCATION_K = None

# Data Generation 
LANCZOS_M = 80 
ARNOLDI_M = 40

# NN architecture
NUM_LAYERS = 8
EMBED_DIM = 16
HIDDEN_DIM = 32
DROP_RATE = 0.0

# NN training
BATCH_SIZE = 16
EPOCHS = 2000
LEARNING_RATE = 1e-3

TRAIN_OFFLINE = True    # train on pre-harvested data from generate_pcg_dataset.py
OFFLINE_DATASET_DIR = './data/pcg_harvested'  # Directory where harvested datasets are stored

# Harvesting parameters (used when auto-generating dataset)
HARVEST_NUM_RUNS = 50      # Number of random PCG runs to harvest
HARVEST_MAX_ITERS = 100    # Max iterations per PCG run
HARVEST_RTOL = 1e-8       # Convergence tolerance for harvesting

#==================SYSTEM/PATHS & ENVIRONMENT CONFIG==================#
import os
NUM_WORKERS = 0
DEFAULT_DUMP_PATH = './dump/'
SUITE_SPARSE_PATH = os.getenv('SUITESPARSE_PATH', './data')

MODE = 'eval'
SOLVERS = ['FGMRES', 'PCG']   #['all'] or ['FCG', 'PolakRibiereCG', 'PCG', 'FGMRES', ...]
NETWORK_OVERRIDE = 'SplitResGCN'
PROBLEM_PATH = 'FIDAP/ex10hs'
TIE_WEIGHTS = True
CLASSICAL = False

# Baseline solver for unpreconditioned / classical comparisons
# Use 'PCG' for SPD matrices, 'FGMRES' for non-symmetric matrices
BASELINE_SOLVER = 'PCG'

# for heatmap generation during eval
TRACK_ORTHOGONALITY = True
ORTHOGONALITY_SAMPLE_RATE = 1  # Downsample rate (1 = every direction, 2 = every other, etc.)
#=====================================================================#

# --- SOLVER REGISTRY ---
# Maps string keys to their Solver Class, Network Architecture, and description.
# Class names are stored as strings to avoid circular imports.
# Use GNP.factory.get_solver_and_network() to resolve the actual classes.
SOLVER_REGISTRY = {
    'FGMRES': {
        'solver_cls': 'GMRES',
        'use_lanczos': True,
        'net_cls': 'ResGCN',
        'description': 'Flexible GMRES with Standard ResGCN'
    },
    'FCG': {
        'solver_cls': 'FCG',
        'use_lanczos': True,
        'net_cls': 'SplitResGCN',
        'description': 'Flexible CG (Notay 2000) with full orthogonalization and SplitResGCN'
    },
    'PolakRibiereCG': {
        'solver_cls': 'PolakRibiereCG',
        'use_lanczos': True,
        'net_cls': 'SplitResGCN',
        'description': 'Nonlinear CG with Polak-Ribiere beta formula and SplitResGCN'
    },
    'PCG': {
        'solver_cls': 'PCG',
        'use_lanczos': True,
        'net_cls': 'SplitResGCN',
        'description': 'Standard PCG (Baseline) with SplitResGCN'
    }
}