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

# Neural Network Architecture (Default)
NUM_LAYERS = 8
EMBED_DIM = 16
HIDDEN_DIM = 32
DROP_RATE = 0.0

# Neural Network Training
BATCH_SIZE = 16
EPOCHS = 2000
LEARNING_RATE = 1e-3

#==================SYSTEM/PATHS & ENVIRONMENT CONFIG==================#
import os
NUM_WORKERS = 0
DEFAULT_DUMP_PATH = './dump/'
SUITE_SPARSE_PATH = os.getenv('SUITESPARSE_PATH', './data')

MODE = 'eval'
SOLVERS = ['FCG', 'FGMRES']   #['all'] or ['FCG', 'PolakRibiereCG', 'PCG', 'FGMRES', ...]
NETWORK_OVERRIDE = 'SplitResGCN'
PROBLEM_PATH = 'HB/bcsstk16'
TIE_WEIGHTS = True
CLASSICAL = True

# Baseline solver for unpreconditioned / classical comparisons
# Use 'PCG' for SPD matrices, 'FGMRES' for non-symmetric matrices
BASELINE_SOLVER = 'PCG'

# Debugging: Track A-orthogonality of search directions
# Set to True to generate heatmaps showing conjugacy loss
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