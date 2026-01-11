"""
Solver and Training hyperparameters.
"""
SEED = 42

# Solver Configurations
RESTART = 10
MAX_ITERS = 1000
TOLERANCE = 1e-6

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
EPOCHS = 50  # Full epochs (1 epoch = full pass through ~50k samples)
LEARNING_RATE = 1e-3

# Data generation
NUM_DATA_SAMPLES = 50  # Number of random PCG runs to harvest for training

TRAIN_OFFLINE = True   # Use streaming training (set True to use pre-harvested data)
OFFLINE_DATASET_DIR = './data/pcg_harvested'  # Directory where harvested datasets are stored

# Harvesting parameters (used when auto-generating dataset)
HARVEST_NUM_RUNS = 50     # Number of random PCG runs to harvest
HARVEST_MAX_ITERS = 1000   # Max iterations per PCG run
HARVEST_RTOL = 1e-6        # Convergence tolerance for harvesting

#==================SYSTEM/PATHS & ENVIRONMENT CONFIG==================#
import os
NUM_WORKERS = 0
DEFAULT_DUMP_PATH = './dump/'
SUITE_SPARSE_PATH = os.getenv('SUITESPARSE_PATH', './data')

MODE = 'eval'
SOLVERS = ['FGMRES', 'PCG']   #['all'] or ['FCG', 'PolakRibiereCG', 'PCG', 'FGMRES', ...]
NETWORK_OVERRIDE = 'SplitResGCN'
PROBLEM_PATH = 'HB/plat1919'
TIE_WEIGHTS = True

# Baseline solver for unpreconditioned / classical comparisons
# Use 'PCG' for SPD matrices, 'FGMRES' for non-symmetric matrices
BASELINE_SOLVER = 'PCG'

# for heatmap generation during eval
TRACK_ORTHOGONALITY = True
ORTHOGONALITY_SAMPLE_RATE = 1  # Downsample rate (1 = every direction, 2 = every other, etc.)
#=====================================================================#

# --- EXPERIMENTS LIST ---
# Defines every solver/preconditioner pair to run during evaluation.
# Each experiment specifies: name, solver, preconditioner type, optional kwargs, and plot style.
# This is the SINGLE SOURCE OF TRUTH for what gets run and plotted.
EXPERIMENTS = [
    {
        'name': 'FGMRES (GNP)',
        'solver': 'FGMRES',
        'precond': 'GNP',
        'precond_kwargs': {},
        'style': {'color': 'blue', 'linestyle': '-', 'linewidth': 2}
    },
    {
        'name': 'PCG (GNP)',
        'solver': 'PCG',
        'precond': 'GNP',
        'precond_kwargs': {},
        'style': {'color': 'orange', 'linestyle': '-', 'linewidth': 2}
    },
    {
        'name': 'PCG (No Precond)',
        'solver': 'PCG',
        'precond': None,
        'precond_kwargs': {},
        'style': {'color': 'green', 'linestyle': '-', 'linewidth': 2}
    },
    {
        'name': 'PCG (IChol)',
        'solver': 'PCG',
        'precond': 'IChol',
        'precond_kwargs': {'shift': 1e-3},  # Diagonal shift for stability
        'style': {'color': 'red', 'linestyle': '-', 'linewidth': 2}
    },
    # {
    #     'name': 'PCG (ILU)',
    #     'solver': 'PCG',
    #     'precond': 'ILU',
    #     'precond_kwargs': {},
    #     'style': {'color': 'purple', 'linestyle': '-', 'linewidth': 2}
    # },
    # {
    #     'name': 'PCG (AMG)',
    #     'solver': 'PCG',
    #     'precond': 'AMG',
    #     'precond_kwargs': {},
    #     'style': {'color': 'brown', 'linestyle': '-', 'linewidth': 2}
    # },
]

# --- SOLVER CONFIG ---
# Maps solver names to their properties. Used by factory.get_solver_info().
# - solver_cls: Actual class name in GNP.solver module
# - use_lanczos: True for CG-family (uses Lanczos), False for GMRES (uses Arnoldi)
# - default_net: Default network architecture for GNP preconditioner
SOLVER_CONFIG = {
    'FGMRES': {'solver_cls': 'GMRES', 'use_lanczos': False, 'default_net': 'ResGCN'},
    'GMRES':  {'solver_cls': 'GMRES', 'use_lanczos': False, 'default_net': 'ResGCN'},
    'FCG':    {'solver_cls': 'FCG',   'use_lanczos': True,  'default_net': 'SplitResGCN'},
    'PCG':    {'solver_cls': 'PCG',   'use_lanczos': True,  'default_net': 'SplitResGCN'},
    'PolakRibiereCG': {'solver_cls': 'PolakRibiereCG', 'use_lanczos': True, 'default_net': 'SplitResGCN'},
}