import os

# ===========================GLOBAL & SYSTEM CONFIGURATION===========================
SEED = 42
NUM_WORKERS = 8
MODE = 'both'  # Options: 'train', 'eval', 'both'

# Keep filesystem defaults on the same /work volume as Slurm logs.
WORK_ROOT = os.getenv('GNP_WORK_ROOT', '/work/hdd/bdyf/vterrelonge/GNP-with-CG')
DATA_ROOT = os.getenv('GNP_DATA_PATH', os.path.join(WORK_ROOT, 'data'))

DEFAULT_DUMP_PATH = os.getenv('GNP_DUMP_PATH', os.path.join(WORK_ROOT, 'dump'))
SUITE_SPARSE_PATH = os.getenv('SUITESPARSE_PATH', os.path.join(DATA_ROOT, 'SuiteSparse'))
CHECKPOINT_PATH = os.getenv('GNP_CHECKPOINT_PATH', os.path.join(WORK_ROOT, 'checkpoints'))
PROBLEM_PATH = None  

# ============================NEURAL NETWORK ARCHITECTURE============================
NETWORK_OVERRIDE = 'LinearMGGNN'  
NUM_LAYERS = 8
EMBED_DIM = 16
HIDDEN_DIM = 32
DROP_RATE = 0.0
TIE_WEIGHTS = True

# UNetGCN / MGGNN multigrid hierarchy
NUM_LEVELS = None          # None = auto (min(8, ceil(log2(n))))
LAYERS_PER_LEVEL = 2       # GCN layers per resolution level

# MGGNN-specific (parallel multi-scale architecture)
NUM_BLOCKS = 4             # Number of stacked MG-GNN blocks
TAGCONV_K = 3              # TAGConv polynomial order (K-hop neighbourhood)
CROSS_LEVEL_WIDTH = 128    # Paper: 2 FC layers of size 128 for inter-level MLPs
NUM_V_CYCLES = 2           # Legacy — kept for checkpoint compat

# LinearMGGNN-specific (strictly linear V-cycle for standard PCG compatibility)
LINEAR_MGGNN_NUM_VCYCLES = 2       # Number of V-cycles per forward pass
LINEAR_MGGNN_SMOOTHER_K = 3        # Polynomial degree for Chebyshev-like smoother
LINEAR_MGGNN_COARSEST_K = 5        # Polynomial degree for coarsest level solve
LINEAR_MGGNN_SHARE_SMOOTHERS = True  # Share pre/post smoothers (ensures symmetric operator)

# ICholSparseTensorNet-specific (frozen IChol sparse-tensor core + trainable scalars)
ICHOL_SPARSE_DROP_TOL = 1e-3
ICHOL_SPARSE_SHIFT = 1e-3
ICHOL_SPARSE_INIT_ALPHA = 0.95
ICHOL_SPARSE_INIT_GAIN = 1.0
ICHOL_SPARSE_MIN_ALPHA = 0.9
ICHOL_SPARSE_GAIN_DELTA = 0.1

# ---- MGGNN v2: Neural Multigrid Generator features ----
# Phase 1 — Differentiable coarsening (MinCutPool-style learned topology)
LEARNED_COARSENING = False
LAMBDA_CUT = 1.0           # Weight for MinCut loss  -Tr(S^T A S)/Tr(S^T D S)
LAMBDA_ORTH = 1.0          # Weight for orthogonality loss ||S^T S/n - I/k||_F

# Phase 2 — Petrov-Galerkin asymmetric projection (R ≠ P^T)
PETROV_GALERKIN = False
LAMBDA_SYM = 0.1           # Soft symmetry penalty ||R - P^T||_F^2 (SPD mode)

# Phase 3 — Anisotropic message passing (GATv2 attention)
USE_ATTENTION = False
NUM_ATTENTION_HEADS = 4

# Phase 4 — Learned Neural Smoother (per-node Jacobi damping)
USE_NEURAL_SMOOTHER = False
NUM_SMOOTH_STEPS = 1       # Number of pre/post-smooth Jacobi iterations
SYMMETRIC_SMOOTH = True    # Apply post-smooth too (preserves SPD for PCG)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4                # Adam weight decay regularization
TRAIN_VAL_SPLIT = 0.9              # Fraction of data for training (rest for validation)
GRAD_CLIP_NORM = 1.0               # Maximum gradient norm for clipping

# Spectral radius loss (unsupervised training)
SPECTRAL_NUM_VECTORS = 32      # Random probe vectors per optimizer step
SPECTRAL_POWER_ITERS = 10      # Power iteration depth for rho estimation
SPECTRAL_STEPS_PER_EPOCH = 50  # Optimizer steps per epoch (no DataLoader)

# ===============================DATASET & HARVESTING================================
TRAIN_OFFLINE = True   # Set True to use pre-harvested data; False for streaming
OFFLINE_DATASET_DIR = os.getenv('GNP_OFFLINE_DATASET_DIR', os.path.join(DATA_ROOT, 'pcg_harvested'))
RANDOM_RATIO = 0.0  # Fraction of dataset that is white-noise vectors (0.0 = pure PCG, 1.0 = pure noise)

HARVEST_DATASET_PATH = None 
HARVEST_NUM_RUNS = 200
HARVEST_MAX_ITERS = 1000
HARVEST_RTOL = 1e-10
HARVEST_CHUNK_SIZE = 50  # Chunk size for memory-efficient harvesting

LOG_SAMPLING = False
LOG_SAMPLING_BASE = 2.0
    
# ===========================SOLVER MATHEMATICAL SETTINGS===========================
RESTART = 10
MAX_ITERS = 2000
TOLERANCE = 1e-10

LANCZOS_M = 80
ARNOLDI_M = 40

# FCG Specific
TRUNCATION_K = None

BASELINE_SOLVER = 'PCG' # 'PCG' for SPD matrices, 'FGMRES' for non-symmetric
SOLVERS = ['FGMRES', 'PCG']

# ==============================EVALUATION & ANALYSIS==============================
TRACK_ORTHOGONALITY = True
ORTHOGONALITY_SAMPLE_RATE = 1  

# =====================ADVANCED CONFIGURATIONS (DICTS & LISTS)=====================
SOLVER_CONFIG = {
    'FGMRES':         {'solver_cls': 'GMRES',          'use_lanczos': False, 'default_net': 'ResGCN'},
    'GMRES':          {'solver_cls': 'GMRES',          'use_lanczos': False, 'default_net': 'ResGCN'},
    'FCG':            {'solver_cls': 'FCG',            'use_lanczos': True,  'default_net': 'SplitResGCN'},
    'PCG':            {'solver_cls': 'PCG',            'use_lanczos': True,  'default_net': 'UNetGCN'},
    'PolakRibiereCG': {'solver_cls': 'PolakRibiereCG', 'use_lanczos': True,  'default_net': 'SplitResGCN'},
}

def _get_arch_label():
    """Derive a short display label from NETWORK_OVERRIDE for experiment names."""
    return NETWORK_OVERRIDE if NETWORK_OVERRIDE else 'GNP'

_ARCH_LABEL = _get_arch_label()

EXPERIMENTS = [
    {
        'name': f'FGMRES ({_ARCH_LABEL})',
        'solver': 'FGMRES',
        'precond': 'GNP',
        'precond_kwargs': {},
        'style': {'color': 'blue', 'linestyle': '-', 'linewidth': 2}
    },
    {
        'name': f'PCG ({_ARCH_LABEL})',
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
    {
        'name': 'GMRES (AMG)',
        'solver': 'GMRES',
        'precond': 'AMG',
        'precond_kwargs': {},
        'style': {'color': 'purple', 'linestyle': '-', 'linewidth': 2}
    },
]