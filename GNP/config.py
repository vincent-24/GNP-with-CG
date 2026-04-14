import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ===========================GLOBAL & SYSTEM CONFIGURATION===========================
SEED = 42
NUM_WORKERS = 8
MODE = 'both'  # Options: 'train', 'eval', 'both'

WORK_ROOT = os.getenv('WORK_ROOT', '../tests/')

DATA_ROOT = os.getenv('DATA_PATH', os.path.join(WORK_ROOT, 'data'))
DEFAULT_DUMP_PATH = os.getenv('DUMP_PATH', os.path.join(WORK_ROOT, 'dump'))
SUITE_SPARSE_PATH = os.getenv('SUITESPARSE_PATH', os.path.join(DATA_ROOT, 'SuiteSparse'))

PROBLEM_PATH = None

# ============================NEURAL NETWORK ARCHITECTURE============================
NETWORK_OVERRIDE = 'TwoLevelMGGNN'
NUM_LAYERS = 8
EMBED_DIM = 64
HIDDEN_DIM = 64
DROP_RATE = 0.0

# Multigrid hierarchy
NUM_LEVELS = None          # None = auto (min(8, ceil(log2(n))))
CROSS_LEVEL_WIDTH = 128    # Paper: 2 FC layers of size 128 for inter-level MLPs

# TwoLevelMGGNN-specific (paper-faithful parallel cross-scale architecture)
TWO_LEVEL_NUM_STEPS = 6            # Number of implicit correction steps
TWO_LEVEL_FINE_K = 5               # TAGConv polynomial order for fine level
TWO_LEVEL_COARSE_K = 7             # TAGConv polynomial order for coarse level
TWO_LEVEL_SHARE_BLOCKS = False     # Share block weights across steps (True = fewer params)
TWO_LEVEL_CROSS_WIDTH = 128        # Hidden dim for cross-level MLPs
SPD_JACOBI_EPS = 1e-6              # Jacobi floor for SPD enforcement: M^{-1}(r) = N^{T}N(r) + \epsilon D^{-1}r

BATCH_SIZE = 16
EPOCHS = 50                       # Hutchinson loss is 160x cheaper per step → can afford 10x more epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0                # Adam weight decay regularization
TRAIN_VAL_SPLIT = 0.9             
GRAD_CLIP_NORM = 5.0               # Maximum gradient norm for clipping

# Spectral loss (unsupervised training)
# Loss types: 'rho'        = ρ(I-M⁻¹A) via power iteration
#              'kappa'      = log κ(M⁻¹A) via two-phase power iteration
#              'hutchinson' = ||I - M⁻¹A||²_F via Hutchinson trace estimator (160x cheaper)
#              'curriculum' = hutchinson first, then warm-started kappa refinement
SPECTRAL_LOSS_TYPE = 'hutchinson'
SPECTRAL_NUM_VECTORS = 16      # Probe vectors per step (16 for hutchinson, 64 for kappa)
SPECTRAL_POWER_ITERS = 20      # Power iteration depth (only used by rho/kappa)
SPECTRAL_STEPS_PER_EPOCH = 100  # Optimizer steps per epoch (no DataLoader)

# Curriculum schedule (only used when SPECTRAL_LOSS_TYPE = 'curriculum')
CURRICULUM_SWITCH_FRAC = 0.8    # Fraction of total steps using hutchinson before switching to warm kappa
CURRICULUM_WARM_POWER_ITERS = 3 # Power iterations for warm-started kappa phase
CURRICULUM_WARM_NUM_VECTORS = 16 # Probe vectors for warm-started kappa phase

# Learning rate scheduler
LR_SCHEDULER = 'cosine'        # 'none', 'cosine' (CosineAnnealingWarmRestarts)
LR_COSINE_T0 = 5               # Restart period in epochs for cosine scheduler
LR_COSINE_ETA_MIN = 1e-5       # Minimum learning rate

# ===============================DATASET & HARVESTING================================
TRAIN_OFFLINE = False  # True = supervised (harvested PCG dataset), False = spectral (rho minimization)
OFFLINE_DATASET_DIR = os.getenv('OFFLINE_DATASET_DIR', os.path.join(DATA_ROOT, 'pcg_harvested'))
RANDOM_RATIO = 0.5  # Fraction of dataset that is white-noise vectors

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

BASELINE_SOLVER = 'PCG' # 'PCG' for SPD matrices, 'FGMRES' for non-symmetric

# ==============================EVALUATION & ANALYSIS==============================
TRACK_ORTHOGONALITY = True
ORTHOGONALITY_SAMPLE_RATE = 1
X_GROUND_TRUTH = 'random'   # Options: 'random', 'ones', 'ramp', 'sine', 'alternating'

PCG_DIAGNOSTICS = 3            # 0=off, 1=termination reason, 2=+per-iter scalars, 3=+preconditioner internals
PCG_DIAG_SYMMETRY_PERIOD = 10  # Symmetry probe every N iters (level 3 only)

# =====================ADVANCED CONFIGURATIONS (DICTS & LISTS)=====================
SOLVER_CONFIG = {
    'FGMRES': {'solver_cls': 'GMRES', 'use_lanczos': False, 'default_net': NETWORK_OVERRIDE},
    'GMRES':  {'solver_cls': 'GMRES', 'use_lanczos': False, 'default_net': NETWORK_OVERRIDE},
    'PCG':    {'solver_cls': 'PCG',   'use_lanczos': True,  'default_net': NETWORK_OVERRIDE},
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
        'name': 'CG',
        'solver': 'PCG',
        'precond': None,
        'precond_kwargs': {},
        'style': {'color': 'green', 'linestyle': '-', 'linewidth': 2}
    },
    {
        'name': 'CG (IChol)',
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
