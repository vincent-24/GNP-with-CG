import os

# ===========================GLOBAL & SYSTEM CONFIGURATION===========================
SEED = 42
NUM_WORKERS = 8
MODE = 'both'  # Options: 'train', 'eval', 'both'

DEFAULT_DUMP_PATH = './dump/'
SUITE_SPARSE_PATH = os.getenv('SUITESPARSE_PATH', './data')
PROBLEM_PATH = None  

# ============================NEURAL NETWORK ARCHITECTURE============================
NETWORK_OVERRIDE = 'SplitResGCN'
NUM_LAYERS = 8
EMBED_DIM = 16
HIDDEN_DIM = 32
DROP_RATE = 0.0
TIE_WEIGHTS = True

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

# ===============================DATASET & HARVESTING================================
TRAIN_OFFLINE = True   # Set True to use pre-harvested data; False for streaming
OFFLINE_DATASET_DIR = './data/pcg_harvested'

HARVEST_DATASET_PATH = None 
HARVEST_NUM_RUNS = 200
HARVEST_MAX_ITERS = 1000
HARVEST_RTOL = 1e-10

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
    'PCG':            {'solver_cls': 'PCG',            'use_lanczos': True,  'default_net': 'SplitResGCN'},
    'PolakRibiereCG': {'solver_cls': 'PolakRibiereCG', 'use_lanczos': True,  'default_net': 'SplitResGCN'},
}

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
    {
        'name': 'PCG (AMG)',
        'solver': 'PCG',
        'precond': 'AMG',
        'precond_kwargs': {},
        'style': {'color': 'purple', 'linestyle': '-', 'linewidth': 2}
    },
]