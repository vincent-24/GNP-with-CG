"""
Solver and Training hyperparameters.
"""

# Solver Configurations
RESTART = 10

# Maximum solver iterations
MAX_ITERS = 100

# Convergence tolerance
TOLERANCE = 1e-8

# FCG Specific Settings
TRUNCATION_K = 5

# Data Generation 
LANCZOS_M = 80 

# Krylov subspace size for Arnoldi 
ARNOLDI_M = 40

# Neural Network Training
BATCH_SIZE = 16
EPOCHS = 2000
LEARNING_RATE = 1e-3

# System & Paths
NUM_WORKERS = 0
DEFAULT_DUMP_PATH = './dump/'
SUITE_SPARSE_PATH = '~/data/SuiteSparse/ssget/mat'