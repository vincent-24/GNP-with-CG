"""Algebraic Multigrid (AMG) Preconditioner using PyAMG."""

import torch
from scipy import sparse
from pyamg.blackbox import solver_configuration, make_csr
from pyamg.classical import air_solver
from pyamg.aggregation import smoothed_aggregation_solver

from .base import CPUPreconditioner

def _build_amg_solver(A, config, use_air_for_nonsymmetric: bool = True):
    """Build an AMG solver given matrix A and configuration.

    Args:
        A: Scipy sparse matrix in CSR format
        config: Configuration dict from solver_configuration()
        use_air_for_nonsymmetric: If True, use AIR solver for non-hermitian matrices

    Returns:
        PyAMG multilevel solver
    """
    A = make_csr(A)

    if config['symmetry'] == 'hermitian':
        return smoothed_aggregation_solver(
            A,
            B=config['B'],
            BH=config['BH'],
            smooth=config['smooth'],
            strength=config['strength'],
            max_levels=config['max_levels'],
            max_coarse=config['max_coarse'],
            coarse_solver=config['coarse_solver'],
            symmetry=config['symmetry'],
            aggregate=config['aggregate'],
            presmoother=config['presmoother'],
            postsmoother=config['postsmoother'],
            keep=config['keep']
        )
    else:
        if use_air_for_nonsymmetric:
            # AIR solver for non-symmetric matrices
            return air_solver(
                A,
                presmoother=config['presmoother'],
                postsmoother=config['postsmoother']
            )
        else:
            # Fall back to smoothed aggregation even for non-symmetric
            return smoothed_aggregation_solver(
                A,
                B=config['B'],
                BH=config['BH'],
                smooth=config['smooth'],
                strength=config['strength'],
                max_levels=config['max_levels'],
                max_coarse=config['max_coarse'],
                coarse_solver=config['coarse_solver'],
                symmetry=config['symmetry'],
                aggregate=config['aggregate'],
                presmoother=config['presmoother'],
                postsmoother=config['postsmoother'],
                keep=config['keep']
            )

class AMGPreconditioner(CPUPreconditioner):
    """Algebraic Multigrid preconditioner using PyAMG.

    For symmetric positive definite matrices, uses smoothed aggregation.
    For non-symmetric matrices, can optionally use AIR (Approximate Ideal
    Restriction) which is better suited for non-symmetric problems.

    Args:
        A: System matrix as torch sparse CSC tensor
        use_air_for_nonsymmetric: If True (default), use AIR solver for
            non-hermitian matrices. If False, use smoothed aggregation for all.
    """
    def __init__(self, A: torch.Tensor, use_air_for_nonsymmetric: bool = True):
        if A.layout != torch.sparse_csc:
            raise ValueError('AMGPreconditioner requires A to be sparse CSC')

        super().__init__(device=A.device)

        # Convert to scipy sparse
        A_cpu = A.to('cpu')
        n = A_cpu.shape[0]
        spA = sparse.csc_array(
            (A_cpu.values().numpy(),
             A_cpu.row_indices().numpy(),
             A_cpu.ccol_indices().numpy()),
            shape=(n, n)
        )
        spA = sparse.csr_matrix(spA)

        # Build AMG solver
        config = solver_configuration(spA, verb=False)
        mg = _build_amg_solver(spA, config, use_air_for_nonsymmetric)
        self.M = mg.aspreconditioner()

    def _apply_numpy(self, r):
        """Apply AMG V-cycle."""
        return self.M * r
