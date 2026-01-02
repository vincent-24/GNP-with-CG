"""
Base class for solvers.
Provides common initialization and utility methods.
"""
import time
import torch
import numpy as np
from tqdm import tqdm
from GNP import config

class IterativeSolver:
    def _prepare_solve(self, b, x0, max_iters, desc, progress_bar=True):
        # Initialize x
        if x0 is None: x = torch.zeros_like(b)
        else: x = x0.clone()

        # Compute norm_b for relative residual
        norm_b = torch.linalg.norm(b)

        if norm_b == 0: norm_b = 1.0

        hist_abs = []
        hist_rel = []
        hist_energy = []
        hist_time = []
        
        # Initialize orthogonality tracking (only if enabled)
        self.search_directions = []
        self._direction_count = 0
        
        # Timer & Progress Bar
        tic = time.time()
        pbar = None

        if progress_bar: pbar = tqdm(total=max_iters, desc=desc)

        return x, norm_b, (hist_abs, hist_rel, hist_energy, hist_time), tic, pbar

    def _apply_M(self, M, r):
        if M is not None: return M.apply(r)
        return r.clone()

    def _update_history(self, r, norm_b, tic, history_tuple, energy_val=None):
        hist_abs, hist_rel, hist_energy, hist_time = history_tuple
    
        abs_res = torch.linalg.norm(r)
        rel_res = abs_res / norm_b
        hist_abs.append(abs_res.item())
        hist_rel.append(rel_res.item())
        hist_time.append(time.time() - tic)
        
        if energy_val is not None: hist_energy.append(energy_val.item())
            
        return abs_res, rel_res

    def _record_direction(self, d):
        """Record search direction for orthogonality analysis (if enabled)."""
        if not config.TRACK_ORTHOGONALITY:
            return
        
        self._direction_count += 1
        if self._direction_count % config.ORTHOGONALITY_SAMPLE_RATE == 0:
            # Clone, detach, move to CPU to save GPU memory
            self.search_directions.append(d.detach().cpu().clone())

    def _compute_orthogonality(self, A):
        """Compute normalized A-orthogonality matrix of recorded search directions.
        
        Returns:
            H: numpy array where H[i,j] = |d_i^T A d_j| / (||d_i||_A * ||d_j||_A)
               Diagonal is 1.0, off-diagonal measures loss of A-conjugacy.
               Returns None if no directions were recorded.
        """
        if not config.TRACK_ORTHOGONALITY or len(self.search_directions) == 0:
            return None
        
        D = torch.stack(self.search_directions, dim=1)  # Matrix D where columns are d_0, d_1...
        A_cpu = A.to('cpu')
        AD = A_cpu @ D                  # Matrix where columns are (A * d_j)
        M = torch.abs(D.T @ AD)         # Computes D^T * (A * D)
        diag = torch.diag(M)            # Extracts the diagonal (d_i^T A d_i)
        N = torch.sqrt(diag + 1e-15)    # Calculates the Length: sqrt(...)
        outer_N = torch.outer(N, N)     # Calculates Length(i) * Length(j) for the whole grid
        H = M / (outer_N + 1e-15)
        
        return H.numpy()
