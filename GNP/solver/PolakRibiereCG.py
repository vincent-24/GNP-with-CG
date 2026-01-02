import time
import torch
import numpy as np
from tqdm import tqdm
from .base import IterativeSolver
from GNP import config

class PolakRibiereCG(IterativeSolver):
    """
    Nonlinear Conjugate Gradient Solver using the Polak-Ribiere formula.
    
    Theoretical Basis:
    Uses the Polak-Ribiere formula for computing the beta parameter, which 
    provides some robustness to variable preconditioners. However, this is 
    NOT the same as Flexible CG (Notay 2000), which requires full 
    orthogonalization against previous search directions.
    
    This implementation assumes A is Symmetric Positive Definite (SPD).
    
    Note: For truly variable/nonlinear preconditioners (like Neural Networks),
    use the FCG (Flexible Conjugate Gradient) solver instead, which guarantees
    A-conjugacy of search directions through explicit orthogonalization.
    """
    def solve(self, A, b, M=None, x0=None, max_iters=100, rtol=1e-8, progress_bar=True): 
        
        x, norm_b, hists, tic, progress_bar = self._prepare_solve(b, x0, max_iters, 'PolakRibiereCG Solve', progress_bar)
        hist_abs, hist_rel, hist_energy, hist_time = hists

        iters = 0
        r = b - A @ x
        z = self._apply_M(M, r)
        d = z.clone()
        delta_rz = torch.dot(r, z)
        
        abs_res, rel_res = self._update_history(r, norm_b, tic, hists)
        
        while iters < max_iters:
            if rel_res < rtol: break
            
            self._record_direction(d)

            q = A @ d
            dAq = torch.dot(d, q)
            
            if dAq <= 1e-15: break
                
            alpha = delta_rz / dAq
            x = x + alpha * d
            r_old = r.clone()
            r = r - alpha * q
            
            abs_res, rel_res = self._update_history(r, norm_b, tic, hists)
            
            if rel_res < rtol:
                iters += 1
                if progress_bar: progress_bar.update()
                break

            z_new = self._apply_M(M, r)
            beta = max(0.0, (torch.dot(z_new, (r - r_old))) / torch.dot(r_old, z))
            d = z_new + beta * d
            
            if torch.dot(d, r) <= 0:
                d = z_new
                beta = 0.0

            z = z_new
            delta_rz = torch.dot(r, z)
            iters += 1

            if progress_bar: progress_bar.update()

        if progress_bar: progress_bar.close()
        
        ortho_map = self._compute_orthogonality(A)

        return x, iters, hist_abs, hist_rel, hist_time, ortho_map