import time
import torch
import numpy as np
from tqdm import tqdm
from .base import IterativeSolver
from GNP import config

class PCG(IterativeSolver):
    """
    Preconditioned Conjugate Gradient (PCG) Solver.
    
    Theoretical Basis:
    Based on Algorithm B3 from J.R. Shewchuk, 
    "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain".
    
    Assumption:
    1. A is Symmetric Positive Definite (SPD).
    2. M is Symmetric Positive Definite (SPD).
    
    NOTE ON NEURAL NETWORKS:
    Standard PCG assumes M is a LINEAR operator. Neural Networks (with ReLU) are 
    NONLINEAR. Using this solver with a Neural Network is a heuristic. 
    It serves as a baseline to demonstrate why Flexible CG (FCG) is often required 
    for Neural Preconditioning.
    """

    def solve(self, A, b, M=None, x0=None, max_iters=100, rtol=1e-8, progress_bar=True): 
        x, norm_b, hists, tic, progress_bar = self._prepare_solve(b, x0, max_iters, 'PCG Solve', progress_bar)
        hist_abs, hist_rel, hist_energy, hist_time = hists

        iters = 0
        r = b - A @ x
        d = self._apply_M(M, r)
        delta_new = torch.dot(r, d)
        abs_res, rel_res = self._update_history(r, norm_b, tic, hists)
        
        while iters < max_iters:
            if rel_res < rtol: break
            
            self._record_direction(d)

            q = A @ d
            dAq = torch.dot(d, q)
            
            # Safety for numerical breakdown
            if dAq <= 1e-15: break
                
            alpha = delta_new / dAq
            x = x + alpha * d
            r = r - alpha * q
            
            abs_res, rel_res = self._update_history(r, norm_b, tic, hists)
            
            if rel_res < rtol:
                iters += 1
                if progress_bar: progress_bar.update()
                break

            s = self._apply_M(M, r)
            
            delta_old = delta_new
            delta_new = torch.dot(r, s)
            beta = delta_new / (delta_old + 1e-15)
            d = s + beta * d
            iters += 1

            if progress_bar: progress_bar.update()

        if progress_bar: progress_bar.close()
        
        ortho_map = self._compute_orthogonality(A)

        return x, iters, hist_abs, hist_rel, hist_time, ortho_map