import time
import torch
from .base import IterativeSolver
from GNP import config

class FCG(IterativeSolver):
    """
    Flexible Conjugate Gradient (FCG).
    Implementation of Algorithm 2.1 from Notay (2000).
    
    BEST SETTINGS:
    - truncation_k = None (Full Orthogonalization) is recommended for 
      Neural Preconditioners to prevent stagnation.
    """
    
    def __init__(self, truncation_k=1):
        self.truncation_k = truncation_k

    def solve(self, A, b, M=None, x0=None, max_iters=100, rtol=1e-8, progress_bar=True):
        x, norm_b, hists, tic, pbar = self._prepare_solve(b, x0, max_iters, "FCG Solve", progress_bar)
        
        r = b - A @ x
        n = b.shape[0]
        device = b.device
        dtype = b.dtype
    
        capacity = self.truncation_k if self.truncation_k is not None else max_iters
        D_mem = torch.zeros(n, capacity, device=device, dtype=dtype)   # Search directions
        AD_mem = torch.zeros(n, capacity, device=device, dtype=dtype)  # A @ d_j
        dAd_vec = torch.zeros(capacity, device=device, dtype=dtype)    # d_j^T A d_j
        num_dirs = 0  # Number of stored directions
        
        self._update_history(r, norm_b, tic, hists)
        
        while len(hists[1]) - 1 < max_iters: 
            if hists[1][-1] < rtol:
                if pbar: pbar.close()
                break
                
            w = self._apply_M(M, r)
            d = w.clone()
            
            if num_dirs > 0:
                AD_active = AD_mem[:, :num_dirs]  # (n, num_dirs)
                D_active = D_mem[:, :num_dirs]    # (n, num_dirs)
                dAd_active = dAd_vec[:num_dirs]   # (num_dirs,)
                alphas = (AD_active.T @ w) / (dAd_active + 1e-15)  # (num_dirs,)
                d = d - D_active @ alphas  # (n,)
            
            self._record_direction(d)
    
            q = A @ d
            d_Aq = torch.dot(d, q)
            
            if d_Aq <= 1e-15: break # Breakdown safety
            
            alpha = torch.dot(d, r) / d_Aq
            
            x = x + alpha * d
            # r = r - alpha * q
            r = b - A @ x 
            
            if self.truncation_k is not None and num_dirs >= self.truncation_k:
                D_mem[:, :-1] = D_mem[:, 1:].clone()
                AD_mem[:, :-1] = AD_mem[:, 1:].clone()
                dAd_vec[:-1] = dAd_vec[1:].clone()
                D_mem[:, -1] = d
                AD_mem[:, -1] = q
                dAd_vec[-1] = d_Aq
            else:
                D_mem[:, num_dirs] = d
                AD_mem[:, num_dirs] = q
                dAd_vec[num_dirs] = d_Aq
                num_dirs += 1

            self._update_history(r, norm_b, tic, hists)

            if pbar: pbar.update()

        if pbar: pbar.close()
        hist_abs, hist_rel, hist_energy, hist_time = hists
        ortho_map = self._compute_orthogonality(A)
        
        return x, len(hist_rel)-1, hist_abs, hist_rel, hist_time, ortho_map