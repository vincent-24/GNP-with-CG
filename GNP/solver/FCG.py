import time
import torch
import numpy as np
from tqdm import tqdm

# Flexible Conjugate Gradient (FCG)
# Reference: Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
class FCG():

    def solve(self, A, b, M=None, x0=None, restart=None, max_iters=100,
              timeout=None, rtol=1e-8, progress_bar=True):
        
        if progress_bar:
            if timeout is None:
                pbar = tqdm(total=max_iters, desc='Solve')
                pbar.update()
            else:
                pbar = tqdm(desc='Solve')
                pbar.update()
                        
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0
        norm_b = torch.linalg.norm(b)
        hist_abs_res = []
        hist_rel_res = []
        hist_time = []
        
        tic = time.time()

        # Initial setup
        r = b - A @ x
        
        # Apply preconditioner M to residual r
        if M is not None:
            z = M.apply(r)
        else:
            z = r.clone()
            
        p = z.clone()
        
        # Track residual norms
        abs_res = torch.linalg.norm(r)
        rel_res = abs_res / norm_b
        hist_abs_res.append(abs_res.item())
        hist_rel_res.append(rel_res.item())
        hist_time.append(time.time() - tic)
        
        iters = 0
        
        # Main Loop
        while True:
            # Check convergence at start
            if (rel_res < rtol) or \
               (timeout is None and iters == max_iters) or \
               (timeout is not None and hist_time[-1] >= timeout):
                break

            # Matrix-vector product
            Ap = A @ p
            
            # Flexible CG step size: alpha = (z, r) / (p, Ap)
            gamma = torch.dot(z, r)
            
            # Safety check for breakdown (e.g. if A is indefinite or p is zero)
            denom = torch.dot(p, Ap)
            if denom.abs() < 1e-12:
                break
                
            alpha = gamma / denom
            
            # Update solution and residual
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            # Update history
            abs_res = torch.linalg.norm(r_new)
            rel_res = abs_res / norm_b
            hist_abs_res.append(abs_res.item())
            hist_rel_res.append(rel_res.item())
            hist_time.append(time.time() - tic)
            iters += 1
            
            if progress_bar:
                pbar.update()
                
            # Check convergence immediately after update
            if (rel_res < rtol) or \
               (timeout is None and iters == max_iters) or \
               (timeout is not None and hist_time[-1] >= timeout):
                break

            # Apply preconditioner (Flexible step: M can vary)
            if M is not None:
                z_new = M.apply(r_new)
            else:
                z_new = r_new.clone()
            
            # --- RESTART LOGIC ---
            # If the preconditioner is non-linear, conjugacy is lost over time.
            # Restarting (setting beta = 0) effectively resets the method to 
            # a preconditioned steepest descent step to clear history.
            if restart is not None and iters % restart == 0:
                beta = 0.0
            else:
                # Polak-Ribiere Beta Update for Flexible CG
                # beta = (z_new, r_new - r) / (z, r)
                gamma_new = torch.dot(z_new, r_new - r)
                beta = gamma_new / gamma
            
            # Update search direction
            p = z_new + beta * p
            
            # Update pointers
            r = r_new
            z = z_new

        if progress_bar:
            pbar.close()

        return x, iters, hist_abs_res, hist_rel_res, hist_time