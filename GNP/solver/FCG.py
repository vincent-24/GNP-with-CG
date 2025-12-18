import time
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

class FCG():
    def solve(self, A, b, M=None, x0=None, restart=None, max_iters=100,
              timeout=None, rtol=1e-8, progress_bar=True, truncation_k=5): 
        
        if progress_bar:
            pbar = tqdm(total=max_iters, desc='Solve')
                        
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0
            
        norm_b = torch.linalg.norm(b)
        hist_abs_res = []
        hist_rel_res = []
        hist_time = []
        
        tic = time.time()

        # initial setup
        r = b - A @ x
        if M is not None:
            z = M.apply(r)
        else:
            z = r.clone()
            
        p = z.clone()
        
        # --- OPTIMIZATION: Pre-allocate buffers for vectorization ---
        # store history as lists of tensors, then stack them for the projection step
        P_buffer = []
        AP_buffer = []
        AP_P_dot = [] # Cache denominator (p_j^T A p_j) to save compute
        
        # initial residuals
        abs_res = torch.linalg.norm(r)
        rel_res = abs_res / norm_b
        hist_abs_res.append(abs_res.item())
        hist_rel_res.append(rel_res.item())
        hist_time.append(time.time() - tic)
        
        iters = 0
        
        while True:
            if (rel_res < rtol) or (iters == max_iters):
                break

            # 1. Compute Ap
            Ap = A @ p
            
            # 2. Store for orthogonalization (Manage sliding window)
            denom_p = torch.dot(p, Ap) # We need this for alpha anyway
            
            if len(P_buffer) == truncation_k:
                P_buffer.pop(0)
                AP_buffer.pop(0)
                AP_P_dot.pop(0)
            
            P_buffer.append(p)
            AP_buffer.append(Ap)
            AP_P_dot.append(denom_p)

            # 3. Calculate Step Size (alpha)
            gamma = torch.dot(z, r)
            
            if denom_p.abs() < 1e-12: break
            alpha = gamma / denom_p
            
            # 4. Update Solution
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            # Convergence Check
            abs_res = torch.linalg.norm(r_new)
            rel_res = abs_res / norm_b
            hist_abs_res.append(abs_res.item())
            hist_rel_res.append(rel_res.item())
            hist_time.append(time.time() - tic)
            iters += 1
            if progress_bar: pbar.update()
            
            if (rel_res < rtol) or (iters == max_iters):
                break

            # 5. Preconditioner
            if M is not None:
                z_new = M.apply(r_new)
            else:
                z_new = r_new.clone()
                
            # 6. Polak-Ribiere with Auto-Restart
            gamma_new = torch.dot(z_new, r_new - r)
            beta = max(0.0, gamma_new / gamma)
            
            # 7. Update Search Direction 
            p_new = z_new + beta * p
            
            # 8. --- VECTORIZED TRUNCATED ORTHOGONALIZATION ---
            # Project p_new against all vectors in P_buffer at once
            if len(P_buffer) > 0:
                # Stack buffers: (N, k)
                # NOTE: In production code, pre-allocating a tensor is faster than stacking lists
                # But stacking is safer for dynamic sliding windows in Python
                AP_mat = torch.stack(AP_buffer, dim=1) 
                P_mat = torch.stack(P_buffer, dim=1)
                denoms = torch.stack(AP_P_dot)
                
                # Compute overlaps: (p_new, AP_j) for all j
                # Result shape: (k,)
                numerators = p_new.unsqueeze(0) @ AP_mat 
                numerators = numerators.squeeze(0)
                
                # Compute coefficients: coeffs = num / denom
                coeffs = numerators / (denoms + 1e-12)
                
                # Subtract projections: p_new = p_new - sum(coeff_j * P_j)
                # We use matrix-vector multiplication: (N, k) @ (k, 1) -> (N, 1)
                correction = P_mat @ coeffs.unsqueeze(1)
                p_new = p_new - correction.squeeze(1)

            # Update pointers
            p = p_new
            r = r_new
            z = z_new

        if progress_bar: pbar.close()

        return x, iters, hist_abs_res, hist_rel_res, hist_time