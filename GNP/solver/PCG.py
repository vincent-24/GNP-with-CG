import time
import torch
import numpy as np

from tqdm import tqdm
from .base import IterativeSolver
from GNP import config

class PCG(IterativeSolver):
    def solve(self, A, b, M=None, x0=None, max_iters=100, rtol=1e-8, progress_bar=True, return_trajectory=False):
        x, norm_b, hists, tic, progress_bar = self._prepare_solve(b, x0, max_iters, 'PCG Solve', progress_bar)
        hist_abs, hist_rel, hist_energy, hist_time = hists
        history_x = []
        iters = 0
        r = b - A @ x
        d = self._apply_M(M, r)
        delta_new = torch.dot(r, d)
        abs_res, rel_res = self._update_history(r, norm_b, tic, hists)

        if return_trajectory: 
            history_x.append(x.detach().clone())

        while iters < max_iters:
            if rel_res < rtol: 
                break

            self._record_direction(d)
            q = A @ d
            dAq = torch.dot(d, q)

            # Safety for numerical breakdown (use abs to handle non-SPD preconditioners)
            if abs(dAq) <= 1e-15: 
                break

            alpha = delta_new / dAq
            x = x + alpha * d

            # True residual for convergence check and history
            r_true = b - A @ x
            r = r_true.clone()

            if return_trajectory: 
                history_x.append(x.detach().clone())

            abs_res, rel_res = self._update_history(r_true, norm_b, tic, hists)

            if rel_res < rtol:
                iters += 1

                if progress_bar: 
                    progress_bar.update()

                break

            s = self._apply_M(M, r)
            delta_old = delta_new
            delta_new = torch.dot(r, s)

            # Safety for numerical breakdown (use abs to handle non-SPD preconditioners)
            if abs(delta_old) <= 1e-15: 
                break

            beta = delta_new / delta_old
            d = s + beta * d
            iters += 1

            if progress_bar: 
                progress_bar.update()

        if progress_bar: 
            progress_bar.close()

        ortho_map = self._compute_orthogonality(A)

        if return_trajectory: 
            return x, iters, hist_abs, hist_rel, hist_time, ortho_map, history_x

        return x, iters, hist_abs, hist_rel, hist_time, ortho_map