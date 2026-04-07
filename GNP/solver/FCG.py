import time
import torch
import numpy as np

from tqdm import tqdm
from .base import IterativeSolver
from GNP import config

class FCG(IterativeSolver):
    """Flexible Conjugate Gradient solver (Polak-Ribière variant).

    Handles variable (iteration-dependent) SPD preconditioners correctly.
    Reduces to standard PCG when the preconditioner is constant.

    Reference: Notay (2000), "Flexible Conjugate Gradients",
               SIAM J. Sci. Comput. 22(4), 1444-1460.
    """
    def solve(self, A, b, M=None, x0=None, max_iters=100, rtol=1e-8, progress_bar=True, return_trajectory=False):
        x, norm_b, hists, tic, progress_bar = self._prepare_solve(b, x0, max_iters, 'FCG Solve', progress_bar)
        hist_abs, hist_rel, hist_energy, hist_time = hists
        history_x = []

        # --- Diagnostics setup ---
        diag = None
        if config.PCG_DIAGNOSTICS >= 1:
            from GNP.solver.diagnostics import PCGDiagnostics
            diag = PCGDiagnostics(config.PCG_DIAGNOSTICS)
            diag.max_iters = max_iters
            diag.rtol = rtol

            if M is not None and hasattr(M, '_set_diagnostics'):
                M._set_diagnostics(diag)

        iters = 0
        r = b - A @ x
        d = self._apply_M(M, r)
        delta_new = torch.dot(r, d)
        r_old = r.clone()
        abs_res, rel_res = self._update_history(r, norm_b, tic, hists)

        if return_trajectory:
            history_x.append(x.detach().clone())

        while iters < max_iters:
            if rel_res < rtol:
                if diag: 
                    diag.set_termination('converged')

                break

            self._record_direction(d)
            q = A @ d
            dAq = torch.dot(d, q)

            if abs(dAq) <= 1e-15:
                if diag: 
                    diag.set_termination('breakdown_dAq')

                break

            alpha = delta_new / dAq
            x = x + alpha * d

            r_true = b - A @ x
            r = r_true.clone()

            if return_trajectory:
                history_x.append(x.detach().clone())

            prev_abs = abs_res.item() if torch.is_tensor(abs_res) else abs_res
            abs_res, rel_res = self._update_history(r_true, norm_b, tic, hists)

            if rel_res < rtol:
                iters += 1

                if diag:
                    diag.set_termination('converged')

                    if diag.level >= 2:
                        diag.record_iteration(
                            iters, alpha.item(), 0.0,
                            delta_new.item(), 0.0, dAq.item(),
                            abs_res.item(), rel_res.item(), prev_abs)

                if progress_bar:
                    progress_bar.update()

                break

            s = self._apply_M(M, r)
            delta_old = delta_new
            delta_new = torch.dot(r, s)

            if abs(delta_old) <= 1e-15:
                if diag: 
                    diag.set_termination('breakdown_delta')
                    
                break

            # Polak-Ribière beta: correct for variable preconditioners
            beta = torch.dot(s, r - r_old) / delta_old
            r_old = r.clone()
            d = s + beta * d
            iters += 1

            if diag and diag.level >= 2:
                diag.record_iteration(
                    iters, alpha.item(), beta.item(),
                    delta_new.item(), delta_old.item(), dAq.item(),
                    abs_res.item() if torch.is_tensor(abs_res) else abs_res,
                    rel_res.item() if torch.is_tensor(rel_res) else rel_res,
                    prev_abs)

            if progress_bar:
                progress_bar.update()

        if diag and diag.termination_reason is None:
            diag.set_termination('max_iters')

        if progress_bar:
            progress_bar.close()

        if diag:
            diag.print_summary()

        self._last_diagnostics = diag
        ortho_map = self._compute_orthogonality(A)

        if return_trajectory:
            return x, iters, hist_abs, hist_rel, hist_time, ortho_map, history_x

        return x, iters, hist_abs, hist_rel, hist_time, ortho_map
