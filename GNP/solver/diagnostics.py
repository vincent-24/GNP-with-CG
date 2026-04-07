"""PCG diagnostics collector.

Levels:
    0 — off (no overhead)
    1 — termination reason only
    2 — per-iteration PCG scalars (alpha, beta, dAq, delta, residual)
    3 — per-iteration preconditioner decomposition (||NtNr||, ||Jacobi||, r^T M^{-1}r, symmetry)
"""
import math

def _append_table(lines, title, header, data, fmt_row):
    """Append a titled, formatted table to *lines*."""
    sep = '-' * len(header)
    lines.append(title)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    
    for d in data:
        lines.append(fmt_row(d))

    lines.append('')

class PCGDiagnostics:
    def __init__(self, level: int):
        self.level = level
        self.termination_reason = None
        self.max_iters = None
        self.rtol = None
        self.total_iters = 0
        self.final_rel_res = None
        self.iter_data = []
        self.precond_data = []

    def set_termination(self, reason: str, iters: int = None, rel_res: float = None):
        self.termination_reason = reason
        if iters is not None:
            self.total_iters = iters
        if rel_res is not None:
            self.final_rel_res = rel_res

    def record_iteration(self, i, alpha, beta, delta_new, delta_old, dAq, abs_res, rel_res, prev_abs_res):
        if self.level < 2:
            return

        drop = prev_abs_res / abs_res if abs_res > 0 else float('inf')
        self.iter_data.append({
            'iter': i,
            'alpha': alpha,
            'beta': beta,
            'delta_new': delta_new,
            'delta_old': delta_old,
            'dAq': dAq,
            'abs_res': abs_res,
            'rel_res': rel_res,
            'drop': drop,
            'alpha_neg': alpha < 0,
            'beta_neg': beta < 0,
        })

    def record_precond(self, i, r_norm, Nr_norm, NtNr_norm, jacobi_norm, rTMr, symmetry_err=None):
        if self.level < 3:
            return

        ratio = NtNr_norm / max(jacobi_norm, 1e-30)
        self.precond_data.append({
            'iter': i,
            'r_norm': r_norm,
            'Nr_norm': Nr_norm,
            'NtNr_norm': NtNr_norm,
            'jacobi_norm': jacobi_norm,
            'ratio': ratio,
            'rTMr': rTMr,
            'symmetry_err': symmetry_err,
        })

    def summary_dict(self):
        final_rel = self.iter_data[-1]['rel_res'] if self.iter_data else self.final_rel_res
        total_iters = len(self.iter_data) if self.iter_data else self.total_iters
        neg_alpha = sum(1 for d in self.iter_data if d.get('alpha_neg'))
        neg_beta = sum(1 for d in self.iter_data if d.get('beta_neg'))
        min_dAq = min((d['dAq'] for d in self.iter_data), default=None)
        jacobi_dom = sum(1 for d in self.precond_data if d['ratio'] < 1.0)
        jacobi_frac = jacobi_dom / max(len(self.precond_data), 1)

        return {
            'termination': self.termination_reason,
            'total_iters': total_iters,
            'final_rel_res': final_rel,
            'negative_alpha_count': neg_alpha,
            'negative_beta_count': neg_beta,
            'min_dAq': min_dAq,
            'jacobi_dominance_frac': jacobi_frac if self.precond_data else None,
        }

    def print_summary(self):
        reason = self.termination_reason or 'unknown'
        n = len(self.iter_data) if self.iter_data else self.total_iters
        rel = self.iter_data[-1]['rel_res'] if self.iter_data else self.final_rel_res
        max_it = self.max_iters or '?'
        rtol = self.rtol or '?'
        rel_str = f'{rel:.2e}' if isinstance(rel, float) else '?'

        print(f"\nPCG TERMINATION: {reason}")
        print(f"  Iterations: {n} / {max_it} | "
              f"Final rel residual: {rel_str} (target: {rtol})")

        if self.level >= 2:
            neg_a = sum(1 for d in self.iter_data if d.get('alpha_neg'))
            neg_b = sum(1 for d in self.iter_data if d.get('beta_neg'))

            if neg_a:
                print(f"  WARNING: {neg_a} iterations had negative alpha (non-SPD indicator)")
            if neg_b:
                print(f"  WARNING: {neg_b} iterations had negative beta")

        if self.precond_data:
            jdom = sum(1 for d in self.precond_data if d['ratio'] < 1.0)
            frac = jdom / len(self.precond_data)

            if frac > 0.5:
                print(f"  WARNING: Jacobi floor dominated learned component "
                      f"in {jdom}/{len(self.precond_data)} iterations ({frac:.0%})")

            spd_fail = sum(1 for d in self.precond_data if d['rTMr'] <= 0)

            if spd_fail:
                print(f"  WARNING: r^T M^{{-1}}r <= 0 at {spd_fail} iterations (SPD violation!)")

    def write_log(self, path: str):
        lines = []
        lines.append('=' * 90)
        lines.append('PCG DIAGNOSTICS LOG')
        lines.append('=' * 90)

        reason = self.termination_reason or 'unknown'
        n = len(self.iter_data) if self.iter_data else self.total_iters
        max_it = self.max_iters or '?'
        rtol = self.rtol or '?'
        rel = self.iter_data[-1]['rel_res'] if self.iter_data else self.final_rel_res
        rel_str = f'{rel:.2e}' if isinstance(rel, float) else '?'

        lines.append(f'Termination: {reason}')
        lines.append(f'Iterations:  {n} / {max_it}')
        lines.append(f'Final rel residual: {rel_str} (target: {rtol})')
        lines.append('')

        # Per-iteration PCG scalars
        if self.iter_data:
            hdr = (f'{"Iter":>5}  {"alpha":>12}  {"beta":>12}  {"dAq":>12}  '
                   f'{"delta_new":>12}  {"delta_old":>12}  {"rel_res":>12}  {"drop":>8}')

            def _fmt_iter(d):
                a_str = f'{d["alpha"]:>12.4e}' if not math.isnan(d['alpha']) else f'{"---":>12}'
                return (f'{d["iter"]:>5}  {a_str}  {d["beta"]:>12.4e}  '
                        f'{d["dAq"]:>12.4e}  {d["delta_new"]:>12.4e}  '
                        f'{d["delta_old"]:>12.4e}  {d["rel_res"]:>12.4e}  '
                        f'{d["drop"]:>8.2f}')

            _append_table(lines, 'PCG ITERATION SCALARS', hdr, self.iter_data, _fmt_iter)

        # Per-iteration preconditioner decomposition
        if self.precond_data:
            hdr = (f'{"Iter":>5}  {"||r||":>12}  {"||Nr||":>12}  {"||NtNr||":>12}  '
                   f'{"||Jac||":>12}  {"ratio":>10}  {"r^TMr":>12}  {"sym_err":>10}')

            def _fmt_precond(d):
                sym = f'{d["symmetry_err"]:.2e}' if d['symmetry_err'] is not None else '---'
                return (f'{d["iter"]:>5}  {d["r_norm"]:>12.4e}  {d["Nr_norm"]:>12.4e}  '
                        f'{d["NtNr_norm"]:>12.4e}  {d["jacobi_norm"]:>12.4e}  '
                        f'{d["ratio"]:>10.4f}  {d["rTMr"]:>12.4e}  {sym:>10}')

            _append_table(lines, 'PRECONDITIONER DECOMPOSITION', hdr, self.precond_data, _fmt_precond)

        # Warnings
        warnings = []
        if self.iter_data:
            neg_a = sum(1 for d in self.iter_data if d.get('alpha_neg'))
            neg_b = sum(1 for d in self.iter_data if d.get('beta_neg'))

            if neg_a:
                warnings.append(f'{neg_a} iterations had negative alpha (non-SPD indicator)')
            if neg_b:
                warnings.append(f'{neg_b} iterations had negative beta')

            # Stagnation detection
            if len(self.iter_data) >= 5:
                last5 = self.iter_data[-5:]

                if all(d['drop'] < 1.05 for d in last5):
                    warnings.append(f'Residual stagnated: drop ratio < 1.05 for last 5 iterations')

        if self.precond_data:
            jdom = sum(1 for d in self.precond_data if d['ratio'] < 1.0)

            if jdom > len(self.precond_data) * 0.5:
                warnings.append(
                    f'Jacobi floor dominated learned component in '
                    f'{jdom}/{len(self.precond_data)} iterations')
            spd_fail = sum(1 for d in self.precond_data if d['rTMr'] <= 0)

            if spd_fail:
                warnings.append(f'r^T M^{{-1}}r <= 0 at {spd_fail} iterations (SPD violation)')

        if warnings:
            lines.append('WARNINGS')
            lines.append('-' * 40)

            for w in warnings:
                lines.append(f'  - {w}')
                
            lines.append('')

        lines.append('=' * 90)

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
