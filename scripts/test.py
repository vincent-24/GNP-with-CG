import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from GNP import config
from GNP.factory import get_solver_info
from scripts.utils import get_preconditioner, plot_results

def eval_routine(A, A_csc, b, x_gt, device, args, plot_dir, master_ckpt_path=None, run_id=None):
    results = {}
    
    print(f"\n{'='*60}")
    print(f"RUNNING {len(config.EXPERIMENTS)} EXPERIMENTS")
    print(f"{'='*60}")
    
    for exp in config.EXPERIMENTS:
        name = exp['name']
        solver_name = exp['solver']
        precond_type = exp.get('precond')
        precond_kwargs = exp.get('precond_kwargs', {}).copy() 
        
        print(f"\n--- {name} ---")
        
        try:
            solver_cls, cfg = get_solver_info(solver_name)
            precond = None

            if precond_type:
                precond = get_preconditioner(
                    precond_type=precond_type,
                    A=A,
                    A_csc=A_csc,
                    device=device,
                    args=args,
                    master_ckpt_path=master_ckpt_path,
                    solver_name=solver_name,
                    **precond_kwargs
                )
            
            solver = solver_cls()
            x0 = torch.zeros(A.shape[0], dtype=A.dtype, device=device)
            
            if solver_name in ('GMRES', 'FGMRES'):
                result = solver.solve(
                    A=A,
                    b=b,
                    M=precond,
                    x0=x0,
                    restart=config.RESTART,
                    rtol=config.TOLERANCE,
                    max_iters=config.MAX_ITERS,
                    progress_bar=True
                )
            else:
                result = solver.solve(
                    A=A,
                    b=b,
                    M=precond,
                    x0=x0,
                    rtol=config.TOLERANCE,
                    max_iters=config.MAX_ITERS,
                    progress_bar=True
                )
            
            sol, iters, hist_abs, hist_rel, hist_time, ortho_map = result
            final_res = torch.norm(A @ sol - b) / torch.norm(b)
            
            results[name] = {
                'res_history': hist_rel,
                'time_history': hist_time,
                'ortho_map': ortho_map,
                'final_res': final_res.item(),
                'iters': iters,
            }
            
            print(f"  Iterations: {results[name]['iters']}")
            print(f"  Final residual: {final_res:.2e}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {
                'res_history': [],
                'time_history': [],
                'ortho_map': None,
                'final_res': float('inf'),
                'iters': 0,
                'error': str(e)
            }
    
    plot_results(results, args, plot_dir, run_id)
    
    print(f"\n{'='*60}")
    print(f"{'Experiment':<30} {'Iters':>8} {'Final Res':>12}")
    print(f"{'='*60}")

    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<30} {'ERROR':>8} {res['error'][:12]:>12}")
        else:
            print(f"{name:<30} {res['iters']:>8} {res['final_res']:>12.2e}")
            
    print(f"{'='*60}")
    
    return results
