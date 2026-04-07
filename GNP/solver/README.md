# Solver Module

Iterative solvers for sparse linear systems `Ax = b`, with built-in diagnostics for neural preconditioners.

## Solvers

| File | Class | Purpose |
|------|-------|---------|
| `base.py` | `IterativeSolver` | Base class — residual history, timing, A-orthogonality tracking |
| `PCG.py` | `PCG` | Preconditioned Conjugate Gradient (SPD systems) |
| `GMRES.py` | `GMRES` | Flexible GMRES with restart (general systems) |
| `Lanczos.py` | `Lanczos` | Lanczos iteration for eigenvalue estimation |
| `diagnostics.py` | `PCGDiagnostics` | Per-iteration diagnostic collector for PCG |

## PCG Diagnostics

Controlled by `config.PCG_DIAGNOSTICS` (0–3). Diagnostics run **during evaluation only** — training is unaffected.

### Levels

| Level | What it records | Overhead |
|-------|----------------|----------|
| 0 | Nothing (default) | Zero |
| 1 | Termination reason, total iterations, final residual | Negligible |
| 2 | Level 1 + per-iteration PCG scalars | ~microseconds/iter |
| 3 | Level 2 + preconditioner decomposition | ~2x preconditioner cost/iter |

Each level is cumulative (level 3 includes everything from 1 and 2).

### Termination Reasons (Level 1+)

PCG can stop for four reasons:

| Reason | Meaning |
|--------|---------|
| `converged` | Relative residual dropped below `rtol` — normal, successful exit |
| `max_iters` | Hit the iteration limit without converging |
| `breakdown_dAq` | `d^T A d ≈ 0` — the search direction became A-conjugate to itself, making the step size undefined. Indicates loss of A-conjugacy, often caused by a preconditioner that is not truly SPD |
| `breakdown_delta` | `r^T M^{-1} r ≈ 0` — the preconditioned residual inner product collapsed. The preconditioner mapped the residual to near-zero or to a direction nearly orthogonal to it |

When a breakdown occurs, the iteration data for the breakdown step is still recorded (with `NaN` for quantities that could not be computed), so the scalar values leading to the collapse are visible in the log.

### Per-Iteration PCG Scalars (Level 2+)

Recorded each iteration and written to the diagnostic log file. Values that could not be computed (e.g., beta when PCG converges before the conjugation step) are recorded as `NaN`.

| Column | Formula | What it tells you |
|--------|---------|-------------------|
| `alpha` | `delta_new / (d^T A d)` | Step size along search direction. Negative alpha means the preconditioner is not SPD (d^T A d should always be positive for SPD A with A-conjugate directions) |
| `beta` | `delta_new / delta_old` | Conjugation coefficient. Determines how much of the previous direction to mix in. Negative beta also signals non-SPD preconditioner behavior |
| `dAq` | `d^T A d` | Denominator of alpha. Must be positive for SPD A with valid directions. Values collapsing toward zero signal impending `breakdown_dAq` |
| `delta_new` | `r^T M^{-1} r` | Preconditioned residual inner product. For an SPD preconditioner this must be positive. Tracks how much "energy" the preconditioned residual has |
| `delta_old` | Previous iteration's `delta_new` | Denominator of beta. Collapse toward zero signals impending `breakdown_delta` |
| `rel_res` | `\|\|r\|\| / \|\|b\|\|` | Relative residual norm. This is the convergence metric — it should decrease monotonically toward `rtol` |
| `drop` | `prev_abs_res / abs_res` | Residual reduction ratio. Values > 1 mean the residual decreased. Values near 1.0 for many consecutive iterations indicate **stagnation** — the solver is making no progress |

**What to look for:**
- `alpha < 0` at any iteration: the preconditioner is not behaving as SPD
- `dAq` or `delta_old` trending toward zero: breakdown is coming
- `drop ≈ 1.0` for many iterations: stagnation, the preconditioner is not helping
- `rel_res` plateauing far above `rtol`: the preconditioner quality caps the achievable accuracy
- `NaN` in a row: that iteration hit a breakdown or early convergence before the quantity was computed

### Preconditioner Decomposition (Level 3)

The neural preconditioner computes `M^{-1}(r) = N^T N(r) + eps * D^{-1} * r`, where:
- **N^T N(r)**: the learned component (neural network, made PSD via autograd adjoint)
- **eps * D^{-1} * r**: the Jacobi floor (safety net ensuring strict positive definiteness)

Level 3 decomposes these terms at every PCG iteration. The decomposition accounts for the scale-equivariant wrapping applied by the preconditioner (`r -> s*r` where `s = sqrt(n)/||r||`), so the reported norms match the actual preconditioner output.

| Column | What it measures |
|--------|-----------------|
| `\|\|r\|\|` | Input residual norm — the "problem" handed to the preconditioner |
| `\|\|Nr\|\|` | Output of the raw network N(r) before SPD enforcement |
| `\|\|NtNr\|\|` | Norm of the learned SPD component N^T N(r) |
| `\|\|Jac\|\|` | Norm of the Jacobi floor `eps * D^{-1} * r` |
| `ratio` | `\|\|NtNr\|\| / \|\|Jac\|\|` — is the network or the safety floor doing the work? |
| `r^T Mr` | `r^T M^{-1}(r)` — must be > 0 for the preconditioner to be SPD. A non-positive value is a definitive SPD violation |
| `sym_err` | Relative symmetry error: `\|r1^T M^{-1} r2 - r2^T M^{-1} r1\| / (\|r1^T M^{-1} r2\| + \|r2^T M^{-1} r1\|)`. Measured every `PCG_DIAG_SYMMETRY_PERIOD` iterations (default 10). Should be near machine epsilon (~1e-15) for a truly symmetric operator |

**What to look for:**
- `ratio < 1`: the Jacobi floor dominates — the network learned nothing useful for this residual
- `ratio < 1` at ALL iterations: training failed entirely; the network contribution is weaker than a trivially scaled diagonal preconditioner
- `r^T Mr <= 0`: SPD violation — this should never happen with the N^T N + eps*D^{-1} construction. If it does, there's a numerical issue
- `sym_err >> 1e-12`: the preconditioner is not symmetric — could cause PCG to behave unpredictably (PCG assumes M^{-1} is SPD)
- `||Nr||` collapsing to zero over iterations: the network is dying / outputting near-zero corrections

### Output

Diagnostics are written to:
- **stdout**: one-line summary with termination reason and warnings
- **log file**: `diag_<experiment_name>.txt` in the experiment's plot directory, containing full columnar tables

### Quick Reference: Post-Training Diagnostics

These are separate from the PCG diagnostics above. They run once after `train_spectral()` finishes (in `GNP/precond/GNP.py`) and appear in the training log:

| Diagnostic | What it answers |
|-----------|-----------------|
| **[1] kappa(A) via Lanczos** | How ill-conditioned is the original system? Higher = harder to solve |
| **[2] Jacobi floor dominance** | Did the network learn anything stronger than the safety floor? ratio >> 1 is good, < 1 is bad |
| **[3] Gradient norm** | Is the optimizer stuck (grad ≈ 0) or unstable (grad >> 1e3)? Moderate values mean more training could help |
| **[4] High-fidelity rho** | What is the spectral radius rho(I - M^{-1}A)? This gives a conservative (worst-case) convergence rate. rho < 1 required for convergence; closer to 0 is better |

### Configuration

```python
# In GNP/config.py
PCG_DIAGNOSTICS = 0            # 0=off, 1=termination, 2=+scalars, 3=+preconditioner
PCG_DIAG_SYMMETRY_PERIOD = 10  # Symmetry check frequency at level 3
```
