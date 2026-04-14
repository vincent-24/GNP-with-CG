import os
import math
import numpy as np
import scipy.sparse as sp
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from GNP import config

class OfflineDataset(Dataset):
    def __init__(self, dataset_path, A=None):
        super().__init__()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        data = torch.load(dataset_path, weights_only=False)

        if isinstance(data, dict):
            self.errors = data['e']
            self.metadata = data.get('metadata', {})
            self._has_legacy_residuals = 'r' in data

            if self._has_legacy_residuals:
                self._legacy_residuals = data['r']
        else:
            raise ValueError("Dataset must be a dict with 'e' key")
        
        if self.errors.dtype != torch.float64:
            self.errors = self.errors.to(torch.float64)

            if self._has_legacy_residuals:
                self._legacy_residuals = self._legacy_residuals.to(torch.float64)
        
        self.n_samples = self.errors.shape[0]
        self.n_dim = self.errors.shape[1]
        self.A_cpu = None

        if A is not None:
            self._set_matrix(A)
        elif not self._has_legacy_residuals:
            raise ValueError("Matrix A is required for datasets without pre-computed residuals")
        
        print(f"Loaded offline dataset: {self.n_samples} samples, dim={self.n_dim}")

        if self._has_legacy_residuals and A is None:
            print(f"  Mode: Legacy (using pre-stored residuals)")
        else:
            print(f"  Mode: On-the-fly residual computation (r = A @ e)")

        if self.metadata:
            print(f"  Problem: {self.metadata.get('problem', 'unknown')}")
            ratio = self.metadata.get('random_ratio')

            if ratio is not None:
                print(f"  Random ratio: {ratio:.2f}")
                
            print(f"  From {self.metadata.get('num_runs', 'unknown')} PCG runs, "
                  f"{self.metadata.get('total_samples', self.n_samples)} total samples")
    
    def _set_matrix(self, A):
        target_dtype = self.errors.dtype
        
        if sp.issparse(A):
            coo = A.tocoo()
            indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
            values = torch.from_numpy(coo.data).to(target_dtype)
            shape = torch.Size(coo.shape)
            self.A_cpu = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        elif torch.is_tensor(A):
            self.A_cpu = A.cpu().to(target_dtype)

            if A.is_sparse:
                self.A_cpu = self.A_cpu.coalesce()
        else:
            self.A_cpu = torch.tensor(A, dtype=target_dtype)
            
    def __len__(self): 
        return self.n_samples
    
    def __getitem__(self, idx):
        e = self.errors[idx]
        
        if self._has_legacy_residuals and self.A_cpu is None:
            r = self._legacy_residuals[idx]
        else:
            e_col = e.unsqueeze(1) 
            
            if self.A_cpu.is_sparse:
                r = torch.sparse.mm(self.A_cpu, e_col).squeeze(1)
            else:
                r = torch.mv(self.A_cpu, e)
        
        return r, e

class GNP():
    def __init__(self, A, net, device):
        self.A = A
        self.net = net
        self.device = device
        self.dtype = net.dtype
        self.n = A.shape[0]
        self.A_torch = None
        self._diag = None
        self._diag_iter = 0

    def _prepare_A_torch(self):
        if self.A_torch is not None:
            return

        if torch.is_tensor(self.A):
            # Convert CSC/CSR to COO for better CUDA compatibility
            if self.A.layout in (torch.sparse_csc, torch.sparse_csr):
                self.A_torch = self.A.to_sparse_coo().coalesce().to(torch.float64).to(self.device)
            else:
                self.A_torch = self.A.to(torch.float64).to(self.device)
        elif sp.issparse(self.A):
            coo = self.A.tocoo()
            indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
            values = torch.from_numpy(coo.data)
            shape = torch.Size(coo.shape)
            self.A_torch = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float64)
            self.A_torch = self.A_torch.to(self.device)
        else:
            self.A_torch = torch.tensor(self.A, dtype=torch.float64).to(self.device)

    def _scale_equivariant_forward(self, b):
        input_dtype = b.dtype
        norms = torch.linalg.norm(b, dim=0, keepdim=True)
        norms = norms.clamp(min=1e-12)
        scaling_factor = math.sqrt(self.n) / norms
        b_scaled = b * scaling_factor
        x_scaled = self.net(b_scaled)
        x = x_scaled / scaling_factor
        # Cast back to input dtype to avoid mismatches with A_torch (float64)

        return x.to(input_dtype)

    def _compute_supervised_loss(self, e_pred, e_true):
        e_pred = e_pred.to(torch.float64)
        e_true = e_true.to(torch.float64)
        loss = torch.nn.MSELoss()(e_pred, e_true)

        return loss

    def train_spectral(self, epochs, optimizer, scheduler=None, checkpoint_path=None, num_vectors=32,
        power_iters=10, steps_per_epoch=50, progress_bar=True):
        """Train the preconditioner by minimising a spectral loss.

        Loss type is selected by ``config.SPECTRAL_LOSS_TYPE``:
        - ``'rho'``:        minimise rho(I - M^{-1}A)    (Richardson convergence rate)
        - ``'kappa'``:      minimise log kappa(M^{-1}A)   (PCG convergence rate)
        - ``'hutchinson'``: minimise ||I - M^{-1}A||_F^2  (160x cheaper, all-eigenvalue signal)
        - ``'curriculum'``: hutchinson first, then warm-started kappa refinement

        Returns
            hist_train_loss : list[float]
            best_loss : float
            best_epoch : int
        """
        from GNP.nn.losses import (spectral_radius_loss, condition_number_loss,
                                   hutchinson_frobenius_loss, condition_number_loss_warm)

        self._prepare_A_torch()
        hist_train_loss = []
        best_loss = float('inf')
        best_epoch = -1

        loss_type = getattr(config, 'SPECTRAL_LOSS_TYPE', 'rho')
        M_inv = self._scale_equivariant_forward

        # Curriculum state: warm-start eigenvector buffers
        _v_warm, _w_warm = None, None
        total_steps = epochs * steps_per_epoch
        switch_step = int(getattr(config, 'CURRICULUM_SWITCH_FRAC', 0.8) * total_steps)
        warm_power_iters = getattr(config, 'CURRICULUM_WARM_POWER_ITERS', 3)
        warm_num_vectors = getattr(config, 'CURRICULUM_WARM_NUM_VECTORS', 16)

        print(f"Starting spectral training: {epochs} epochs, "
              f"{steps_per_epoch} steps/epoch, "
              f"{num_vectors} vectors, {power_iters} power iters, "
              f"loss={loss_type}")

        global_step = 0
        for epoch in range(epochs):
            self.net.train()
            epoch_losses = []
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", leave=False, disable=not progress_bar)

            for step in pbar:
                optimizer.zero_grad()

                # --- Select loss function based on type and curriculum phase ---
                if loss_type == 'hutchinson':
                    loss = hutchinson_frobenius_loss(self.A_torch, M_inv, self.n, num_vectors=num_vectors)
                    pbar.set_postfix({'||I-M⁻¹A||²': f'{loss.item():.4f}'})

                elif loss_type == 'curriculum':
                    if global_step < switch_step:
                        # Phase 1: cheap Hutchinson
                        loss = hutchinson_frobenius_loss(self.A_torch, M_inv, self.n, num_vectors=num_vectors)
                        pbar.set_postfix({'hutch': f'{loss.item():.4f}', 'phase': 'H'})
                    else:
                        # Phase 2: warm-started kappa refinement
                        loss, _v_warm, _w_warm = condition_number_loss_warm(
                            self.A_torch, M_inv, self.n,
                            num_vectors=warm_num_vectors,
                            power_iters=warm_power_iters,
                            v_init=_v_warm, w_init=_w_warm,
                        )
                        kval = math.exp(loss.item())
                        pbar.set_postfix({
                            'log(κ)': f'{loss.item():.2f}',
                            'κ': f'{kval:.1f}' if kval < 1e4 else f'{kval:.2e}',
                            'phase': 'K',
                        })

                elif loss_type == 'kappa':
                    loss = condition_number_loss(self.A_torch, M_inv, self.n,
                                                num_vectors=num_vectors, power_iters=power_iters)
                    kval = math.exp(loss.item())
                    pbar.set_postfix({
                        'log(κ)': f'{loss.item():.2f}',
                        'κ': f'{kval:.1f}' if kval < 1e4 else f'{kval:.2e}',
                    })

                else:  # 'rho'
                    loss = spectral_radius_loss(self.A_torch, M_inv, self.n,
                                               num_vectors=num_vectors, power_iters=power_iters)
                    pbar.set_postfix({'rho': f'{loss.item():.4f}'})

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=config.GRAD_CLIP_NORM)
                optimizer.step()
                epoch_losses.append(loss.item())
                global_step += 1

            if scheduler is not None:
                scheduler.step()

            avg_loss = np.mean(epoch_losses)
            hist_train_loss.append(avg_loss)
            improved = avg_loss < best_loss

            if improved:
                best_loss = avg_loss
                best_epoch = epoch + 1

                if checkpoint_path is not None:
                    torch.save(self.net.state_dict(), checkpoint_path)

            status = " (saved)" if improved else ""

            if loss_type in ('kappa', 'curriculum'):
                kval = math.exp(avg_loss) if avg_loss < 50 else float('inf')
                kstr = f'{kval:.1f}' if kval < 1e4 else f'{kval:.2e}'
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"loss: {avg_loss:.4f} | κ ≈ {kstr}{status}")
            else:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"loss: {avg_loss:.6f}{status}")

        print(f"\nBest loss: {best_loss:.6f} at epoch {best_epoch}")

        # --- Post-training diagnostic report ---
        self._log_training_diagnostics(optimizer, num_vectors, power_iters, best_loss, best_epoch, epochs)

        return hist_train_loss, best_loss, best_epoch

    def _log_training_diagnostics(self, optimizer, num_vectors, power_iters, best_loss, best_epoch, total_epochs):
        """Print diagnostic report after spectral training.

        Reports:
        1. Condition number of A (Lanczos estimate).
        2. ||N^T N(r)|| vs ||eps * D^{-1} r|| ratio (Jacobi floor dominance).
        3. Gradient norm of network parameters (vanishing gradient check).
        4. High-fidelity rho estimate (more power iterations + more probes).
        """
        from GNP.solver.Lanczos import Lanczos

        print(f"\n{'='*60}")
        print("POST-TRAINING DIAGNOSTICS")
        print(f"{'='*60}")
        n = self.n
        M_inv = self._scale_equivariant_forward

        # ------------------------------------------------------------------
        # 1. Condition number estimate via Lanczos on A
        # ------------------------------------------------------------------
        try:
            lanczos = Lanczos()
            m_lanc = min(80, n - 1)
            with torch.no_grad():
                _, barT = lanczos.build(self.A_torch, m=m_lanc)

            T_sq = barT[:m_lanc, :m_lanc]
            eigs_T = torch.linalg.eigvalsh(T_sq)
            eigs_pos = eigs_T[eigs_T > 1e-15]

            if len(eigs_pos) > 0:
                lam_min = eigs_pos.min().item()
                lam_max = eigs_pos.max().item()
                kappa = lam_max / lam_min

                print(f"\n[1] Condition number estimate (Lanczos, m={m_lanc}):")
                print(f"    lambda_min ≈ {lam_min:.6e}")
                print(f"    lambda_max ≈ {lam_max:.6e}")
                print(f"    kappa(A)   ≈ {kappa:.6e}")
            else:
                print(f"\n[1] Condition number: could not estimate (no positive Ritz values)")
        except Exception as e:
            print(f"\n[1] Condition number: FAILED ({e})")

        # ------------------------------------------------------------------
        # 2. Jacobi floor dominance: ||N^T N(r)|| vs ||eps * D^{-1} r||
        # ------------------------------------------------------------------
        try:
            self.net.eval()
            num_probes = min(32, n)
            r_probes = torch.randn(n, num_probes, dtype=self.A_torch.dtype, device=self.device)
            r_probes = r_probes / torch.linalg.norm(r_probes, dim=0, keepdim=True).clamp(min=1e-12)

            # Full forward gives NtN(r) + eps * D^{-1} r; compute each term
            # NtN(r) via autograd
            with torch.enable_grad():
                r_ad = r_probes.detach().requires_grad_(True)
                z = self.net._forward_raw(r_ad)
                NtN_r = torch.autograd.grad(outputs=z, inputs=r_ad, grad_outputs=z.detach(), create_graph=False)[0].detach()

            spd_eps = self.net.spd_eps
            D_inv = self.net._D_inv
            jacobi_r = spd_eps * D_inv.unsqueeze(1) * r_probes

            norm_NtN = torch.linalg.norm(NtN_r, dim=0)      # (num_probes,)
            norm_jacobi = torch.linalg.norm(jacobi_r, dim=0)  # (num_probes,)
            ratio = (norm_NtN / norm_jacobi.clamp(min=1e-15))

            print(f"\n[2] Jacobi floor dominance (eps={spd_eps:.1e}, {num_probes} random unit probes):")
            print(f"    ||N^T N(r)|| / ||eps D^{{-1}} r||:")
            print(f"      mean  = {ratio.mean().item():.6f}")
            print(f"      min   = {ratio.min().item():.6f}")
            print(f"      max   = {ratio.max().item():.6f}")

            if ratio.mean().item() < 1.0:
                print(f"    ** WARNING: Jacobi floor DOMINATES the learned component (ratio < 1).")
                print(f"       The network's contribution is smaller than the safety floor.")
                print(f"       Consider reducing SPD_JACOBI_EPS (currently {spd_eps:.1e}).")
            elif ratio.mean().item() < 10.0:
                print(f"    Note: Learned component is comparable to Jacobi floor.")
            else:
                print(f"    OK: Learned component dominates the Jacobi floor.")
        except Exception as e:
            print(f"\n[2] Jacobi floor dominance: FAILED ({e})")

        # ------------------------------------------------------------------
        # 3. Gradient norm check (one step with current params)
        # ------------------------------------------------------------------
        try:
            from GNP.nn.losses import spectral_radius_loss

            self.net.train()
            torch.set_grad_enabled(True)
            rho = spectral_radius_loss(self.A_torch, M_inv, n, num_vectors=num_vectors, power_iters=power_iters)
            rho.backward()

            total_norm = 0.0
            num_zero = 0
            num_params = 0

            for p in self.net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    num_params += 1

                    if p.grad.data.abs().max().item() < 1e-15:
                        num_zero += 1
                else:
                    num_zero += 1
                    num_params += 1

            total_norm = total_norm ** 0.5

            for p in self.net.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            print(f"\n[3] Gradient norm (single rho backward pass):")
            print(f"    Total gradient L2 norm: {total_norm:.6e}")
            print(f"    Parameter groups with grad: {num_params - num_zero}/{num_params}")

            if total_norm < 1e-8:
                print(f"    ** WARNING: Gradient near zero — network may be in a flat region.")
            elif total_norm > 1e3:
                print(f"    ** WARNING: Gradient very large — potential instability.")
            else:
                print(f"    OK: Gradient magnitude is in a reasonable range.")
        except Exception as e:
            print(f"\n[3] Gradient norm: FAILED ({e})")
        finally:
            torch.set_grad_enabled(True)
            self.net.eval()

        # ------------------------------------------------------------------
        # 4. High-fidelity rho estimate (more probes, more power iters)
        # ------------------------------------------------------------------
        try:
            from GNP.nn.losses import spectral_radius_loss

            hifi_probes = min(64, n)
            hifi_power = max(power_iters * 3, 30)
            self.net.eval()
            with torch.no_grad():
                rho_hifi = spectral_radius_loss(self.A_torch, M_inv, n, num_vectors=hifi_probes, power_iters=hifi_power)

            print(f"\n[4] High-fidelity rho estimate ({hifi_probes} probes, {hifi_power} power iters):")
            rho_val = rho_hifi.item()
            print(f"    rho(I - M^{{-1}}A) = {rho_val:.6f}")

            if rho_val >= 1.0:
                print(f"    ** WARNING: rho >= 1.0 — preconditioner DIVERGES.")
            else:
                iters_per_decade = math.log(10) / math.log(1.0 / rho_val)
                print(f"    Estimated PCG iterations per decade of residual reduction: {iters_per_decade:.0f}")
                print(f"    Estimated iterations for 1e-10 tolerance: ~{int(10 * iters_per_decade)}")
        except Exception as e:
            print(f"\n[4] High-fidelity rho: FAILED ({e})")

        print(f"\n{'='*60}")
        print(f"END DIAGNOSTICS")
        print(f"{'='*60}\n")

    def train(self, train_loader, val_loader, epochs, optimizer, scheduler=None, checkpoint_path=None, progress_bar=True):
        self._prepare_A_torch()
        hist_train_loss = []
        hist_val_loss = []
        best_val_loss = float('inf')
        best_epoch = -1
        step_losses = []      
        val_steps = []        
        val_losses_step = []  
        global_step = 0
        batches_per_epoch = len(train_loader)
        
        print(f"Starting training: {epochs} epochs, {batches_per_epoch} batches/epoch")
        
        train_miniters = max(1, batches_per_epoch // 100)
        
        for epoch in range(epochs):
            self.net.train()
            epoch_train_losses = []
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, disable=not progress_bar, miniters=train_miniters, mininterval=0)
            
            for batch_idx, (r_batch, e_batch) in enumerate(train_pbar):
                r_batch = r_batch.to(self.device).to(self.dtype)
                e_batch = e_batch.to(self.device).to(self.dtype)
                r_in = r_batch.T
                e_true = e_batch.T
                
                optimizer.zero_grad()
                e_pred = self._scale_equivariant_forward(r_in)
                loss = self._compute_supervised_loss(e_pred, e_true)
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                epoch_train_losses.append(loss_val)
                step_losses.append(loss_val)     
                global_step += 1

                if batch_idx % train_miniters == 0 or batch_idx == len(train_loader) - 1:
                    train_pbar.set_postfix({'loss': f'{loss_val:.2e}'})
            
            avg_train_loss = np.mean(epoch_train_losses)
            hist_train_loss.append(avg_train_loss)
            
            if scheduler is not None:
                scheduler.step()
            
            self.net.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for r_batch, e_batch in val_loader:
                    r_batch = r_batch.to(self.device).to(self.dtype)
                    r_in = r_batch.T
                    e_batch = e_batch.to(self.device).to(self.dtype)
                    e_true = e_batch.T
                    e_pred = self._scale_equivariant_forward(r_in)
                    loss = self._compute_supervised_loss(e_pred, e_true)
                    epoch_val_losses.append(loss.item())
            
            avg_val_loss = np.mean(epoch_val_losses)
            hist_val_loss.append(avg_val_loss)
            val_steps.append(global_step)
            val_losses_step.append(avg_val_loss)
            improved = avg_val_loss < best_val_loss

            if improved:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                
                if checkpoint_path is not None:
                    torch.save(self.net.state_dict(), checkpoint_path)
            
            status = " (saved)" if improved else ""
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.4e} | "
                  f"Val: {avg_val_loss:.4e} {status}")
        
        print(f"\nBest validation loss: {best_val_loss:.4e} at epoch {best_epoch}")

        step_data = {
            'step_losses': step_losses,
            'val_steps': val_steps,
            'val_losses': val_losses_step,
            'batches_per_epoch': batches_per_epoch,
        }
        
        return hist_train_loss, hist_val_loss, best_val_loss, best_epoch, step_data

    def _set_diagnostics(self, diag):
        self._diag = diag
        self._diag_iter = 0

    @torch.no_grad()
    def _apply_raw(self, r):
        """Apply preconditioner without diagnostics (used for symmetry probes)."""
        self.net.eval()
        r = r.to(self.dtype)
        r_in = r.view(-1, 1)
        z_out = self._scale_equivariant_forward(r_in)
        return z_out.view(-1).double()

    @torch.no_grad()
    def apply(self, r):
        self.net.eval()
        r = r.to(self.dtype)
        r_in = r.view(-1, 1)
        z_out = self._scale_equivariant_forward(r_in)
        z = z_out.view(-1).double()

        if self._diag is not None and self._diag.level >= 3:
            self._record_precond_diagnostics(r, z)

        return z

    def _record_precond_diagnostics(self, r_orig, z):
        from GNP import config

        r_in = r_orig.view(-1, 1).to(self.dtype)

        # Apply the same scaling as _scale_equivariant_forward so that
        # the decomposition matches the actual preconditioner output.
        norms = torch.linalg.norm(r_in, dim=0, keepdim=True).clamp(min=1e-12)
        scale = math.sqrt(self.n) / norms
        r_scaled = r_in * scale

        with torch.enable_grad():
            r_ad = r_scaled.detach().requires_grad_(True)
            Nr_scaled = self.net._forward_raw(r_ad)
            NtNr_scaled = torch.autograd.grad(
                outputs=Nr_scaled, inputs=r_ad,
                grad_outputs=Nr_scaled.detach(),
                create_graph=False,
            )[0].detach()

        # Undo scaling: actual NtN contribution = NtNr_scaled / scale
        NtNr = NtNr_scaled / scale
        Nr = Nr_scaled / scale
        jacobi_r = self.net.spd_eps * self.net._D_inv.unsqueeze(1) * r_in

        r_norm = torch.linalg.norm(r_orig).item()
        Nr_norm = torch.linalg.norm(Nr).item()
        NtNr_norm = torch.linalg.norm(NtNr).item()
        jacobi_norm = torch.linalg.norm(jacobi_r).item()
        rTMr = torch.dot(r_orig.view(-1).double(), z.view(-1).double()).item()

        symmetry_err = None
        if self._diag_iter % config.PCG_DIAG_SYMMETRY_PERIOD == 0:
            r2 = torch.randn_like(r_orig)
            z2 = self._apply_raw(r2)
            sym1 = torch.dot(r_orig.view(-1).double(), z2.view(-1).double()).item()
            sym2 = torch.dot(r2.view(-1).double(), z.view(-1).double()).item()
            symmetry_err = abs(sym1 - sym2) / (abs(sym1) + abs(sym2) + 1e-30)

        self._diag.record_precond(self._diag_iter, r_norm, Nr_norm, NtNr_norm, jacobi_norm, rTMr, symmetry_err)
        self._diag_iter += 1