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

    def _matmul(self, A, x):
        # Check for any sparse layout (COO returns is_sparse=True, CSC/CSR need layout check)
        if A.is_sparse or A.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc):
            return torch.sparse.mm(A, x)
        else:
            return torch.matmul(A, x)

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

    def _compute_spectral_radius_loss(self, num_vectors=32, power_iters=10):
        """Estimate rho(I - M^{-1}A) via power iteration and minimise it.

        Only the **last** power iteration builds a computational graph;
        all preceding iterations run under ``torch.no_grad()`` so that
        VRAM usage equals a single forward pass regardless of
        ``power_iters``.

        Returns
        -------
        rho : torch.Tensor  (scalar, differentiable)
            Estimated spectral radius of the error propagation operator.
        """
        n = self.n
        # Sample random unit vectors: (n, num_vectors) - use A's dtype for consistency
        v = torch.randn(n, num_vectors, dtype=self.A_torch.dtype, device=self.device)
        v = v / torch.linalg.norm(v, dim=0, keepdim=True).clamp(min=1e-12)

        # --- Detached inner iterations (no autograd graph) ---
        with torch.no_grad():
            for _ in range(power_iters - 1):
                Av = self._matmul(self.A_torch, v)
                M_inv_Av = self._scale_equivariant_forward(Av)
                Ev = v - M_inv_Av
                norms = torch.linalg.norm(Ev, dim=0, keepdim=True).clamp(min=1e-12)
                v = Ev / norms

        # --- Final iteration WITH gradients ---
        Av = self._matmul(self.A_torch, v)
        M_inv_Av = self._scale_equivariant_forward(Av)
        Ev = v - M_inv_Av

        # Rayleigh-quotient estimate per probe vector:  rho_i = ||E v_i|| / ||v_i||
        # v_i are unit-norm so this simplifies to ||E v_i||
        rho_estimates = torch.linalg.norm(Ev, dim=0)  # (num_vectors,)
        rho = rho_estimates.mean()

        return rho

    def train_spectral(self, epochs, optimizer, scheduler=None,
                       checkpoint_path=None, num_vectors=32,
                       power_iters=10, steps_per_epoch=50,
                       progress_bar=True):
        """Train the preconditioner by minimising spectral radius rho(I - M^{-1}A).

        No harvested data is needed.  Random probe vectors are sampled
        each step and the spectral radius is estimated via power iteration.

        Returns
        -------
        hist_train_loss : list[float]
        best_loss : float
        best_epoch : int
        """
        self._prepare_A_torch()
        hist_train_loss = []
        best_loss = float('inf')
        best_epoch = -1

        print(f"Starting spectral radius training: {epochs} epochs, "
              f"{steps_per_epoch} steps/epoch, "
              f"{num_vectors} vectors, {power_iters} power iters")

        for epoch in range(epochs):
            self.net.train()
            epoch_losses = []
            pbar = tqdm(range(steps_per_epoch),
                        desc=f"Epoch {epoch+1}/{epochs}",
                        leave=False, disable=not progress_bar)

            for step in pbar:
                optimizer.zero_grad()
                rho = self._compute_spectral_radius_loss(
                    num_vectors=num_vectors,
                    power_iters=power_iters,
                )
                rho.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=config.GRAD_CLIP_NORM)
                optimizer.step()
                epoch_losses.append(rho.item())
                pbar.set_postfix({'rho': f'{rho.item():.4f}'})

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
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"rho: {avg_loss:.6f}{status}")

        print(f"\nBest spectral radius: {best_loss:.6f} at epoch {best_epoch}")

        return hist_train_loss, best_loss, best_epoch

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

    @torch.no_grad()
    def apply(self, r): 
        self.net.eval()
        r = r.to(self.dtype)
        r_in = r.view(-1, 1)
        z_out = self._scale_equivariant_forward(r_in)
        z = z_out.view(-1)
        z = z.double()
        
        return z