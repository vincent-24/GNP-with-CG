import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
import scipy.sparse as sp
import math
from tqdm import tqdm

class OfflineDataset(Dataset):
    """
    Dataset for offline training with pre-harvested error vectors.
    
    Storage Optimization: Only error vectors (e) are stored on disk.
    Residuals (r) are computed on-the-fly via r = A @ e during training.
    
    Args:
        dataset_path: Path to .pt file containing {'e': tensor, 'metadata': dict}
        A: System matrix (scipy sparse or torch tensor). Kept on CPU for 
           multi-process DataLoader compatibility.
    """
    def __init__(self, dataset_path, A=None):
        super().__init__()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        data = torch.load(dataset_path, weights_only=False)
        if isinstance(data, dict):
            # Support both legacy format (with 'r') and new format (without 'r')
            self.errors = data['e']
            self.metadata = data.get('metadata', {})
            self._has_legacy_residuals = 'r' in data
            if self._has_legacy_residuals:
                self._legacy_residuals = data['r']
        else:
            raise ValueError("Dataset must be a dict with 'e' key")
        
        self.n_samples = self.errors.shape[0]
        self.n_dim = self.errors.shape[1]
        
        # Store A on CPU for multi-process data loading safety
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
            print(f"  From {self.metadata.get('num_runs', 'unknown')} PCG runs")
    
    def _set_matrix(self, A):
        """Convert and store A as a CPU torch sparse tensor matching errors dtype."""
        # Get dtype from errors tensor (typically float64 from harvesting)
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
        
        # Use legacy residuals if available and no A provided
        if self._has_legacy_residuals and self.A_cpu is None:
            r = self._legacy_residuals[idx]
        else:
            # Compute r = A @ e on-the-fly (on CPU)
            e_col = e.unsqueeze(1)  # (n,) -> (n, 1)
            if self.A_cpu.is_sparse:
                r = torch.sparse.mm(self.A_cpu, e_col).squeeze(1)
            else:
                r = torch.mv(self.A_cpu, e)
        
        return r, e

class GNP():
    """
    Graph Neural Preconditioner.
    
    Wraps a GNN that learns to approximate A^{-1} by minimizing
    the physics residual loss: ||r - A @ e_pred||
    """
    def __init__(self, A, training_data, m, net, device, use_lanczos=False):
        self.A = A
        self.training_data = training_data
        self.m = m
        self.net = net
        self.device = device
        self.dtype = net.dtype
        self.use_lanczos = use_lanczos
        self.n = A.shape[0]
        self.A_torch = None
        self.loss_params = torch.nn.Parameter(torch.zeros(2, device=device))

    def _prepare_A_torch(self):
        """Convert matrix A to torch tensor on device (lazy initialization)."""
        if self.A_torch is not None:
            return

        if torch.is_tensor(self.A):
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
        """Adaptive matrix-vector multiplication (handles sparse and dense)."""
        if A.is_sparse:
            return torch.sparse.mm(A, x)
        else:
            return torch.matmul(A, x)

    def _scale_equivariant_forward(self, b):
        """Scale-equivariant forward pass through the network."""
        norms = torch.linalg.norm(b, dim=0, keepdim=True)
        norms = norms.clamp(min=1e-12)
        scaling_factor = math.sqrt(self.n) / norms
        b_scaled = b * scaling_factor
        x_scaled = self.net(b_scaled)
        x = x_scaled / scaling_factor
        return x

    # loss 1. ||r - A @ e_pred|| 
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        loss = torch.mean(diff_norm_sq)
        return loss
    '''

    # loss 2. ||r - Ae||^2 / ||b||
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        ref_norm = torch.norm(r_in, dim=0)
        ref_norm = torch.where(ref_norm < 1e-12, torch.ones_like(ref_norm), ref_norm)
        loss = torch.mean(diff_norm_sq / ref_norm)
        return loss
    '''

    # loss 2.4u. loss 2 unnormalized with soft stability (lambda = 1e-4)
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        ref_norm = torch.norm(r_in, dim=0)
        ref_norm = torch.where(ref_norm < 1e-12, torch.ones_like(ref_norm), ref_norm)
        term_error = torch.mean(diff_norm_sq / ref_norm)
        term_reg = torch.mean(torch.norm(e_pred, dim=0) ** 2)
        loss = term_error + (1e-4 * term_reg)
        return loss
    '''

    # loss 2.4n. loss 2 normalized Tikhonov 
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        numerator = diff_norm_sq + (1e-3 * e_norm_sq)
        ref_norm = torch.norm(r_in, dim=0)
        ref_norm = torch.where(ref_norm < 1e-12, torch.ones_like(ref_norm), ref_norm)
        loss = torch.mean(numerator / ref_norm)
        return loss
    '''

    # loss 3. ||r - Ae||^2 / ||A||_F
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        # For sparse tensors (COO or CSR), compute Frobenius norm from values
        if self.A_torch.is_sparse or self.A_torch.layout in (torch.sparse_csr, torch.sparse_csc):
            A_norm = torch.norm(self.A_torch.values())
        else:
            A_norm = torch.norm(self.A_torch)
        loss = torch.mean(diff_norm_sq) / (A_norm + 1e-12)
        return loss
    '''

    # loss 4. ||r - Ae||^2 + ||e||^2
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        loss = torch.mean(diff_norm_sq + e_norm_sq)
        return loss
    '''

    # loss 5. ||r - Ae||^2 + ||Ae||^2
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        recon_norm_sq = torch.norm(Ae, dim=0) ** 2
        loss = torch.mean(diff_norm_sq + recon_norm_sq)
        return loss
    '''

    # loss 6. ||Ae||^2 / ||b||^2
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)
        recon_norm_sq = torch.norm(Ae, dim=0) ** 2
        ref_norm_sq = torch.norm(r_in, dim=0) ** 2
        ref_norm_sq = torch.where(ref_norm_sq < 1e-12, torch.ones_like(ref_norm_sq), ref_norm_sq)
        loss = torch.mean(recon_norm_sq / ref_norm_sq)
        return loss
    '''

    # log 1
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)

        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        
        numerator = diff_norm_sq + (1e-4 * e_norm_sq)
        
        ref_norm_sq = torch.norm(r_in, dim=0) ** 2
        ref_norm_sq = torch.where(ref_norm_sq < 1e-12, torch.ones_like(ref_norm_sq), ref_norm_sq)
        
        ratio = numerator / ref_norm_sq
        
        epsilon = 1e-16 
        loss = torch.mean(torch.log(ratio + epsilon))
        
        return loss
    '''

    # log 2
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)

        ref_norm_sq = torch.norm(r_in, dim=0) ** 2
        safe_ref = torch.where(ref_norm_sq < 1e-12, torch.ones_like(ref_norm_sq), ref_norm_sq)

        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        ratio_res = diff_norm_sq / safe_ref
        loss_res = torch.mean(torch.log(ratio_res + 1e-16))

        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        ratio_reg = e_norm_sq / safe_ref
        loss_reg = torch.mean(ratio_reg)
        
        loss = loss_res + (1e-4 * loss_reg)
        
        return loss
    '''

    # log 3
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)

        ref_norm_sq = torch.norm(r_in, dim=0) ** 2
        safe_ref = torch.where(ref_norm_sq < 1e-12, torch.ones_like(ref_norm_sq), ref_norm_sq)

        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        ratio_res = diff_norm_sq / safe_ref
        
        loss_res = torch.mean(torch.log(ratio_res + 1e-16))
        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        ratio_reg = e_norm_sq / safe_ref
        loss_reg = torch.mean(torch.log1p(ratio_reg))
        
        loss = loss_res + (0.1 * loss_reg)
        
        return loss
    '''

    # log 4
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)

        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        Ae_norm_sq = torch.norm(Ae, dim=0) ** 2
        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        numerator = diff_norm_sq + (1e-5 * Ae_norm_sq) + (1e-9 * e_norm_sq)
        ref_norm_sq = torch.norm(r_in, dim=0) ** 2
        safe_ref = torch.where(ref_norm_sq < 1e-12, torch.ones_like(ref_norm_sq), ref_norm_sq)
        ratio = numerator / safe_ref
        loss = torch.mean(torch.log(ratio + 1e-16))
        
        return loss
    '''

    # log 5
    '''
    def _compute_physics_loss(self, r_in, e_pred):
        e_pred_double = e_pred.to(torch.float64)
        Ae = self._matmul(self.A_torch, e_pred_double)
        Ae = Ae.to(self.dtype)

        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        numerator = diff_norm_sq + (1e-4 * e_norm_sq)

        ref_norm_sq = torch.norm(r_in, dim=0) ** 2
        safe_ref = torch.where(ref_norm_sq < 1e-12, torch.ones_like(ref_norm_sq), ref_norm_sq)
        
        diff_norm_sq = torch.norm(r_in - Ae, dim=0) ** 2
        e_norm_sq = torch.norm(e_pred, dim=0) ** 2
        ratio_coupled = (diff_norm_sq + 1e-4 * e_norm_sq) / safe_ref
        loss_1 = torch.mean(torch.log(ratio_coupled + 1e-16))

        ratio_pure = diff_norm_sq / safe_ref
        loss_2 = torch.mean(torch.log(ratio_pure + 1e-16))
        
        s1 = self.loss_params[0]
        s2 = self.loss_params[1]
        
        weighted_loss = (torch.exp(-s1) * loss_1 + s1) + \
                        (torch.exp(-s2) * loss_2 + s2)
        
        return weighted_loss
    '''

    def _compute_supervised_loss(self, e_pred, e_true):
        # Cast to float64 for precision
        e_pred = e_pred.to(torch.float64)
        e_true = e_true.to(torch.float64)
        
        # Standard MSE: L = || e_pred - e_true ||^2
        # We assume e_true is the exact vector needed to solve Ax=b
        loss = torch.nn.MSELoss()(e_pred, e_true)
        return loss

    def train(self, train_loader, val_loader, epochs, optimizer, scheduler=None, 
              checkpoint_path=None, progress_bar=True):
        """
        Train the neural preconditioner with proper epoch-based loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of full passes through training data
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            checkpoint_path: Path to save best model checkpoint
            progress_bar: Show progress bar
            
        Returns:
            hist_train_loss: List of average training loss per epoch
            hist_val_loss: List of average validation loss per epoch
            best_val_loss: Best validation loss achieved
            best_epoch: Epoch with best validation loss
        """
        self._prepare_A_torch()
        
        hist_train_loss = []
        hist_val_loss = []
        best_val_loss = float('inf')
        best_epoch = -1
        
        print(f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch")
        
        train_miniters = max(1, len(train_loader) // 100)
        
        # Epoch loop
        for epoch in range(epochs):
            # ==================== TRAINING ====================
            self.net.train()
            epoch_train_losses = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                              leave=False, disable=not progress_bar, 
                              miniters=train_miniters, mininterval=0)
            
            for batch_idx, (r_batch, e_batch) in enumerate(train_pbar):
                r_batch = r_batch.to(self.device).to(self.dtype)
                e_batch = e_batch.to(self.device).to(self.dtype)
                r_in = r_batch.T
                e_true = e_batch.T
                
                # Forward pass
                optimizer.zero_grad()
                e_pred = self._scale_equivariant_forward(r_in)
                
                # Physics loss
                # loss = self._compute_physics_loss(r_in, e_pred)
                loss = self._compute_supervised_loss(e_pred, e_true)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
                # Only update postfix at miniters intervals to reduce log verbosity
                if batch_idx % train_miniters == 0 or batch_idx == len(train_loader) - 1:
                    train_pbar.set_postfix({'loss': f'{loss.item():.2e}'})
            
            avg_train_loss = np.mean(epoch_train_losses)
            hist_train_loss.append(avg_train_loss)
            
            if scheduler is not None:
                scheduler.step()
            
            # ==================== VALIDATION ====================
            self.net.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for r_batch, e_batch in val_loader:
                    r_batch = r_batch.to(self.device).to(self.dtype)
                    r_in = r_batch.T
                    e_batch = e_batch.to(self.device).to(self.dtype)
                    e_true = e_batch.T
                    
                    e_pred = self._scale_equivariant_forward(r_in)
                    # loss = self._compute_physics_loss(r_in, e_pred)
                    loss = self._compute_supervised_loss(e_pred, e_true)
                    
                    epoch_val_losses.append(loss.item())
            
            avg_val_loss = np.mean(epoch_val_losses)
            hist_val_loss.append(avg_val_loss)
            
            # ==================== CHECKPOINTING ====================
            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                if checkpoint_path is not None:
                    torch.save(self.net.state_dict(), checkpoint_path)
            
            # ==================== LOGGING ====================
            status = " (saved)" if improved else ""
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.4e} | "
                  f"Val: {avg_val_loss:.4e} {status}")
        
        print(f"\nBest validation loss: {best_val_loss:.4e} at epoch {best_epoch}")
        
        return hist_train_loss, hist_val_loss, best_val_loss, best_epoch

    @torch.no_grad()
    def apply(self, r): 
        """Apply the trained preconditioner to a residual vector."""
        self.net.eval()
        r = r.to(self.dtype)
        r_in = r.view(-1, 1)
        z_out = self._scale_equivariant_forward(r_in)
        z = z_out.view(-1)
        z = z.double()
        return z