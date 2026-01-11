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
    """Dataset for offline training with pre-harvested (residual, error) pairs."""
    def __init__(self, dataset_path):
        super().__init__()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
        if isinstance(data, dict):
            self.residuals = data['r']
            self.errors = data['e']
            self.metadata = data.get('metadata', {})
        else:
            raise ValueError("Dataset must be a dict with 'r' and 'e' keys")
        self.n_samples = self.residuals.shape[0]
        self.n_dim = self.residuals.shape[1]
        print(f"Loaded offline dataset: {self.n_samples} samples, dim={self.n_dim}")
        if self.metadata:
            print(f"  Problem: {self.metadata.get('problem', 'unknown')}")
            print(f"  From {self.metadata.get('num_runs', 'unknown')} PCG runs")
            
    def __len__(self): 
        return self.n_samples
    
    def __getitem__(self, idx): 
        return self.residuals[idx], self.errors[idx]


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

    def _compute_physics_loss(self, r_in, e_pred):
        """
        Compute physics-based loss: ||r - A @ e_pred|| / ||r||
        
        Args:
            r_in: Input residuals (n, batch)
            e_pred: Predicted errors (n, batch)
            
        Returns:
            Scalar loss value
        """
        e_pred_double = e_pred.to(torch.float64)
        r_recon = self._matmul(self.A_torch, e_pred_double)
        r_recon = r_recon.to(self.dtype)
        
        diff_norm = torch.norm(r_in - r_recon, dim=0)
        ref_norm = torch.norm(r_in, dim=0)
        ref_norm = torch.where(ref_norm < 1e-12, torch.ones_like(ref_norm), ref_norm)
        
        loss = torch.mean((diff_norm / ref_norm) ** 2)
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
        
        # Epoch loop
        for epoch in range(epochs):
            # ==================== TRAINING ====================
            self.net.train()
            epoch_train_losses = []
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                              leave=False, disable=not progress_bar)
            
            for batch_idx, (r_batch, e_batch) in enumerate(train_pbar):
                # Move to device and transpose: (batch, n) -> (n, batch)
                r_batch = r_batch.to(self.device).to(self.dtype)
                r_in = r_batch.T
                
                # Forward pass
                optimizer.zero_grad()
                e_pred = self._scale_equivariant_forward(r_in)
                
                # Physics loss
                loss = self._compute_physics_loss(r_in, e_pred)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
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
                    
                    e_pred = self._scale_equivariant_forward(r_in)
                    loss = self._compute_physics_loss(r_in, e_pred)
                    
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
            status = "✓ (saved)" if improved else ""
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