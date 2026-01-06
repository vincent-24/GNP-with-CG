import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
import math
from tqdm import tqdm

from GNP.solver import Arnoldi, Lanczos

class OfflineDataset(Dataset):
    """
    Offline dataset for GNP training using harvested PCG trajectories.
    
    Loads (residual, error) pairs from a pre-generated .pt file.
    The GNN learns to predict the error from the residual: r_i -> e_i.
    
    Args:
        dataset_path: Path to the .pt file containing {'r': residuals, 'e': errors}
    """
    def __init__(self, dataset_path):
        super().__init__()
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        data = torch.load(dataset_path, weights_only=False)
        
        # Support both flat format and format with metadata
        if isinstance(data, dict):
            self.residuals = data['r']  # Shape: (N_samples, n)
            self.errors = data['e']      # Shape: (N_samples, n)
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
        """Returns (residual, error) pair."""
        return self.residuals[idx], self.errors[idx]

class StreamingDataset(IterableDataset):
    def __init__(self, n, batch_size, training_data, m, Q=None, evals=None):
        """
        Streaming dataset for GNP training.
        
        Args:
            n: Matrix dimension
            batch_size: Batch size for training
            training_data: Type of training data ('x_normal', 'x_subspace', 'x_mix')
            m: Krylov subspace dimension
            Q: Pre-computed subspace matrix (required for 'x_subspace' and 'x_mix')
            evals: Pre-computed eigenvalues (required for 'x_subspace' and 'x_mix')
        """
        super().__init__()
        self.n = n
        self.m = m
        self.batch_size = batch_size
        self.training_data = training_data

        if training_data == 'x_subspace' or training_data == 'x_mix':
            if Q is None or evals is None:
                raise ValueError(
                    f"Q and evals must be provided for training_data='{training_data}'. "
                    "Pre-compute them using Lanczos/Arnoldi before creating the dataset."
                )
            self.Q = Q.to('cpu')
            self.evals = evals.to('cpu')

    def generate(self):
        while True:
            if self.training_data == 'x_normal':
                x = torch.normal(0, 1, size=(self.n, self.batch_size), dtype=torch.float64)
                yield x 
            elif self.training_data == 'x_subspace':
                e = torch.normal(0, 1, size=(self.m, self.batch_size), dtype=torch.float64)
                safe_evals = self.evals.view(-1, 1).clamp(min=1e-6)
                weighted_e = e / safe_evals
                x = self.Q @ weighted_e
                yield x
            elif self.training_data == 'x_mix':
                batch_size1 = self.batch_size // 2
                e = torch.normal(0, 1, size=(self.m, batch_size1), dtype=torch.float64)
                safe_evals = self.evals.view(-1, 1).clamp(min=1e-6)
                x1 = self.Q @ (e / safe_evals)
                batch_size2 = self.batch_size - batch_size1
                x2 = torch.normal(0, 1, size=(self.n, batch_size2), dtype=torch.float64)
                x = torch.cat([x1, x2], dim=1)
                yield x
            else: 
                b = torch.normal(0, 1, size=(self.n, self.batch_size), dtype=torch.float64)
                yield b
            
    def __iter__(self):
        return iter(self.generate())

class GNP():
    def __init__(self, A, training_data, m, net, device, use_lanczos=False):
        self.A = A
        self.training_data = training_data
        self.m = m
        self.net = net
        self.device = device
        self.dtype = net.dtype
        self.use_lanczos = use_lanczos
        self.n = A.shape[0]

    def _scale_equivariant_forward(self, b):
        norms = torch.linalg.norm(b, dim=0, keepdim=True)
        norms = norms.clamp(min=1e-12)
        scaling_factor = math.sqrt(self.n) / norms
        b_scaled = b * scaling_factor
        x_scaled = self.net(b_scaled)
        x = x_scaled / scaling_factor
        
        return x

    def train(self, batch_size, grad_accu_steps, epochs, optimizer, scheduler=None, num_workers=0, checkpoint_prefix_with_path=None, progress_bar=True, dataset_path=None):
        """
        Train the GNP neural preconditioner.
        
        Args:
            batch_size: Number of samples per batch
            grad_accu_steps: Gradient accumulation steps
            epochs: Number of training epochs
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            num_workers: DataLoader workers
            checkpoint_prefix_with_path: Path prefix for saving checkpoints
            progress_bar: Show training progress bar
            dataset_path: Path to offline dataset (.pt file). If provided, uses
                         supervised training on harvested PCG trajectories.
                         If None, uses streaming unsupervised training (default).
        """
        self.net.train()
        optimizer.zero_grad()
        
        # Determine training mode: offline (supervised) vs streaming (unsupervised)
        use_offline = dataset_path is not None
        
        if use_offline:
            # Offline mode: Load pre-harvested (residual, error) pairs
            dataset = OfflineDataset(dataset_path)
            loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers, 
                pin_memory=True
            )
            # IMPORTANT: `epochs` always means "total optimization steps" in both modes
            # Convert to required passes over the dataset
            batches_per_pass = len(loader)
            total_steps_desired = epochs  # epochs argument = total steps
            required_passes = math.ceil(total_steps_desired / batches_per_pass)
            print(f"Offline training: {len(dataset)} samples, {batches_per_pass} batches/pass")
            print(f"  Converting {total_steps_desired} steps -> {required_passes} epochs (will stop at exactly {total_steps_desired} steps)")
        else:
            # Streaming mode: Generate synthetic data on-the-fly
            Q = None
            evals = None
            if self.training_data == 'x_subspace' or self.training_data == 'x_mix':
                if self.use_lanczos:
                    solver_gen = Lanczos()
                    Vm1, T = solver_gen.build(self.A, m=self.m)
                    vals, evecs = torch.linalg.eigh(T[:self.m, :self.m])
                    Q = Vm1[:, :self.m] @ evecs
                else:
                    arnoldi = Arnoldi()
                    Vm1, barHm = arnoldi.build(self.A, m=self.m)
                    U, vals, Vh = torch.linalg.svd(barHm, full_matrices=False)
                    Q = (Vm1[:, :-1] @ Vh.T)
                evals = vals
            
            dataset = StreamingDataset(self.n, batch_size, self.training_data, self.m, Q=Q, evals=evals)
            loader = DataLoader(dataset, num_workers=num_workers, pin_memory=True)
        
        hist_loss = []
        best_loss = np.inf
        best_epoch = -1
        checkpoint_file = None
        global_step = 0
            
        if progress_bar:
            # In both modes, epochs = total steps desired
            pbar = tqdm(total=epochs, desc='Train')

        if use_offline:
            # Offline training loop: iterate until we reach total_steps_desired
            training_complete = False
            for epoch_pass in range(required_passes):
                if training_complete:
                    break
                    
                for batch_idx, (r_batch, e_batch) in enumerate(loader):
                    # Check if we've reached the target number of steps
                    if global_step >= total_steps_desired:
                        training_complete = True
                        break
                    
                    # Move to device and convert dtype
                    r_batch = r_batch.to(self.device).to(self.dtype)  # Shape: (batch, n)
                    e_batch = e_batch.to(self.device).to(self.dtype)  # Shape: (batch, n)
                    
                    # Transpose to (n, batch) for network input
                    r_in = r_batch.T  # Shape: (n, batch)
                    e_gt = e_batch.T  # Shape: (n, batch)
                    
                    # Forward pass: predict error from residual
                    e_pred = self._scale_equivariant_forward(r_in)
                    
                    # Supervised loss: directly compare predicted vs true error
                    loss = F.mse_loss(e_pred, e_gt)
                    hist_loss.append(loss.item())

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_epoch = global_step

                        if checkpoint_prefix_with_path is not None:
                            checkpoint_file = checkpoint_prefix_with_path + 'best.pt'
                            torch.save(self.net.state_dict(), checkpoint_file)

                    loss.backward()

                    if (global_step+1) % grad_accu_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                        if scheduler is not None:
                            scheduler.step()

                    if progress_bar:
                        pbar.set_description(f'Step {global_step+1}/{total_steps_desired} loss {loss:.1e}')
                        pbar.update()
                    
                    global_step += 1
                    
        else:
            # Streaming training loop (original behavior)
            for epoch, x_or_b in enumerate(loader):
                if self.training_data != 'no_x':
                    x_gt = x_or_b[0].to(self.device)
                    b_in = self.A @ x_gt
                    b_in, x_gt = b_in.to(self.dtype), x_gt.to(self.dtype)
                else: 
                    b_in = x_or_b[0].to(self.device).to(self.dtype)
                    raise NotImplementedError("GNP requires (x, b) pairs for training")

                x_pred = self._scale_equivariant_forward(b_in)
                b_pred = (self.A @ x_pred.to(torch.float64)).to(self.dtype)
                loss = F.l1_loss(b_pred, b_in)
                hist_loss.append(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch

                    if checkpoint_prefix_with_path is not None:
                        checkpoint_file = checkpoint_prefix_with_path + 'best.pt'
                        torch.save(self.net.state_dict(), checkpoint_file)

                loss.backward()

                if (epoch+1) % grad_accu_steps == 0 or epoch == epochs - 1:
                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                if progress_bar:
                    pbar.set_description(f'Train loss {loss:.1e}')
                    pbar.update()

                if epoch == epochs - 1:
                    break

        if progress_bar:
            pbar.close()

        if checkpoint_file is not None:
            checkpoint_file_old = checkpoint_file
            checkpoint_file = checkpoint_prefix_with_path + f'epoch_{best_epoch}.pt'
            os.rename(checkpoint_file_old, checkpoint_file)
            
        return hist_loss, best_loss, best_epoch, checkpoint_file

    @torch.no_grad()
    def apply(self, r): 
        self.net.eval()
        r = r.to(self.dtype)
        r_in = r.view(-1, 1)
        z_out = self._scale_equivariant_forward(r_in)
        z = z_out.view(-1)
        z = z.double()

        return z