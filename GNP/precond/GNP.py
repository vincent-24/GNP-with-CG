import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
from tqdm import tqdm

# IMPORTANT: Ensure Lanczos is imported here!
from GNP.solver import Arnoldi, Lanczos

class StreamingDataset(IterableDataset):
    def __init__(self, A, batch_size, training_data, m, use_lanczos=False):
        super().__init__()
        self.n = A.shape[0]
        self.m = m
        self.batch_size = batch_size
        self.training_data = training_data

        if training_data == 'x_subspace' or training_data == 'x_mix':
            if use_lanczos:
                # Lanczos for SPD/FCG
                solver_gen = Lanczos()
                Vm1, T = solver_gen.build(A, m=m)
                # Eigendecomposition of tridiagonal T
                evals, evecs = torch.linalg.eigh(T[:m, :m])
                # Project back to full space
                Q = Vm1[:, :m] @ evecs
                self.Q = Q.to('cpu')
            else:
                # Arnoldi for FGMRES
                arnoldi = Arnoldi()
                Vm1, barHm = arnoldi.build(A, m=m)
                W, S, Zh = torch.linalg.svd(barHm, full_matrices=False)
                Q = ( Vm1[:,:-1] @ Zh.T ) / S.view(1, m)
                self.Q = Q.to('cpu')

    def generate(self):
        while True:
            if self.training_data == 'x_normal':
                x = torch.normal(0, 1, size=(self.n, self.batch_size), dtype=torch.float64)
                yield x
            elif self.training_data == 'x_subspace':
                e = torch.normal(0, 1, size=(self.m, self.batch_size), dtype=torch.float64)
                x = self.Q @ e
                yield x
            elif self.training_data == 'x_mix':
                batch_size1 = self.batch_size // 2
                e = torch.normal(0, 1, size=(self.m, batch_size1), dtype=torch.float64)
                x = self.Q @ e
                batch_size2 = self.batch_size - batch_size1
                x2 = torch.normal(0, 1, size=(self.n, batch_size2), dtype=torch.float64)
                x = torch.cat([x, x2], dim=1)
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

    def train(self, batch_size, grad_accu_steps, epochs, optimizer,
              scheduler=None, num_workers=0, checkpoint_prefix_with_path=None,
              progress_bar=True):

        self.net.train()
        optimizer.zero_grad()
        dataset = StreamingDataset(self.A, batch_size, self.training_data, self.m, 
                                   use_lanczos=self.use_lanczos)
        loader = DataLoader(dataset, num_workers=num_workers, pin_memory=True)
        
        hist_loss = []
        best_loss = np.inf
        best_epoch = -1
        checkpoint_file = None
            
        if progress_bar:
            pbar = tqdm(total=epochs, desc='Train')

        for epoch, x_or_b in enumerate(loader):
            if self.training_data != 'no_x':
                x = x_or_b[0].to(self.device)
                b = self.A @ x
                b, x = b.to(self.dtype), x.to(self.dtype)
            else: 
                b = x_or_b[0].to(self.device).to(self.dtype)

            x_out = self.net(b)
            b_out = (self.A @ x_out.to(torch.float64)).to(self.dtype)
            loss = F.l1_loss(b_out, b)

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

        if checkpoint_file is not None:
            checkpoint_file_old = checkpoint_file
            checkpoint_file = checkpoint_prefix_with_path + f'epoch_{best_epoch}.pt'
            os.rename(checkpoint_file_old, checkpoint_file)
            
        return hist_loss, best_loss, best_epoch, checkpoint_file

    @torch.no_grad()
    def apply(self, r): 
        self.net.eval()
        r = r.to(self.dtype)
        r = r.view(-1, 1)
        z = self.net(r)
        z = z.view(-1)
        z = z.double()
        return z