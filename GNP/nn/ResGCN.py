import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from GNP.utils import scale_A_by_spectral_radius

#-----------------------------------------------------------------------------
# An MLP layer.
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, hidden, drop_rate,
                 use_batchnorm=False, is_output_layer=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.is_output_layer = is_output_layer

        self.lin = nn.ModuleList()
        self.lin.append( nn.Linear(in_dim, hidden) )
        for i in range(1, num_layers-1):
            self.lin.append( nn.Linear(hidden, hidden) )
        self.lin.append( nn.Linear(hidden, out_dim) )
        if use_batchnorm:
            self.batchnorm = nn.ModuleList()
            for i in range(0, num_layers-1):
                self.batchnorm.append( nn.BatchNorm1d(hidden) )
            if not is_output_layer:
                self.batchnorm.append( nn.BatchNorm1d(out_dim) )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, R):
        assert len(R.shape) >= 2
        for i in range(self.num_layers):
            R = self.lin[i](R)
            if i != self.num_layers-1 or not self.is_output_layer:
                if self.use_batchnorm:
                    shape = R.shape
                    R = R.view(-1, shape[-1])
                    R = self.batchnorm[i](R)
                    R = R.view(shape)
                R = self.dropout(F.relu(R))
        return R

#-----------------------------------------------------------------------------
# A GCN layer.
class GCNConv(nn.Module):
    def __init__(self, AA, in_dim, out_dim):
        super().__init__()
        self.AA = AA  # normalized A
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, R):
        assert len(R.shape) == 3
        n, batch_size, in_dim = R.shape
        if in_dim > self.out_dim:
            R = self.fc(R)
            R = R.view(n, batch_size * self.out_dim)
            R = self.AA @ R
            R = R.view(n, batch_size, self.out_dim)
        else:
            R = R.view(n, batch_size * in_dim)
            R = self.AA @ R
            R = R.view(n, batch_size, in_dim)
            R = self.fc(R)
        return R

#-----------------------------------------------------------------------------
# Standard ResGCN (Use for FGMRES / Non-Symmetric)
class ResGCN(nn.Module):
    def __init__(self, A, num_layers, embed, hidden, drop_rate,
                 scale_input=True, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.num_layers = num_layers
        self.embed = embed
        self.scale_input = scale_input
        self.AA = scale_A_by_spectral_radius(A).to(dtype)

        self.mlp_initial = MLP(1, embed, 4, hidden, drop_rate)
        self.mlp_final = MLP(embed, 1, 4, hidden, drop_rate, is_output_layer=True)
        
        self.gconv = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        for i in range(num_layers):
            self.gconv.append( GCNConv(self.AA, embed, embed) )
            self.skip.append( nn.Linear(embed, embed) )
            self.batchnorm.append( nn.BatchNorm1d(embed) )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):
        n, batch_size = r.shape
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            scaling = torch.where(scaling < 1e-12, torch.tensor(1.0, device=r.device, dtype=r.dtype), scaling)
            r = r / scaling
        
        r = r.view(n, batch_size, 1)
        R = self.mlp_initial(r)
        
        for i in range(self.num_layers):
            R = self.gconv[i](R) + self.skip[i](R)
            R = R.view(n * batch_size, self.embed)
            R = self.batchnorm[i](R)
            R = R.view(n, batch_size, self.embed)
            R = self.dropout(F.relu(R))
            
        z = self.mlp_final(R)
        z = z.view(n, batch_size)
        
        if self.scale_input:
            z = z * scaling
        return z

#-----------------------------------------------------------------------------
# SplitResGCN (Use for FCG / SPD Matrices)
# Enforces M = Enc^T * Enc, guaranteeing SPD preconditioning
class SplitResGCN(nn.Module):
    
    def __init__(self, A, num_layers, embed, hidden, drop_rate,
                 scale_input=True, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.scale_input = scale_input
        self.AA = scale_A_by_spectral_radius(A).to(dtype)
        
        # Encoder (approximates L^T)
        self.enc_mlp = MLP(1, embed, 2, hidden, drop_rate)
        self.enc_gconv = nn.ModuleList()
        self.enc_skip = nn.ModuleList()   
        self.enc_bn = nn.ModuleList()
        
        # Decoder (approximates L)
        self.dec_mlp = MLP(embed, 1, 2, hidden, drop_rate, is_output_layer=True)
        self.dec_gconv = nn.ModuleList()
        self.dec_skip = nn.ModuleList()   
        self.dec_bn = nn.ModuleList()
        
        self.half_layers = num_layers // 2
        
        for i in range(self.half_layers):
            # Encoder Layers
            self.enc_gconv.append(GCNConv(self.AA, embed, embed))
            self.enc_skip.append(nn.Linear(embed, embed)) 
            self.enc_bn.append(nn.BatchNorm1d(embed))
            
            # Decoder Layers
            self.dec_gconv.append(GCNConv(self.AA, embed, embed))
            self.dec_skip.append(nn.Linear(embed, embed)) 
            self.dec_bn.append(nn.BatchNorm1d(embed))
            
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):
        n, batch_size = r.shape
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            scaling = torch.where(scaling < 1e-12, torch.tensor(1.0, device=r.device, dtype=r.dtype), scaling)
            r = r / scaling

        # --- ENCODER (L^T) ---
        r = r.view(n, batch_size, 1)
        R = self.enc_mlp(r)
        
        for i in range(self.half_layers):
            # Apply GCN + Residual Skip
            R_conv = self.enc_gconv[i](R)
            R_skip = self.enc_skip[i](R)
            R = R_conv + R_skip         
            
            R = R.view(n * batch_size, -1)
            R = self.enc_bn[i](R)
            R = R.view(n, batch_size, -1)
            R = self.dropout(F.relu(R))
            
        # --- DECODER (L) ---
        for i in range(self.half_layers):
            # Apply GCN + Residual Skip
            R_conv = self.dec_gconv[i](R)
            R_skip = self.dec_skip[i](R)
            R = R_conv + R_skip         

            R = R.view(n * batch_size, -1)
            R = self.dec_bn[i](R)
            R = R.view(n, batch_size, -1)
            R = self.dropout(F.relu(R))

        z = self.dec_mlp(R)
        z = z.view(n, batch_size)

        if self.scale_input:
            z = z * scaling
        return z