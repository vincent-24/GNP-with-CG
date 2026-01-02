import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from GNP.utils import scale_A_by_spectral_radius
from .layers import MLP, GCNConv

#-----------------------------------------------------------------------------
# SplitResGCN (Use for FCG / SPD Matrices)
class SplitResGCN(nn.Module):
    
    def __init__(self, A, num_layers, embed, hidden, drop_rate,
                 scale_input=True, dtype=torch.float32, tie_weights=False):
        super().__init__()
        self.dtype = dtype
        self.scale_input = scale_input
        self.tie_weights = tie_weights
        self.AA = scale_A_by_spectral_radius(A).to(dtype)
        self.embed = embed
        
        # Encoder (approximates L^T)
        self.enc_mlp = MLP(1, embed, 2, hidden, drop_rate)
        self.enc_gconv = nn.ModuleList()
        self.enc_skip = nn.ModuleList()   
        self.enc_bn = nn.ModuleList()
        
        # Decoder (approximates L)
        self.dec_bn = nn.ModuleList()
        
        if not tie_weights:
            self.dec_mlp = MLP(embed, 1, 2, hidden, drop_rate, is_output_layer=True)
            self.dec_gconv = nn.ModuleList()
            self.dec_skip = nn.ModuleList()
        else:
            self.dec_mlp = MLP(embed, 1, 2, hidden, drop_rate, is_output_layer=True)
        
        self.half_layers = num_layers // 2
        
        for i in range(self.half_layers):
            self.enc_gconv.append(GCNConv(self.AA, embed, embed))
            self.enc_skip.append(nn.Linear(embed, embed)) 
            self.enc_bn.append(nn.BatchNorm1d(embed))
            self.dec_bn.append(nn.BatchNorm1d(embed))
            
            if not tie_weights:
                self.dec_gconv.append(GCNConv(self.AA, embed, embed))
                self.dec_skip.append(nn.Linear(embed, embed))
            
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, r):
        n, batch_size = r.shape
        if self.scale_input:
            scaling = torch.linalg.vector_norm(r, dim=0) / np.sqrt(n)
            scaling = torch.where(scaling < 1e-12, torch.tensor(1.0, device=r.device, dtype=r.dtype), scaling)
            r = r / scaling

        r = r.view(n, batch_size, 1)
        R = self.enc_mlp(r)
        
        for i in range(self.half_layers):
            R = self.enc_gconv[i](R) + self.enc_skip[i](R)         
            R = R.view(n * batch_size, -1)
            R = self.enc_bn[i](R)
            R = R.view(n, batch_size, -1)
            R = self.dropout(F.relu(R))
            
        # --- DECODER (L) ---
        for i in range(self.half_layers):
            if self.tie_weights:
                enc_idx = self.half_layers - 1 - i
                W_gcn_T = self.enc_gconv[enc_idx].fc.weight.t()
                R_flat = R.view(n, batch_size * self.embed)
                R_flat = self.AA @ R_flat
                R_flat = R_flat.view(n, batch_size, self.embed)
                R_conv = F.linear(R_flat, weight=self.enc_gconv[enc_idx].fc.weight)
                R_skip = F.linear(R, weight=self.enc_skip[enc_idx].weight)
                R = R_conv + R_skip
            else:
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