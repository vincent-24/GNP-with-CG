import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from GNP.utils import scale_A_by_spectral_radius
from .layers import MLP, GCNConv

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