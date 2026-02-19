import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from GNP.utils import scale_A_by_spectral_radius
from .layers import MLP, GCNConv

class ResGCN(nn.Module):
    def __init__(self, A, num_layers, embed, hidden, drop_rate, scale_input=True, dtype=torch.float32):
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
            self.gconv.append(GCNConv(self.AA, embed, embed))
            self.skip.append(nn.Linear(embed, embed))
            self.batchnorm.append(nn.BatchNorm1d(embed))

        self.dropout = nn.Dropout(drop_rate)
        
        if dtype == torch.float64:
            self._cast_to_float64()

    def _cast_to_float64(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.BatchNorm1d)):
                module.weight.data = module.weight.data.double()
                if module.bias is not None:
                    module.bias.data = module.bias.data.double()
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean = module.running_mean.double() if module.running_mean is not None else None
                module.running_var = module.running_var.double() if module.running_var is not None else None

    def forward(self, r):
        n, batch_size = r.shape
        r = r.to(self.dtype)
        
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