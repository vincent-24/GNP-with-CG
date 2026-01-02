import torch
from torch import nn
import torch.nn.functional as F

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