import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from GNP.data.graph_hierarchy import GraphHierarchy

def add_self_loops_simple(
    edge_index: torch.Tensor, 
    edge_weight: Optional[torch.Tensor] = None,
    fill_value: float = 1.0,
    num_nodes: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    loop_index = torch.arange(0, num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index_out = torch.cat([edge_index, loop_index], dim=1)
    
    if edge_weight is not None:
        loop_weight = torch.full((num_nodes,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device)
        edge_weight_out = torch.cat([edge_weight, loop_weight], dim=0)
    else:
        edge_weight_out = torch.ones(edge_index_out.size(1), device=edge_index.device)
    
    return edge_index_out, edge_weight_out


def compute_degree(
    index: torch.Tensor, 
    num_nodes: int,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=dtype, device=index.device)
    ones = torch.ones(index.size(0), dtype=dtype, device=index.device)
    deg.scatter_add_(0, index, ones)
    return deg

class TAGConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 3):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) 
            for _ in range(K + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        
        edge_index_sl, edge_weight_sl = add_self_loops_simple(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )
        
        row_sl, col_sl = edge_index_sl
        deg = compute_degree(col_sl, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row_sl] * edge_weight_sl * deg_inv_sqrt[col_sl]
        
        out = self.lins[0](x)
        h = x
        
        for k in range(1, self.K + 1):
            h_new = torch.zeros_like(h)
            h_new.index_add_(0, row_sl, h[col_sl] * norm.unsqueeze(-1))
            h = h_new
            out = out + self.lins[k](h)
        
        return out + self.bias

class MGLayer(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        num_levels: int,
        K: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.intra_convs = nn.ModuleList([
            TAGConvLayer(hidden_dim, hidden_dim, K=K)
            for _ in range(num_levels)
        ])
        
        self.restrict_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_levels - 1)
        ])
        
        self.interp_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_levels - 1)
        ])
        
        self.combine = nn.ModuleList([
            nn.Linear(3 * hidden_dim if 0 < i < num_levels - 1 else 2 * hidden_dim, hidden_dim)
            for i in range(num_levels)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_levels)])
    
    def forward(
        self, 
        x_levels: List[torch.Tensor],
        hierarchy: GraphHierarchy
    ) -> List[torch.Tensor]:
        num_levels = len(x_levels)
        conv_features = []
        for l in range(num_levels):
            edge_index = hierarchy.levels[l].edge_index
            edge_weight = hierarchy.levels[l].edge_weight
            h = self.intra_convs[l](x_levels[l], edge_index, edge_weight)
            conv_features.append(h)
        
        restricted = [None] * num_levels
        for l in range(num_levels - 1):
            R = hierarchy.restriction_matrices[l]
            h_fine = self.restrict_mlps[l](conv_features[l])
            restricted[l + 1] = torch.sparse.mm(R, h_fine)
        
        interpolated = [None] * num_levels
        for l in range(num_levels - 1):
            P = hierarchy.interpolation_matrices[l]
            h_coarse = self.interp_mlps[l](conv_features[l + 1])
            interpolated[l] = torch.sparse.mm(P, h_coarse)
        
        out_features = []
        for l in range(num_levels):
            features_to_cat = [conv_features[l]]
            
            if l > 0 and restricted[l] is not None:
                features_to_cat.append(restricted[l])
            
            if l < num_levels - 1 and interpolated[l] is not None:
                features_to_cat.append(interpolated[l])
            
            if l == 0: 
                if interpolated[l] is not None:
                    combined = torch.cat([conv_features[l], interpolated[l]], dim=-1)
                else:
                    combined = torch.cat([conv_features[l], conv_features[l]], dim=-1)
            elif l == num_levels - 1: 
                if restricted[l] is not None:
                    combined = torch.cat([conv_features[l], restricted[l]], dim=-1)
                else:
                    combined = torch.cat([conv_features[l], conv_features[l]], dim=-1)
            else:  
                combined = torch.cat(features_to_cat, dim=-1)
                if len(features_to_cat) < 3:
                    pad = torch.zeros_like(conv_features[l])
                    combined = torch.cat([combined, pad], dim=-1)
            
            h = self.combine[l](combined)
            h = self.norm[l](h)
            h = F.relu(h)
            h = self.dropout(h)
            
            h = h + x_levels[l]
            out_features.append(h)
        
        return out_features

class MGGNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_levels: int = 2,
        K: int = 3,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.dtype = dtype
        
        self.input_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_levels)
        ])
        
        self.mg_layers = nn.ModuleList([
            MGLayer(hidden_dim, num_levels, K=K, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.P_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.L_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus() 
        )
        
        if dtype == torch.float64:
            self._cast_to_float64()
    
    def _cast_to_float64(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.double()
                if module.bias is not None:
                    module.bias.data = module.bias.data.double()
            if isinstance(module, nn.LayerNorm):
                module.weight.data = module.weight.data.double()
                if module.bias is not None:
                    module.bias.data = module.bias.data.double()
    
    def forward(
        self,
        hierarchy: GraphHierarchy,
        node_features: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        num_levels = hierarchy.num_levels
        
        if node_features is None:
            node_features = self._init_node_features(hierarchy)
        
        x_levels = []
        for l in range(num_levels):
            x = node_features[l].to(self.dtype)
            x = self.input_embeds[l](x)
            x_levels.append(x)
        
        for mg_layer in self.mg_layers:
            x_levels = mg_layer(x_levels, hierarchy)
        
        P_weights_list = []
        for l in range(num_levels - 1):
            c2f_edges = hierarchy.coarse_to_fine_edges[l]
            coarse_idx, fine_idx = c2f_edges[0], c2f_edges[1]
            
            coarse_feats = x_levels[l + 1][coarse_idx]
            fine_feats = x_levels[l][fine_idx]
            
            edge_feats = torch.cat([coarse_feats, fine_feats], dim=-1)
            weights = self.P_head(edge_feats).squeeze(-1)
            P_weights_list.append(weights)
        
        edge_index = hierarchy.levels[0].edge_index
        row, col = edge_index
        
        src_feats = x_levels[0][row]
        dst_feats = x_levels[0][col]
        edge_feats = torch.cat([src_feats, dst_feats], dim=-1)
        L_weights = self.L_head(edge_feats).squeeze(-1)
        
        return {
            'P_weights': P_weights_list,
            'L_weights': L_weights,
            'level_features': x_levels
        }
    
    def _init_node_features(self, hierarchy: GraphHierarchy) -> List[torch.Tensor]:
        features = []
        for level in hierarchy.levels:
            num_nodes = level.num_nodes
            edge_index = level.edge_index
            
            deg = compute_degree(edge_index[1], num_nodes, dtype=self.dtype)
            deg = deg / (deg.max() + 1e-8) 
            features.append(deg.unsqueeze(-1))  
        
        return features

class MGGNNWithResidual(MGGNN):
    def __init__(
        self,
        input_dim: int = 2,  
        **kwargs
    ):
        super().__init__(input_dim=input_dim, **kwargs)
    
    def forward_with_residual(
        self,
        hierarchy: GraphHierarchy,
        residual: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        node_features = []
        for l, level in enumerate(hierarchy.levels):
            num_nodes = level.num_nodes
            edge_index = level.edge_index
            
            deg = compute_degree(edge_index[1], num_nodes, dtype=self.dtype)
            deg = deg / (deg.max() + 1e-8)
            
            if l == 0:
                res_feat = torch.abs(residual)
            else:
                res_feat = residual
                for k in range(l):
                    R = hierarchy.restriction_matrices[k]
                    res_feat = torch.sparse.mm(R, res_feat.unsqueeze(-1)).squeeze(-1)
                res_feat = torch.abs(res_feat)
            
            res_feat = res_feat / (res_feat.max() + 1e-8)  
            
            features = torch.stack([deg, res_feat], dim=-1)
            node_features.append(features)
        
        return self.forward(hierarchy, node_features)
