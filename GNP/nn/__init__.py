from .layers import MLP, GCNConv, GATv2Conv
from .ResGCN import ResGCN
from .SplitResGCN import SplitResGCN
from .MGGNN import MGGNN, UNetGCN  # UNetGCN is alias for MGGNN

# FNO is optional - only import if implemented
try:
    from .FNO import FNO
    __all__ = ['MLP', 'GCNConv', 'GATv2Conv', 'ResGCN', 'SplitResGCN', 'UNetGCN', 'MGGNN', 'FNO']
except ImportError:
    FNO = None
    __all__ = ['MLP', 'GCNConv', 'GATv2Conv', 'ResGCN', 'SplitResGCN', 'UNetGCN', 'MGGNN']