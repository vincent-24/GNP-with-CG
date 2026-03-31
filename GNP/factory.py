"""
Factory module for resolving solver and network classes from string names.
This module handles the lazy imports to avoid circular dependencies.
"""
from GNP import config


def get_solver_class(solver_name: str):
    from GNP.solver import GMRES, FCG, PCG, PolakRibiereCG

    solver_classes = {
        'GMRES': GMRES,
        'FCG': FCG,
        'PCG': PCG,
        'PolakRibiereCG': PolakRibiereCG,
    }

    if solver_name not in solver_classes:
        raise ValueError(f"Unknown solver: {solver_name}. Available: {list(solver_classes.keys())}")

    return solver_classes[solver_name]


def get_network_class(net_name: str):
    from GNP.nn import ResGCN, SplitResGCN, UNetGCN, MGGNN, LinearMGGNN

    net_classes = {
        'ResGCN': ResGCN,
        'SplitResGCN': SplitResGCN,
        'UNetGCN': UNetGCN,
        'MGGNN': MGGNN,
        'LinearMGGNN': LinearMGGNN,
    }

    if net_name not in net_classes:
        raise ValueError(f"Unknown network: {net_name}. Available: {list(net_classes.keys())}")

    return net_classes[net_name]


def get_solver_info(solver_name: str):
    if solver_name not in config.SOLVER_CONFIG:
        raise ValueError(f"Solver '{solver_name}' not found. Available: {list(config.SOLVER_CONFIG.keys())}")

    cfg = config.SOLVER_CONFIG[solver_name]
    solver_cls = get_solver_class(cfg['solver_cls'])

    return solver_cls, cfg
