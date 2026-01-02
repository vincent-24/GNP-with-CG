"""
Factory module for resolving solver and network classes from string names.
This module handles the lazy imports to avoid circular dependencies.
"""

from GNP import config


def get_solver_class(solver_name: str):
    """
    Get the solver class from a string name.
    
    Args:
        solver_name: Name of the solver class (e.g., 'GMRES', 'FCG', 'PCG', 'PolakRibiereCG')
    
    Returns:
        The solver class object
    """
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
    """
    Get the network class from a string name.
    
    Args:
        net_name: Name of the network class (e.g., 'ResGCN', 'SplitResGCN')
    
    Returns:
        The network class object
    """
    from GNP.nn import ResGCN, SplitResGCN
    
    net_classes = {
        'ResGCN': ResGCN,
        'SplitResGCN': SplitResGCN,
    }
    
    if net_name not in net_classes:
        raise ValueError(f"Unknown network: {net_name}. Available: {list(net_classes.keys())}")
    
    return net_classes[net_name]


def get_solver_and_network(solver_name: str):
    """
    Get the solver class and network class for a registered solver configuration.
    
    Args:
        solver_name: Key in SOLVER_REGISTRY (e.g., 'FGMRES', 'FCG', 'PCG')
    
    Returns:
        tuple: (solver_class, network_class, config_dict)
            - solver_class: The solver class object
            - network_class: The network class object  
            - config_dict: The full configuration dictionary from SOLVER_REGISTRY
    """
    if solver_name not in config.SOLVER_REGISTRY:
        raise ValueError(
            f"Solver '{solver_name}' not found in registry. "
            f"Available: {list(config.SOLVER_REGISTRY.keys())}"
        )
    
    cfg = config.SOLVER_REGISTRY[solver_name]
    solver_cls = get_solver_class(cfg['solver_cls'])
    net_cls = get_network_class(cfg['net_cls'])
    
    return solver_cls, net_cls, cfg
