"""
GNP Data Module.

Contains data generation and loading utilities for training.
"""

from .graph_hierarchy import (
    GraphLevel,
    GraphHierarchy,
    lloyd_aggregation,
    build_graph_hierarchy,
    generate_subdomains,
    get_boundary_edges
)

__all__ = [
    'GraphLevel',
    'GraphHierarchy',
    'lloyd_aggregation',
    'build_graph_hierarchy',
    'generate_subdomains',
    'get_boundary_edges'
]
