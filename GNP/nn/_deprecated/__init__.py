"""
Deprecated neural network architectures.

These modules are kept for reference but should not be used in new code.
Use LinearMGGNN instead for PCG-compatible neural preconditioners.
"""
import warnings

def _warn_deprecated(name: str):
    warnings.warn(
        f"{name} is deprecated. Use LinearMGGNN for standard PCG compatibility.",
        DeprecationWarning,
        stacklevel=3
    )
