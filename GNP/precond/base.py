"""Base classes for preconditioners."""

import torch
import numpy as np
from abc import ABC, abstractmethod

class BasePreconditioner(ABC):
    """Abstract base class for all preconditioners.

    Defines the interface that all preconditioners must implement.
    """
    @abstractmethod
    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioner to residual vector r.

        Args:
            r: Residual vector on the device (n,) or (n, batch_size)

        Returns:
            z: Preconditioned residual z = M^{-1} r
        """
        pass

class CPUPreconditioner(BasePreconditioner):
    """Base class for preconditioners that require CPU/scipy operations.

    Many classical preconditioners (ILU, AMG, etc.) use scipy routines that
    only work on CPU. This base class handles the device transfer pattern.

    Subclasses should implement _apply_numpy() instead of apply().
    """
    def __init__(self, device: torch.device):
        """Initialize with target device.

        Args:
            device: The device where input tensors will come from and
                    output tensors should be returned to.
        """
        self.device = device

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """Apply preconditioner with automatic CPU transfer.

        Args:
            r: Residual vector on self.device

        Returns:
            z: Preconditioned residual on self.device
        """
        r_np = r.detach().cpu().numpy()
        z_np = self._apply_numpy(r_np)
        return torch.from_numpy(z_np).to(device=self.device, dtype=r.dtype)

    @abstractmethod
    def _apply_numpy(self, r: np.ndarray) -> np.ndarray:
        """Apply the preconditioner in numpy/scipy space.

        Args:
            r: Residual vector as numpy array

        Returns:
            z: Preconditioned residual as numpy array
        """
        pass
