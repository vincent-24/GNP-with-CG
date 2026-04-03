from GNP.utils import extract_diagonal
from .base import BasePreconditioner

class Jacobi(BasePreconditioner):
    """Jacobi (diagonal) preconditioner: M = diag(A)."""

    def __init__(self, A):
        self.D = extract_diagonal(A)
        self.D[self.D == 0] = 1

    def apply(self, r):
        return r / self.D
