import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

class IChol():
    def __init__(self, A, drop_tol=1e-4, fill_factor=None, shift=0.0):
        self.device = A.device
        self.dtype = A.dtype

        if A.layout != torch.sparse_csc:
            raise Exception('To use IChol, A must be sparse csc')
            
        A_cpu = A.to('cpu')
        n = A_cpu.shape[0]

        spA = sparse.csc_matrix(
            (A_cpu.values().numpy(), A_cpu.row_indices().numpy(), A_cpu.ccol_indices().numpy()), 
            shape=(n, n)
        )
        
        if shift > 0:
            spA = spA + shift * sparse.eye(n, format='csc')
        
        try:
            ilu = sparse.linalg.spilu(spA, drop_tol=drop_tol, diag_pivot_thresh=0.0, permc_spec='NATURAL')
        except Exception as e:
            print(f"IChol Warning: NATURAL ordering failed ({e}). Falling back to default pivoting.")
            if fill_factor is not None:
                ilu = sparse.linalg.spilu(spA, drop_tol=drop_tol, fill_factor=fill_factor)
            else:
                ilu = sparse.linalg.spilu(spA, drop_tol=drop_tol)

        self.perm_r = ilu.perm_r  
        self.perm_c = ilu.perm_c  
        L = ilu.L.tocsr()
        U = ilu.U.tocsr()
        diag_U = U.diagonal()
        
        if np.any(diag_U < 0):
            print("IChol Warning: Negative diagonals found in U. Preconditioner is not SPD.")
            
        diag_U = np.abs(diag_U)
        diag_U = np.maximum(diag_U, 1e-12) 
        sqrt_diag = np.sqrt(diag_U)
        self.L = L.multiply(sqrt_diag.reshape(1, -1)).tocsr()
        self.LT = self.L.T.tocsr()

    def apply(self, r):
        r_np = r.detach().to('cpu').numpy().astype(np.float64)
        r_perm = r_np[self.perm_r]
        y = spsolve_triangular(self.L, r_perm, lower=True, unit_diagonal=False)
        z_perm = spsolve_triangular(self.LT, y, lower=False, unit_diagonal=False)
        z_np = np.zeros_like(z_perm)
        z_np[self.perm_c] = z_perm
        z = torch.from_numpy(z_np).to(dtype=self.dtype, device=self.device)
        
        return z