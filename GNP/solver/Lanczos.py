import torch

# Lanczos iteration with Full Re-orthogonalization
# Used for data generation with symmetric matrices (required for FCG/CG)
class Lanczos():

    def build(self, A, v0=None, m=100):
        n = A.shape[0]
        if v0 is None:
            v0 = torch.normal(0, 1, size=(n,), dtype=A.dtype).to(A.device)
        beta = torch.linalg.norm(v0)
        
        # V stores the orthonormal basis
        V = torch.zeros(n, m+1, dtype=A.dtype).to(A.device)
        # T stores the tridiagonal matrix (diagonal alpha, off-diagonal beta)
        # We store it as a dense matrix to match the API of Arnoldi's H
        T = torch.zeros(m+1, m, dtype=A.dtype).to(A.device)

        V[:,0] = v0 / beta
        
        for j in range(m):
            w = A @ V[:,j]
            
            # Orthogonalize against v_j (alpha)
            alpha = torch.dot(V[:,j], w)
            w = w - alpha * V[:,j]
            T[j, j] = alpha
            
            # Orthogonalize against v_{j-1} (beta)
            if j > 0:
                beta_prev = T[j, j-1] # Retreive beta_{j-1}
                w = w - beta_prev * V[:, j-1]
                # Note: T[j-1, j] was already set in previous iter
            
            # Full Re-orthogonalization (Crucial for training stability)
            # In exact arithmetic Lanczos only needs 2 terms, but for 
            # accurate subspace generation we re-orthogonalize against ALL previous vectors.
            for k in range(j+1):
                coeff = torch.dot(V[:, k], w)
                w = w - coeff * V[:, k]
                
            beta_next = torch.linalg.norm(w)
            
            if j < m:
                T[j+1, j] = beta_next
                if j < m - 1: # Symmetric part
                    T[j, j+1] = beta_next
            
            # Check for breakdown
            if beta_next < 1e-12:
                break
                
            V[:,j+1] = w / beta_next

        Vm1 = V
        barTm = T 
        return Vm1, barTm