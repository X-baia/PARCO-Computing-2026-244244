import numpy as np
import scipy.sparse as sp
import scipy.io as io

def poisson3d(n):
    N = n*n*n
    main = 6*np.ones(N)
    off1 = -1*np.ones(N-1)
    offn = -1*np.ones(N-n)
    offn2 = -1*np.ones(N-n*n)

    A = sp.diags(
        [main, off1, off1, offn, offn, offn2, offn2],
        [0, -1, 1, -n, n, -n*n, n*n],
        shape=(N, N),
        format="csr"
    )
    return A

base_rows_per_rank = 10000

for p in [1,2,4,8,9,16,25,32,36,49,64,81,100,121,128]:
    total_rows = base_rows_per_rank * p
    n = int(round(total_rows ** (1/3)))
    A = poisson3d(n)
    io.mmwrite(f"poisson_weak_{p}.mtx", A)
