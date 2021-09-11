import numpy as np
import scipy as sp
from time import time

import warnings

## ADI solvers

def adi_pr(H, V, b, n_iter=10, u_init=None, p=None, q=None, store_every=1, verbose=True):
    """
    Vanilla ADI solver for Peaceman-Rachford model problem.

    Returns an array of iterates and of elapsed time.
    """
    n = H.shape[0]
    # n_iter = int(n_iter) # if n_iter is not an int convert it
    
    if store_every <= 0:
        store_every = 1
        warnings.warn("store_every set to 1.")

    if p is None:
        p = np.zeros(n_iter)
    elif isinstance(p, float) or isinstance(p, int):
        p = p*np.ones(n_iter)
    if q is None:
        q = np.zeros(n_iter)
    elif isinstance(q, float) or isinstance(q, int):
        q = q*np.ones(n_iter)

    u = np.zeros(n) if u_init is None else u_init
    u_list = [u]

    if verbose:
        print("--------------------------------------------")
        print("  Iteration   |    Epoch     |   Time (s)   ")
        print("--------------------------------------------")

    t_list = [0.]
    iter_list = [0]
    t0 = time()
    for j in range(1, n_iter+1):
        # First half-step
        p_j = p[j-1]
        A = H + p_j*np.eye(n)                     # H + pI
        c = b + (p_j*np.eye(n) - V) @ u           # b + (pI - V) @ u_{j}
#         u = sp.linalg.solve(A, c, assume_a="pos") # u_{j+1/2} = solution of (H+pI) u = b + (pI-V) @ u_{j}
        u = sp.linalg.solve(A, c, assume_a="sym") # u_{j+1/2} = solution of (H+pI) u = b + (pI-V) @ u_{j}

        # Second half-step
        q_j = q[j-1]
        A = V + q_j*np.eye(n)                     # V + qI
        c = b + (q_j*np.eye(n) - H) @ u           # b + (qI - H) @ u_{j+1/2}
#         u = sp.linalg.solve(A, c, assume_a="pos") # u_{j+1} = solution of (V+qI) u = b + (qI-H) @ u_{j+1/2}
        u = sp.linalg.solve(A, c, assume_a="sym") # u_{j+1} = solution of (V+qI) u = b + (qI-H) @ u_{j+1/2}

        if (j % store_every == 0) or (j == n_iter):
            t_list.append(time() - t0)
            u_list.append(u)
            iter_list.append(j)
            
        if verbose and (j % store_every == 0):
            print(f"{j:^14}|{j:^14}|{time() - t0:^14.2e}")

    if verbose and not (n_iter % store_every == 0):
        print(f"{j:^14}|{j:^14}|{time() - t0:^14.2e}\n")

    epoch_list = iter_list
    return np.array(u_list), np.array(t_list), np.array(iter_list), np.array(epoch_list)


def sap_adi_pr(H, V, b, n_iter=10, u_init=None, p=None, q=None, sketch_size=None, store_every=1, verbose=True):
    """
    Sketch-and-Project (SAP) ADI solver for Peaceman-Rachford model problem.

    With block row sketching.

    Returns an array of iterates and of elapsed time.
    """
    n = H.shape[0]

    if store_every <= 0:
        store_every = 1
        warnings.warn("store_every set to 1.")
    
    if sketch_size is None:
        sketch_size = n

    if p is None:
        p = np.zeros(n_iter)
    elif isinstance(p, float) or isinstance(p, int):
        p = p*np.ones(n_iter)
    if q is None:
        q = np.zeros(n_iter)
    elif isinstance(q, float) or isinstance(q, int):
        q = q*np.ones(n_iter)

    u = np.zeros(n) if u_init is None else u_init
    u_list = [u]

    if verbose:
        print("--------------------------------------------")
        print("  Iteration   |    Epoch     |   Time (s)   ")
        print("--------------------------------------------")

    t_list = [0.]
    iter_list = [0]
    epoch_list = [0]
    t0 = time()
    for j in range(1, n_iter+1):
        p_j, q_j = p[j-1], q[j-1]

        # First half-step
        sampled_idx = np.random.choice(n, size=sketch_size, replace=False)
        A = H[sampled_idx, :] + p_j*np.eye(n)[sampled_idx, :]            # S_{1/2}^T (H+pI)=(H+pI)_{R:}
        c = (H[sampled_idx, :] + V[sampled_idx, :]) @ u - b[sampled_idx] # (H+V)_{R:} u - b_{R:}
        u = u - A.T @ sp.linalg.solve(A @ A.T, c, assume_a="pos")        # u = u - A^T (AA^T)^{-1} c

        # Second half-step
        sampled_idx = np.random.choice(n, size=sketch_size, replace=False)
        A = V[sampled_idx, :] + q_j*np.eye(n)[sampled_idx, :]            # S_{1}^T (V+pI)=(H+pI)_{R:}
        c = (H[sampled_idx, :] + V[sampled_idx, :]) @ u - b[sampled_idx] # (H+V)_{R:} u - b_{R:}
        u = u - A.T @ sp.linalg.solve(A @ A.T, c, assume_a="pos")        # u = u - A^T (AA^T)^{-1} c

        if (j % store_every == 0) or (j == n_iter):
            t_list.append(time() - t0)
            u_list.append(u)
            iter_list.append(j)
            epoch_list.append(j*sketch_size/n)

        if verbose and (j % store_every == 0):
            print(f"{j:^14}|{j*sketch_size/n:^14}|{time() - t0:^14.2e}")

    if verbose and not (n_iter % store_every == 0):
        print(f"{j:^14}|{j*sketch_size/n:^14}|{time() - t0:^14.2e}\n")

    return np.array(u_list), np.array(t_list), np.array(iter_list), np.array(epoch_list)