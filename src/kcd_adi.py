import numpy as np
import scipy as sp
from time import time

import warnings


# SADI solver for Sylvester equation using both left and right sketching
# KCD ADI stands for "Kaczmarz-Coordinate Descent ADI"
# Cyclical sampling

def kcd_adi_syl(A, B, C, n_iter=10, X_init=None, p=None, q=None,
                sketch_size=None, sketch_frac=None, store_every=1, assume_a="pos", verbose=True):
    """
    Sketch-and-Project (SAP) ADI solver for solving Sylvester equation:
    AX - XB = C

    First solves for X_{j+1/2} using the Sketched-and-Project method
    (A − p_j I) X_{j+1/2} = F − X_j (B − p_j I)
    Then solving for X_{j+1} using the Sketched-and-Project method
    X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}

    With both single row & column sketching.
    See corresponding note.

    Returns an array of iterates and of elapsed time.
    """
    # n_iter = int(n_iter) # if n_iter is not an int convert it

    if store_every <= 0:
        store_every = 1
        warnings.warn("store_every set to 1.")

    m = A.shape[0]
    n = B.shape[0]
    min_size = min(m, n)

    if sketch_size is not None and sketch_frac is not None:
        raise ValueError("Sketch size and sketch fraction should not be set simultaneously.")

    if sketch_size is None:
        if sketch_frac is None:
            print("Default sketch size equals 90% of the rows of each matrix A and B\n")
            sketch_frac = .9  # default 90% of the rows
            sketch_size_A = int(sketch_frac * m)
            sketch_size_B = int(sketch_frac * n)
        elif sketch_frac <= 0. or sketch_frac > 1.:
            raise ValueError("sketch fraction must be between 0 and 1.")
        else:
            print(f"Sketch size equals {sketch_frac:.0%} of the rows of each matrix\n")
            sketch_size_A = int(sketch_frac * m)
            sketch_size_B = int(sketch_frac * n)
    elif not isinstance(sketch_size, int) or sketch_size < 1 or sketch_size > min_size:
        raise ValueError("Sketch size must be an int between 1 and the smallest number of rows of A and B.")
    else:
        print(f"Sketch size equals {sketch_size} for both matrices A and B\n")
        sketch_size_A = sketch_size
        sketch_size_B = sketch_size
        sketch_frac = sketch_size/min_size

    if p is None:
        p = np.zeros(n_iter)
    elif isinstance(p, float) or isinstance(p, int):
        p = p*np.ones(n_iter)
    if q is None:
        q = np.zeros(n_iter)
    elif isinstance(q, float) or isinstance(q, int):
        q = q*np.ones(n_iter)

    X = np.zeros((m, n)) if X_init is None else X_init
    X_list = [X]

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
        # fixed_row_idx = (j - 1) % m
        fixed_row_idx = np.random.randint(sketch_size_A)

        # G = S_{1/2}^T (A-pI) = (A-pI)_{R:}
        G = A[fixed_row_idx, :] - p_j*np.eye(m)[fixed_row_idx, :]
        squared_norm_A_r = np.linalg.norm(G) ** 2

        # Going through all columns of X cyclically
        for c in range(n):
            # U = C_{R:} - A_{R:} X + X_{R:} B
            u = C[fixed_row_idx, c] - A[fixed_row_idx, :].dot(X[:, c]) + X[fixed_row_idx, :].dot(B[:, c])

            # X = X + G^T (G G^T)^{-1} U
            X[:, c] += (u / squared_norm_A_r) * G.T

        # Second half-step
        # fixed_col_idx = (j - 1) % n
        fixed_col_idx = np.random.randint(sketch_size_A)

        # H = (B-qI) S_{1} = (B-qI)_{:R}
        H = B[:, fixed_col_idx] - q_j*np.eye(n)[:, fixed_col_idx]
        squared_norm_B_c = np.linalg.norm(H) ** 2

        # Going through all rows of X cyclically
        for r in range(m):
            # V = C_{:R} - A X_{:R} + X B_{:R}
            v = C[r, fixed_col_idx] - A[r, :].dot(X[:, fixed_col_idx]) + X[r, :].dot(B[:, fixed_col_idx])

            # X = X - V (H^T H)^{-1} H^T
            X[r, :] -= (v / squared_norm_B_c) * H


        if (j % store_every == 0) or (j == n_iter):
            t_list.append(time() - t0)
            X_list.append(X)
            iter_list.append(j)
            epoch_list.append(j*sketch_frac)

        if verbose and (j % store_every == 0):
            print(f"{j:^14}|{j*sketch_frac:^14}|{time() - t0:^14.2e}")

    if verbose and not (n_iter % store_every == 0):
        print(f"{j:^14}|{j*sketch_frac:^14}|{time() - t0:^14.2e}")

    return np.array(X_list), np.array(t_list), np.array(iter_list), np.array(epoch_list)