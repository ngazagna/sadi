import numpy as np
import scipy as sp
from time import time

import warnings


# ADI solvers for Sylvester equation

def adi_syl(A, B, C, n_iter=10, X_init=None, p=None, q=None,
            store_every=1, verbose=True):
    """
    Vanilla ADI solver for solving Sylvester equation:
    AX - XB = C

    First solves for X_{j+1/2}
    (A − p_j I) X_{j+1/2} = F − X_j (B − p_j I)
    Then solving for X_{j+1}
    X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}

    Returns an array of iterates and of elapsed time.
    """
    m = A.shape[0]
    n = B.shape[0]
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

    X = np.zeros((m, n)) if X_init is None else X_init
    X_list = [X]

    if verbose:
        print("--------------------------------------------")
        print("  Iteration   |    Epoch     |   Time (s)   ")
        print("--------------------------------------------")

    t_list = [0.]
    iter_list = [0]
    t0 = time()
    for j in range(1, n_iter+1):
        p_j, q_j = p[j-1], q[j-1]

        # First half-step
        # A - pI
        left = A - p_j*np.eye(m)
        # X_{j} @ (B - pI) + C
        right = X @ (B - p_j*np.eye(n)) + C
        # X_{j+1/2} = solution in X of (A - pI) X = X_{j} (B - pI) + C
        X = sp.linalg.solve(left, right)

        # Second half-step
        # B - qI
        left = B - q_j*np.eye(n)
        # (A - qI) @ X_{j+1/2} - C
        right = (A - q_j*np.eye(m)) @ X - C
        # X_{j} = solution in X of X @ (B - qI) = (A - qI) @ X_{j+1/2} - C
        X = sp.linalg.solve(left.T, right.T).T

        if (j % store_every == 0) or (j == n_iter):
            t_list.append(time() - t0)
            X_list.append(X)
            iter_list.append(j)

        if verbose and (j % store_every == 0):
            print(f"{j:^14}|{j:^14}|{time() - t0:^14.2e}")

    if verbose and not (n_iter % store_every == 0):
        print(f"{j:^14}|{j:^14}|{time() - t0:^14.2e}")

    epoch_list = iter_list # 1 epoch per iteration for full ADI
    return np.array(X_list), np.array(t_list), np.array(iter_list), np.array(epoch_list)


def sap_adi_syl(A, B, C, n_iter=10, X_init=None, p=None, q=None,
                sketch_size=None, sketch_frac=None, store_every=1, assume_a="pos", verbose=True):
    """
    Sketch-and-Project (SAP) ADI solver for solving Sylvester equation:
    AX - XB = C

    First solves for X_{j+1/2} using the Sketched-and-Project method
    (A − p_j I) X_{j+1/2} = F − X_j (B − p_j I)
    Then solving for X_{j+1} using the Sketched-and-Project method
    X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}

    With block row / column sketching.

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
        sampled_idx = np.random.choice(m, size=sketch_size_A, replace=False)
        # G = S_{1/2}^T (A-pI) = (A-pI)_{R:}
        G = A[sampled_idx, :] - p_j*np.eye(m)[sampled_idx, :]
        # U = C_{R:} - A_{R:} X + X_{R:} B
        U = C[sampled_idx, :] - A[sampled_idx, :] @ X + X[sampled_idx, :] @ B

        # X = X + G^T (G G^T)^{-1} U
#         if np.any(np.linalg.eigvalsh(G @ G.T) < 1e-3):
#             print("---> G @ G.T is NOT PD\n")
#             print("min eigval = ", min(np.linalg.eigvalsh(G @ G.T)), "\n")
#         X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a="pos")
        X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a=assume_a)

        # Second half-step
        sampled_idx = np.random.choice(n, size=sketch_size_B, replace=False)
        # H = (B-qI) S_{1} = (B-qI)_{:R}
        H = B[:, sampled_idx] - q_j*np.eye(n)[:, sampled_idx]
        # V = C_{:R} - A X_{:R} + X B_{:R}
        V = C[:, sampled_idx] - A @ X[:, sampled_idx] + X @ B[:, sampled_idx]

        # X = X - V (H^T H)^{-1} H^T
        # X = X - (H (H^T H)^{-1} V^T)^T
#         if np.any(np.linalg.eigvalsh(H.T @ H) < 1e-3):
#             print("---> H.T @ H is NOT PD\n")
#             print("min eigval = ", min(np.linalg.eigvalsh(H.T @ H)), "\n")
#         X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a="pos")
        # TODO: which update is faster ?
#         X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a=assume_a)
        X = X - (H @ sp.linalg.solve(H.T @ H, V.T, assume_a=assume_a)).T

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


def sap_adi_syl_sym(A, B, C, n_iter=10, X_init=None, p=None, q=None,
                    sketch_size=None, store_every=1, assume_a="pos", verbose=True):
    """
    Sketch-and-Project (SAP) ADI solver for solving Sylvester equation:
    AX - XB = C

    First solves for X_{j+1/2} using the Sketched-and-Project method
    (A − p_j I) X_{j+1/2} = F − X_j (B − p_j I)
    Then solving for X_{j+1} using the Sketched-and-Project method
    X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}

    With block row / column sketching.

    Returns an array of iterates and of elapsed time.
    """
    m = A.shape[0]
    n = B.shape[0]
    min_size = min(m, n)
    # n_iter = int(n_iter) # if n_iter is not an int convert it

    if store_every <= 0:
        store_every = 1
        warnings.warn("store_every set to 1.")

    if sketch_size is None:
        print("Default sketch size equals 90% of the rows\n")
        sketch_size = int(.9 * min_size) # default 90% of the rows

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

        # Same sampled indices for both half-steps
        sampled_idx = np.random.choice(m, size=sketch_size, replace=False)

        # First half-step
        # G = S_{1/2}^T (A-pI) = (A-pI)_{R:}
        G = A[sampled_idx, :] - p_j*np.eye(m)[sampled_idx, :]
        # U = C_{R:} - A_{R:} X + X_{R:} B
        U = C[sampled_idx, :] - A[sampled_idx, :] @ X + X[sampled_idx, :] @ B
        # X = X + G^T (G G^T)^{-1} U
#         if np.any(np.linalg.eigvalsh(G @ G.T) < 1e-3):
#             print("---> G @ G.T is NOT PD\n")
#             print("min eigval = ", min(np.linalg.eigvalsh(G @ G.T)), "\n")
#         X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a="pos")
        X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a=assume_a)

        # Second half-step
        # H = (B-qI) S_{1} = (B-qI)_{:R}
        H = B[:, sampled_idx] - q_j*np.eye(n)[:, sampled_idx]
        # V = C_{:R} - A X_{:R} + X B_{:R}
        V = C[:, sampled_idx] - A @ X[:, sampled_idx] + X @ B[:, sampled_idx]
        # X = X - V (H^T H)^{-1} H^T
#         if np.any(np.linalg.eigvalsh(H.T @ H) < 1e-3):
#             print("---> H.T @ H is NOT PD\n")
#             print("min eigval = ", min(np.linalg.eigvalsh(H.T @ H)), "\n")
#         X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a="pos")
        X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a=assume_a)

        if (j % store_every == 0) or (j == n_iter):
            t_list.append(time() - t0)
            X_list.append(X)
            iter_list.append(j)
            epoch_list.append(j*sketch_size/min_size)

        if verbose and (j % store_every == 0):
            print(f"{j:^14}|{j*sketch_size/min_size:^14}|{time() - t0:^14.2e}")

    if verbose and not (n_iter % store_every == 0):
        print(f"{j:^14}|{j*sketch_size/min_size:^14}|{time() - t0:^14.2e}")

    return np.array(X_list), np.array(t_list), np.array(iter_list), np.array(epoch_list)


def sap_adi_syl_decrease(A, B, C, lr=.999, n_iter=10, X_init=None, p=None, q=None,
                         sketch_size=None, store_every=1, assume_a="pos", verbose=True):
    """
    Sketch-and-Project (SAP) ADI solver for solving Sylvester equation:
    AX - XB = C

    with decreasing shifts through a multiplicative learning rate.

    First solves for X_{j+1/2} using the Sketched-and-Project method
    (A − p_j I) X_{j+1/2} = F − X_j (B − p_j I)
    Then solving for X_{j+1} using the Sketched-and-Project method
    X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}

    With block row / column sketching.

    Returns an array of iterates and of elapsed time.
    """
    m = A.shape[0]
    n = B.shape[0]
    min_size = min(m, n)
    # n_iter = int(n_iter) # if n_iter is not an int convert it

    if store_every <= 0:
        store_every = 1
        warnings.warn("store_every set to 1.")

    if sketch_size is None:
        print("Default sketch size equals 90% of the rows\n")
        sketch_size = int(.9 * min_size) # default 90% of the rows

    if p is None:
        p = np.zeros(n_iter)
    if q is None:
        q = np.zeros(n_iter)

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
        p_j = p * (lr**(j-1))
        q_j = -p_j

        # First half-step
        sampled_idx = np.random.choice(m, size=sketch_size, replace=False)
        # G = S_{1/2}^T (A-pI) = (A-pI)_{R:}
        G = A[sampled_idx, :] - p_j*np.eye(m)[sampled_idx, :]
        # U = C_{R:} - A_{R:} X + X_{R:} B
        U = C[sampled_idx, :] - A[sampled_idx, :] @ X + X[sampled_idx, :] @ B
        # X = X + G^T (G G^T)^{-1} U
#         if np.any(np.linalg.eigvalsh(G @ G.T) < 1e-3):
#             print("---> G @ G.T is NOT PD\n")
#             print("min eigval = ", min(np.linalg.eigvalsh(G @ G.T)), "\n")
#         X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a="pos")
        X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a=assume_a)

        # Second half-step
        sampled_idx = np.random.choice(n, size=sketch_size, replace=False)
        # H = (B-qI) S_{1} = (B-qI)_{:R}
        H = B[:, sampled_idx] - q_j*np.eye(n)[:, sampled_idx]
        # V = C_{:R} - A X_{:R} + X B_{:R}
        V = C[:, sampled_idx] - A @ X[:, sampled_idx] + X @ B[:, sampled_idx]
        # X = X - V (H^T H)^{-1} H^T
#         if np.any(np.linalg.eigvalsh(H.T @ H) < 1e-3):
#             print("---> H.T @ H is NOT PD\n")
#             print("min eigval = ", min(np.linalg.eigvalsh(H.T @ H)), "\n")
#         X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a="pos")
        X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a=assume_a)

        if (j % store_every == 0) or (j == n_iter):
            t_list.append(time() - t0)
            X_list.append(X)
            iter_list.append(j)
            epoch_list.append(j*sketch_size/min_size)

        if verbose and (j % store_every == 0):
            print(f"{j:^14}|{j*sketch_size/min_size:^14}|{time() - t0:^14.2e}")

    if verbose and not (n_iter % store_every == 0):
        print(f"{j:^14}|{j*sketch_size/min_size:^14}|{time() - t0:^14.2e}")

    return np.array(X_list), np.array(t_list), np.array(iter_list), np.array(epoch_list)


# def sap_adi_syl_adaptative(A, B, C, lr_init=.999, n_iter=10, X_init=None, p=None, q=None,
#                            sketch_size=None, store_every=1, assume_a="pos", verbose=True):
#     """
#     Sketch-and-Project (SAP) ADI solver for solving Sylvester equation:
#     AX - XB = C

#     with decreasing shifts through a multiplicative learning rate.

#     First solves for X_{j+1/2} using the Sketched-and-Project method
#     (A − p_j I) X_{j+1/2} = F − X_j (B − p_j I)
#     Then solving for X_{j+1} using the Sketched-and-Project method
#     X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}

#     With block row / column sketching.

#     Returns an array of iterates and of elapsed time.
#     """
#     m = A.shape[0]
#     n = B.shape[0]
#     min_size = min(m, n)

#     if store_every <= 0:
#         store_every = 1
#         warnings.warn("store_every set to 1.")

#     if sketch_size is None:
#         print("Default sketch size equals 90% of the rows\n")
#         sketch_size = int(.9 * min_size) # default 90% of the rows

#     if p is None:
#         p = np.zeros(n_iter)
#     if q is None:
#         q = np.zeros(n_iter)

#     X = np.zeros((m, n)) if X_init is None else X_init
#     X_list = [X]

#     if verbose:
#         print("--------------------------------------------")
#         print("  Iteration   |    Epoch     |   Time (s)   ")
#         print("--------------------------------------------")

#     t_list = [0.]
#     iter_list = [0]
#     epoch_list = [0]
#     t0 = time()
#     for j in range(1, n_iter+1):

#         while delta >= 0:
#             p_j = p * (lr**(j-1))
#             q_j = -p_j

#         # First half-step
#         sampled_idx = np.random.choice(m, size=sketch_size, replace=False)
#         # G = S_{1/2}^T (A-pI) = (A-pI)_{R:}
#         G = A[sampled_idx, :] - p_j*np.eye(m)[sampled_idx, :]
#         # U = C_{R:} - A_{R:} X + X_{R:} B
#         U = C[sampled_idx, :] - A[sampled_idx, :] @ X + X[sampled_idx, :] @ B
#         # X = X + G^T (G G^T)^{-1} U
#         X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a=assume_a)

#         # Second half-step
#         sampled_idx = np.random.choice(n, size=sketch_size, replace=False)
#         # H = (B-qI) S_{1} = (B-qI)_{:R}
#         H = B[:, sampled_idx] - q_j*np.eye(n)[:, sampled_idx]
#         # V = C_{:R} - A X_{:R} + X B_{:R}
#         V = C[:, sampled_idx] - A @ X[:, sampled_idx] + X @ B[:, sampled_idx]
#         # X = X - V (H^T H)^{-1} H^T
#         X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a=assume_a)

#         if (j % store_every == 0) or (j == n_iter):
#             t_list.append(time() - t0)
#             X_list.append(X)
#             iter_list.append(j)
#             epoch_list.append(j*sketch_size/min_size)

#         if verbose and (j % store_every == 0):
#             print(f"{j:^14}|{j*sketch_size/min_size:^14}|{time() - t0:^14.2e}")

#     if verbose and not (n_iter % store_every == 0):
#         print(f"{j:^14}|{j*sketch_size/min_size:^14}|{time() - t0:^14.2e}")

#     return np.array(X_list), np.array(t_list), np.array(iter_list), np.array(epoch_list)


def bsadi_lstsq(A, B, C, n_iter=10, X_init=None, p=None, q=None,
                sketch_size=None, sketch_frac=None, store_every=1, assume_a="pos", verbose=True):
    """
    Sketch-and-Project (SAP) ADI solver for solving Sylvester equation:
    AX - XB = C

    First solves for X_{j+1/2} using the Sketched-and-Project method
    (A − p_j I) X_{j+1/2} = F − X_j (B − p_j I)
    Then solving for X_{j+1} using the Sketched-and-Project method
    X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}

    With block row / column sketching.

    Uses an update of sketch-and-project that should be faster.

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
        sampled_idx = np.random.choice(m, size=sketch_size_A, replace=False)
        # G = S_{1/2}^T (A-pI) = (A-pI)_{R:}
        G = A[sampled_idx, :] - p_j*np.eye(m)[sampled_idx, :]
        # U = C_{R:} - A_{R:} X + X_{R:} B
        U = C[sampled_idx, :] - A[sampled_idx, :] @ X + X[sampled_idx, :] @ B

        # X = X + G^T (G G^T)^{-1} U
        # Robert's "smart update"
#         X = X + G.T @ sp.linalg.solve(G @ G.T, U, assume_a=assume_a)

        # Using the least norm solution: DIVERGES
        Z, _, _, _ = np.linalg.lstsq(G, U, rcond=None)

        # 5 steps of CG: rhs must be a vector
        # Z, _ = sp.sparse.linalg.cg(G, U, x0=X, maxiter=5)
        X = X + Z

        # Second half-step
        sampled_idx = np.random.choice(n, size=sketch_size_B, replace=False)
        # H = (B-qI) S_{1} = (B-qI)_{:R}
        H = B[:, sampled_idx] - q_j*np.eye(n)[:, sampled_idx]
        # V = C_{:R} - A X_{:R} + X B_{:R}
        V = C[:, sampled_idx] - A @ X[:, sampled_idx] + X @ B[:, sampled_idx]

        # X = X - V (H^T H)^{-1} H^T
        # X = X - (H (H^T H)^{-1} V)^T
#         X = X - V @ sp.linalg.solve(H.T @ H, H.T, assume_a=assume_a)
#         X = X - (H @ sp.linalg.solve(H.T @ H, V.T, assume_a=assume_a)).T

        # Using the least norm solution: DIVERGES
        Z, _, _, _ = np.linalg.lstsq(H.T, V.T, rcond=None)

        # 5 steps of CG: rhs must be a vector
        # Z, _ = sp.sparse.linalg.cg(H.T, V.T, x0=X, maxiter=5)
        X = X - Z

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
