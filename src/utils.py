import numpy as np


## General tools compute errors

def rel_err_pr(u_array, H, V, b, u_true):
    """Relative errors for PR ADI problem"""
    err_init = np.linalg.norm(u_array[0] - u_true, 2)
    rel_err_iter = np.array([np.linalg.norm(u - u_true, 2) / err_init for u in u_array])

    rel_residual = rel_res_pr(u_array, H, V, b)
    return rel_err_iter, rel_residual

def rel_res_pr(u_array, H, V, b):
    """Relative residuals for PR ADI problem"""
    residual_init = res_norm_pr(u_array[0], H, V, b)
    rel_residual = np.array([res_norm_pr(u, H, V, b) / residual_init for u in u_array])
    return rel_residual

def res_norm_pr(u, H, V, b):
    """2-norm of (H + V)u - b"""
    return np.linalg.norm((H+V) @ u - b, 2)


def rel_err_syl(X_array, A, B, C, X_true):
    """Relative errors for Sylvester ADI problem"""
    err_init = np.linalg.norm(X_array[0] - X_true, 2)
    rel_err_iter = np.array([np.linalg.norm(X - X_true, 2) / err_init for X in X_array])

    rel_residual = rel_res_syl(X_array, A, B, C)
    return rel_err_iter, rel_residual

def rel_res_syl(X_array, A, B, C):
    """Relative residuals for Sylvester ADI problem"""
    residual_init = res_norm_syl(X_array[0], A, B, C)
    rel_residual = np.array([res_norm_syl(X, A, B, C) / residual_init for X in X_array])
    return rel_residual

def res_norm_syl(X, A, B, C):
    """2-norm of AX - XB - C"""
    return np.linalg.norm(A @ X - X @ B - C, 2) # largest singular value


## General tool functions

def is_symmetric(M, rtol=1e-05, atol=1e-08):
    return np.allclose(M, M.T, rtol=rtol, atol=atol)

def is_pos_def(M):
    return np.all(np.linalg.eigvalsh(M) > 0)

def is_normal(M, rtol=1e-05, atol=1e-08):
    return np.allclose(M @ M.T, M.T @ M, rtol=rtol, atol=atol)

def is_commutative(H, V, rtol=1e-05, atol=1e-08):
    return np.allclose(H @ V, V @ H, rtol=rtol, atol=atol)
