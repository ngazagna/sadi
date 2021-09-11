import numpy as np
from scipy.stats import ortho_group  # Requires version 0.18 of scipy
from scipy.linalg import solve_sylvester

import warnings

from utils import res_norm_syl


# Functions to generate a Peaceman-Rachford ADI problem
def gen_artifical_pr(N, spectra):
    """
    Generate matrices for Peaceman-Rachford problem
    
    H and V are symmetric, positive definite and commutative matrices
    """
    np.random.seed(0)
    H, V = gen_adi_matrices(N, spectra)
    u_true = np.random.rand(N)
    b = (H + V) @ u_true
    return H, V, b, u_true

def gen_adi_matrices(N, spectra):
    U = ortho_group.rvs(dim=N)

    if spectra == "close":
        eigen_vals_H = 1e-4 + np.exp(-.01*np.arange(N)) / 1e0
        eigen_vals_V = 1.5e-4 + np.exp(-.01*np.arange(N)) / 1e0
    elif spectra == "sidebyside":
        eigen_vals_H = 1. + np.exp(-.01*np.arange(N)) / 1e0
        eigen_vals_V = .1 + np.exp(-.01*np.arange(N)) / 1e0
    elif spectra == "separated":
        eigen_vals_H = 1e3 + np.exp(-.02*np.arange(N)) / 1e-7
        eigen_vals_V = 1.5e-4 + np.exp(-.02*np.arange(N)) / 1e0
    elif spectra == "negative":
        init = 8.
        mu = .1
        power = 1e-3
        eigen_vals_H = init + np.exp(-mu*np.arange(N)) / power
        
        init = -.005
        mu = .087
        power = 1e-2
        eigen_vals_V = init - np.exp(-mu*np.arange(N)) / power

#         eigen_vals_H = 1e2 + np.exp(-.02*np.arange(N)) / 1e-5
#         eigen_vals_V = 1.5e-5 + np.exp(-.01*np.arange(N)) / 1e2

#     eigen_vals_H[-1] = 1e-4
#     eigen_vals_V = eigen_vals_H - 5e-7
        
    H = (U * eigen_vals_H) @ U.T
    V = (U * eigen_vals_V) @ U.T
    return H, V


# Chosen spectrum + symmetric + orthogonal ?
def gen_matrix_adi(n, eigen_vals=None):
    U = ortho_group.rvs(dim=n)
    if eigen_vals is None:
        eigen_vals = np.arange(1, n+1)
    return (U * eigen_vals) @ U.T


# Disjoint spectrum
def gen_adi_disjoint_spectrum(n, lmbda=.001):
    eigen_vals_H = np.exp(-.1 * np.arange(n))
    H = gen_matrix_adi(n, eigen_vals=eigen_vals_H)

    # eigen_vals_V = 2 ** (np.arange(n)) + 1000
    eigen_vals_V = 1.2 + np.exp(- np.arange(n))
    V = gen_matrix_adi(n, eigen_vals=eigen_vals_V)

    u_true = np.random.rand(n)
    b = (H + V) @ u_true
    return H, V, b, u_true

# H, V, b, u_true = gen_adi_disjoint_spectrum(n) # OLD: separated spectrum
# H, V, b, u_true = gen_adi(n) # OLD: uncontrolled spectrum


# Functions to generate a Sylvester equation problem
def gen_sylvester_chosen_spectra(n, a, b, c, d):
    r"""
    Generate Sylvester equation matrices such that:
    $AX - XB = C$
    where $A$ and $B$ are normal with disjoint spectrum
    $sp(A) = [a, b]$
    $sp(B) = [c, d]$
    """
    eigen_vals_A = np.linspace(a, b, n)
    eigen_vals_B = np.linspace(c, d, n)  # - eigen_vals_A
    U = ortho_group.rvs(dim=n)
    A = (U * eigen_vals_A) @ U.T
    B = (U * eigen_vals_B) @ U.T

    X_true = np.random.rand(n, n)

    C = A @ X_true - X_true @ B
    return A, B, C, X_true

def gen_sylvester_close_spectrum(N):
    r"""
    Generate Sylvester equation matrices such that:
    $AX - XB = C$
    where $A$ and $B$ are normal with interlaced spectrum
    $sp(A) = [1, 3, 5, ..., 2*N-1]$
    $sp(B) = [2, 4, 6, ..., 2*N]$
    """
    # interlaced
#     eigen_vals_A = np.arange(1, 2*N+1, 2)
#     eigen_vals_B = np.arange(2, 2*N+2, 2)
#     eigen_vals_A = np.arange(-1, 0, 1/N)
#     eigen_vals_B = np.arange(-0.11, 9.89, 10/N)
    # bonded
    eigen_vals_A = np.arange(-1, 0, 1/N)
    eigen_vals_B = np.arange(-1/N + 1e-6, 10 - 1/N + 1e-6, 10/N)
#     eigen_vals_B = np.geomspace(1e-6, 1000, num=N)
    U = ortho_group.rvs(dim=N)
    A = (U * eigen_vals_A) @ U.T
    B = (U * eigen_vals_B) @ U.T

    X_true = np.random.rand(N, N)

    C = A @ X_true - X_true @ B
    return A, B, C, X_true

def gen_sylvester_interlaced_matrices(N, step=50):
    r"""
    Generate Sylvester equation matrices such that:
    $AX - XB = C$
    where $A$ and $B$ are normal with interlaced spectrum
    $sp(A) = [1, 3, 5, ..., 2*N-1]$
    $sp(B) = [2, 4, 6, ..., 2*N]$
    """
    # interlaced
    l = np.arange(1, step*N + 1)
    eigen_vals_A = l[::step]
    eigen_vals_B = l[int(step/2)::step]

    U = ortho_group.rvs(dim=N)
    A = (U * eigen_vals_A) @ U.T
    B = (U * eigen_vals_B) @ U.T

    X_true = np.random.rand(N, N)

    C = A @ X_true - X_true @ B
    return A, B, C, X_true

# Poisson equation
def gen_discrete_poisson_filtered(N, omega=10, compute_sol=False):
    r"""
    Generate matrices with filtered large eigenvalues comming from the
    discretization of the Poisson equation:
    $D_2 X + X D_2^\top = F$
    which can be rewritten
    $D_2 X - X (-D_2^\top) = F$
    """
    A, B, C, _ = gen_discrete_poisson(N, omega=10, compute_sol=False)
    
    delta, S = np.linalg.eig(B)
    delta = np.sort(delta)

    # index threshold
    k = int(np.ceil(2 * N / np.pi))

    # replace eigvals with these of continuous problem 
    delta_filtered = delta.copy()
    delta_filtered[k:] = (np.pi ** 2 / 4) * np.arange(k, N) ** 2

    B_filtered = S @ np.diag(delta_filtered) @ np.linalg.inv(S)
    A_filtered = -B_filtered.T
        
    if compute_sol:
        X_sol = solve_sylvester(A_filtered, -B_filtered, C)
        if res_norm_syl(X_sol, A_filtered, B_filtered, C) >= 1e-3:
            warnings.warn("Impossible to compute solution of the Sylvester equation.")
            X_sol = np.array([])
    else:
        warnings.warn("Solution of the Sylvester equation not computed.")
        X_sol = np.array([])
    
    return A_filtered, B_filtered, C, X_sol

def gen_discrete_poisson(N, omega=10, compute_sol=False):
    r"""
    Generate matrices comming from the discretization of the Poisson equation:
    $D_2 X + X D_2^\top = F$
    which can be rewritten
    $D_2 X - X (-D_2^\top) = F$
    """
    # Chebyshev differentiation matrix of order 2 over [-1, 1]
    D_2, x = cheb_diff_mat(N+2, k=2)
    D_2 = D_2[1:-1, 1:-1]
    # x = x[1:-1]

    # discretization over [-1, 1] of the function
    # f(x, y) = cos(omega * x) * sin(omega * y)
    X, Y = np.meshgrid(x, x)
    F = poisson_func(X, Y, omega)
    F = F[1:-1, 1:-1]

    if compute_sol:
        X_sol = solve_sylvester(D_2, D_2.T, F)
        if res_norm_syl(X_sol, D_2, -D_2.T, F) >= 1e-3:
            warnings.warn("Impossible to compute solution of the Sylvester equation.")
            X_sol = np.array([])
    else:
        warnings.warn("Solution of the Sylvester equation not computed.")
        X_sol = np.array([])
    
    return D_2, -D_2.T, F, X_sol

def poisson_func(x, y, omega):
    return np.cos(omega * x) * np.sin(omega * y)


def cheb_diff_mat(N, k=2):
    """
    Chebyshev polynomial differentiation matrix.

    Adapted from
    https://github.com/nikola-m/another-chebpy/blob/master/chebPy.py

    Parameters
    ----------
    N : int
        Number of discretization points.

    k : int, default=2
        Order of the differentiation, 1 or 2.

    Returns
    -------
    D : ndarray of shape (N, N)
        Chebyshev differentiation matrix of order k over N points.

    x : ndarray of shape (N, 1)
        Chebyshev differentiation N points (which ones ?) over [-1, 1].
        From -1 to 1.

    References
    ----------
        Trefethen's 'Spectral Methods in MATLAB' book.
    """
    if k not in [1, 2]:
        raise ValueError("The order of the differentiation must be 1 or 2.")

    x = np.cos(np.pi * np.linspace(N-1, 0, N)/(N-1))
#    x[N/2]=0.0 # only when N is even!
    c = np.zeros(N)
    c[0] = 2.
    c[1:N-1] = 1.
    c[N-1] = 2.
    c = c * (-1) ** np.linspace(0, N-1, N)
    X = np.tile(x, (N, 1))
    dX = X - X.T
    D = np.dot(c.reshape(N, 1), (1./c).reshape(1, N))
    D = D / (dX+np.eye(N))
    D = D - np.diag(D.T.sum(axis=0))

    if k == 2:
        # order 2 is obtained by squaring the differentiation matrix
        D = D @ D

    return D, x
