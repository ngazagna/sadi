import os
from os.path import join
import numpy as np
from scipy.io import mmread
from scipy.linalg import solve_sylvester


def load_lanpro(dir_path="./datasets/", dense=False):
    """
    Load LANPRO matrices for Sylvester equation

    Sylvester equation matrices such that:
    $AX - XB = C$
    where A, B and C are mxm, nxn and mxn matrices.
    Moreover, C = G F^T, where both G and F are random vectors.

    Parameters
    ----------
    dir_path : string
        Path to the nos6.mtx and nos5.mtx files.
        Default is the subfolder 'datasets'

    dense : boolean
        Default is False, in this case A and B are resp. CSR and CSC matrices
        If True, A and B are converted to dense numpy arrays

    Returns
    -------
    A : scipy.sparse_matrix.csr_matrix or numpy.ndarray
        First matrix of the Sylvester equation
        A (675×675) is NOS6.

    B : scipy.sparse_matrix.csc_matrix or numpy.ndarray
        Second matrix of the Sylvester equation
        B (468×468) is negative NOS5

    C : numpy.ndarray
        Right-hand side matrix of the Sylvester equation
        Randomly generated with set seed

    X_true : numpy.ndarray
        Solution of this Sylvester equation

    References
    ----------
    Benner, Li, Truhar. "On the ADI method for Sylvester equations". 2009
    """
    # Loading the matrices
    A = mmread(join(dir_path, "nos6.mtx"))
    B = -mmread(join(dir_path, "nos5.mtx"))

    folder = os.path.join(os.getcwd(), "datasets/syl/")
    filename_rhs = "lanpro_rhs.npy"
    filename_sol = "lanpro_solution.npy"
    path_rhs = os.path.join(folder, filename_rhs)
    path_sol = os.path.join(folder, filename_sol)

    if os.path.exists(path_rhs) and os.path.exists(path_sol):
        # Load right-hand side term and the solution if available
        C, X_true = load_lanpro_rhs_and_sol()
        print("Matrices loaded from\n", path_rhs, "\n", path_sol)
    else:
        # Else generate and save both of them
        m = A.shape[0]
        n = B.shape[0]

        np.random.seed(0)
        # Random vectors sampled from uniform distribution
        # G = np.random.rand(m, 1)
        # F = np.random.rand(n, 1)

        # Random vectors sampled from normal distribution
        G = np.random.randn(m, 1)
        F = np.random.randn(n, 1)
        C = G @ F.T

        X_true = solve_sylvester(A.toarray(), -B.toarray(), C)
        save_lanpro_rhs_and_sol(C, X_true)
        print("Matrices saved in\n", path_rhs, "\n", path_sol)
        
    # Setting the right format dense or sparse for A and B
    if dense:
        A = A.toarray()
        B = B.toarray()
    else:
        A = A.tocsr()
        B = B.tocsc()

    return A, B, C, X_true


def save_lanpro_rhs_and_sol(C, X):
    """Save the right-hand side and solution of Lanpro Sylvester equation"""
    folder = os.path.join(os.getcwd(), "datasets/syl/")
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = "lanpro_rhs.npy"
    np.save(os.path.join(folder, filename), C)
    
    filename = "lanpro_solution.npy"
    np.save(os.path.join(folder, filename), X)
    
    
def load_lanpro_rhs_and_sol():
    """Load the right-hand side and the solution of Lanpro Sylvester equation saved in the datasets folder"""
    folder = os.path.join(os.getcwd(), "datasets/syl/")
    
    filename = "lanpro_rhs.npy"
    C = np.load(os.path.join(folder, filename))
    
    filename = "lanpro_solution.npy"
    X_sol = np.load(os.path.join(folder, filename))
    return C, X_sol
