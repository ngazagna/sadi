import numpy as np
from math import ceil, pi
from scipy.special import ellipk, ellipj
from mpmath import mp


## Shift parameters for PR problem

def shifts_pr(a, b, n_iter):
    """
    Compute Peaceman-Rachford (1955) shift parameters.

    From Section 1.3 in Wachspress's book:
    p_j = b (a/b)^((2j-1)/2J)
    where J is the number of iterations
    and j the index of current step

    Parameters
    ----------
    a : float
        Lower bound of the smallest eigenvalue of H and V.
        Should be positive, since H and V are assumed to be SPD
    b : float
        Upper bound of the largest eigenvalue of H and V.
        Should be positive
    n_iter : int
        Number of iterations

    Returns
    -------
    p : numpy.ndarray
        Array containing shift parameters

    References
    ----------
    Wachspress, "ADI Model Problem". 2013
    """
    p = b * (a/b) ** ((2*np.arange(1, n_iter+1) - 1) / (2*n_iter))
    return p, p

def shifts_w(a, b, n_iter):
    """
    Compute Wachspress (1957) shift parameters.

    From Section 1.3 in Wachspress's book:
    p_j = b (a/b)^((j-1)/(J-1))
    where J is the number of iterations
    and j the index of current step

    Parameters
    ----------
    a : float
        Lower bound of the smallest eigenvalue of H and V.
        Should be positive, since H and V are assumed to be SPD.
    b : float
        Upper bound of the largest eigenvalue of H and V.
        Should be positive.
    n_iter : int
        Number of iterations.

    Returns
    -------
    p : numpy.ndarray
        Array containing shift parameters

    References
    ----------
    Wachspress, "ADI Model Problem". 2013
    """
    p = b * (a/b) ** ((np.arange(1, n_iter+1) - 1) / (n_iter - 1))
    return p, p


## Shift parameters for Sylvester problem

def shifts_adi_syl(a, b, c, d, tol):
    """
    Return shifts for AX - XB = C when the eigenvalues of A (B) 
    are in [a, b] and the eigenvalues of B (A) are in [c, d]. 

    WLOG, we require that a<b<c<d and 0<tol<1.

    /!\ WARNING: signs of p and q depend on the order
    of the half ADI steps. See commented code at the
    end of this function.
    
    Parameters
    ----------
    a : float
        Smallest eigenvalue of A
    b : float
        Largest eigenvalue of A
    c : float
        Smallest eigenvalue of B
    d : float
        Largest eigenvalue of B

    Returns
    -------
    p : numpy.ndarray
        Array containing shift parameters for the
        first half-step
    q : numpy.ndarray
        Array containing shift parameters for the
        second half-step
    J : int
        Number of iterations

    References
    ----------
    D. Fortunato and A. Townsend, Fast Poisson Solvers For Spectral Methods (2017)
    """
    gamma = (c-a)*(d-b)/(c-b)/(d-a) # cross-ratio of a, b, c, d
    
    # Calculate Mobius transform T :{-alpha, -1, 1, alpha} -> {a, b, c, d} for some alpha:
    alpha = -1 + 2*gamma + 2*np.sqrt(gamma**2 - gamma)  # Mobius exists with this t
    A = np.linalg.det(np.array([[-a*alpha, a, 1], [-b, b, 1], [c, c, 1]]))  # determinant formulae for Mobius
    B = np.linalg.det(np.array([[-a*alpha, -alpha, a], [-b, -1, b], [c, 1, c]]))
    C = np.linalg.det(np.array([[-alpha, a, 1], [-1, b, 1], [1, c, 1]]))
    D = np.linalg.det(np.array([[-a*alpha, -alpha, 1], [-b, -1, 1], [c, 1, 1]]))
    T = lambda z: (A*z + B) / (C*z + D)  # Mobius transform
    
    J = int(ceil( np.log(16*gamma)*np.log(4/tol)/(pi**2) ))  # number of ADI iterations
    
    if alpha < 1e7:
        K = ellipk(1 - 1/(alpha**2))
        _, _, dn, _ = ellipj(np.arange(.5, J-.5 + 1) * K/J, 1 - 1/(alpha**2)) 
    else:  
        # Prevent underflow when alpha large
        K = (2*np.log(2) + np.log(alpha)) + (-1 + 2*np.log(2) + np.log(alpha))/(4*alpha**2)
        m1 = 1/(alpha**2)
        u = np.arange(.5, J-.5 + 1) * K/J
        
        sech_vec = np.vectorize(mp.sech)
        sinh_vec = np.vectorize(mp.sinh)
        cosh_vec = np.vectorize(mp.cosh)
        tanh_vec = np.vectorize(mp.tanh)
        
        dn = sech_vec(u) + .25*m1*(sinh_vec(u) * cosh_vec(u) + u) * tanh_vec(u) * sech_vec(u)
        dn = np.array(dn.tolist(), dtype=float)

    # If solving for X_{j+1/2}
    # X_{j+1/2} (B − p_j I) = F − (A − p_j I) X_j
    # Then solving for X_{j+1}
    # (A − q_j I)X_{j+1} = F − X_{j+1/2} (B − q_j I)
    # Shifts are
#     p, q = T(-alpha*dn), T(alpha*dn)  # ADI shifts for [a, b] & [c, d]
    
    # If solving for X_{j+1/2}
    # (A − p_j I)X_{j+1/2} = F − X_j (B − p_j I)
    # Then solving for X_{j+1}
    # X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}
    # Shifts are
    p, q = T(alpha*dn), T(-alpha*dn)  # ADI shifts for [a, b] & [c, d]
    
    return p, q, J


def shifts_adi_syl_one_iter(a, b, c, d, tol):
    """
    Return shifts for AX - XB = C when the eigenvalues of A (B) 
    are in [a, b] and the eigenvalues of B (A) are in [c, d] 
    for one step of ADI method.

    WLOG, we require that a<b<c<d and 0<tol<1.

    /!\ WARNING: signs of p and q depend on the order
    of the half ADI steps. See commented code at the
    end of this function.
    
    Parameters
    ----------
    a : float
        Smallest eigenvalue of A
    b : float
        Smallest eigenvalue of A
    c : float
        Smallest eigenvalue of B
    d : float
        Smallest eigenvalue of B

    Returns
    -------
    p : float
        Optimal shift parameter for the first half-step
    q : float
        Optimal shift parameter for the second half-step
    """
    gamma = (c-a)*(d-b)/(c-b)/(d-a) # cross-ratio of a, b, c, d
    
    # Calculate Mobius transform T :{-alpha, -1, 1, alpha} -> {a, b, c, d} for some alpha:
    alpha = -1 + 2*gamma + 2*np.sqrt(gamma**2 - gamma)  # Mobius exists with this t
    A = np.linalg.det(np.array([[-a*alpha, a, 1], [-b, b, 1], [c, c, 1]]))  # determinant formulae for Mobius
    B = np.linalg.det(np.array([[-a*alpha, -alpha, a], [-b, -1, b], [c, 1, c]]))
    C = np.linalg.det(np.array([[-alpha, a, 1], [-1, b, 1], [1, c, 1]]))
    D = np.linalg.det(np.array([[-a*alpha, -alpha, 1], [-b, -1, 1], [c, 1, 1]]))
    T = lambda z: (A*z + B) / (C*z + D)  # Mobius transform
    
    J = 1  # number of ADI iterations
    
    if alpha < 1e7:
        K = ellipk(1 - 1/(alpha**2))
        _, _, dn, _ = ellipj(.5 * K, 1 - 1/(alpha**2)) 
    else:  
        # Prevent underflow when alpha large
        K = (2*np.log(2) + np.log(alpha)) + (-1 + 2*np.log(2) + np.log(alpha))/(4*alpha**2)
        m1 = 1/(alpha**2)
        u = .5 * K
        
#         sech_vec = np.vectorize(mp.sech)
#         sinh_vec = np.vectorize(mp.sinh)
#         cosh_vec = np.vectorize(mp.cosh)
#         tanh_vec = np.vectorize(mp.tanh)
        
        dn = mp.sech(u) + .25*m1*(mp.sinh(u) * mp.cosh(u) + u) * mp.tanh(u) * mp.sech(u)
        dn = float(dn)
        
    # If solving for X_{j+1/2}
    # X_{j+1/2} (B − p_j I) = F − (A − p_j I) X_j
    # Then solving for X_{j+1}
    # (A − q_j I)X_{j+1} = F − X_{j+1/2} (B − q_j I)
    # Shifts are
#     p, q = T(-alpha*dn), T(alpha*dn)  # ADI shifts for [a, b] & [c, d]
    
    # If solving for X_{j+1/2}
    # (A − p_j I)X_{j+1/2} = F − X_j (B − p_j I)
    # Then solving for X_{j+1}
    # X_{j+1} (B − q_j I) = F − (A − q_j I) X_{j+1/2}
    # Shifts are
    p, q = T(alpha*dn), T(-alpha*dn)  # ADI shifts for [a, b] & [c, d]
    
    return p, q



















