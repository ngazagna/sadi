import numpy as np
from abc import ABC, abstractmethod

from utils import rel_err_pr, rel_res_pr, rel_err_syl, rel_res_syl


class Result(ABC):
    """Abstract class defining a result object"""

    def __init__(self, algo_name, iterates, times, iterations, epochs):
        self.algo_name = algo_name # name of the solver
        self.iterates = iterates # array of iterates
        self.times = times # array of times
        self.iterations = iterations # array of iterations at which iterates and times are stored
        self.epochs = epochs # array of epochs

        self.rel_err = np.array([]) # array of relative distances to optimum
        self.rel_res = np.array([]) # array of relative residuals

    @abstractmethod
    def compute_errors(self):
        """Compute relative residuals and relative errors to the solution if provided 

        Update:
        --------
            Fill self.rel_res and (if solution is provided) self.rel_err
        """
        pass

class ResultPR(Result):
    """
    Result class flexible for both ADI real problem:
    (H + V)u = b
    """
    def __init__(self, algo, iterates, times, iterations, epochs, u_true):
#         u_true = np.array([])  # by default ?
        super().__init__(algo, iterates, times, iterations, epochs)
        self.u_true = u_true

    def compute_errors(self, H, V, b):
        if self.u_true.size == 0:
            self.rel_res = rel_res_pr(self.iterates, H, V, b)
        else:
            self.rel_err, self.rel_res = rel_err_pr(self.iterates, H, V, b, self.u_true)
        

class ResultSyl(Result):
    """
    Result class flexible for both Sylvester equation:
    AX - XB = C
    """
    def __init__(self, algo, iterates, times, iterations, epochs, X_true):
#         X_true = np.array([])  # by default ?
        super().__init__(algo, iterates, times, iterations, epochs)
        self.X_true = X_true

    def compute_errors(self, A, B, C):
        "For Sylvester equation AX - XB = C"
        if self.X_true.size == 0:
            self.rel_res = rel_res_syl(self.iterates, A, B, C)
        else:
            self.rel_err, self.rel_res = rel_err_syl(self.iterates, A, B, C, self.X_true)

    def compute_residuals(self, A, B, C):
        "For Sylvester equation AX - XB = C"
        self.rel_res = rel_res_syl(self.iterates, A, B, C)