import os
import shutil
import numpy as np
import pandas as pd

from plots import plot_conv_plotly


def save(results, title, prob, exp_name):
        """Saves results and plots of an experiment"""
        if prob not in ["pr", "syl"]:
            raise ValueError("prob not supported, must be 'pr' (Peaceman-Rachford) or 'syl' (Sylvester)")

        folder = os.path.join(os.getcwd(), "outputs/" + prob)
        if not os.path.exists(folder):
            os.makedirs(folder)
        full_path = os.path.join(folder, exp_name)
                
        shutil.rmtree(full_path, ignore_errors=True) # clear old results
        os.makedirs(full_path)
        print("Full path:", full_path)

        print("Save results in csv")
        save_results(results, full_path)
        print("Save plots")
        save_plots(results, title, prob, full_path)

        
def results_to_df(results):
    """Creates a dataframe with all results.

    It returns a dataframe of group of columns of 4
    (iterations, time, relative error to the solution, relative residuals)
    corresponding to the same solver/algo.

    Empty values are filled with NaN to match the
    run with the largest number of iterations"""
    df_list = []
    
    if results[0].rel_err.size == 0:
        columns = ["iterations", "time", "rel_res"]
    else:
        columns = ["iterations", "time", "rel_err", "rel_res"]
    
    for r in results:
        columns = [r.algo_name + "|" + col for col in columns]
        if results[0].rel_err.size == 0:
            data = np.array([r.iterations, r.times, r.rel_res]).T
        else:
            data = np.array([r.iterations, r.times, r.rel_err, r.rel_res]).T
        df_list.append(pd.DataFrame(data=data, columns=columns))
    return pd.concat(df_list, axis=1)


def save_results(results, save_path):
    results_df = results_to_df(results)
    results_df.to_csv(os.path.join(save_path, "results.csv"))

    
def save_plots(results, title, prob, save_path):
    algo_names = [r.algo_name for r in results]
    mask = ["Mobius" in n for n in algo_names]
    idx_not_mobius = [i for i, x in enumerate(mask) if not x]
    
    if len(idx_not_mobius) != len(results):
        print("Save plots with methods using Mobius shifts")
        plot_conv_plotly(results, title, prob=prob, x_axis="iter", save_path=save_path, suffix="with_mobius")
        plot_conv_plotly(results, title, prob=prob, x_axis="epoch", save_path=save_path, suffix="with_mobius")
        plot_conv_plotly(results, title, prob=prob, x_axis="time", save_path=save_path, suffix="with_mobius")
    
    results_not_mobius = [results[i] for i in idx_not_mobius]
    plot_conv_plotly(results_not_mobius, title, prob=prob, x_axis="iter", save_path=save_path)
    plot_conv_plotly(results_not_mobius, title, prob=prob, x_axis="epoch", save_path=save_path)
    plot_conv_plotly(results_not_mobius, title, prob=prob, x_axis="time", save_path=save_path)
    
    
def save_poisson_filtered(A, C, X_sol):
    """Save the filtered Poisson equation matrix
    
    The Chebyshev second-order differentiation matrix
    The right-hand side matrix
    The computed solution matrix (with linalg.solve)
    
    The parameter omega is set to 10.
    """
    N = A.shape[0] # dimension

    folder = os.path.join(os.getcwd(), "datasets/syl/")
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename = "poisson_filtered_" + str(N) + ".npz"
    np.savez(os.path.join(folder, filename), A=A, C=C, X_sol=X_sol)
    
def save_poisson(A, C, X_sol):
    """Save the Poisson equation matrix
    
    The Chebyshev second-order differentiation matrix
    The right-hand side matrix
    The computed solution matrix (with linalg.solve)
    
    The parameter omega is set to 10.
    """
    N = A.shape[0] # dimension

    folder = os.path.join(os.getcwd(), "datasets/syl/")
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename = "poisson_" + str(N) + ".npz"
    np.savez(os.path.join(folder, filename), A=A, C=C, X_sol=X_sol)
    
def load_poisson_filtered(N):
    """Load filtered Poisson equation matrices saved in the datasets folder"""
    folder = os.path.join(os.getcwd(), "datasets/syl/")        
    filename = "poisson_filtered_" + str(N) + ".npz"
    data = np.load(os.path.join(folder, filename))
    return data["A"], data["C"], data["X_sol"]
    
def load_poisson(N):
    """Load Poisson equation matrices saved in the datasets folder"""
    folder = os.path.join(os.getcwd(), "datasets/syl/")        
    filename = "poisson_" + str(N) + ".npz"
    data = np.load(os.path.join(folder, filename))
    return data["A"], data["C"], data["X_sol"]
    
    
def save_artificial_pr(H, V, b, u_sol, spectra):
    """Save the artifical matrices for PR problem"""
    N = H.shape[0] # dimension

    folder = os.path.join(os.getcwd(), "datasets/pr/")
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename = f"artificial_pr_{N}_{spectra}.npz"
    np.savez(os.path.join(folder, filename), H=H, V=V, b=b, u_sol=u_sol)
    
    
def load_artificial_pr(N, spectra):
    """Load the artifical matrices for PR problem saved in the datasets folder"""
    folder = os.path.join(os.getcwd(), "datasets/pr/")
    filename = f"artificial_pr_{N}_{spectra}.npz"
    data = np.load(os.path.join(folder, filename))
    return data["H"], data["V"], data["b"], data["u_sol"]


def save_sylvester_matrices(A, B, C, X_sol, name):
    """Save the Sylvester matrices
    
    AX - XB = C

    The computed solution matrix (with linalg.solve)
    """
    N = A.shape[0] # dimension

    folder = os.path.join(os.getcwd(), "datasets/syl/")
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename = f"sylvester_matrix_{name}_{N}.npz"
    np.savez(os.path.join(folder, filename), A=A, B=B, C=C, X_sol=X_sol)
    
def load_sylvester_matrices(N, name):
    """Load Sylvester matrices in the datasets folder"""
    folder = os.path.join(os.getcwd(), "datasets/syl/")        
    filename = f"sylvester_matrix_{name}_{N}.npz"
    data = np.load(os.path.join(folder, filename))
    return data["A"], data["B"], data["C"], data["X_sol"]
    