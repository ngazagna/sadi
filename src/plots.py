import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


COLORS = px.colors.qualitative.Plotly
# blue, red, green, purple, cyan, pink, ...

LINE_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

SYMBOLS = [
    "circle",
    "square",
    "star",
    "x",
    "triangle-up",
    "pentagon",
    "cross",
]

COLOR_DICT = {
    "PR-pr_shifts": COLORS[0],
    "PR-w_shifts": COLORS[1],
    "PR-cst": COLORS[2],
    "SPR-cst-0.1": COLORS[3],
    "SPR-cst-0.25": COLORS[4],
    "BSPR-cst-0.1": COLORS[3],
    "BSPR-cst-0.25": COLORS[4],
    "ADI-mobius_shifts": COLORS[1],
    "ADI-cst-1_step_mobius": COLORS[0],
    "ADI-cst": COLORS[2],
    "BSADI-cst-0.1": COLORS[3],
    "BSADI-cst-0.25": COLORS[4],
    "BSADI-cst-0.75": COLORS[0],
    "BSADI-cst-0.9": COLORS[5],
    "BSADI-cst-1.0": COLORS[1],
    "BSADI-lstsq-cst-0.1": COLORS[3],
    "BSADI-lstsq-cst-0.9": COLORS[5],
    "BSADI-lstsq-cst-1.0": COLORS[1],
    "KCDADI-cst-0.1": COLORS[3],
    "KCDADI-cst-0.9": COLORS[5],
    "KCDADI-cst-1.0": COLORS[1],
}

LINE_DICT = {
    "PR-pr_shifts": LINE_STYLES[0],
    "PR-w_shifts": LINE_STYLES[1],
    "PR-cst": LINE_STYLES[2],
    "SPR-cst-0.1": LINE_STYLES[3],
    "SPR-cst-0.25": LINE_STYLES[4],
    "BSPR-cst-0.1": LINE_STYLES[3],
    "BSPR-cst-0.25": LINE_STYLES[4],
    "ADI-mobius_shifts": LINE_STYLES[1],
    "ADI-cst-1_step_mobius": LINE_STYLES[0],
    "ADI-cst": LINE_STYLES[2],
    "BSADI-cst-0.1": LINE_STYLES[3],
    "BSADI-cst-0.25": LINE_STYLES[4],
    "BSADI-cst-0.75": LINE_STYLES[5],
    "BSADI-cst-0.9": LINE_STYLES[5],
    "BSADI-cst-1.0": LINE_STYLES[2],
    "BSADI-lstsq-cst-0.1": LINE_STYLES[0],
    "BSADI-lstsq-cst-0.9": LINE_STYLES[2],
    "BSADI-lstsq-cst-1.0": LINE_STYLES[1],
    "KCDADI-cst-0.1": LINE_STYLES[0],
    "KCDADI-cst-0.9": LINE_STYLES[2],
    "KCDADI-cst-1.0": LINE_STYLES[1],
}

SYMBOL_DICT = {
    "PR-pr_shifts": SYMBOLS[0],
    "PR-w_shifts": SYMBOLS[1],
    "PR-cst": SYMBOLS[2],
    "SPR-cst-0.1": SYMBOLS[3],
    "SPR-cst-0.25": SYMBOLS[4],
    "BSPR-cst-0.1": SYMBOLS[3],
    "BSPR-cst-0.25": SYMBOLS[4],
    "ADI-mobius_shifts": SYMBOLS[1],
    "ADI-cst-1_step_mobius": SYMBOLS[0],
    "ADI-cst": SYMBOLS[2],
    "BSADI-cst-0.1": SYMBOLS[3],
    "BSADI-cst-0.25": SYMBOLS[4],
    "BSADI-cst-0.75": SYMBOLS[6],
    "BSADI-cst-0.9": SYMBOLS[5],
    "BSADI-cst-1.0": SYMBOLS[3],
    "BSADI-lstsq-cst-0.1": SYMBOLS[0],
    "BSADI-lstsq-cst-0.9": SYMBOLS[2],
    "BSADI-lstsq-cst-1.0": SYMBOLS[1],
    "KCDADI-cst-0.1": SYMBOLS[0],
    "KCDADI-cst-0.9": SYMBOLS[2],
    "KCDADI-cst-1.0": SYMBOLS[1],
}


def plot_conv_plotly(results, title, prob="pr", x_axis="time", save_path=None, suffix=""):
    if x_axis == "iter":
        x_label = "Iteration"
    elif x_axis == "epoch":
        x_label = "Epochs"
    elif x_axis == "time":
        x_label = "Time (s)"
    else:
        raise ValueError("x_axis not supported, must be 'iter', 'epoch' or 'time'")

    if prob == "pr":
        y_err = "$\|u^k - u^*\|_2 \ / \ \|u^0 - u^*\|_2$"
        y_res = "$\|(H+V)u^k - b \|_2 \ / \ \|(H+V)u^0 - b \|_2$"
    elif prob == "syl":
        y_err = "$\|X^{(k)} - X^*\|_2 \ / \ \|X^{(0)} - X^*\|_2$"
        y_res = "$\|A X^{(k)} - X^{(k)} B - F \|_2 \ / \ \|A X^{(0)} - X^{(0)} B - F \|_2$"
    else:
        raise ValueError("prob not supported, must be 'pr' (Peaceman-Rachford) or 'syl' (Sylvester)")

    # Plot only residuals if the relative error is not available
    fig_rel_res = go.Figure()
    fig_rel_res.update_layout(
        template="plotly_white",
        font=dict(size=20,),
    )

    # Relative residuals
    for result in results:
        if x_axis == "iter":
            x_array = result.iterations
        if x_axis == "epoch":
            x_array = result.epochs
        elif x_axis == "time":
            x_array = result.times

        fig_rel_res.add_trace(
            go.Scatter(
                x=x_array,
                y=result.rel_res,
                name=result.algo_name,
                mode="lines+markers",
                line=dict(color=COLOR_DICT[result.algo_name], dash=LINE_DICT[result.algo_name]),
                marker=dict(symbol=SYMBOL_DICT[result.algo_name], size=10),
            )
        )

    # horizontal line at y=1
#     fig_rel_res.add_hline(y=1., line_dash="dot")

    fig_rel_res.update_layout(
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title=x_label,
        yaxis_title=y_res,
        yaxis_type="log",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)"),
        legend_title="Methods",
    )

    if results[0].rel_err.size != 0:
        # Relative distances to solution
        fig_rel_err = go.Figure()
        fig_rel_err.update_layout(
            template="plotly_white",
            font=dict(size=20,),
        )
        for result in results:
            if x_axis == "iter":
                x_array = result.iterations
            if x_axis == "epoch":
                x_array = result.epochs
            elif x_axis == "time":
                x_array = result.times

            fig_rel_err.add_trace(
                go.Scatter(
                x=x_array,
                y=result.rel_err,
                name=result.algo_name,
                mode="lines+markers",
                line=dict(color=COLOR_DICT[result.algo_name], dash=LINE_DICT[result.algo_name]),
                marker=dict(symbol=SYMBOL_DICT[result.algo_name], size=10),
                )
            )

        # horizontal line at y=1
#         fig_rel_err.add_hline(y=1., line_dash="dot")

        fig_rel_err.update_layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis_title=x_label,
            yaxis_title=y_err,
            yaxis_type="log",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)"),
            legend_title="Methods",
        )

    if save_path is None:
        # Show the plot(s)
        if results[0].rel_err.size == 0:
            fig_rel_res.show()
        else:
            fig_rel_res.show()
            fig_rel_err.show()
    else:
        # Save relative residual plot
        if not suffix:
            filename = f"conv_plot_rel_res_{x_axis}.pdf"
        else:
            filename = f"conv_plot_rel_res_{x_axis}_{suffix}.pdf"
        full_path = os.path.join(save_path, filename)
        fig_rel_res.write_image(full_path)

        # Save relative error plot
        if results[0].rel_err.size != 0:
            if not suffix:
                filename = f"conv_plot_rel_err_{x_axis}.pdf"
            else:
                filename = f"conv_plot_rel_err_{x_axis}_{suffix}.pdf"
            full_path = os.path.join(save_path, filename)
            fig_rel_err.write_image(full_path)


def plot_conv(results, title, prob="pr", x_axis="time", save_path=None, suffix=""):
    if x_axis == "iter":
        x_label = "Iteration"
    elif x_axis == "epoch":
        x_label = "Epochs"
    elif x_axis == "time":
        x_label = "Time (s)"
    else:
        raise ValueError("x_axis not supported, must be 'iter', 'epoch' or 'time'")

    if prob == "pr":
        y_dist = "$\|u^k - u^*\|_2 \ / \ \|u^0 - u^*\|_2$"
        y_res = "$\|(H+V)u^k - b \|_2 \ / \ \|(H+V)u^0 - b \|_2$"
    elif prob == "syl":
        y_dist = "$\|X^{(k)} - X^*\|_2 \ / \ \|X^{(0)} - X^*\|_2$"
        y_res = "$\|A X^{(k)} - X^{(k)} B - C \|_2 \ / \ \|A X^{(0)} - X^{(0)} B - C \|_2$"
    else:
        raise ValueError("prob not supported, must be 'pr' (Peaceman-Rachford) or 'syl' (Sylvester)")

    # Plot only residuals if the relative error is not available
    if results[0].rel_err.size == 0:
        plt.figure(figsize=(10, 5))
        # Relative residuals
        for result in results:
            if x_axis == "iter":
                x_array = result.iterations
            if x_axis == "epoch":
                x_array = result.epochs
            elif x_axis == "time":
                x_array = result.times
            plt.semilogy(x_array, result.rel_res, lw=2, label=result.algo_name)

        plt.axhline(y=1.0, color='k', linestyle='--')
        plt.title(f"{title} (residuals)")
        plt.xlabel(x_label)
        plt.ylabel(y_res)
        plt.legend()

    else:
        plt.figure(figsize=(18, 5))
        # Relative distances to solution
        plt.subplot(1, 2, 1) # plot 1x2
        for result in results:
            if x_axis == "iter":
                x_array = result.iterations
            if x_axis == "epoch":
                x_array = result.epochs
            elif x_axis == "time":
                x_array = result.times

            plt.semilogy(x_array, result.rel_err, lw=2, label=result.algo_name)

        plt.axhline(y=1.0, color='k', linestyle='--')
        plt.title(f"Distance to solution")
        plt.xlabel(x_label)
        plt.ylabel(y_dist)
        plt.legend()

        # Relative residuals
        plt.subplot(1, 2, 2) # plot 2x2
        for result in results:
            if x_axis == "iter":
                x_array = result.iterations
            if x_axis == "epoch":
                x_array = result.epochs
            elif x_axis == "time":
                x_array = result.times
            plt.semilogy(x_array, result.rel_res, lw=2, label=result.algo_name)

        plt.axhline(y=1.0, color='k', linestyle='--')
        plt.title("Residuals")
        plt.xlabel(x_label)
        plt.ylabel(y_res)
        plt.legend()

#         fig.suptitle(title, fontsize=14)
        plt.suptitle(title, fontsize=14)

    if not suffix:
        filename = f"conv_plot_{x_axis}.pdf"
    else:
        filename = f"conv_plot_{x_axis}_{suffix}.pdf"

    if save_path is None:
        plt.show()
    else:
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, bbox_inches="tight")
        plt.close()


def plot_eigenvals(eig_val_M, eig_val_N, prob):
    """Plot the eigenvalues of the two squared matrices"""
    m = eig_val_M.shape[0]
    n = eig_val_N.shape[0]

    if prob == "pr":
        # Peaceman-Rachford ADI model problem
        # Solve in x: (H+V) x = b
        label_1 = "eigenvals(H)"
        label_2 = "eigenvals(V)"
    elif prob == "syl":
        # Sylvester equation ADI model problem
        # Solve in X: AX - XB = C
        label_1 = "eigenvals(A)"
        label_2 = "eigenvals(B)"
    else:
        raise ValueError("Problem name should be either 'pr' or 'syl'.")

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(np.arange(1, m+1), np.sort(eig_val_M)[::-1],
               c="blue", marker="x", alpha=0.5, label=label_1)
    ax.scatter(np.arange(1, n+1), np.sort(eig_val_N)[::-1],
               c="red", marker="+", alpha=0.5, label=label_2)
    ax.legend()