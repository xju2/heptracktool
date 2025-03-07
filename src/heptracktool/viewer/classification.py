"""Plot the performance for classification tasks"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics

fontsize = 16
minor_size = 14


def plot_metrics(
    odd,
    tdd,
    odd_th=0.5,
    tdd_th=0.5,
    outname: Optional[str] = "roc_graph_nets",
    off_interactive=False,
    alternative=True,
    target_efficiency=0.99,
):
    if off_interactive:
        plt.ioff()

    y_pred, y_true = (odd > odd_th), (tdd > tdd_th)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, odd)

    if alternative:
        results = []
        labels = [
            "Accuracy:           ",
            "Precision (purity): ",
            "Recall (efficiency):",
        ]
        thresholds = [0.1, 0.5, 0.8]

        for threshold in thresholds:
            y_p, y_t = (odd > threshold), (tdd > threshold)
            accuracy = sklearn.metrics.accuracy_score(y_t, y_p)
            precision = sklearn.metrics.precision_score(y_t, y_p)
            recall = sklearn.metrics.recall_score(y_t, y_p)
            results.append((accuracy, precision, recall))

        print("{:25.2f} {:7.2f} {:7.2f}".format(*thresholds))
        for idx, lab in enumerate(labels):
            print("{} {:6.4f} {:6.4f} {:6.4f}".format(lab, *[x[idx] for x in results]))

    else:
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
        print("Accuracy:            %.6f" % accuracy)
        print("Precision (purity):  %.6f" % precision)
        print("Recall (efficiency): %.6f" % recall)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    ax0, ax1, ax2, ax3 = axs

    # Plot the model outputs
    # binning=dict(bins=50, range=(0,1), histtype='step', log=True)
    binning = dict(bins=50, histtype="step", log=True)
    ax0.hist(odd[~y_true], lw=2, label="fake", **binning)
    ax0.hist(odd[y_true], lw=2, label="true", **binning)
    ax0.set_xlabel("Model output", fontsize=fontsize)
    ax0.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax0.legend(loc=0, fontsize=fontsize)

    # Plot the ROC curve
    auc = sklearn.metrics.auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0, 1], [0, 1], "--", lw=2)
    ax1.set_xlabel("False positive rate", fontsize=fontsize)
    ax1.set_ylabel("True positive rate", fontsize=fontsize)
    ax1.set_title("ROC curve, AUC = %.4f" % auc, fontsize=fontsize)
    ax1.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    print("AUC: %.4f" % auc)

    # Plot the purity and efficiency
    p, r, t = sklearn.metrics.precision_recall_curve(y_true, odd)
    ax2.plot(t, p[:-1], label="purity", lw=2)
    ax2.plot(t, r[:-1], label="efficiency", lw=2)
    ax2.set_xlabel("Cut on model score", fontsize=fontsize)
    ax2.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax2.legend(fontsize=fontsize, loc="upper right")
    ax2.grid(True)
    # Find the point efficiency = 99% and print the purity
    idx = (np.abs(r[:-1] - target_efficiency)).argmin()
    if r[idx] < target_efficiency:
        idx -= 1
    print(f"Purity at {p[idx]:.4f} efficiency: {r[idx]:.4f} with cut {t[idx]:.4f}")
    # draw dashed lines in the purity and efficiency plots for this point
    ax2.axvline(t[idx], color="k", linestyle="--", lw=2)
    # add a text box with the purity and efficiency
    ax2.text(
        0.11,
        0.6,
        f"Purity: {100 * p[idx]:.2f}%\nEfficiency: {100 * r[idx]:.2f}%\nCut: {t[idx]:.4f}",
        transform=ax2.transAxes,
        fontsize=fontsize,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="green", alpha=0.5),
    )

    ax3.plot(p, r, lw=2)
    ax3.set_xlabel("Purity", fontsize=fontsize)
    ax3.set_ylabel("Efficiency", fontsize=fontsize)
    ax3.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    if outname is not None:
        plt.savefig(outname + ".pdf", dpi=300)
        plt.savefig(outname + ".png")

    if off_interactive:
        plt.close(fig)
    return fig, axs, auc, r[idx], p[idx], t[idx]
