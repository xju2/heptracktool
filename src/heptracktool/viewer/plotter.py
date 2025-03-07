# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import itertools
import scipy

fontsize = 16
minor_size = 14
pt_bins = np.arange(0, 5.0, step=0.5).tolist() + np.arange(5, 11, step=1.0).tolist()
# pt_bins = np.arange(0, 10.2, step=0.2)
pt_configs = {"bins": pt_bins, "histtype": "step", "lw": 2, "log": False}
# %%
eta_bins = np.arange(-4, 4.4, step=0.4)
# %%
eta_configs = {"bins": eta_bins, "histtype": "step", "lw": 2, "log": False}


def get_plot(nrows=1, ncols=1, figsize=6, nominor=False):
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize * ncols, figsize * nrows),
        constrained_layout=True,
    )

    def format(ax):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax

    if nrows * ncols == 1:
        ax = axs
        if not nominor:
            format(ax)
    else:
        ax = [format(x) if not nominor else x for x in axs.flatten()]

    return fig, ax


def add_up_xaxis(ax):
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(["" for x in ax.get_xticks()])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())


def clopper_pearson(passed: float, total: float, level: float = 0.68):
    """
    Estimate the confidence interval for a sampled binomial random variable with Clopper-Pearson.
    `passed` = number of successes; `total` = number trials; `level` = the confidence level.
    The function returns a `(low, high)` pair of numbers indicating the lower and upper error bars.
    """
    alpha = (1 - level) / 2
    lo = scipy.stats.beta.ppf(alpha, passed, total - passed + 1) if passed > 0 else 0.0
    hi = scipy.stats.beta.ppf(1 - alpha, passed + 1, total - passed) if passed < total else 1.0
    average = passed / total
    return (average - lo, hi - average)


def get_ratio(x_vals, y_vals):
    res = [x / y if y != 0 else 0.0 for x, y in zip(x_vals, y_vals)]
    err = np.array(list(itertools.starmap(clopper_pearson, zip(x_vals, y_vals)))).T
    return res, err


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def add_mean_std(array, x, y, ax, color="k", dy=0.3, digits=2, fontsize=12, with_std=True):
    this_mean, this_std = np.mean(array), np.std(array)
    ax.text(x, y, "Mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=12)
    if with_std:
        ax.text(
            x,
            y - dy,
            "Standard Deviation: {0:.{1}f}".format(this_std, digits),
            color=color,
            fontsize=12,
        )


def make_cmp_plot(
    arrays,
    legends,
    configs,
    xlabel,
    ylabel,
    ratio_label,
    ratio_legends,
    outname,
    ymin=0,
):
    _, ax = get_plot()
    vals_list = []
    for array, legend in zip(arrays, legends):
        vals, bins, _ = ax.hist(array, **configs, label=legend)
        vals_list.append(vals)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    add_up_xaxis(ax)
    ax.legend()
    ax.grid(True)
    plt.savefig(f"{outname}.pdf")

    # make a ratio plot
    _, ax = get_plot()
    xvals = [0.5 * (x[1] + x[0]) for x in pairwise(bins)]
    xerrs = [0.5 * (x[1] - x[0]) for x in pairwise(bins)]

    for idx in range(1, len(arrays)):
        ratio, ratio_err = get_ratio(vals_list[-1], vals_list[idx - 1])
        label = None if ratio_legends is None else ratio_legends[idx - 1]
        ax.errorbar(xvals, ratio, yerr=ratio_err, fmt="o", xerr=xerrs, lw=2, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ratio_label)
    add_up_xaxis(ax)

    if ratio_legends is not None:
        ax.legend()
    ax.grid(True)
    plt.savefig(f"{outname}_ratio.pdf")
