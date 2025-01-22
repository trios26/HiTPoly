import numpy as np


def plot_hexbin(pred, targ, ax, scale="log", plot_helper_lines=False):
    if scale == "log":
        pred = np.abs(pred) + 1e-8
        targ = np.abs(targ) + 1e-8

    hb = ax.hexbin(
        pred,
        targ,
        cmap="viridis",
        gridsize=80,
        bins="log",
        mincnt=1,
        edgecolors=None,
        linewidths=(0.1,),
        xscale=scale,
        yscale=scale,
        extent=(
            min(np.min(pred), np.min(targ)) * 1.1,
            max(np.max(pred), np.max(targ)) * 1.1,
            min(np.min(pred), np.min(targ)) * 1.1,
            max(np.max(pred), np.max(targ)) * 1.1,
        ),
    )

    lim_min = min(np.min(pred), np.min(targ)) * 1.1
    lim_max = max(np.max(pred), np.max(targ)) * 1.1

    ax.set_aspect("equal")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    # ax.set_aspect(aspect=1)

    ax.plot(
        (lim_min, lim_max),
        (lim_min, lim_max),
        color="#000000",
        zorder=-1,
        linewidth=0.5,
    )

    return ax, hb
