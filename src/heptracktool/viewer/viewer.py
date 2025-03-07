"""Visualize tracks."""

import numpy as np
import matplotlib.pyplot as plt


def plot(
    track: np.ndarray,
    figsize=(15, 12),
    fontsize=16,
    minorsize=14,
    outname=None,
    **kwargs,
) -> None:
    """Plot a track from different views"""

    if track.shape[1] != 3:
        raise ValueError("Track must have 3 columns for x, y, and z.")

    if "lw" not in kwargs:
        kwargs["lw"] = 2

    x = track[:, 0]
    y = track[:, 1]
    z = track[:, 2]
    r = np.sqrt(x**2 + y**2)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="polar")

    # plot r vs z
    ax1.plot(z, r, **kwargs)
    ax1.set_xlabel("z [m]", fontsize=fontsize)
    ax1.set_ylabel("r [m]", fontsize=fontsize)
    ax1.tick_params(axis="both", which="major", labelsize=minorsize)

    # plot x vs y
    ax2.plot(x, y, **kwargs)
    ax2.set_xlabel("x [m]", fontsize=fontsize)
    ax2.set_ylabel("y [m]", fontsize=fontsize)
    ax2.tick_params(axis="both", which="major", labelsize=minorsize)

    # plot 3D
    ax3.plot3D(x, y, z, **kwargs)
    ax3.set_xlabel("x [m]", fontsize=fontsize)
    ax3.set_ylabel("y [m]", fontsize=fontsize)
    ax3.set_zlabel("z [m]", fontsize=fontsize)
    ax3.tick_params(axis="both", which="major", labelsize=minorsize)
    ax3.scatter(x, y, z, **kwargs)

    # plot x vs y in polar coordinates
    ax4.plot(x, y, **kwargs)
    ax4.set_xlabel("x [m]", fontsize=fontsize)
    ax4.set_ylabel("y [m]", fontsize=fontsize)
    ax4.tick_params(axis="both", which="major", labelsize=minorsize)


def view_graph(
    hits: np.ndarray,
    pids: np.ndarray,
    edges: np.ndarray,
    outname: str | None = None,
    markersize: int = 20,
    max_tracks: int = 10,
    with_legend: bool = False,
):
    """View a graph of hits and edges. If max_tracks is too large,
    we only plot the nodes and edges with the same color.

    Args:
        hits: spacepoint positions in [r, phi, z]
        pids: particle id of each spacepoint
        edges: list of edges in dimension of [2, num-edges]
        outname: name of output file
        markersize: size of markers
        max_tracks: maximum number of tracks for visulization
    """
    unique_particles = np.unique(pids)
    do_only_nodes = False

    if max_tracks is not None and max_tracks < len(unique_particles) - 1:
        sel_pids = unique_particles[1 : max_tracks + 1]
    else:
        sel_pids = unique_particles[1:]
        print("only plot the nodes!")
        do_only_nodes = True
        max_tracks = len(sel_pids) + 1

    print(f"randomly select {max_tracks} particles for display")
    all_sel_hits = hits[np.isin(pids, sel_pids)]
    all_sel_hit_idx = np.where(np.isin(pids, sel_pids))[0]
    print(f"{all_sel_hits.shape[0]:,} out of {hits.shape[0]} hits are selected")

    def get_hit_info(X):
        r = X[:, 0]
        phi = X[:, 1]
        z = X[:, 2]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y, z, r, phi

    _, axs = plt.subplots(1, 2, figsize=(13, 6))
    if not do_only_nodes:
        for pid in sel_pids:
            sel_hits = hits[pids == pid]
            x, y, z, r, _ = get_hit_info(sel_hits)
            axs[0].scatter(x, y, s=markersize, label=str(pid))
            axs[1].scatter(z, r, s=markersize)

        if with_legend:
            axs[0].legend(fontsize=10)

        sel_edges = edges[
            :, np.isin(edges[0], all_sel_hit_idx) & np.isin(edges[1], all_sel_hit_idx)
        ]
        print(f"selected {sel_edges.shape[1]:,} edges from total {edges.shape[1]:,} true edges")

        sel_edges = sel_edges.T
        for iedge in range(sel_edges.shape[0]):
            sel_hits = hits[sel_edges[iedge]]
            x, y, z, r, _ = get_hit_info(sel_hits)
            axs[0].plot(x, y, color="k", lw=2.0, alpha=0.5)
            axs[1].plot(z, r, color="k", lw=2.0, alpha=0.5)
    else:
        x, y, z, r, _ = get_hit_info(hits)
        axs[0].scatter(x, y, s=markersize)
        axs[1].scatter(z, r, s=markersize)
        # plot edges
        sel_edges = edges.T
        for iedge in range(sel_edges.shape[0]):
            sel_hits = hits[sel_edges[iedge]]
            x, y, z, r, _ = get_hit_info(sel_hits)
            axs[0].plot(x, y, color="grey", lw=0.5, alpha=0.35)
            axs[1].plot(z, r, color="grey", lw=0.5, alpha=0.35)

    axs[0].set_xlabel("X [mm]")
    axs[0].set_ylabel("Y [mm]")
    axs[1].set_xlabel("Z [mm]")
    axs[1].set_ylabel("R [mm]")
    if outname is not None:
        plt.savefig(outname)
