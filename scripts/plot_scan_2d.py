"""Animated 2D detector images (2theta vs height) over a McStas scan, read with scippnexus.

In VS Code: run this file, or from a notebook/Interactive Window:

    from mcstas_to_sqe.mcstas.run_folders import latest_run
    from plot_scan_2d import plot_scan_2d
    from work_dir import MCSTAS_DIR, OUTPUT_DIR
    plot_scan_2d(latest_run(MCSTAS_DIR, "*_scan_omega"), scan_par="omega", out=OUTPUT_DIR / "scan.gif")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from work_dir import MCSTAS_DIR, OUTPUT_DIR

from mcstas_to_sqe.mcstas.run_folders import latest_run
from mcstas_to_sqe.reduction import read_scan


def plot_scan_2d(
    folder=None,
    scan_par="omega",
    out=OUTPUT_DIR / "scan.gif",
    bins=(1200, 50),
    fps=3,
    show=True,
):
    """One frame per scan point of `folder` (None: the newest scan in MCSTAS_DIR), the scan angle
    in the title. Saves a gif to `out` (None: don't save); returns the FuncAnimation."""
    points = read_scan(folder or latest_run(MCSTAS_DIR, "*scan*"))
    hists = [
        np.histogram2d(
            ev.coords["gamma"].values,
            ev.coords["y"].values,
            bins=bins,
            weights=ev.values,
        )
        for _, ev in points
    ]
    # same log color scale for all frames; empty bins are masked
    vmax = max(h.max() for h, _, _ in hists)
    vmin = max(min(h[h > 0].min() for h, _, _ in hists), vmax * 1e-6)

    fig, ax = plt.subplots(figsize=(9, 5))
    h, xe, ye = hists[0]
    mesh = ax.pcolormesh(
        xe, ye, np.ma.masked_less_equal(h.T, 0), cmap="jet", norm=LogNorm(vmin, vmax)
    )
    fig.colorbar(mesh, label="intensity (n/s)")
    ax.set_xlabel("2θ (deg)")
    ax.set_ylabel("height y (m)")
    ax.grid(alpha=0.6)
    ax.set_title(f"{scan_par} = {points[0][0][scan_par]:g}°")  # reserve room before tight_layout
    fig.tight_layout()

    def draw(i):
        mesh.set_array(np.ma.masked_less_equal(hists[i][0].T, 0))
        ax.set_title(f"{scan_par} = {points[i][0][scan_par]:g}°")
        return (mesh,)

    anim = FuncAnimation(fig, draw, frames=len(points), blit=False)
    draw(0)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    return anim


if __name__ == "__main__":
    plot_scan_2d()
