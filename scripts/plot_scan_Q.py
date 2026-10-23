"""Animated (Qx, Qz) maps over a McStas scan: intensity summed over Qy, one frame per scan point.

Q is the elastic momentum transfer (scippneutron) rotated into the sample frame with the inverse
goniometer rotation, so the reciprocal lattice stays fixed while omega changes.

In VS Code: run this file, or from a notebook/Interactive Window:

    from mcstas_to_sqe.mcstas.run_folders import latest_run
    from plot_scan_Q import plot_scan_Q
    from work_dir import MCSTAS_DIR, OUTPUT_DIR
    plot_scan_Q(latest_run(MCSTAS_DIR, "*_scan_omega"), scan_par="omega", out=OUTPUT_DIR / "scan_Q.gif")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from work_dir import MCSTAS_DIR, OUTPUT_DIR

from mcstas_to_sqe.mcstas import run_folders
from mcstas_to_sqe.mcstas.run_folders import latest_run
from mcstas_to_sqe.reduction import (
    normalize_scan,
    read_scan,
    run_wavelength,
    to_Q,
    to_sample_frame,
)


def plot_scan_Q(
    folder=None,
    scan_par="omega",
    out=OUTPUT_DIR / "scan_Q.gif",
    bins=(200, 200),
    fps=3,
    accumulate=True,
    normalize=False,
    monitor_file=None,
    wavelength=None,
    show=True,
):
    """Plots the scan in `folder` (None: the newest scan in MCSTAS_DIR). Saves a gif to `out`
    (None: don't save); returns the FuncAnimation.

    With `normalize`, the intensities are divided by the incident monitor and the pixel solid
    angle (see `normalize_scan`), in 1/sr. `monitor_file` None takes the scan's own monitor, or
    for data from `scan_from_sample` the beam run's (`run_folders.monitor_file`). `wavelength` (a
    scipp scalar, e.g. `reduction.wavelength(14)`): None takes the one each scan point stores.

    With `accumulate`, frame i shows the sum of all scan points up to i (previous frames are kept).
    """
    folder = folder or latest_run(MCSTAS_DIR, "*scan*")
    if normalize:
        monitor_file = monitor_file or run_folders.monitor_file(folder)
        scan = normalize_scan(folder, monitor_file=monitor_file)
        points = [
            ({k: v.value for k, v in g["parameters"].items()}, g["events"].bins.sum())
            for g in scan.values()
        ]
    else:
        points = read_scan(folder)
    qs = [to_sample_frame(to_Q(ev, run_wavelength(p, wavelength)), p) for p, ev in points]

    # common Qx/Qz range so frames are comparable
    qx_all = np.concatenate([q.coords["Qx_sample"].values for q in qs])
    qz_all = np.concatenate([q.coords["Qz_sample"].values for q in qs])
    rng = [[qx_all.min(), qx_all.max()], [qz_all.min(), qz_all.max()]]
    # histogramming over (Qx, Qz) with weights sums the intensity over Qy
    hists = [
        np.histogram2d(
            q.coords["Qx_sample"].values,
            q.coords["Qz_sample"].values,
            bins=bins,
            range=rng,
            weights=q.values,
        )[0]
        for q in qs
    ]
    if accumulate:
        hists = list(np.cumsum(hists, axis=0))
    xe = np.linspace(*rng[0], bins[0] + 1)
    ye = np.linspace(*rng[1], bins[1] + 1)

    vmax = max(h.max() for h in hists)
    vmin = max(min(h[h > 0].min() for h in hists), vmax * 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(
        xe,
        ye,
        np.ma.masked_less_equal(hists[0].T, 0),
        cmap="jet",
        norm=LogNorm(vmin, vmax),
    )
    fig.colorbar(
        mesh,
        label="normalized intensity summed over Qy (1/sr)"
        if normalize
        else "intensity summed over Qy (n/s)",
    )
    ax.set_xlabel("Qx sample (1/Å)")
    ax.set_ylabel("Qz sample (1/Å)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.6)
    ax.set_title(f"{scan_par} = {points[0][0][scan_par]:g}°")  # reserve room before tight_layout
    fig.tight_layout()

    def draw(i):
        mesh.set_array(np.ma.masked_less_equal(hists[i].T, 0))
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
    # normalized (Qx, Qz) map of the newest omega scan: a scan (source -> detector) by each
    # point's own monitor, a scan_from_sample scan by the monitor of the beam run it started from
    plot_scan_Q(
        folder=latest_run(MCSTAS_DIR, "*_scan_omega*"),
        out=OUTPUT_DIR / "scan_Q_normalized_132pts.gif",
        normalize=True,
    )
