"""Animated (H, L) maps over a McStas scan in the (2H, H, L) plane: intensity summed over a thin
slice around the plane, one frame per scan point.

The Miller indices come from the sample-frame Q (see plot_scan_Q.py) and a UB matrix
(`reduction.calculate_ub`), so the reciprocal lattice stays fixed while omega changes.

In VS Code: run this file, or from a notebook/Interactive Window:

    from mcstas_to_sqe.mcstas.run_folders import latest_run
    from plot_scan_hkl import UB, plot_scan_hkl
    from work_dir import MCSTAS_DIR, OUTPUT_DIR
    plot_scan_hkl(latest_run(MCSTAS_DIR, "*_scan_omega"), UB, out=OUTPUT_DIR / "scan_hkl.gif")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from work_dir import MCSTAS_DIR, OUTPUT_DIR

from mcstas_to_sqe.mcstas import run_folders
from mcstas_to_sqe.mcstas.run_folders import latest_run
from mcstas_to_sqe.reduction import (
    calculate_ub,
    normalize_scan,
    read_scan,
    run_wavelength,
    to_hkl,
    to_Q,
    to_sample_frame,
)


# the plane: (h, k, l) = H (2, 1, 0) + D (0, 1, 0) + L (0, 0, 1). For the hexagonal lattice
# (a* and b* at 120 deg) (0, 1, 0) is perpendicular to (2, 1, 0), so D is the distance from the
# plane in units of |b*|
def to_plane(hkl):
    """(H, D, L) of the Miller indices `hkl` (rows h, k, l)."""
    h, k, l = hkl
    return h / 2, k - h / 2, l


# UB of the sample from three indexed peaks of output/peaks.txt (sample-frame Q in 1/Å)
UB = calculate_ub(
    [[1.2977, -0.6043, 0.0026], [0.0007, 0.0004, 1.5335], [2.4771, 0.2166, 0.0004]],
    [[1, 0, 0], [0, 0, 6], [2, 1, 0]],
)


def plot_scan_hkl(
    folder=None,
    ub=UB,
    scan_par="omega",
    out=OUTPUT_DIR / "scan_hkl.gif",
    bins=(200, 200),
    dk=0.1,
    fps=3,
    accumulate=True,
    normalize=False,
    monitor_file=None,
    wavelength=None,
    show=True,
):
    """Plots the (2H, H, L) plane of the scan in `folder` (None: the newest scan in MCSTAS_DIR), with
    Miller indices from the UB matrix `ub` (1/Å, see `calculate_ub`). Saves a gif to `out` (None:
    don't save); returns the FuncAnimation.

    Only the pixels within `dk` of the plane are kept (|D| <= `dk`, see `to_plane`); the map
    sums their intensity over D.

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
    hkls = [
        to_hkl(to_sample_frame(to_Q(ev, run_wavelength(p, wavelength)), p), ub) for p, ev in points
    ]
    hdls = [to_plane([d.coords[c].values for c in "hkl"]) for d in hkls]
    sel = [abs(dd) <= dk for _, dd, _ in hdls]
    planes = [(h[s], l[s], d.values[s]) for (h, _, l), s, d in zip(hdls, sel, hkls)]
    if not any(len(w) for *_, w in planes):
        raise ValueError(f"No pixels with |D| <= {dk}: the scan does not reach the plane.")

    # common H/L range so frames are comparable
    h_all = np.concatenate([h for h, _, _ in planes])
    l_all = np.concatenate([l for _, l, _ in planes])
    rng = [[h_all.min(), h_all.max()], [l_all.min(), l_all.max()]]
    # histogramming over (H, L) with weights sums the intensity over D
    hists = [np.histogram2d(h, l, bins=bins, range=rng, weights=w)[0] for h, l, w in planes]
    if accumulate:
        hists = list(np.cumsum(hists, axis=0))
    xe = np.linspace(*rng[0], bins[0] + 1)
    ye = np.linspace(*rng[1], bins[1] + 1)

    vmax = max(h.max() for h in hists)
    vmin = max(min(h[h > 0].min() for h in hists if (h > 0).any()), vmax * 1e-6)

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
        label=f"normalized intensity summed over |D| ≤ {dk:g} (1/sr)"
        if normalize
        else f"intensity summed over |D| ≤ {dk:g} (n/s)",
    )
    ax.set_xlabel("(2H, H, 0) (r.l.u.)")
    ax.set_ylabel("(0, 0, L) (r.l.u.)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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
    # normalized (2H, H, L) map of the newest omega scan (see plot_scan_Q.py for the monitor)
    plot_scan_hkl(
        folder=latest_run(MCSTAS_DIR, "*_scan_omega*"),
        out=OUTPUT_DIR / "scan_hkl_normalized.gif",
        normalize=True,
    )
