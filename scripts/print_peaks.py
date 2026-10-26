"""Print the peak list (sample-frame Q, |Q|, Miller indices, intensity and its error bar) of a
McStas scan, and save it to a text file. The Miller indices come from the UB matrix of
plot_scan_hkl.py.

In VS Code: run this file, or from a notebook/Interactive Window:

    from mcstas_to_sqe.mcstas.run_folders import latest_run
    from print_peaks import print_peaks, save_peaks
    from work_dir import MCSTAS_DIR, OUTPUT_DIR
    peaks = print_peaks(latest_run(MCSTAS_DIR, "*_scan_omega"))
    save_peaks(peaks, OUTPUT_DIR / "peaks.txt")
"""

import sys
from pathlib import Path

import numpy as np
from plot_scan_hkl import UB
from work_dir import MCSTAS_DIR, OUTPUT_DIR

from mcstas_to_sqe.mcstas.run_folders import latest_run, monitor_file
from mcstas_to_sqe.reduction import find_peaks


def format_peaks(peaks, ub=UB):
    """The peak table as a list of text lines: Qx, Qy, Qz, |Q| [1/Å], h, k, l (from the UB
    matrix `ub`, see `reduction.calculate_ub`), Intensity, Errorbar [1/sr]."""
    lines = [
        (
            f"{'Qx':>9} {'Qy':>9} {'Qz':>9} {'|Q|':>9} {'h':>7} {'k':>7} {'l':>7}"
            f" {'Intensity':>12} {'Errorbar':>10}"
        )
    ]
    for p in peaks:
        qx, qy, qz = p.q
        q_abs = np.linalg.norm(p.q)
        h, k, l = np.linalg.solve(2 * np.pi * np.asarray(ub), p.q)
        lines.append(
            f"{qx:9.4f} {qy:9.4f} {qz:9.4f} {q_abs:9.4f} {h:7.3f} {k:7.3f} {l:7.3f}"
            f" {p.intensity:12.4g} {p.intensity_err:10.2g}"
        )
    lines.append(f"{len(peaks)} peaks")
    return lines


def save_peaks(peaks, out=OUTPUT_DIR / "peaks.txt", ub=UB):
    """Write the peak table to the text file `out`; returns the path."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(format_peaks(peaks, ub)) + "\n")
    return out


def print_peaks(folder=None, ub=UB, **kwargs):
    """Print the peaks of a scan (`folder`, None: the newest scan in MCSTAS_DIR), smallest |Q|
    first, with their Miller indices from the UB matrix `ub`. `kwargs` go to `find_peaks`; a
    `scan_from_sample` scan without a `monitor_file` gets the beam run's
    (`run_folders.monitor_file`). Returns the list of peaks (use `save_peaks` to write them to a
    file)."""
    folder = folder or latest_run(MCSTAS_DIR, "*scan*")
    kwargs.setdefault("monitor_file", monitor_file(folder))
    peaks = sorted(find_peaks(folder, **kwargs), key=lambda p: np.linalg.norm(p.q))
    print("\n".join(format_peaks(peaks, ub)))
    return peaks


if __name__ == "__main__":
    # optional command-line use: print_peaks.py [folder] [out.txt]
    peaks = print_peaks(*sys.argv[1:2])
    print(f"Saved to {save_peaks(peaks, *sys.argv[2:3])}")
