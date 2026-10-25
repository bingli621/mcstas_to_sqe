"""UB matrix of a single crystal from indexed peaks, and Miller indices of events.

The convention is scippneutron's (``scippneutron.spectroscopy.crystal``):
``Q_sample = 2 pi UB (h, k, l)``, with ``Q_sample = R^-1 Q_lab`` from :func:`.to_sample_frame`.
"""

import numpy as np
import scipp as sc


def calculate_ub(q: np.ndarray, hkl: np.ndarray) -> np.ndarray:
    """Compute the UB matrix from peaks with known Miller indices.

    Solves ``Q_i = 2 pi UB hkl_i`` for UB. Three peaks with non-coplanar ``hkl`` determine it;
    more peaks give the least-squares fit.

    Parameters
    ----------
    q:
        (N, 3) sample-frame Q of the peaks in 1/Å, e.g. ``np.stack([p.q for p in peaks])`` from
        :func:`.find_peaks`.
    hkl:
        (N, 3) Miller indices of the peaks.

    Returns
    -------
    :
        (3, 3) UB matrix in 1/Å.

    Raises
    ------
    ValueError
        If there are fewer than 3 peaks, or their ``hkl`` are coplanar.
    """
    q = np.asarray(q, dtype=float)
    hkl = np.asarray(hkl, dtype=float)
    if q.shape != hkl.shape or q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(
            f"Expected q and hkl of the same shape (N, 3), got {q.shape}, {hkl.shape}."
        )
    if np.linalg.matrix_rank(hkl) < 3:
        raise ValueError("Need at least 3 peaks with non-coplanar hkl.")
    # rows: hkl_i^T UB^T = q_i^T / (2 pi)
    ub_t, *_ = np.linalg.lstsq(hkl, q / (2 * np.pi), rcond=None)
    return ub_t.T


def to_hkl(events: sc.DataArray, ub: np.ndarray) -> sc.DataArray:
    """Add the Miller indices, ``(h, k, l) = (2 pi UB)^-1 Q_sample``.

    Parameters
    ----------
    events:
        Pixels with the coords ``Qx_sample``, ``Qy_sample``, ``Qz_sample``, from
        :func:`.to_sample_frame`.
    ub:
        (3, 3) UB matrix in 1/Å, from :func:`calculate_ub`.

    Returns
    -------
    :
        A shallow copy of ``events`` with the dimensionless coords ``h``, ``k``, ``l``.
    """
    q_sample = np.stack(
        [
            events.coords[k].to(unit="1/angstrom").values
            for k in ("Qx_sample", "Qy_sample", "Qz_sample")
        ]
    )
    hkl = np.linalg.solve(2 * np.pi * np.asarray(ub), q_sample)
    events = events.copy(deep=False)
    for name, values in zip(("h", "k", "l"), hkl):
        events.coords[name] = sc.array(dims=events.dims, values=values, unit="dimensionless")
    return events
