"""The reduction of a split run, on a small recorded one (tests/data/make_split_run.py)."""

from pathlib import Path

import mcstastox
import numpy as np
import pytest

from mcstas_to_sqe.mcstas.nexus_io import numeric_parameters
from mcstas_to_sqe.reduction import (
    find_peaks,
    normalize_scan,
    read_scan,
    run_wavelength,
    to_Q,
    wavelength,
)
from mcstas_to_sqe.reduction.peaks import load_points

DATA = Path(__file__).parents[1] / "data" / "split_run"
SPLIT = DATA / "split"
MONITOR = DATA / "beam" / "mccode.h5"


def test_the_split_run_stores_its_own_parameters():
    """Every entry records the wavelength and its angles (scan_from_sample writes them as defaults:
    McStas only records a parameter that has one)."""
    params = []
    for entry in (1, 2):
        with mcstastox.Read(str(SPLIT), entry_number=entry) as data:
            params.append(numeric_parameters(data.get_instrument_parameters()))
    assert [p["omega"] for p in params] == [-5.0, 5.0]
    assert all(p["wavelength"] == pytest.approx(2.41726) and p["phi"] == 0 for p in params)


def test_a_split_run_takes_the_missing_parameters_from_the_beam_run():
    scan = normalize_scan(SPLIT, monitor_file=MONITOR)
    params = [{k: v.value for k, v in g["parameters"].items()} for g in scan.values()]
    # the entries' own values win: the beam run had omega 0 and chi 0 (unused before the sample)
    assert [p["omega"] for p in params] == [-5.0, 5.0]
    assert [p["chi"] for p in params] == pytest.approx([-5.893, -5.893])


def test_the_peaks_of_a_split_run_can_be_found():
    points = load_points(SPLIT, monitor_file=MONITOR)  # the wavelength the scan stores
    n = len(points["w"])
    assert n > 0 and points["q"].shape == (n, 3) and np.all(np.isfinite(points["q"]))
    assert set(points["scan"]) == {0, 1}
    assert points["monitor"][0] == points["monitor"][1]  # one beam-run reading for every angle
    given = load_points(SPLIT, wavelength(14), monitor_file=MONITOR)  # ei = 14 meV, as stored
    assert np.allclose(given["q"], points["q"], atol=1e-4)
    assert isinstance(find_peaks(SPLIT, monitor_file=MONITOR), list)


def test_read_scan_reads_a_split_run_without_a_source():
    """The split run has no source: read_scan puts one upstream along the beam, which gives the same
    Q as the normalization's source rebuilt from the beam run."""
    points = read_scan(SPLIT)
    assert [p["omega"] for p, _ in points] == [-5.0, 5.0]
    params, pixels = points[0]
    g = next(iter(normalize_scan(SPLIT, monitor_file=MONITOR).values()))
    q_read = to_Q(pixels, run_wavelength(params))
    q_norm = to_Q(g["events"].bins.sum(), run_wavelength(params))
    for k in ("Qx", "Qy", "Qz"):
        assert np.allclose(q_read.coords[k].values, q_norm.coords[k].values)
