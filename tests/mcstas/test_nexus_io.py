from pathlib import Path

import h5py
import numpy as np
import pytest

from mcstas_to_sqe.mcstas.nexus_io import merge_entries, read_pixels

SPLIT = Path(__file__).parents[1] / "data" / "split_run" / "split"


def test_single_run_files_become_one_entry_each_in_their_order(tmp_path):
    files = []
    for omega in (-5, 5):
        files.append(tmp_path / f"run_{omega}.h5")
        with h5py.File(files[-1], "w") as f:
            f.create_group("entry1").attrs["omega"] = omega

    with h5py.File(merge_entries(files, tmp_path / "scan.h5")) as scan:
        assert list(scan) == ["entry1", "entry2"]
        assert [scan[e].attrs["omega"] for e in scan] == [-5, 5]


def test_read_pixels_needs_the_beam_direction_for_a_file_without_a_source():
    with pytest.raises(ValueError, match="beam_direction"):
        read_pixels(SPLIT, "detector", "source", "sample_pos")


def test_read_pixels_puts_the_missing_source_upstream_along_the_beam_and_adds_no_coords():
    points = read_pixels(SPLIT, "detector", "source", "sample_pos", beam_direction=[0, 0, 2])
    assert [p["omega"] for p, _ in points] == [-5.0, 5.0]
    pixels = points[0][1]
    upstream = pixels.coords["sample_position"].value - pixels.coords["source_position"].value
    assert np.allclose(upstream, [0, 0, 1])  # 1 m, whatever the length of beam_direction
    assert "gamma" not in pixels.coords and "y" not in pixels.coords
