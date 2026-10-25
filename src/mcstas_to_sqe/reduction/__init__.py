"""Data reduction of diffractometer scans.

Reading, momentum transfer, sample frame, solid-angle normalization, peak finding, UB matrix
and Miller indices.
"""

from .alignment import calculate_ub, to_hkl
from .normalization import (
    detector_solid_angle,
    incident_intensity,
    normalize_scan,
)
from .peaks import Peak, centre_of_mass, find_peaks
from .reduction import (
    goniometer_matrix,
    read_scan,
    run_wavelength,
    to_Q,
    to_sample_frame,
    wavelength,
)
from .solid_angle import rectangle_solid_angle

__all__ = [
    "Peak",
    "calculate_ub",
    "centre_of_mass",
    "detector_solid_angle",
    "find_peaks",
    "goniometer_matrix",
    "incident_intensity",
    "normalize_scan",
    "read_scan",
    "rectangle_solid_angle",
    "run_wavelength",
    "to_Q",
    "to_hkl",
    "to_sample_frame",
    "wavelength",
]
