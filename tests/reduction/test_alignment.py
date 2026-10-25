import numpy as np
import pytest
import scipp as sc

from mcstas_to_sqe.reduction import calculate_ub, goniometer_matrix, to_hkl


def hexagonal_b(a, c):
    # B with a* along x, b* at 60 deg in the xy plane, c* along z (1/Å, without 2 pi)
    a_star = 2 / (np.sqrt(3) * a)
    return np.array([[a_star, a_star * 0.5, 0], [0, a_star * np.sqrt(3) / 2, 0], [0, 0, 1 / c]])


def test_calculate_ub_recovers_a_known_ub():
    ub = goniometer_matrix(30, -10, 5) @ hexagonal_b(5.0556, 24.598)
    hkl = np.array([[1, 0, 0], [0, 0, 6], [2, 1, 0], [1, 1, 3]])
    q = (2 * np.pi * ub @ hkl.T).T
    assert calculate_ub(q[:3], hkl[:3]) == pytest.approx(ub)
    assert calculate_ub(q, hkl) == pytest.approx(ub)  # least squares with 4 peaks


def test_calculate_ub_needs_non_coplanar_peaks():
    with pytest.raises(ValueError, match="non-coplanar"):
        calculate_ub(np.ones((3, 3)), [[1, 0, 0], [0, 1, 0], [1, 1, 0]])


def test_ub_from_the_trex_peaks():
    # alignment-trex-test.ipynb: (100) and (006) in the sample frame, (210) at 2theta, omega, chi
    k = 2 * np.pi / np.sqrt(81.80421 / 14)
    t = np.radians(-93.83)
    q_lab = [-k * np.sin(t), 0, k * (1 - np.cos(t))]  # ki - kf
    q3 = goniometer_matrix(-46.92, -5.893, 0).T @ q_lab
    q = np.array([[1.2977, -0.6043, 0.0026], [0.0007, 0.0004, 1.5335], q3])
    ub = calculate_ub(q, [[1, 0, 0], [0, 0, 6], [2, 1, 0]])
    # the lattice: |Q(010)| = |Q(100)| and |Q(001)| = 2 pi / c, to the precision of the peaks
    q_of = lambda hkl: np.linalg.norm(2 * np.pi * ub @ hkl)
    assert q_of([0, 1, 0]) == pytest.approx(q_of([1, 0, 0]), rel=1e-2)
    assert q_of([0, 0, 1]) == pytest.approx(2 * np.pi / 24.598, rel=1e-3)


def test_to_hkl_inverts_the_ub():
    ub = goniometer_matrix(30, -10, 5) @ hexagonal_b(5.0556, 24.598)
    hkl = np.array([[1, 0, 0], [0, 0, 6], [2, 1, 0], [-1, 2, 3]], dtype=float)
    q = 2 * np.pi * ub @ hkl.T
    events = sc.DataArray(
        sc.ones(dims=["pixel_id"], shape=[4]),
        coords={
            name: sc.array(dims=["pixel_id"], values=v, unit="1/angstrom")
            for name, v in zip(("Qx_sample", "Qy_sample", "Qz_sample"), q)
        },
    )
    out = to_hkl(events, ub)
    assert np.stack([out.coords[c].values for c in "hkl"], axis=1) == pytest.approx(hkl)
    assert out.coords["h"].unit == sc.units.one
    assert "h" not in events.coords
