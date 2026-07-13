import numpy as np
import pytest

from tools.avo_tools import _eei_chi_scan


def _logs(n=40, seed=0):
    rng = np.random.RandomState(seed)
    vp = 3000.0 + 800.0 * rng.rand(n)
    vs = vp / 2.0 + 50.0 * rng.rand(n)   # keeps 0 < vs < vp
    rho = 2.2 + 0.3 * rng.rand(n)
    return vp, vs, rho


def test_scan_ai_target_recovers_chi_zero():
    # At chi=0, EEI = Vp*rho (acoustic impedance). With target = Vp*rho, the
    # correlation peaks at chi=0 with |r| = 1.
    vp, vs, rho = _logs()
    target = vp * rho
    res = _eei_chi_scan(vp, vs, rho, target, np.arange(-90.0, 90.0 + 1.0, 1.0))
    assert abs(res["optimal_chi"]) <= 1.0
    assert np.isclose(abs(res["max_correlation"]), 1.0, atol=1e-6)
    assert len(res["chi"]) == len(res["correlation"])
    assert len(res["eei_optimal"]) == len(vp)


def test_scan_recovers_planted_chi():
    # Build a target that IS the EEI log at a chosen chi (same scalar-K formula the
    # scan uses). The scan must then peak exactly at that planted chi with |r| = 1.
    vp, vs, rho = _logs(seed=7)
    planted = 30.0
    K = float(np.mean((vs / vp) ** 2))
    x = np.radians(planted)
    p = np.cos(x) + np.sin(x)
    q = -8.0 * K * np.sin(x)
    r = np.cos(x) - 4.0 * K * np.sin(x)
    target = vp ** p * vs ** q * rho ** r
    res = _eei_chi_scan(vp, vs, rho, target, np.arange(-90.0, 90.0 + 1.0, 1.0))
    assert res["optimal_chi"] == planted
    assert np.isclose(abs(res["max_correlation"]), 1.0, atol=1e-6)


def test_scan_is_shift_scale_invariant():
    # Pearson r is invariant to affine transforms of the target -> same optimal chi.
    vp, vs, rho = _logs(seed=3)
    target = vp * rho
    chi = np.arange(-90.0, 90.0 + 1.0, 1.0)
    a = _eei_chi_scan(vp, vs, rho, target, chi)["optimal_chi"]
    b = _eei_chi_scan(vp, vs, rho, 5.0 * target + 17.0, chi)["optimal_chi"]
    assert a == b


def test_scan_constant_target_raises():
    vp, vs, rho = _logs()
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, np.ones_like(vp), np.arange(-90.0, 91.0, 1.0))


def test_scan_length_mismatch_raises():
    vp, vs, rho = _logs()
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, (vp * rho)[:-1], np.arange(-90.0, 91.0, 1.0))


def test_scan_nonphysical_sample_raises():
    vp, vs, rho = _logs()
    vs = vs.copy()
    vs[0] = vp[0] + 1.0   # vs >= vp -> non-physical
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, vp * rho, np.arange(-90.0, 91.0, 1.0))


def test_scan_empty_chi_raises():
    vp, vs, rho = _logs()
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, vp * rho, np.array([]))


def test_scan_chi_out_of_range_raises():
    vp, vs, rho = _logs()
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, vp * rho, np.array([-90.0, 0.0, 120.0]))


import os

from tools.avo_tools import eei_optimal_chi


def test_eei_optimal_chi_tool_ai_target():
    vp, vs, rho = _logs(seed=1)
    res = eei_optimal_chi(
        vp.tolist(), vs.tolist(), rho.tolist(), (vp * rho).tolist(),
        chi_min=-90, chi_max=90, chi_step=1,
    )
    assert abs(res["optimal_chi"]) <= 1.0
    assert np.isclose(abs(res["max_correlation"]), 1.0, atol=1e-6)
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)
