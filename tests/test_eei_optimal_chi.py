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
