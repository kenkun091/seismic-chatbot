"""create_synthetic_section: oracle vs the 1-D synthetic, guards, angle paths."""
import warnings

import numpy as np
import pytest

from tools.section_tools import create_synthetic_section, validate_section_inputs
from tools.synthetic_tools import create_synthetic_seismogram

VP = [3000.0, 2500.0, 3200.0]
RHO = [2.4, 2.2, 2.5]
VS = [1500.0, 1250.0, 1600.0]
TH = [50.0, 50.0]


def _layer_cake(nx=5, dz=1.0, bottom=60.0):
    """Horizontal 3-layer grid: 50 m / 50 m / bottom m basal layer."""
    rows = [int(TH[0] / dz), int(TH[1] / dz), int(bottom / dz)]
    vp = np.concatenate([np.full(r, v) for r, v in zip(rows, VP)])
    vs = np.concatenate([np.full(r, v) for r, v in zip(rows, VS)])
    rho = np.concatenate([np.full(r, v) for r, v in zip(rows, RHO)])
    tile = lambda a: np.tile(a[:, None], (1, nx))
    return tile(vp), tile(vs), tile(rho)


def test_oracle_every_column_matches_1d_synthetic():
    vp, vs, rho = _layer_cake()
    t, sec, par = create_synthetic_section(vp, vs, rho, dz=1.0, dx=10.0, dt=1.0, pad_time=50.0)
    t1, trace, p1 = create_synthetic_seismogram(TH, VP, RHO, vs=VS, dt=1.0, pad_time=50.0)
    n = min(len(t1), len(t))
    for j in range(sec.shape[1]):
        np.testing.assert_allclose(sec[:n, j], trace[:n], rtol=1e-6, atol=1e-9)
    assert par["n_interfaces"] == 2 * vp.shape[1]
    assert par["max_abs_amplitude"] == pytest.approx(np.max(np.abs(trace)), rel=1e-6)
    assert par["wavelet_label"] == p1["wavelet_label"]


def test_oracle_at_angle_shuey_and_zoeppritz():
    vp, vs, rho = _layer_cake(nx=2)
    for method in ("shuey", "zoeppritz"):
        _, sec, _ = create_synthetic_section(vp, vs, rho, 1.0, 10.0, dt=1.0, angle=20.0, method=method)
        _, trace, _ = create_synthetic_seismogram(TH, VP, RHO, vs=VS, dt=1.0, angle=20.0, method=method)
        n = min(len(trace), sec.shape[0])
        np.testing.assert_allclose(sec[:n, 0], trace[:n], rtol=1e-6, atol=1e-9)


def test_tiny_angle_zoeppritz_agrees_with_acoustic():
    """Exact Zoeppritz at theta -> 0 is the acoustic RC (Shuey's R0 is only linearized)."""
    vp, vs, rho = _layer_cake(nx=2)
    _, a, _ = create_synthetic_section(vp, vs, rho, 1.0, 10.0, angle=0.0)
    _, z, _ = create_synthetic_section(vp, vs, rho, 1.0, 10.0, angle=1e-6, method="zoeppritz")
    np.testing.assert_allclose(a, z, atol=1e-6)


def test_uniform_grid_gives_zero_section():
    vp = np.full((100, 4), 3000.0); vs = vp / 2; rho = np.full((100, 4), 2.4)
    _, sec, par = create_synthetic_section(vp, vs, rho, 1.0, 5.0)
    assert np.all(sec == 0) and par["n_interfaces"] == 0 and par["max_abs_amplitude"] == 0


def test_lateral_variation_is_column_independent():
    vp, vs, rho = _layer_cake(nx=3)
    vp2 = vp.copy(); vp2[50:100, 2] = 3000.0   # remove the contrast in column 2 only
    rho2 = rho.copy(); rho2[50:100, 2] = 2.4
    vs2 = vs.copy(); vs2[50:100, 2] = 1500.0
    _, sec, _ = create_synthetic_section(vp2, vs2, rho2, 1.0, 5.0)
    np.testing.assert_allclose(sec[:, 0], sec[:, 1])
    assert not np.allclose(sec[:, 0], sec[:, 2])


def test_thin_layers_superpose():
    vp = np.full((100, 1), 3000.0); rho = np.full((100, 1), 2.4); vs = vp / 2
    vp[50] = 2000.0; rho[50] = 2.0            # one-cell layer: two interfaces in one sample at dt=1
    _, sec, par = create_synthetic_section(vp, vs, rho, dz=0.5, dx=1.0, dt=1.0)
    assert par["n_interfaces"] == 2
    assert np.max(np.abs(sec)) < 0.05          # near-cancelling RCs superpose (+=), not overwrite


def test_ormsby_wavelet_path():
    vp, vs, rho = _layer_cake(nx=2)
    _, sec, par = create_synthetic_section(vp, vs, rho, 1.0, 10.0, wv_type="ormsby",
                                           ormsby_freq="5,10,40,60")
    assert par["wavelet_freq"] == pytest.approx(25.0) and "Ormsby" in par["wavelet_label"]
    assert np.max(np.abs(sec)) > 0


def test_postcritical_zoeppritz_zeroed_with_warning():
    vp = np.full((40, 1), 1500.0); vp[20:] = 4500.0
    vs = vp / 2; rho = np.full((40, 1), 2.2)
    with pytest.warns(UserWarning, match="post-critical"):
        _, sec, par = create_synthetic_section(vp, vs, rho, 1.0, 1.0, angle=40.0, method="zoeppritz")
    assert np.isfinite(sec).all() and par["n_postcritical_zeroed"] == 1


def test_guards_reject_bad_inputs():
    vp, vs, rho = _layer_cake(nx=2)
    with pytest.raises(ValueError, match="shape"):
        validate_section_inputs(vp, vs[:-1], rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="2-D"):
        validate_section_inputs(vp[:, 0], vs[:, 0], rho[:, 0], 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    bad_vs = vs.copy(); bad_vs[0, 0] = 9999.0
    with pytest.raises(ValueError, match="vs"):
        validate_section_inputs(vp, bad_vs, rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="dz"):
        validate_section_inputs(vp, vs, rho, 0.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="angle"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 95.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="method"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 10.0, "magic", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="domain"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "sideways")
    with pytest.raises(ValueError, match="ormsby"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 0.0, "shuey", "ormsby", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="finite"):
        nan_vp = vp.copy(); nan_vp[0, 0] = np.nan
        validate_section_inputs(nan_vp, vs, rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")


def test_aliasing_warns():
    vp, vs, rho = _layer_cake(nx=1)
    with pytest.warns(UserWarning, match="Nyquist"):
        create_synthetic_section(vp, vs, rho, 1.0, 1.0, wavelet_freq=200.0, dt=1.0)
