"""Depth conversion, the model adapter, and plot_seismic_section."""
import os

import numpy as np
import pytest

from tools.section_tools import (create_synthetic_section, synthetic_section_from_model,
                                 plot_seismic_section, MAX_WIGGLE_TRACES)


def _single_interface(nx=3, nz=200, dz=0.5, z_int=60.0):
    vp = np.full((nz, nx), 2500.0); vp[int(z_int / dz):] = 3500.0
    vs = vp / 2; rho = np.full((nz, nx), 2.3); rho[int(z_int / dz):] = 2.6
    return vp, vs, rho


def _model(nx=3, nz=200, dz=0.5):
    vp, vs, rho = _single_interface(nx, nz, dz)
    facies = np.zeros((nz, nx), int); facies[int(60.0 / dz):] = 1
    return {"vp": vp, "vs": vs, "rho": rho, "dz": dz, "dx": 2.0, "facies": facies,
            "legend": {0: {"lithology": "shale", "label": "background"},
                       1: {"lithology": "limestone", "label": "lime"}},
            "z": (np.arange(nz) + 0.5) * dz, "x": np.arange(nx) * 2.0}


def test_depth_domain_peak_sits_at_interface_depth():
    vp, vs, rho = _single_interface()
    z, sec, par = create_synthetic_section(vp, vs, rho, 0.5, 2.0, domain="depth")
    assert par["domain"] == "depth" and sec.shape == vp.shape and z.shape == (vp.shape[0],)
    peak_z = z[np.argmax(np.abs(sec[:, 0]))]
    assert abs(peak_z - 60.0) <= 0.5 + 1e-9          # within one cell (zero-phase wavelet)


def test_time_domain_peak_sits_at_interface_time():
    vp, vs, rho = _single_interface()
    t, sec, par = create_synthetic_section(vp, vs, rho, 0.5, 2.0, pad_time=50.0)
    expected = 50.0 + 2000.0 * 60.0 / 2500.0
    assert abs(t[np.argmax(np.abs(sec[:, 0]))] - expected) <= par["dt"]


def test_model_adapter_matches_direct_call():
    m = _model()
    a1, s1, p1 = synthetic_section_from_model(m, wavelet_freq=25.0)
    a2, s2, p2 = create_synthetic_section(m["vp"], m["vs"], m["rho"], m["dz"], m["dx"], wavelet_freq=25.0)
    np.testing.assert_allclose(s1, s2); np.testing.assert_allclose(a1, a2)
    assert p1 == p2


def test_model_adapter_requires_model():
    with pytest.raises(ValueError, match="earth model first"):
        synthetic_section_from_model(None)


@pytest.mark.parametrize("display", ["image", "wiggle", "both"])
def test_plot_modes_write_png(display):
    m = _model()
    axis, sec, par = synthetic_section_from_model(m)
    png = plot_seismic_section(sec, par, axis=axis, model=m, display=display)
    try:
        assert png.endswith(".png") and os.path.getsize(png) > 0
    finally:
        os.remove(png)


def test_plot_without_model_and_without_axis():
    m = _model()
    axis, sec, par = synthetic_section_from_model(m, domain="depth")
    png = plot_seismic_section(sec, par)      # axis reconstructed from parameters
    try:
        assert os.path.getsize(png) > 0
    finally:
        os.remove(png)


def test_plot_bad_display_rejected():
    m = _model()
    axis, sec, par = synthetic_section_from_model(m)
    with pytest.raises(ValueError, match="display"):
        plot_seismic_section(sec, par, display="hologram")


def test_wiggle_decimation_step():
    from tools.section_tools import _wiggle_step
    assert _wiggle_step(50) == 1 and _wiggle_step(MAX_WIGGLE_TRACES) == 1
    assert _wiggle_step(MAX_WIGGLE_TRACES + 1) == 2 and _wiggle_step(401) == 6
