import os

import numpy as np
import pytest

from tools.avo_tools import extended_elastic_impedance, plot_extended_elastic_impedance


def test_chi_zero_is_acoustic_impedance():
    # At chi=0: p=1, q=0, r=1 -> EEI = Vp*rho (acoustic impedance), raw.
    vp, vs, rho = 3000.0, 1500.0, 2.3
    eei = extended_elastic_impedance(vp, vs, rho, chi=[0.0])
    assert np.isclose(eei[0], vp * rho)


def test_closed_form_at_chi_30():
    vp, vs, rho = 3000.0, 1500.0, 2.3
    chi_deg = 30.0
    x = np.radians(chi_deg)
    K = (vs / vp) ** 2
    p = np.cos(x) + np.sin(x)
    q = -8.0 * K * np.sin(x)
    r = np.cos(x) - 4.0 * K * np.sin(x)
    expected = vp ** p * vs ** q * rho ** r
    eei = extended_elastic_impedance(vp, vs, rho, chi=[chi_deg])
    assert np.isclose(eei[0], expected)


def test_eei_varies_with_chi():
    eei = extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[-45.0, 0.0, 45.0])
    assert eei.shape == (3,)
    assert not np.allclose(eei, eei[0])


def test_normalization_anchors_at_chi_zero_differs_elsewhere():
    vp, vs, rho = 3000.0, 1500.0, 2.3
    # A background reference different from the sample.
    ref = dict(vp0=2800.0, vs0=1400.0, rho0=2.2)
    chi = [0.0, 45.0]
    raw = extended_elastic_impedance(vp, vs, rho, chi=chi)
    norm = extended_elastic_impedance(vp, vs, rho, chi=chi, **ref)
    # chi=0: normalization scale is 1 -> same as raw (== Vp*rho).
    assert np.isclose(norm[0], raw[0])
    assert np.isclose(norm[0], vp * rho)
    # chi=45: reference rescales the value -> differs from raw.
    assert not np.isclose(norm[1], raw[1])


def test_k_override_changes_result():
    vp, vs, rho = 3000.0, 1500.0, 2.3
    default = extended_elastic_impedance(vp, vs, rho, chi=[45.0])
    overridden = extended_elastic_impedance(vp, vs, rho, chi=[45.0], k=0.1)
    assert not np.isclose(default[0], overridden[0])


def test_guard_rejects_vs_ge_vp():
    # vs >= vp -> unphysical medium (caught by require_elastic_medium).
    with pytest.raises(ValueError, match="vs"):
        extended_elastic_impedance(2000.0, 2200.0, 2.3, chi=[0.0])


def test_guard_rejects_chi_out_of_range():
    with pytest.raises(ValueError, match="rotation angle"):
        extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[91.0])


def test_guard_rejects_partial_reference():
    # Only vp0 supplied -> all-or-nothing rule.
    with pytest.raises(ValueError, match="all-or-nothing"):
        extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[0.0], vp0=2800.0)


def test_guard_rejects_nonpositive_reference():
    with pytest.raises(ValueError, match="must be positive"):
        extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[0.0],
                                   vp0=2800.0, vs0=1400.0, rho0=-1.0)


def test_eei_plot_returns_png_path():
    chi = np.linspace(-90, 90, 37)
    eei = extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=chi)
    path = plot_extended_elastic_impedance(chi, eei)
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.exists(path)
    os.remove(path)


def test_eei_registered_and_chained():
    from core.tool_registry import REGISTRY_BY_NAME, AUTO_PLOT, TOOL_SCHEMAS

    assert "extended_elastic_impedance" in REGISTRY_BY_NAME
    assert "plot_extended_elastic_impedance" in REGISTRY_BY_NAME
    assert AUTO_PLOT.get("extended_elastic_impedance") == "plot_extended_elastic_impedance"
    spec = REGISTRY_BY_NAME["extended_elastic_impedance"]
    assert set(spec.required) == {"vp", "vs", "rho", "chi"}
    assert spec.validator is None
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "extended_elastic_impedance" in names and "plot_extended_elastic_impedance" in names
