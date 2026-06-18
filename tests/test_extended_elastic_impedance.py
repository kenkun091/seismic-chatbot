import numpy as np
import pytest

from tools.avo_tools import extended_elastic_impedance


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
