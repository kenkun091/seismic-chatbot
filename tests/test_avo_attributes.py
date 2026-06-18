import numpy as np

from tools.avo_tools import _shuey_coefficients, shuey_reflectivity


def test_shuey_coefficients_intercept_matches_closed_form():
    args = dict(vp1=2400, vs1=1200, rho1=2.35, vp2=2000, vs2=1300, rho2=2.0)
    R0, G, F = _shuey_coefficients(**args)
    d_vp, d_rho = args["vp2"] - args["vp1"], args["rho2"] - args["rho1"]
    avg_vp, avg_rho = 0.5 * (args["vp1"] + args["vp2"]), 0.5 * (args["rho1"] + args["rho2"])
    assert np.isclose(R0, 0.5 * (d_vp / avg_vp + d_rho / avg_rho))
    # Intercept == zero-angle reflectivity.
    assert np.isclose(shuey_reflectivity(angles=[0.0], **args)[0], R0)


def test_shuey_reflectivity_unchanged_by_refactor():
    args = dict(vp1=2400, vs1=1200, rho1=2.35, vp2=3000, vs2=1500, rho2=2.4)
    angles = [0.0, 10.0, 20.0, 30.0]
    R0, G, F = _shuey_coefficients(**args)
    th = np.radians(angles)
    expected = R0 + G * np.sin(th) ** 2 + F * (np.tan(th) ** 2 - np.sin(th) ** 2)
    got = shuey_reflectivity(angles=angles, **args)
    assert np.allclose(got, expected)
