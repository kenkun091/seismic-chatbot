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


import pytest

from tools.avo_tools import avo_attributes


def test_intercept_gradient_match_helper():
    args = dict(vp1=2400, vs1=1200, rho1=2.35, vp2=2000, vs2=1300, rho2=2.0)
    R0, G, _ = _shuey_coefficients(**args)
    res = avo_attributes(**args)
    assert np.isclose(res["intercept"], R0)
    assert np.isclose(res["gradient"], G)
    # Independently pin the gradient against a hand-computed value (closes a
    # coverage note: G must not be silently corrupted inside the helper).
    d_vp, d_vs, d_rho = -400, 100, -0.35
    avg_vp, avg_vs, avg_rho = 2200.0, 1250.0, 2.175
    expected_G = 0.5 * d_vp / avg_vp - 2 * (avg_vs ** 2 / avg_vp ** 2) * (d_rho / avg_rho + 2 * d_vs / avg_vs)
    assert np.isclose(res["gradient"], expected_G)


def test_class_iii_gas_sand():
    # Shale over gas sand: Vp and rho both drop -> A<0; gradient B<0 -> Class III.
    res = avo_attributes(vp1=2400, vs1=1100, rho1=2.35, vp2=2000, vs2=1250, rho2=2.0)
    assert res["intercept"] < 0 and res["gradient"] < 0
    assert res["avo_class"] == "III"


def test_class_i_hard_event():
    # Soft shale over hard limestone: A>0, B<0 -> Class I.
    res = avo_attributes(vp1=2500, vs1=1200, rho1=2.3, vp2=4000, vs2=2200, rho2=2.55)
    assert res["intercept"] > 0 and res["gradient"] < 0
    assert res["avo_class"] == "I"


def test_class_iv_soft_sand_low_shear():
    # Hard cap over soft gas sand with lower Vs: A<0, B>0 -> Class IV.
    res = avo_attributes(vp1=3000, vs1=1700, rho1=2.4, vp2=2600, vs2=1100, rho2=2.15)
    assert res["intercept"] < 0 and res["gradient"] > 0
    assert res["avo_class"] == "IV"


def test_class_ii_near_zero_intercept():
    # Tuned so |A| <= 0.02 -> Class II.
    res = avo_attributes(vp1=2500, vs1=1200, rho1=2.30, vp2=2560, vs2=1250, rho2=2.28)
    assert abs(res["intercept"]) <= 0.02
    assert res["avo_class"] in ("II", "IIp")


def test_avo_attributes_rejects_unphysical_medium():
    with pytest.raises(ValueError):
        avo_attributes(vp1=2000, vs1=2200, rho1=2.3, vp2=2500, vs2=1200, rho2=2.4)  # vs1>=vp1


from tools.avo_tools import _classify_avo


def test_classify_iip_negative_intercept_in_band():
    # A<0 within the near-zero band -> IIp (true polarity reversal).
    cls, desc = _classify_avo(-0.01, -0.2)
    assert cls == "IIp"
    assert "reversal" in desc.lower()


def test_classify_i_star_positive_intercept_nonneg_gradient():
    # A>0, B>=0 -> I* (amplitude rises with offset).
    cls, _ = _classify_avo(0.1, 0.05)
    assert cls == "I*"
    assert _classify_avo(0.1, 0.0)[0] == "I*"  # B==0 boundary


def test_classify_flat_gradient_negative_intercept_is_class_iii():
    # A<0, B==0: bright spot that holds with offset -> III (not the mislabeled I*).
    cls, desc = _classify_avo(-0.1, 0.0)
    assert cls == "III"
    assert "positive intercept" not in desc.lower()


import os

from tools.avo_tools import plot_avo_crossplot


def test_crossplot_returns_png_path():
    path = plot_avo_crossplot(0.1, -0.2, "I")
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.exists(path)
    os.remove(path)


def test_crossplot_without_class_label():
    path = plot_avo_crossplot(-0.15, -0.1)
    assert os.path.exists(path)
    os.remove(path)


def test_crossplot_origin_point_not_degenerate():
    # A point at the origin must still produce a valid figure (minimum extent).
    path = plot_avo_crossplot(0.0, 0.0, "II")
    assert os.path.exists(path)
    os.remove(path)


def test_avo_attributes_registered_and_chained():
    from core.tool_registry import REGISTRY_BY_NAME, AUTO_PLOT, TOOL_SCHEMAS

    assert "avo_attributes" in REGISTRY_BY_NAME
    assert "plot_avo_crossplot" in REGISTRY_BY_NAME
    assert AUTO_PLOT.get("avo_attributes") == "plot_avo_crossplot"
    spec = REGISTRY_BY_NAME["avo_attributes"]
    assert set(spec.required) == {"vp1", "vs1", "rho1", "vp2", "vs2", "rho2"}
    assert spec.validator is None
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "avo_attributes" in names and "plot_avo_crossplot" in names
