"""Wedge/wavelet correctness fixes: honored num_traces/dt, Ormsby dominant
frequency for tuning, and single-angle (not RC-averaged) angle response."""
import numpy as np

from tools.wedge_tools import create_wedge_model
from tools.avo_tools import shuey_reflectivity

BASE = dict(max_thickness=60, v1=2500, v2=3000, v3=3500, rho1=2.2, rho2=2.3, rho3=2.4)


def test_num_traces_is_respected():
    _, _, synth, params = create_wedge_model(num_traces=41, **BASE)
    assert 41 in synth.shape
    assert params["num_traces"] == 41


def test_dt_is_respected():
    _, _, synth, params = create_wedge_model(dt=0.5, **BASE)
    assert params["dt"] == 0.5


def test_ormsby_tuning_uses_dominant_frequency():
    # Ormsby corners 5,10,40,50 -> dominant ~ (f2+f3)/2 = 25 Hz, NOT f1=5.
    _, _, _, params = create_wedge_model(
        wv_type="ormsby", ormsby_freq="5,10,40,50", **BASE
    )
    assert abs(params["wavelet_freq"] - 25.0) < 1e-6


def test_multi_angle_uses_first_angle_not_mean():
    # The wedge is a single-angle product; a list must use one angle, not the
    # (physically meaningless) mean of reflection coefficients across angles.
    _, _, _, params = create_wedge_model(incident_angle=[10, 20, 30], **BASE)
    expected_rc1 = float(shuey_reflectivity(
        vp1=2500, vs1=1250, rho1=2.2, vp2=3000, vs2=1500, rho2=2.3, angles=[10]
    )[0])
    assert abs(params["rc1"] - expected_rc1) < 1e-9
