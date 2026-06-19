import numpy as np
import pytest

from workflows.types import Layer
from workflows.adapters import _reduce, predict_layer
from tools.avo_tools import shuey_reflectivity
from workflows.adapters import build_interface


def test_reduce_mean_median_index():
    assert _reduce([1.0, 3.0], "mean") == 2.0
    assert _reduce([1.0, 2.0, 9.0], "median") == 2.0
    assert _reduce([5.0, 6.0, 7.0], 0) == 5.0


def test_reduce_empty_raises():
    with pytest.raises(ValueError):
        _reduce([], "mean")


def test_predict_layer_scalar_known_answer():
    # Han et al. (1986), water, vclay=0, phit=0.2:
    #   vp = (5.59 - 6.93*0.2) * 1000 = 4204 m/s
    #   vs = (3.52 - 4.91*0.2) * 1000 = 2538 m/s
    ly = predict_layer(0.2, 0.0, fluid="water")
    assert isinstance(ly, Layer)
    assert np.isclose(ly.vp, 4204.0, rtol=1e-3)
    assert np.isclose(ly.vs, 2538.0, rtol=1e-3)
    assert 0 < ly.vs < ly.vp and ly.rho > 0


def test_predict_layer_reduces_array_log_to_scalar():
    # A two-sample log; mean of phit 0.1 & 0.3 reproduces the phit=0.2 scalar
    # (the Han model is linear in phit), and the result is a plain float (G3).
    ly = predict_layer([0.1, 0.3], [0.0, 0.0], fluid="water", reduce="mean")
    assert isinstance(ly.vp, float)
    assert np.isclose(ly.vp, 4204.0, rtol=1e-3)


def test_build_interface_keys_and_values():
    upper = Layer(2500.0, 1200.0, 2.30, "shale")
    lower = Layer(3200.0, 1800.0, 2.15, "sand")
    iface = build_interface(upper, lower)
    assert iface == {
        "vp1": 2500.0, "vs1": 1200.0, "rho1": 2.30,
        "vp2": 3200.0, "vs2": 1800.0, "rho2": 2.15,
    }


def test_build_interface_feeds_shuey():
    # The whole point of G1: the assembled dict must satisfy shuey_reflectivity's
    # exact kwargs, proving the contract connects end-to-end.
    upper = Layer(2500.0, 1200.0, 2.30, "shale")
    lower = Layer(3200.0, 1800.0, 2.15, "sand")
    rc = shuey_reflectivity(**build_interface(upper, lower), angles=[0, 10, 20, 30])
    assert np.asarray(rc).shape == (4,)


def test_build_interface_rejects_nonphysical_layer():
    bad = Layer(3000.0, 3200.0, 2.2, "bad")  # vs >= vp is non-physical
    good = Layer(2500.0, 1200.0, 2.3, "ok")
    with pytest.raises(ValueError):
        build_interface(good, bad)
