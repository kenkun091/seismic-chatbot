import numpy as np
import pytest

from workflows.types import Layer
from workflows.adapters import _reduce, predict_layer


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
