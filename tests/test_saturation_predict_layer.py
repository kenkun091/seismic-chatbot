import numpy as np
import pytest

from workflows.adapters import predict_layer


def test_predict_layer_sw_none_unchanged():
    a = predict_layer(0.25, 0.20, fluid="gas")
    b = predict_layer(0.25, 0.20, fluid="gas", sw=None)
    assert (a.vp, a.vs, a.rho) == (b.vp, b.vs, b.rho)


def test_predict_layer_sw1_matches_brine():
    sat = predict_layer(0.25, 0.20, fluid="gas", sw=1.0)
    brine = predict_layer(0.25, 0.20, fluid="water")
    assert np.isclose(sat.vp, brine.vp) and np.isclose(sat.rho, brine.rho)


def test_predict_layer_sw0_matches_gas():
    sat = predict_layer(0.25, 0.20, fluid="gas", sw=0.0)
    gas = predict_layer(0.25, 0.20, fluid="gas")
    assert np.isclose(sat.vp, gas.vp) and np.isclose(sat.rho, gas.rho)


def test_predict_layer_sw_requires_hydrocarbon_fluid():
    with pytest.raises(ValueError):
        predict_layer(0.25, 0.20, fluid="water", sw=0.5)
