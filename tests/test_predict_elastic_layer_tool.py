import numpy as np

from core import tool_registry as reg
from workflows.adapters import predict_elastic_layer


def test_predict_elastic_layer_returns_plain_dict():
    out = predict_elastic_layer(0.2, 0.0, fluid="water")
    assert set(out) == {"vp", "vs", "rho", "vp_vs"}
    assert np.isclose(out["vp"], 4204.0, rtol=1e-3)
    assert np.isclose(out["vp_vs"], out["vp"] / out["vs"], rtol=1e-9)


def test_predict_elastic_layer_registered():
    assert "predict_elastic_layer" in reg.REGISTRY_BY_NAME
    assert "predict_elastic_layer" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"predict_elastic_layer"}
