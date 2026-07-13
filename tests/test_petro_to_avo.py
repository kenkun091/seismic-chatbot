import os

import numpy as np

from workflows.adapters import predict_layer
from workflows.recipes.petro_to_avo import petro_to_avo


def test_petro_to_avo_keys_and_layers():
    res = petro_to_avo(
        phit_sand=0.25, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluid_sand="brine", method="shuey",
    )
    # The compute recipe returns at least these keys (image_path added later).
    assert {"upper", "lower", "angles", "rc", "intercept", "gradient",
            "avo_class", "method"} <= set(res)
    # Layers match predict_layer exactly (upper = shale, lower = sand).
    up = predict_layer(0.10, 0.50, fluid="water", label="shale")
    lo = predict_layer(0.25, 0.10, fluid="brine", label="sand")
    assert np.isclose(res["upper"]["vp"], up.vp)
    assert np.isclose(res["lower"]["vp"], lo.vp)
    assert res["upper"]["label"] == "shale"
    assert res["lower"]["label"] == "sand"


def test_petro_to_avo_shuey_intercept_consistency():
    # For Shuey, R(theta=0) is exactly the intercept A (same Aki-Richards R0
    # that avo_attributes reports). This pins the reflectivity curve and the
    # attributes to the same physics.
    res = petro_to_avo(
        phit_sand=0.25, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluid_sand="brine", method="shuey",
    )
    assert len(res["rc"]) == 4
    assert np.isclose(res["rc"][0], res["intercept"], rtol=1e-6, atol=1e-9)
    assert res["avo_class"] in {"I", "I*", "II", "IIp", "III", "IV"}


def test_petro_to_avo_rejects_bad_method():
    import pytest
    with pytest.raises(ValueError):
        petro_to_avo(0.25, 0.10, 0.10, 0.50, [0, 10], method="bogus")


def test_petro_to_avo_returns_image_path():
    res = petro_to_avo(
        phit_sand=0.25, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluid_sand="gas", method="shuey",
    )
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)
