import os

import numpy as np
import pytest

from workflows.recipes.fluid_scenario import fluid_scenario


def test_fluid_scenario_keys_and_cases():
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluids=["brine", "gas"], fluid_in="brine",
    )
    assert {"shale", "fluids", "fluid_in", "cases", "angles", "method"} <= set(res)
    assert set(res["cases"]) == {"brine", "gas"}
    for f in ("brine", "gas"):
        c = res["cases"][f]
        assert {"layer", "rc", "intercept", "gradient", "avo_class"} <= set(c)
        assert len(c["rc"]) == 4


def test_fluid_scenario_gas_softens_sand():
    # Gassmann brine->gas: Vp down, Vs up (shear-independent, lower density), rho down.
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluids=["brine", "gas"], fluid_in="brine",
    )
    brine = res["cases"]["brine"]["layer"]
    gas = res["cases"]["gas"]["layer"]
    assert gas["vp"] < brine["vp"]
    assert gas["vs"] > brine["vs"]
    assert gas["rho"] < brine["rho"]


def test_fluid_scenario_shuey_intercept_consistency():
    # Per case, Shuey R(0) == that case's intercept; classes are valid labels.
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], method="shuey",  # default fluids ["brine","gas"]
    )
    for c in res["cases"].values():
        assert np.isclose(c["rc"][0], c["intercept"], rtol=1e-6, atol=1e-9)
        assert c["avo_class"] in {"I", "I*", "II", "IIp", "III", "IV"}


def test_fluid_scenario_rejects_bad_method():
    with pytest.raises(ValueError):
        fluid_scenario(0.28, 0.10, 0.10, 0.50, [0, 10], method="bogus")


def test_fluid_scenario_rejects_empty_fluids():
    with pytest.raises(ValueError):
        fluid_scenario(0.28, 0.10, 0.10, 0.50, [0, 10], fluids=[])


def test_fluid_scenario_returns_image_path():
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluids=["brine", "gas"],
    )
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)
