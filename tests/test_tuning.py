import os

import numpy as np
import pytest

from workflows.adapters import predict_layer
from workflows.recipes.tuning import tuning


def test_tuning_keys_and_shapes():
    res = tuning(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        max_thickness=40.0, wavelet_freq=30.0, num_traces=41,
    )
    assert {"sand", "shale", "tuning_thickness", "tuning_amplitude",
            "resolution_limit", "thicknesses", "max_amplitudes",
            "wavelet_freq", "max_thickness"} <= set(res)
    assert len(res["thicknesses"]) == 41
    assert len(res["max_amplitudes"]) == 41
    assert all(np.isfinite(res["max_amplitudes"]))
    assert res["tuning_thickness"] > 0
    assert res["resolution_limit"] > 0


def test_tuning_thickness_known_answer():
    # analyze_wedge defines tuning_thickness = v2/(4f), resolution_limit = v2/(8f),
    # where v2 is the sand Vp (layer 2 from build_earth_model). This pins the recipe
    # to BOTH the rock-physics prediction AND the correct shale/sand/shale mapping.
    res = tuning(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        max_thickness=40.0, wavelet_freq=30.0,
    )
    sand = predict_layer(0.28, 0.10, fluid="brine", label="sand")
    expected = sand.vp / (4.0 * 30.0)
    assert np.isclose(res["tuning_thickness"], expected, rtol=1e-6)
    assert np.isclose(res["resolution_limit"], expected / 2.0, rtol=1e-6)
    assert res["sand"]["vp"] == pytest.approx(sand.vp)


def test_tuning_higher_freq_resolves_thinner():
    # tuning_thickness = v2/(4f): higher frequency -> thinner tuning / better resolution.
    lo = tuning(0.28, 0.10, 0.10, 0.50, max_thickness=40.0, wavelet_freq=20.0)
    hi = tuning(0.28, 0.10, 0.10, 0.50, max_thickness=40.0, wavelet_freq=50.0)
    assert hi["tuning_thickness"] < lo["tuning_thickness"]
    assert hi["resolution_limit"] < lo["resolution_limit"]


def test_tuning_returns_image_path():
    res = tuning(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        max_thickness=40.0, wavelet_freq=30.0, num_traces=41,
    )
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)


def test_tuning_fluid_sand_gas():
    res = tuning(
        phit_sand=0.25, vclay_sand=0.15,
        phit_shale=0.10, vclay_shale=0.30,
        max_thickness=50, wavelet_freq=30, num_traces=21,
        fluid_sand="gas",
    )
    assert res["tuning_thickness"] > 0
    assert res["image_path"].endswith(".png")
    os.remove(res["image_path"])


def test_tuning_zero_max_thickness_raises():
    with pytest.raises(ValueError):
        tuning(
            phit_sand=0.25, vclay_sand=0.15,
            phit_shale=0.10, vclay_shale=0.30,
            max_thickness=0,
        )


def test_tuning_num_traces_one_raises():
    with pytest.raises(ValueError):
        tuning(
            phit_sand=0.25, vclay_sand=0.15,
            phit_shale=0.10, vclay_shale=0.30,
            max_thickness=50, num_traces=1,
        )
