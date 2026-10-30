import math

import numpy as np
import pytest

from interfaces.serialize import (to_jsonable, model_summary, section_payload,
                                  interpretation_caps)


def test_to_jsonable_rounds_arrays_and_scalars():
    out = to_jsonable({"a": np.array([1.23456789, 2.0]), "b": np.float64(3.14159265),
                       "c": (1, 2), "d": np.int64(7), "e": float("nan"), "f": np.inf})
    assert out == {"a": [1.235, 2.0], "b": 3.142, "c": [1, 2], "d": 7, "e": None, "f": None}


def test_to_jsonable_keeps_small_and_zero_values():
    assert to_jsonable(0.0) == 0.0
    assert to_jsonable(0.000123456) == 0.0001235
    assert to_jsonable("s") == "s" and to_jsonable(True) is True
    assert to_jsonable({1: "x"}) == {"1": "x"}


def test_model_summary_drops_grids_and_stringifies_legend():
    model = {"facies": np.zeros((3, 2)), "vp": np.ones((3, 2)), "vs": np.ones((3, 2)),
             "rho": np.ones((3, 2)), "z": np.arange(3), "x": np.arange(2),
             "legend": {0: {"lithology": "shale", "label": "background"}},
             "regions": [{"id": 1, "label": "sand", "lithology": "sandstone"}],
             "dz": 0.5, "dx": 1.0, "nz": 3, "nx": 2, "height_m": 20.0, "width_m": 40.0,
             "pad_m": 5.0, "image_top_m": 5.0, "scale_source": "user",
             "scale_confidence": "high", "background_lithology": "shale",
             "image_path": "/tmp/x.png"}
    out = model_summary(model)
    assert set(out) == {"height_m", "width_m", "image_top_m", "dz", "dx", "nz", "nx",
                        "pad_m", "scale_source", "scale_confidence",
                        "background_lithology", "legend", "regions"}
    assert out["legend"] == {"0": {"lithology": "shale", "label": "background"}}


def test_section_payload_transposes_and_adds_photo_extent():
    section = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # nz=3, nx=2
    last = {"axis": np.array([0.25, 0.75, 1.25]), "section": section,
            "parameters": {"domain": "depth", "dx": 1.0, "nx": 2, "wavelet_freq": 30.0,
                           "angle": 0.0, "method": "shuey", "max_abs_amplitude": 6.0}}
    out = section_payload(last, {"image_top_m": 5.0, "height_m": 20.0, "width_m": 2.0})
    assert out["z"] == [0.25, 0.75, 1.25]
    assert out["traces"] == [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    assert out["image_top_m"] == 5.0 and out["height_m"] == 20.0 and out["width_m"] == 2.0
    assert out["domain"] == "depth" and out["max_abs_amplitude"] == 6.0


def test_section_payload_requires_depth_domain():
    last = {"axis": np.zeros(2), "section": np.zeros((2, 1)),
            "parameters": {"domain": "time", "dx": 1.0, "nx": 1, "wavelet_freq": 30.0,
                           "angle": 0.0, "method": "shuey", "max_abs_amplitude": 0.0}}
    with pytest.raises(ValueError, match="depth"):
        section_payload(last, None)


def test_interpretation_caps():
    interpretation_caps({"regions": [{"points": [[0, 0]] * 10}]})
    with pytest.raises(ValueError, match="regions"):
        interpretation_caps({"regions": [{}] * 201})
    with pytest.raises(ValueError, match="points"):
        interpretation_caps({"regions": [{"points": [[0, 0]] * 2001}]})
    interpretation_caps("not a dict")  # non-dicts are left for validate_interpretation
