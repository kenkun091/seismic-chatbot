"""outcrop_to_seismic recipe: end-to-end with scripted vision; registration; sweep."""
import json
import os

import numpy as np
import pytest

from tools import outcrop_tools as ot
from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic

INTERP = {
    "regions": [
        {"id": 1, "label": "sand bed", "lithology": "sandstone",
         "geometry": {"type": "band", "y_top": 0.3, "y_bottom": 0.5}},
        {"id": 2, "label": "lime lens", "lithology": "limestone",
         "geometry": {"type": "polygon", "points": [[0.3, 0.6], [0.7, 0.6], [0.7, 0.8], [0.3, 0.8]]}},
    ],
    "scale": {"estimated_height_m": 20, "reference": "hammer", "confidence": "medium"},
    "background_lithology": "shale", "mode": "polygons",
}


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "SEISMIC_UPLOAD_DIR", str(tmp_path))


def _cleanup(res):
    for p in [res.get("image_path")] + list(res.get("extra_image_paths") or []):
        if p and os.path.exists(p):
            os.remove(p)


def test_end_to_end(outcrop_image, fake_vision_factory):
    res = outcrop_to_seismic(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]),
                             num_traces=41)
    try:
        assert res["n_regions"] == 2 and res["scale"] == {"height_m": 20.0, "source": "vision",
                                                         "confidence": "medium"}
        assert res["grid_shape"][1] == 41
        assert res["section"]["section"].shape[1] == 41
        assert res["max_abs_amplitude"] > 0 and res["n_interfaces"] > 0
        assert res["image_path"].endswith(".png") and os.path.getsize(res["image_path"]) > 0
        assert len(res["extra_image_paths"]) == 1 and os.path.getsize(res["extra_image_paths"][0]) > 0
        assert res["interpretation"]["regions"][1]["lithology"] == "limestone"
        assert res["model"]["facies"].shape == tuple(res["grid_shape"])
    finally:
        _cleanup(res)


def test_height_override_and_gas_override(outcrop_image, fake_vision_factory):
    mk = lambda: fake_vision_factory([json.dumps(INTERP)])
    base = outcrop_to_seismic(outcrop_image, vision_client=mk(), num_traces=21)
    gas = outcrop_to_seismic(outcrop_image, vision_client=mk(), num_traces=21, height_m=40.0,
                             overrides={"sand bed": {"fluid": "gas"}})
    try:
        assert gas["scale"]["source"] == "user" and gas["model"]["height_m"] == 40.0
        sand = [r for r in gas["regions"] if r["id"] == 1][0]
        assert sand["fluid"] == "gas"
        assert gas["model"]["nz"] > base["model"]["nz"]
    finally:
        _cleanup(base); _cleanup(gas)


def test_depth_domain_and_wiggle(outcrop_image, fake_vision_factory):
    res = outcrop_to_seismic(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]),
                             num_traces=21, domain="depth", display="wiggle")
    try:
        assert res["domain"] == "depth"
        assert res["section"]["section"].shape[0] == res["model"]["nz"]
    finally:
        _cleanup(res)


def test_requires_image(fake_vision_factory):
    with pytest.raises(ValueError, match="upload an outcrop photo"):
        outcrop_to_seismic(None, vision_client=fake_vision_factory([]))


def test_registered_as_workflow_and_tool():
    from workflows.engine import WORKFLOW_REGISTRY_BY_NAME
    from core import tool_registry as reg
    spec = WORKFLOW_REGISTRY_BY_NAME["outcrop_to_seismic"]
    assert spec.required == []
    assert spec.defaults["image_path"] is None
    assert "outcrop_to_seismic" in reg.REGISTRY_BY_NAME
    assert "vision_client" not in spec.params


def test_run_sweep_over_frequency(outcrop_image, fake_vision_factory, monkeypatch):
    """run_sweep takes a recipe NAME and runs it through the engine, so the
    vision builder is patched instead of passing a client."""
    from workflows.sweep import run_sweep
    fake = fake_vision_factory([json.dumps(INTERP)] * 2)
    monkeypatch.setattr("core.vision_client.build_vision_client", lambda: fake)
    res = run_sweep("outcrop_to_seismic", {"wavelet_freq": [20.0, 40.0]}, "max_abs_amplitude",
                    fixed={"image_path": outcrop_image, "num_traces": 11})
    assert res["stats"]["kind"] == "numeric"
    assert len(res["rows"]) == 2
    assert len(fake.calls) == 2


def test_recipe_default_display_is_overlay(outcrop_image, fake_vision_factory):
    res = outcrop_to_seismic(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]),
                             num_traces=11)
    try:
        assert res["section"]["parameters"]["display"] == "overlay"
    finally:
        _cleanup(res)
