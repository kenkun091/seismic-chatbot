"""interpret_outcrop (scripted vision) and the interpretation overlay plot."""
import json
import os

import pytest

from tools import outcrop_tools as ot

GOOD = {
    "regions": [
        {"id": 1, "label": "dark shale band", "lithology": "shale",
         "geometry": {"type": "band", "y_top": 0.2, "y_bottom": 0.4}},
        {"id": 2, "label": "sand lens", "lithology": "sandstone",
         "geometry": {"type": "polygon",
                      "points": [[0.25, 0.55], [0.75, 0.55], [0.75, 0.85], [0.25, 0.85]]},
         "porosity": 0.22},
        {"id": 3, "label": "bushes", "lithology": "cover",
         "geometry": {"type": "polygon", "points": [[0, 0], [0.1, 0], [0.1, 0.1]]}},
    ],
    "scale": {"estimated_height_m": 25, "reference": "person", "confidence": "medium"},
    "background_lithology": "shale",
    "mode": "polygons",
}


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "SEISMIC_UPLOAD_DIR", str(tmp_path))


def test_extract_json_strips_fences_and_prose():
    text = "Here you go:\n```json\n{\"a\": 1}\n```\nThanks."
    assert ot.extract_json(text) == {"a": 1}


def test_extract_json_failure_raises():
    with pytest.raises(ValueError, match="JSON"):
        ot.extract_json("no braces here")


def test_interpret_outcrop_happy_path(outcrop_image, fake_vision_factory):
    fake = fake_vision_factory([json.dumps(GOOD)])
    out = ot.interpret_outcrop(outcrop_image, vision_client=fake)
    assert out["image_path"] == os.path.abspath(outcrop_image)
    assert out["image_size"] == [400, 200]
    assert [r["lithology"] for r in out["regions"]] == ["shale", "sandstone", "cover"]
    assert out["scale"]["estimated_height_m"] == 25.0
    assert "25" in out["summary"] and "sandstone" in out["summary"]
    mime, prompt = fake.calls[0]
    assert mime == "image/jpeg"
    assert prompt == ot.OUTCROP_PROMPT
    assert "sandstone" in prompt and "estimated_height_m" in prompt


def test_interpret_outcrop_retries_once_with_error(outcrop_image, fake_vision_factory):
    fake = fake_vision_factory(["garbage", json.dumps(GOOD)])
    out = ot.interpret_outcrop(outcrop_image, vision_client=fake)
    assert len(fake.calls) == 2
    assert "previous answer was invalid" in fake.calls[1][1]
    assert len(out["regions"]) == 3


def test_interpret_outcrop_fails_after_second_bad_answer(outcrop_image, fake_vision_factory):
    bad = json.dumps({"regions": [{"lithology": "unobtainium",
                                   "geometry": {"type": "band", "y_top": 0, "y_bottom": 1}}]})
    fake = fake_vision_factory(["garbage", bad])
    with pytest.raises(ValueError, match="could not interpret image"):
        ot.interpret_outcrop(outcrop_image, vision_client=fake)
    assert len(fake.calls) == 2


def test_interpret_outcrop_without_image_asks_for_upload(fake_vision_factory):
    with pytest.raises(ValueError, match="upload an outcrop photo"):
        ot.interpret_outcrop(None, vision_client=fake_vision_factory([]))


def test_interpret_outcrop_rejects_path_outside_sandbox(tmp_path, fake_vision_factory, monkeypatch):
    monkeypatch.setattr(ot, "SEISMIC_UPLOAD_DIR", str(tmp_path / "elsewhere"))
    from PIL import Image
    p = tmp_path / "x.png"
    Image.new("RGB", (10, 10)).save(p)
    with pytest.raises(ValueError, match="outside"):
        ot.interpret_outcrop(str(p), vision_client=fake_vision_factory([]))


def test_interpret_outcrop_unconfigured_vision_raises(outcrop_image, monkeypatch):
    import core.vision_client as vc
    for name in ("VISION_PROVIDER", "ANTHROPIC_API_KEY", "VISION_API_KEY", "VISION_BASE_URL"):
        monkeypatch.setattr(vc, name, None)
    with pytest.raises(RuntimeError, match="vision provider not configured"):
        ot.interpret_outcrop(outcrop_image)


def test_plot_overlay_writes_png(outcrop_image, fake_vision_factory):
    out = ot.interpret_outcrop(outcrop_image, vision_client=fake_vision_factory([json.dumps(GOOD)]))
    png = ot.plot_outcrop_interpretation(out)
    try:
        assert png.endswith(".png") and os.path.getsize(png) > 0
    finally:
        os.remove(png)
