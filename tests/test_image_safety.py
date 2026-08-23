"""Inbound image sandbox: extension allow-list, size cap, confinement, staging."""
import os

import pytest
from PIL import Image

from tools.image_safety import (safe_image_path, stage_upload,
                                downscale_for_vision, image_size)


def _make_png(path, w=64, h=32):
    Image.new("RGB", (w, h), (120, 100, 80)).save(path)
    return str(path)


def test_valid_png_inside_base_is_returned_absolute(tmp_path):
    p = _make_png(tmp_path / "a.png")
    assert safe_image_path(p, str(tmp_path)) == os.path.abspath(p)


def test_path_outside_base_rejected(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    p = _make_png(other / "a.png")
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError, match="outside"):
        safe_image_path(p, str(base))


def test_traversal_rejected(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    p = _make_png(tmp_path / "a.png")
    with pytest.raises(ValueError):
        safe_image_path(os.path.join(str(base), "..", "a.png"), str(base))


def test_bad_extension_rejected(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("not an image")
    with pytest.raises(ValueError, match="extension"):
        safe_image_path(str(p), str(tmp_path))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        safe_image_path(str(tmp_path / "nope.png"), str(tmp_path))


def test_size_cap_rejected(tmp_path):
    p = _make_png(tmp_path / "a.png")
    with pytest.raises(ValueError, match="MB"):
        safe_image_path(p, str(tmp_path), max_mb=0.00001)


def test_stage_upload_copies_into_session_dir(tmp_path):
    src = _make_png(tmp_path / "src.JPG".lower())
    base = tmp_path / "uploads"
    staged = stage_upload(src, str(base), "sess1")
    assert staged.startswith(os.path.join(str(base), "sess1"))
    assert staged.endswith(".jpg")
    assert os.path.getsize(staged) == os.path.getsize(src)
    # the staged path passes the sandbox check
    assert safe_image_path(staged, str(base)) == staged


def test_stage_upload_rejects_bad_extension(tmp_path):
    src = tmp_path / "x.gif"
    src.write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="extension"):
        stage_upload(str(src), str(tmp_path / "u"), "s")


def test_downscale_limits_long_edge(tmp_path):
    p = _make_png(tmp_path / "big.png", w=4000, h=1000)
    data, mime = downscale_for_vision(p, max_edge=800)
    assert mime == "image/jpeg"
    from io import BytesIO
    w, h = Image.open(BytesIO(data)).size
    assert max(w, h) == 800 and w == 800 and h == 200


def test_downscale_keeps_small_image(tmp_path):
    p = _make_png(tmp_path / "small.png", w=300, h=200)
    data, mime = downscale_for_vision(p, max_edge=1568)
    from io import BytesIO
    assert Image.open(BytesIO(data)).size == (300, 200)


def test_image_size(tmp_path):
    assert image_size(_make_png(tmp_path / "a.png", 64, 32)) == (64, 32)
