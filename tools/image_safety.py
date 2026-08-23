"""Inbound image sandbox for user-uploaded outcrop photos.

The outbound twin is tools/path_safety.py (export CSVs). Here we confine the
paths that image-consuming tools READ: every image must sit inside
SEISMIC_UPLOAD_DIR, carry an allow-listed extension, and stay under a size cap.
"""
import os
import shutil
import uuid
from io import BytesIO
from typing import Tuple

from PIL import Image

ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MIME_BY_EXTENSION = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _check_extension(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"unsupported image extension {ext or '(none)'!r}; "
            f"use one of {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
        )
    return ext


def _check_size(path: str, max_mb: float) -> None:
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    if size_mb > max_mb:
        raise ValueError(f"image is {size_mb:.1f} MB; the limit is {max_mb:g} MB")


def safe_image_path(image_path: str, base_dir: str, max_mb: float = 10.0) -> str:
    """Validate ``image_path`` and return it as an absolute path inside ``base_dir``.

    Raises ``ValueError`` for a missing file, a disallowed extension, a file over
    ``max_mb``, or any path (absolute, relative, or via ``..``) that resolves
    outside ``base_dir``.
    """
    if not image_path:
        raise ValueError("no image path given")
    base_abs = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.normpath(image_path))
    if candidate != base_abs and not candidate.startswith(base_abs + os.sep):
        raise ValueError(
            f"image path is outside the upload directory: {image_path!r}"
        )
    if not os.path.isfile(candidate):
        raise ValueError(f"image not found: {image_path!r}")
    _check_extension(candidate)
    _check_size(candidate, max_mb)
    return candidate


def stage_upload(src_path: str, base_dir: str, session_id: str,
                 max_mb: float = 10.0) -> str:
    """Copy an uploaded file into ``base_dir/session_id/<uuid><ext>``.

    The source may live anywhere (Gradio's temp dir); the copy is what the
    tools are allowed to read. Returns the staged absolute path.
    """
    if not src_path or not os.path.isfile(src_path):
        raise ValueError(f"uploaded image not found: {src_path!r}")
    ext = _check_extension(src_path)
    _check_size(src_path, max_mb)
    dest_dir = os.path.join(os.path.abspath(base_dir), str(session_id))
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{uuid.uuid4().hex}{ext}")
    shutil.copyfile(src_path, dest)
    return dest


def image_size(image_path: str) -> Tuple[int, int]:
    """(width_px, height_px) of the image."""
    with Image.open(image_path) as im:
        return im.size


def downscale_for_vision(image_path: str, max_edge: int = 1568) -> Tuple[bytes, str]:
    """Return ``(jpeg_bytes, "image/jpeg")`` with the long edge <= ``max_edge``.

    Always re-encodes as JPEG (RGB) so the vision request is small and the
    MIME type is predictable.
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max(w, h) / float(max_edge)
        if scale > 1.0:
            im = im.resize((max(1, int(round(w / scale))),
                            max(1, int(round(h / scale)))), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"
