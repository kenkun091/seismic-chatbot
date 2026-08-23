# Outcrop Photo → Seismic Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload an outcrop photo, interpret it into facies regions with a vision LLM, establish a scale, map facies to elastic properties on a shale background, and convolve a 2-D reflectivity model into a seismic section (wiggle or image, time or depth).

**Architecture:** Four staged registry tools hand results through the per-session `ContextManager` (`interpret_outcrop` → `outcrop_to_model` → `synthetic_section` → auto-plot), so only step 1 touches the vision model and corrections re-run offline. A generic 2-D convolutional model (`tools/section_tools.py`) has no outcrop knowledge; outcrop-specific code (prompt, schema, lithology table, rasterization) lives in `tools/outcrop_tools.py`; a `VisionClient` protocol with Anthropic and OpenAI-compatible backends lives in `core/vision_client.py`. A one-shot `outcrop_to_seismic` workflow recipe chains everything.

**Tech Stack:** Python 3.9 (no `X | Y` unions, no `match`), numpy 1.26, scipy, matplotlib (`matplotlib.path.Path` for polygon fill), Pillow 8.4, `openai` SDK (present), `anthropic` SDK (new, lazily imported), Gradio 3.50, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-outcrop-to-seismic-design.md`

## Global Constraints

- Run everything from inside `geo-mcp/seismic_chatbot/` (absolute top-level imports: `from config.settings import ...`). Commit with `git` from this directory (it is its own repo).
- Python 3.9 syntax only: use `typing.Optional/List/Dict/Tuple`, never `X | None`.
- Tests must be offline: no network, no API keys. Anything touching the chatbot uses `fake_llm_factory` (`tests/conftest.py`); anything touching vision uses the `FakeVisionClient` added in Task 2.
- Units: velocities m/s, density g/cc, lengths m, **time in ms** (`dt` in ms, as `create_synthetic_seismogram` and `gen_wavelet` already use). Normalized image coordinates: x → right, y → down, in [0, 1].
- Every LLM-facing tool is declared **only** in `core/tool_registry.py::REGISTRY` (plus `WORKFLOW_REGISTRY` for recipes). Never add parallel schema tables.
- Per-conversation state goes on `ContextManager` (keys `last_image`, `last_outcrop`, `last_earth_model`, `last_section`); never on shared components or module globals.
- Guards: REJECT with `ValueError` (clear message, surfaced to chat); WARN with `warnings.warn` and proceed. Guards live in the compute functions (recipes bypass registry validators).
- Plot functions return a PNG path created with `tempfile.mkstemp(suffix=".png")` and `plt.close(fig)`.
- `tests/test_tool_registry.py::test_registry_nonempty` pins `len(REGISTRY)`; update it in the task that adds tools (33 → 38 in Task 8, → 39 in Task 9).
- Deviations from the spec, decided while planning (each noted where it applies): `VisionClient.interpret_image` returns raw text and JSON parsing/retry lives in `outcrop_tools`; shale default `vclay` is 0.50 (Han's range ceiling — the spec's 0.70 would warn-and-clip on every run); `dt` default is 1.0 ms (not 0.001 s); the wedge oracle is covered transitively (the 1-D tool is already oracle-tested against the wedge), so Task 6 tests against the 1-D tool only; the recipe returns the section plot as `image_path` and the interpretation overlay in `extra_image_paths`.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `config/settings.py` | modify | `SEISMIC_UPLOAD_DIR`, `MAX_IMAGE_MB`, `VISION_PROVIDER`, `ANTHROPIC_API_KEY`, `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL` |
| `tools/image_safety.py` | create | upload sandbox (`safe_image_path`, `stage_upload`), `downscale_for_vision`, `image_size` |
| `core/vision_client.py` | create | `VisionClient` protocol, `AnthropicVisionClient`, `OpenAIVisionClient`, `resolve_vision_backend`, `build_vision_client` |
| `tools/outcrop_tools.py` | create | `LITHOLOGY_TABLE`, `validate_interpretation`, `resolve_lithology`, `OUTCROP_PROMPT`, `extract_json`, `interpret_outcrop`, `plot_outcrop_interpretation`, `apply_overrides`, `outcrop_to_model` |
| `tools/section_tools.py` | create | `validate_section_inputs`, `create_synthetic_section`, `depth_convert`, `synthetic_section_from_model`, `plot_seismic_section` |
| `workflows/recipes/outcrop_to_seismic.py` | create | one-shot recipe |
| `workflows/engine.py` | modify | `WorkflowSpec` for `outcrop_to_seismic` |
| `core/tool_registry.py` | modify | five `ToolSpec`s + auto-plot entries |
| `core/chatbot_tool_use.py` | modify | `session_id`, `attach_image`, `_inject_context_inputs`, context/auto-chain/harvest branches, intent short-circuit, prompt bullets |
| `interfaces/gradio_interface.py` | modify | `gr.Image` upload, `prepare_turn` |
| `requirements.txt` | modify | `anthropic>=0.40.0` |
| `tests/conftest.py` | modify | `FakeVisionClient` + `fake_vision_factory` + `outcrop_image` fixtures |
| `tests/test_image_safety.py`, `test_vision_client.py`, `test_outcrop_interpretation.py`, `test_outcrop_tools.py`, `test_outcrop_model.py`, `test_section_tools.py`, `test_section_plot.py`, `test_outcrop_registry.py`, `test_outcrop_to_seismic.py`, `test_chatbot_outcrop.py`, `test_gradio_upload.py` | create | per-task tests |
| `test_outcrop_vision.py` (package root) | create | credential-gated real-VLM smoke script (not in suite) |
| `CLAUDE.md`, `config/example_prompts.py`, `interfaces/web_interface.html`, `docs/superpowers/specs/2026-06-15-scientific-completeness-roadmap.md` | modify | docs + prompts sync |

---

### Task 1: Image upload sandbox (`tools/image_safety.py`)

**Files:**
- Create: `tools/image_safety.py`
- Modify: `config/settings.py` (append after the Databricks block)
- Test: `tests/test_image_safety.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ALLOWED_IMAGE_EXTENSIONS: frozenset`, `MIME_BY_EXTENSION: dict`
  - `safe_image_path(image_path: str, base_dir: str, max_mb: float = 10.0) -> str` — absolute path inside `base_dir`, or `ValueError`.
  - `stage_upload(src_path: str, base_dir: str, session_id: str, max_mb: float = 10.0) -> str` — copies an arbitrary local file into `base_dir/session_id/<uuid>.<ext>`, returns the new path.
  - `downscale_for_vision(image_path: str, max_edge: int = 1568) -> Tuple[bytes, str]` — `(encoded_bytes, mime)`.
  - `image_size(image_path: str) -> Tuple[int, int]` — `(width_px, height_px)`.
  - Settings: `SEISMIC_UPLOAD_DIR: str`, `MAX_IMAGE_MB: float`.

- [ ] **Step 1: Add settings**

Append to `config/settings.py` after the `DATABRICKS_BASE_URL` line:

```python
import tempfile  # (place this import at the top of the file, after `import os`)

# Outcrop-photo upload sandbox (tools/image_safety.py). Absolute paths outside
# this directory are rejected by every image-consuming tool.
SEISMIC_UPLOAD_DIR = os.environ.get("SEISMIC_UPLOAD_DIR") or os.path.join(
    tempfile.gettempdir(), "seismic_uploads"
)
MAX_IMAGE_MB = float(os.environ.get("MAX_IMAGE_MB", "10"))
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_image_safety.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_image_safety.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.image_safety'`

- [ ] **Step 4: Implement `tools/image_safety.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_image_safety.py -q`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add tools/image_safety.py tests/test_image_safety.py config/settings.py
git commit -m "feat(image): inbound upload sandbox — safe_image_path, stage_upload, downscale_for_vision"
```

---

### Task 2: Vision client with two backends (`core/vision_client.py`)

**Files:**
- Create: `core/vision_client.py`
- Modify: `config/settings.py` (append), `requirements.txt`, `tests/conftest.py`
- Test: `tests/test_vision_client.py`

**Interfaces:**
- Consumes: settings from Task 1 pattern.
- Produces:
  - `class VisionClient(Protocol)`: `interpret_image(self, image_bytes: bytes, mime: str, prompt: str) -> str` (raw model text; JSON handling is the caller's job — spec deviation noted in Global Constraints).
  - `AnthropicVisionClient(api_key, model=None, max_tokens=2048, client=None)`
  - `OpenAIVisionClient(api_key, base_url, model=None, max_tokens=2048, client=None)`
  - `DEFAULT_VISION_MODELS = {"anthropic": "claude-sonnet-5", "openai": "gpt-4o"}`
  - `resolve_vision_backend(provider, anthropic_key, vision_key, vision_url) -> Optional[str]` — `"anthropic"`, `"openai"`, or `None` (not configured); raises `RuntimeError` if an explicit provider lacks credentials.
  - `build_vision_client() -> VisionClient` — from settings; `RuntimeError` when unconfigured.
  - Settings: `VISION_PROVIDER`, `ANTHROPIC_API_KEY`, `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL`.
  - conftest: `FakeVisionClient(responses)` with `.calls` list of `(mime, prompt)`; fixture `fake_vision_factory`; fixture `outcrop_image(tmp_path)` → path of a 400×200 PNG with a dark band (rows 40–80) and a light lens.

- [ ] **Step 1: Add settings and dependency**

Append to `config/settings.py`:

```python
# Vision provider for outcrop-photo interpretation (core/vision_client.py).
# Optional: when nothing is set, interpret_outcrop raises a clear RuntimeError
# at call time and every other tool keeps working.
VISION_PROVIDER = os.environ.get("VISION_PROVIDER")          # "anthropic" | "openai" | None (auto)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
VISION_API_KEY = os.environ.get("VISION_API_KEY")            # OpenAI-compatible vision endpoint
VISION_BASE_URL = os.environ.get("VISION_BASE_URL")
VISION_MODEL = os.environ.get("VISION_MODEL")                # provider default when unset
```

Append to `requirements.txt` under `# Core dependencies`:

```
anthropic>=0.40.0
```

Then run: `pip install "anthropic>=0.40.0"`

- [ ] **Step 2: Add fakes to `tests/conftest.py`**

Append:

```python
class FakeVisionClient:
    """Scripted vision responses (raw text); records (mime, prompt) per call."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def interpret_image(self, image_bytes, mime, prompt):
        self.calls.append((mime, prompt))
        if not self._responses:
            raise AssertionError("FakeVisionClient: no scripted response left")
        return self._responses.pop(0)


@pytest.fixture
def fake_vision_factory():
    return lambda responses: FakeVisionClient(responses)


@pytest.fixture
def outcrop_image(tmp_path):
    """400x200 synthetic 'outcrop': grey background, dark band rows 40-80, pale lens."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (400, 200), (150, 140, 130))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 40, 399, 80], fill=(60, 55, 50))
    d.ellipse([100, 110, 300, 170], fill=(220, 210, 190))
    path = tmp_path / "outcrop.png"
    im.save(path)
    return str(path)
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_vision_client.py`:

```python
"""VisionClient backends: selection, fail-fast, and request shape (no network)."""
import pytest

from core import vision_client as vc


def test_resolve_auto_prefers_anthropic():
    assert vc.resolve_vision_backend(None, "ak", "vk", "https://v") == "anthropic"


def test_resolve_auto_openai_when_no_anthropic():
    assert vc.resolve_vision_backend(None, None, "vk", "https://v") == "openai"


def test_resolve_auto_none_when_unconfigured():
    assert vc.resolve_vision_backend(None, None, None, None) is None


def test_resolve_explicit_provider_missing_creds_raises():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        vc.resolve_vision_backend("anthropic", None, "vk", "https://v")
    with pytest.raises(RuntimeError, match="VISION_API_KEY"):
        vc.resolve_vision_backend("openai", "ak", None, None)


def test_resolve_unknown_provider_raises():
    with pytest.raises(RuntimeError, match="VISION_PROVIDER"):
        vc.resolve_vision_backend("gemini", "ak", None, None)


def test_build_client_unconfigured_raises(monkeypatch):
    monkeypatch.setattr(vc, "VISION_PROVIDER", None)
    monkeypatch.setattr(vc, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(vc, "VISION_API_KEY", None)
    monkeypatch.setattr(vc, "VISION_BASE_URL", None)
    with pytest.raises(RuntimeError, match="vision provider not configured"):
        vc.build_vision_client()


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _AnthropicFake:
    def __init__(self):
        self.kwargs = None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Msg", (), {"content": [_Block('{"ok": 1}')]})()


def test_anthropic_backend_sends_base64_image_and_prompt():
    fake = _AnthropicFake()
    client = vc.AnthropicVisionClient("key", client=fake)
    out = client.interpret_image(b"\xff\xd8abc", "image/jpeg", "describe")
    assert out == '{"ok": 1}'
    assert fake.kwargs["model"] == vc.DEFAULT_VISION_MODELS["anthropic"]
    content = fake.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[0]["source"]["type"] == "base64"
    assert content[1] == {"type": "text", "text": "describe"}


class _OpenAIFake:
    def __init__(self):
        self.kwargs = None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = type("M", (), {"content": '{"ok": 2}'})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def test_openai_backend_sends_data_url_and_prompt():
    fake = _OpenAIFake()
    client = vc.OpenAIVisionClient("key", "https://v", model="my-vlm", client=fake)
    out = client.interpret_image(b"abc", "image/jpeg", "describe")
    assert out == '{"ok": 2}'
    assert fake.kwargs["model"] == "my-vlm"
    content = fake.kwargs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_build_client_anthropic(monkeypatch):
    monkeypatch.setattr(vc, "VISION_PROVIDER", None)
    monkeypatch.setattr(vc, "ANTHROPIC_API_KEY", "ak")
    monkeypatch.setattr(vc, "VISION_MODEL", None)
    monkeypatch.setattr(vc.AnthropicVisionClient, "_make_sdk_client",
                        staticmethod(lambda api_key: _AnthropicFake()))
    client = vc.build_vision_client()
    assert isinstance(client, vc.AnthropicVisionClient)
    assert client.model == "claude-sonnet-5"


def test_fake_vision_fixture_records_calls(fake_vision_factory):
    fake = fake_vision_factory(["{}"])
    assert fake.interpret_image(b"", "image/png", "p") == "{}"
    assert fake.calls == [("image/png", "p")]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_vision_client.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.vision_client'`

- [ ] **Step 5: Implement `core/vision_client.py`**

```python
"""Vision-capable LLM client for image interpretation.

The chat loop stays on DeepSeek (text-only); this module is used ONLY by
tools/outcrop_tools.py::interpret_outcrop. Two backends behind one protocol:

- AnthropicVisionClient  (anthropic SDK, lazily imported)
- OpenAIVisionClient     (openai SDK pointed at any vision-capable base_url,
                          e.g. OpenAI GPT-4o or a Databricks-served VLM)

`interpret_image` returns the model's raw text; JSON extraction, validation
and the single retry live in the caller, which owns the prompt.
"""
import base64
import logging
from typing import Optional

from typing import Protocol

from config.settings import (VISION_PROVIDER, ANTHROPIC_API_KEY, VISION_API_KEY,
                             VISION_BASE_URL, VISION_MODEL)

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODELS = {"anthropic": "claude-sonnet-5", "openai": "gpt-4o"}


class VisionClient(Protocol):
    def interpret_image(self, image_bytes: bytes, mime: str, prompt: str) -> str: ...


class AnthropicVisionClient:
    def __init__(self, api_key: str, model: Optional[str] = None,
                 max_tokens: int = 2048, client=None):
        self.model = model or DEFAULT_VISION_MODELS["anthropic"]
        self.max_tokens = max_tokens
        self._client = client if client is not None else self._make_sdk_client(api_key)

    @staticmethod
    def _make_sdk_client(api_key: str):
        import anthropic  # lazy: optional dependency
        return anthropic.Anthropic(api_key=api_key)

    def interpret_image(self, image_bytes: bytes, mime: str, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", "") == "text")


class OpenAIVisionClient:
    def __init__(self, api_key: str, base_url: str, model: Optional[str] = None,
                 max_tokens: int = 2048, client=None):
        self.model = model or DEFAULT_VISION_MODELS["openai"]
        self.max_tokens = max_tokens
        self._client = client if client is not None else self._make_sdk_client(api_key, base_url)

    @staticmethod
    def _make_sdk_client(api_key: str, base_url: str):
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url)

    def interpret_image(self, image_bytes: bytes, mime: str, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        return resp.choices[0].message.content or ""


def resolve_vision_backend(provider, anthropic_key, vision_key, vision_url) -> Optional[str]:
    """Pick "anthropic" / "openai" / None from configuration (pure; unit-tested).

    Auto mode prefers Anthropic. An explicit provider without its credentials
    raises RuntimeError naming the missing variables.
    """
    p = (provider or "").strip().lower() or None
    if p == "anthropic":
        if not anthropic_key:
            raise RuntimeError("VISION_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        return "anthropic"
    if p == "openai":
        if not (vision_key and vision_url):
            raise RuntimeError(
                "VISION_PROVIDER=openai but VISION_API_KEY and/or VISION_BASE_URL is not set."
            )
        return "openai"
    if p is not None:
        raise RuntimeError(f"Unknown VISION_PROVIDER {provider!r}; use 'anthropic' or 'openai'.")
    if anthropic_key:
        return "anthropic"
    if vision_key and vision_url:
        return "openai"
    return None


def build_vision_client() -> VisionClient:
    """Construct the configured backend, or raise a clear RuntimeError."""
    backend = resolve_vision_backend(VISION_PROVIDER, ANTHROPIC_API_KEY,
                                     VISION_API_KEY, VISION_BASE_URL)
    if backend is None:
        raise RuntimeError(
            "vision provider not configured: set ANTHROPIC_API_KEY (Anthropic) or "
            "VISION_API_KEY + VISION_BASE_URL (OpenAI-compatible) to interpret photos."
        )
    if backend == "anthropic":
        return AnthropicVisionClient(ANTHROPIC_API_KEY, model=VISION_MODEL)
    return OpenAIVisionClient(VISION_API_KEY, VISION_BASE_URL, model=VISION_MODEL)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_vision_client.py tests/test_llm_credentials.py -q`
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add core/vision_client.py config/settings.py requirements.txt tests/conftest.py tests/test_vision_client.py
git commit -m "feat(vision): VisionClient protocol with Anthropic and OpenAI-compatible backends"
```

---

### Task 3: Interpretation schema + lithology table (`tools/outcrop_tools.py`, part 1)

**Files:**
- Create: `tools/outcrop_tools.py`
- Test: `tests/test_outcrop_interpretation.py`

**Interfaces:**
- Consumes: `workflows.adapters.predict_layer(phit, vclay, fluid=..., label=...) -> Layer(vp, vs, rho, label)`.
- Produces:
  - `LITHOLOGY_TABLE: dict` — `name -> {"route": "han", "phit", "vclay", "fluid"}` or `{"route": "direct", "vp", "vs", "rho"}` or `{"route": "background"}` (only `"cover"`).
  - `LITHOLOGY_COLORS: dict` — name → matplotlib colour.
  - `CONFIDENCE_LEVELS = ("low", "medium", "high")`
  - `validate_interpretation(data: dict) -> dict` — normalized copy; every region has `id:int, label:str, lithology:str, points:list[[x,y]], geometry_type:"polygon"|"band", porosity:Optional[float], vclay:Optional[float], confidence:str, notes:str`; `scale = {"estimated_height_m": Optional[float], "reference": Optional[str], "confidence": str}`; `background_lithology:str`; `mode:str`. Raises `ValueError` with a specific message.
  - `resolve_lithology(lithology, porosity=None, vclay=None, fluid=None) -> dict` — `{"vp","vs","rho","route","phit","vclay","fluid"}` (`phit/vclay/fluid` are `None` on the direct route).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_outcrop_interpretation.py`:

```python
"""OutcropInterpretation validation/normalization and lithology resolution."""
import pytest

from tools.outcrop_tools import (validate_interpretation, resolve_lithology,
                                 LITHOLOGY_TABLE, CONFIDENCE_LEVELS)


def _region(**kw):
    base = {"id": 1, "label": "sand", "lithology": "sandstone",
            "geometry": {"type": "polygon",
                         "points": [[0.1, 0.2], [0.6, 0.2], [0.6, 0.5], [0.1, 0.5]]}}
    base.update(kw)
    return base


def _interp(**kw):
    base = {"regions": [_region()],
            "scale": {"estimated_height_m": 30, "reference": "hammer", "confidence": "low"},
            "background_lithology": "shale", "mode": "polygons"}
    base.update(kw)
    return base


def test_valid_polygon_normalizes():
    out = validate_interpretation(_interp())
    r = out["regions"][0]
    assert r["id"] == 1 and r["lithology"] == "sandstone"
    assert r["geometry_type"] == "polygon" and len(r["points"]) == 4
    assert r["porosity"] is None and r["vclay"] is None
    assert r["confidence"] == "medium" and r["notes"] == ""
    assert out["scale"]["estimated_height_m"] == 30.0
    assert out["background_lithology"] == "shale"


def test_band_becomes_full_width_rectangle():
    out = validate_interpretation(_interp(
        regions=[_region(geometry={"type": "band", "y_top": 0.2, "y_bottom": 0.35})],
        mode="bands"))
    r = out["regions"][0]
    assert r["geometry_type"] == "band"
    assert r["points"] == [[0.0, 0.2], [1.0, 0.2], [1.0, 0.35], [0.0, 0.35]]


def test_missing_ids_are_assigned_sequentially():
    a = _region(); del a["id"]
    b = _region(label="b"); del b["id"]
    out = validate_interpretation(_interp(regions=[a, b]))
    assert [r["id"] for r in out["regions"]] == [1, 2]


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_interpretation(_interp(regions=[_region(), _region()]))


def test_unknown_lithology_rejected():
    with pytest.raises(ValueError, match="lithology"):
        validate_interpretation(_interp(regions=[_region(lithology="kryptonite")]))


def test_lithology_is_case_and_space_insensitive():
    out = validate_interpretation(_interp(regions=[_region(lithology="Clean Sandstone")]))
    assert out["regions"][0]["lithology"] == "clean_sandstone"


def test_polygon_needs_three_points_in_unit_square():
    with pytest.raises(ValueError, match="at least 3"):
        validate_interpretation(_interp(regions=[_region(
            geometry={"type": "polygon", "points": [[0, 0], [1, 1]]})]))
    with pytest.raises(ValueError, match="0, 1"):
        validate_interpretation(_interp(regions=[_region(
            geometry={"type": "polygon", "points": [[0, 0], [1.2, 0], [1, 1]]})]))


def test_band_needs_top_above_bottom():
    with pytest.raises(ValueError, match="y_top"):
        validate_interpretation(_interp(regions=[_region(
            geometry={"type": "band", "y_top": 0.5, "y_bottom": 0.4})]))


def test_porosity_and_vclay_hints_validated():
    out = validate_interpretation(_interp(regions=[_region(porosity=0.22, vclay=0.05)]))
    assert out["regions"][0]["porosity"] == 0.22
    with pytest.raises(ValueError, match="porosity"):
        validate_interpretation(_interp(regions=[_region(porosity=1.5)]))


def test_scale_null_height_allowed_and_confidence_defaulted():
    out = validate_interpretation(_interp(scale={"estimated_height_m": None}))
    assert out["scale"] == {"estimated_height_m": None, "reference": None,
                            "confidence": "low"}


def test_scale_bad_confidence_rejected():
    with pytest.raises(ValueError, match="confidence"):
        validate_interpretation(_interp(scale={"estimated_height_m": 10,
                                               "confidence": "certain"}))


def test_background_cannot_be_cover():
    with pytest.raises(ValueError, match="background"):
        validate_interpretation(_interp(background_lithology="cover"))


def test_regions_must_be_list():
    with pytest.raises(ValueError, match="regions"):
        validate_interpretation({"regions": "none"})


def test_table_has_both_routes_and_cover():
    routes = {v["route"] for v in LITHOLOGY_TABLE.values()}
    assert routes == {"han", "direct", "background"}
    assert LITHOLOGY_TABLE["cover"]["route"] == "background"
    assert "low" in CONFIDENCE_LEVELS


def test_resolve_han_route_matches_predict_layer():
    from workflows.adapters import predict_layer
    got = resolve_lithology("sandstone")
    exp = predict_layer(0.20, 0.10, fluid="brine", label="sandstone")
    assert got["route"] == "han"
    assert got["vp"] == pytest.approx(exp.vp) and got["vs"] == pytest.approx(exp.vs)
    assert got["fluid"] == "brine" and got["phit"] == 0.20


def test_resolve_han_route_with_gas_lowers_vp():
    brine = resolve_lithology("sandstone", fluid="brine")
    gas = resolve_lithology("sandstone", fluid="gas")
    assert gas["vp"] < brine["vp"] and gas["vs"] > brine["vs"]


def test_resolve_direct_route_returns_table_values():
    got = resolve_lithology("limestone")
    assert got == {"vp": 5000.0, "vs": 2700.0, "rho": 2.55, "route": "direct",
                   "phit": None, "vclay": None, "fluid": None}


def test_resolve_direct_route_rejects_petro_overrides():
    with pytest.raises(ValueError, match="limestone"):
        resolve_lithology("limestone", fluid="gas")
    with pytest.raises(ValueError, match="limestone"):
        resolve_lithology("limestone", porosity=0.1)


def test_resolve_cover_rejected():
    with pytest.raises(ValueError, match="cover"):
        resolve_lithology("cover")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_outcrop_interpretation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.outcrop_tools'`

- [ ] **Step 3: Implement the schema + table (first half of `tools/outcrop_tools.py`)**

```python
"""Outcrop photo -> facies interpretation -> 2-D elastic earth model.

Pipeline (each step is a registry tool; results hand off via ContextManager):
  interpret_outcrop     photo -> OutcropInterpretation (vision LLM; the ONLY
                        function here that touches a network)
  outcrop_to_model      interpretation + scale + lithology table -> EarthModel2D
The generic 2-D convolution lives in tools/section_tools.py and knows nothing
about outcrops.
"""
import json
import os
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

from config.settings import SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB
from tools.image_safety import safe_image_path, downscale_for_vision, image_size
from workflows.adapters import predict_layer

CONFIDENCE_LEVELS = ("low", "medium", "high")

# Han (1986)/Gassmann route for clastics; literature "typical" values (Mavko et
# al., Rock Physics Handbook; Bourbie et al.) for rocks Han does not model.
# Shale vclay is 0.50 — the ceiling of Han's calibrated clay range — so the
# default background never triggers warn-and-clip.
LITHOLOGY_TABLE: Dict[str, Dict[str, Any]] = {
    "shale":           {"route": "han", "phit": 0.10, "vclay": 0.50, "fluid": "brine"},
    "mudstone":        {"route": "han", "phit": 0.10, "vclay": 0.50, "fluid": "brine"},
    "siltstone":       {"route": "han", "phit": 0.15, "vclay": 0.40, "fluid": "brine"},
    "sandstone":       {"route": "han", "phit": 0.20, "vclay": 0.10, "fluid": "brine"},
    "clean_sandstone": {"route": "han", "phit": 0.25, "vclay": 0.02, "fluid": "brine"},
    "conglomerate":    {"route": "han", "phit": 0.15, "vclay": 0.05, "fluid": "brine"},
    "limestone":       {"route": "direct", "vp": 5000.0, "vs": 2700.0, "rho": 2.55},
    "dolomite":        {"route": "direct", "vp": 5800.0, "vs": 3200.0, "rho": 2.75},
    "chalk":           {"route": "direct", "vp": 3500.0, "vs": 1900.0, "rho": 2.20},
    "salt":            {"route": "direct", "vp": 4500.0, "vs": 2600.0, "rho": 2.10},
    "coal":            {"route": "direct", "vp": 2400.0, "vs": 1200.0, "rho": 1.40},
    "basalt":          {"route": "direct", "vp": 5500.0, "vs": 3100.0, "rho": 2.80},
    "cover":           {"route": "background"},
}

LITHOLOGY_COLORS = {
    "shale": "#6b705c", "mudstone": "#7a7d6e", "siltstone": "#b5a37a",
    "sandstone": "#e9c46a", "clean_sandstone": "#f4e285", "conglomerate": "#d4a373",
    "limestone": "#8ecae6", "dolomite": "#6a9fb5", "chalk": "#dbe9ee",
    "salt": "#f2b5d4", "coal": "#222222", "basalt": "#5c4b51", "cover": "#9ccc65",
}


def _norm_lithology(name: Any) -> str:
    key = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    if key not in LITHOLOGY_TABLE:
        raise ValueError(
            f"unknown lithology {name!r}; use one of {sorted(LITHOLOGY_TABLE)}"
        )
    return key


def _opt_fraction(value: Any, name: str, rid: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"region {rid}: {name} must be a number in [0, 1] (got {value!r})")
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"region {rid}: {name} must be in [0, 1] (got {v})")
    return v


def _confidence(value: Any, default: str, where: str) -> str:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v not in CONFIDENCE_LEVELS:
        raise ValueError(f"{where}: confidence must be one of {CONFIDENCE_LEVELS} (got {value!r})")
    return v


def _points_from_geometry(geom: Any, rid: Any) -> Tuple[str, List[List[float]]]:
    if not isinstance(geom, dict):
        raise ValueError(f"region {rid}: geometry must be an object")
    gtype = str(geom.get("type", "polygon")).lower()
    if gtype == "band":
        try:
            y_top = float(geom["y_top"]); y_bot = float(geom["y_bottom"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"region {rid}: band geometry needs numeric y_top and y_bottom")
        if not (0.0 <= y_top < y_bot <= 1.0):
            raise ValueError(f"region {rid}: band needs 0 <= y_top < y_bottom <= 1 "
                             f"(got y_top={y_top}, y_bottom={y_bot})")
        return "band", [[0.0, y_top], [1.0, y_top], [1.0, y_bot], [0.0, y_bot]]
    if gtype != "polygon":
        raise ValueError(f"region {rid}: geometry type must be 'polygon' or 'band' (got {gtype!r})")
    pts = geom.get("points")
    if not isinstance(pts, list) or len(pts) < 3:
        raise ValueError(f"region {rid}: polygon needs at least 3 points")
    out = []
    for p in pts:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"region {rid}: each point must be [x, y]")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"region {rid}: point coordinates must be in [0, 1] (got {p})")
        out.append([x, y])
    return "polygon", out


def validate_interpretation(data: Any) -> Dict[str, Any]:
    """Validate + normalize a raw OutcropInterpretation dict (see spec).

    Returns a new dict; raises ValueError naming the offending field. Bands are
    converted to full-width rectangles so rasterization has one geometry path.
    """
    if not isinstance(data, dict):
        raise ValueError("interpretation must be a JSON object")
    raw_regions = data.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError("interpretation.regions must be a list")

    regions = []
    seen_ids = set()
    next_id = 1
    for i, r in enumerate(raw_regions):
        if not isinstance(r, dict):
            raise ValueError(f"regions[{i}] must be an object")
        rid = r.get("id")
        if rid is None:
            rid = next_id
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            raise ValueError(f"regions[{i}]: id must be an integer (got {r.get('id')!r})")
        if rid in seen_ids:
            raise ValueError(f"duplicate region id {rid}")
        seen_ids.add(rid)
        next_id = max(next_id, rid + 1)
        lith = _norm_lithology(r.get("lithology"))
        gtype, pts = _points_from_geometry(r.get("geometry"), rid)
        regions.append({
            "id": rid,
            "label": str(r.get("label") or lith),
            "lithology": lith,
            "geometry_type": gtype,
            "points": pts,
            "porosity": _opt_fraction(r.get("porosity"), "porosity", rid),
            "vclay": _opt_fraction(r.get("vclay"), "vclay", rid),
            "confidence": _confidence(r.get("confidence"), "medium", f"region {rid}"),
            "notes": str(r.get("notes") or ""),
        })

    raw_scale = data.get("scale") or {}
    if not isinstance(raw_scale, dict):
        raise ValueError("interpretation.scale must be an object")
    height = raw_scale.get("estimated_height_m")
    if height is not None:
        try:
            height = float(height)
        except (TypeError, ValueError):
            raise ValueError("scale.estimated_height_m must be a number or null")
        if height <= 0:
            raise ValueError("scale.estimated_height_m must be positive")
    ref = raw_scale.get("reference")
    scale = {
        "estimated_height_m": height,
        "reference": (str(ref) if ref else None),
        "confidence": _confidence(raw_scale.get("confidence"), "low", "scale"),
    }

    background = _norm_lithology(data.get("background_lithology") or "shale")
    if LITHOLOGY_TABLE[background]["route"] == "background":
        raise ValueError("background_lithology cannot be 'cover'")

    mode = str(data.get("mode") or "polygons").lower()
    if mode not in ("polygons", "bands"):
        raise ValueError("mode must be 'polygons' or 'bands'")

    out = {"regions": regions, "scale": scale,
           "background_lithology": background, "mode": mode}
    for passthrough in ("image_path", "image_size", "summary"):
        if passthrough in data:
            out[passthrough] = data[passthrough]
    return out


def resolve_lithology(lithology: str, porosity: Optional[float] = None,
                      vclay: Optional[float] = None,
                      fluid: Optional[str] = None) -> Dict[str, Any]:
    """Lithology (+ optional petro overrides) -> {vp, vs, rho, route, phit, vclay, fluid}.

    Han route: predict_layer(phit, vclay, fluid). Direct route: table values;
    any of porosity/vclay/fluid raises (Han/Gassmann is not valid for them).
    """
    key = _norm_lithology(lithology)
    entry = LITHOLOGY_TABLE[key]
    route = entry["route"]
    if route == "background":
        raise ValueError("'cover' is not a rock; it is rasterized as the background lithology")
    if route == "direct":
        if porosity is not None or vclay is not None or fluid is not None:
            raise ValueError(
                f"{key} uses fixed literature Vp/Vs/density; porosity, vclay and fluid "
                f"overrides only apply to clastic (Han/Gassmann) lithologies"
            )
        return {"vp": float(entry["vp"]), "vs": float(entry["vs"]),
                "rho": float(entry["rho"]), "route": "direct",
                "phit": None, "vclay": None, "fluid": None}
    phit = float(entry["phit"] if porosity is None else porosity)
    vcl = float(entry["vclay"] if vclay is None else vclay)
    fl = str(fluid or entry["fluid"]).lower()
    layer = predict_layer(phit, vcl, fluid=fl, label=key)
    return {"vp": float(layer.vp), "vs": float(layer.vs), "rho": float(layer.rho),
            "route": "han", "phit": phit, "vclay": vcl, "fluid": fl}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_outcrop_interpretation.py -q`
Expected: 20 passed (a Han range warning from `predict_layer` may print; that is fine)

- [ ] **Step 5: Commit**

```bash
git add tools/outcrop_tools.py tests/test_outcrop_interpretation.py
git commit -m "feat(outcrop): interpretation schema validation + lithology table with Han/direct routes"
```

---

### Task 4: `interpret_outcrop` + `plot_outcrop_interpretation` (`tools/outcrop_tools.py`, part 2)

**Files:**
- Modify: `tools/outcrop_tools.py` (append)
- Test: `tests/test_outcrop_tools.py`

**Interfaces:**
- Consumes: `VisionClient.interpret_image(bytes, mime, prompt) -> str` (Task 2); `safe_image_path`, `downscale_for_vision`, `image_size` (Task 1); `validate_interpretation` (Task 3).
- Produces:
  - `OUTCROP_PROMPT: str`
  - `extract_json(text: str) -> dict` — strips ``` fences / prose, `ValueError` on failure.
  - `interpret_outcrop(image_path=None, vision_client=None, upload_dir=None) -> dict` — validated interpretation plus `image_path` (sandboxed absolute), `image_size` `[w, h]`, `summary` (str). Raises `ValueError("Please upload an outcrop photo first.")` if `image_path` is None.
  - `plot_outcrop_interpretation(interpretation: dict, output_path=None) -> str` (PNG path).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_outcrop_tools.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_outcrop_tools.py -q`
Expected: FAIL — `AttributeError: module 'tools.outcrop_tools' has no attribute 'extract_json'`

- [ ] **Step 3: Append the prompt, JSON extraction, `interpret_outcrop`, and the overlay plot**

Append to `tools/outcrop_tools.py`:

```python
# ---------------------------------------------------------------------------
# Vision interpretation
# ---------------------------------------------------------------------------

_ROCK_NAMES = sorted(k for k, v in LITHOLOGY_TABLE.items() if v["route"] != "background")

OUTCROP_PROMPT = f"""You are a field geologist interpreting an outcrop photograph for seismic forward modeling.

Return ONLY a JSON object (no prose, no markdown fences) with this exact shape:
{{
  "regions": [
    {{"id": 1, "label": "short name", "lithology": "<one of: {', '.join(_ROCK_NAMES)}, cover>",
     "geometry": {{"type": "polygon", "points": [[x, y], [x, y], [x, y]]}},
     "porosity": 0.2, "vclay": 0.1, "confidence": "low|medium|high", "notes": "texture, bedding"}}
  ],
  "scale": {{"estimated_height_m": 30, "reference": "what you measured against", "confidence": "low|medium|high"}},
  "background_lithology": "shale",
  "mode": "polygons"
}}

Rules:
- Coordinates are fractions of the image: x from 0 (left) to 1 (right), y from 0 (top) to 1 (bottom).
- Outline every distinct rock body or bed as a polygon (3+ points, clockwise). If the exposure is a simple
  horizontal layer-cake, you may instead use "mode": "bands" with geometry {{"type": "band", "y_top": 0.2, "y_bottom": 0.35}}.
- Everything you do not outline is treated as the background lithology (default shale).
- Mark sky, vegetation, soil, talus, water, people and equipment as lithology "cover" so they are ignored.
- "porosity" and "vclay" are optional fractions (0-1); include them only when texture (grain size, sorting,
  cementation, mud content) justifies a value different from a typical rock of that lithology.
- Scale: look for a scale bar, hammer (~0.3 m), person (~1.7 m), lens cap, vehicle, or any labelled dimension,
  and estimate the total height of the photographed exposure in metres. If nothing gives a reference, set
  "estimated_height_m": null and "confidence": "low". Never invent a scale.
- Use integer ids starting at 1. Keep labels short.
"""


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first {...} JSON object out of model text (fences/prose tolerated)."""
    if not isinstance(text, str):
        raise ValueError("vision model returned no text")
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the vision model's answer")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"vision model answer is not valid JSON: {exc}")


def _summarize(interp: Dict[str, Any]) -> str:
    rocks = [r for r in interp["regions"] if r["lithology"] != "cover"]
    parts = [f"{len(rocks)} rock region(s) on a {interp['background_lithology']} background"]
    for r in rocks:
        parts.append(f"#{r['id']} {r['label']} ({r['lithology']}, {r['confidence']})")
    s = interp["scale"]
    if s["estimated_height_m"] is None:
        parts.append("scale: none found — please give the outcrop height in metres")
    else:
        parts.append(f"scale: ~{s['estimated_height_m']:g} m high from {s['reference'] or 'unknown reference'} "
                     f"({s['confidence']} confidence)")
    return "; ".join(parts)


def interpret_outcrop(image_path: Optional[str] = None, vision_client=None,
                      upload_dir: Optional[str] = None) -> Dict[str, Any]:
    """Photo -> validated OutcropInterpretation via the vision LLM (one retry).

    `image_path` is filled by the chatbot from the uploaded photo when omitted.
    The user's free text is never injected into the vision prompt; guidance
    goes through outcrop_to_model(overrides=...).
    """
    if not image_path:
        raise ValueError("Please upload an outcrop photo first.")
    base = upload_dir or SEISMIC_UPLOAD_DIR
    path = safe_image_path(image_path, base, MAX_IMAGE_MB)
    if vision_client is None:
        from core.vision_client import build_vision_client
        vision_client = build_vision_client()

    img_bytes, mime = downscale_for_vision(path)
    prompt = OUTCROP_PROMPT
    last_err = None
    interp = None
    for _attempt in range(2):
        text = vision_client.interpret_image(img_bytes, mime, prompt)
        try:
            interp = validate_interpretation(extract_json(text))
            break
        except ValueError as exc:
            last_err = exc
            prompt = (OUTCROP_PROMPT
                      + f"\n\nYour previous answer was invalid: {exc}\n"
                        "Return only the corrected JSON object.")
    if interp is None:
        raise ValueError(f"could not interpret image: {last_err}")

    w, h = image_size(path)
    interp["image_path"] = path
    interp["image_size"] = [int(w), int(h)]
    interp["summary"] = _summarize(interp)
    return interp


def plot_outcrop_interpretation(interpretation: Dict[str, Any],
                                output_path: Optional[str] = None) -> str:
    """Photo with semi-transparent facies polygons, ids, legend and scale note."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    from PIL import Image
    with Image.open(interpretation["image_path"]) as im:
        img = np.asarray(im.convert("RGB"))
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10 * h / float(w)))
    ax.imshow(img)
    used = {}
    for r in interpretation["regions"]:
        color = LITHOLOGY_COLORS.get(r["lithology"], "#ff00ff")
        pts = np.array(r["points"]) * [w, h]
        ax.add_patch(MplPolygon(pts, closed=True, facecolor=color, edgecolor="k",
                                alpha=0.35 if r["lithology"] != "cover" else 0.15, lw=1.2))
        cx, cy = pts.mean(axis=0)
        ax.text(cx, cy, f"#{r['id']} {r['label']}", ha="center", va="center",
                fontsize=8, color="k",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
        used[r["lithology"]] = color
    used.setdefault(interpretation["background_lithology"],
                    LITHOLOGY_COLORS[interpretation["background_lithology"]])
    handles = [MplPolygon([[0, 0], [1, 0], [1, 1]], facecolor=c, edgecolor="k", alpha=0.5,
                          label=n) for n, c in used.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    s = interpretation["scale"]
    scale_txt = ("scale: not found" if s["estimated_height_m"] is None
                 else f"~{s['estimated_height_m']:g} m high ({s['confidence']}, {s['reference'] or '?'})")
    ax.set_title(f"Outcrop interpretation — background {interpretation['background_lithology']}; {scale_txt}")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_outcrop_tools.py tests/test_outcrop_interpretation.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add tools/outcrop_tools.py tests/test_outcrop_tools.py
git commit -m "feat(outcrop): interpret_outcrop via VisionClient (retry-once) + interpretation overlay plot"
```

---

### Task 5: `outcrop_to_model` — rasterize + elastic grids (`tools/outcrop_tools.py`, part 3)

**Files:**
- Modify: `tools/outcrop_tools.py` (append)
- Test: `tests/test_outcrop_model.py`

**Interfaces:**
- Consumes: `validate_interpretation`, `resolve_lithology` (Task 3); interpretation dict with optional `image_size` (Task 4).
- Produces:
  - `apply_overrides(regions: list, overrides: Optional[dict]) -> list` — new region list; keys are region id (int or digit-string) or label (case-insensitive); allowed fields `lithology`, `fluid`, `porosity`, `vclay`; unknown key or field → `ValueError`.
  - `outcrop_to_model(interpretation=None, height_m=None, overrides=None, background_lithology=None, num_traces=101, wavelet_freq=30.0, pad_m=None, nz_target=400) -> dict` — **EarthModel2D**: `facies (nz×nx int ndarray, 0 = background)`, `legend {id: {"lithology","label"}}` (id 0 = background), `vp`, `vs`, `rho` (nz×nx float ndarrays), `z` (nz, m, cell centres, 0 at top of pad), `x` (nx, m), `dz`, `dx`, `nz`, `nx`, `height_m`, `width_m`, `pad_m`, `image_top_m`, `scale_source` (`"user"|"vision"`), `scale_confidence`, `regions` (provenance list: id, label, lithology, route, phit, vclay, fluid, vp, vs, rho, n_cells), `image_path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_outcrop_model.py`:

```python
"""outcrop_to_model: scale policy, overrides, rasterization, padding, provenance."""
import numpy as np
import pytest

from tools.outcrop_tools import (outcrop_to_model, apply_overrides,
                                 validate_interpretation, resolve_lithology)


def _interp(regions, height=25.0, image_size=(400, 200)):
    d = {"regions": regions,
         "scale": {"estimated_height_m": height, "reference": "person", "confidence": "medium"},
         "background_lithology": "shale", "mode": "polygons",
         "image_size": list(image_size)}
    return validate_interpretation(d)


BAND = {"id": 1, "label": "sand bed", "lithology": "sandstone",
        "geometry": {"type": "band", "y_top": 0.2, "y_bottom": 0.4}}
LENS = {"id": 2, "label": "lime lens", "lithology": "limestone",
        "geometry": {"type": "polygon",
                     "points": [[0.25, 0.6], [0.75, 0.6], [0.75, 0.9], [0.25, 0.9]]}}
COVER = {"id": 3, "label": "bush", "lithology": "cover",
         "geometry": {"type": "polygon", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.1]]}}


def test_scale_from_vision_when_height_not_given():
    m = outcrop_to_model(_interp([BAND]))
    assert m["height_m"] == 25.0 and m["scale_source"] == "vision"
    assert m["scale_confidence"] == "medium"


def test_explicit_height_overrides_vision():
    m = outcrop_to_model(_interp([BAND]), height_m=40.0)
    assert m["height_m"] == 40.0 and m["scale_source"] == "user"


def test_no_scale_anywhere_asks_for_height():
    with pytest.raises(ValueError, match="height in metres"):
        outcrop_to_model(_interp([BAND], height=None))


def test_missing_interpretation_asks_to_interpret_first():
    with pytest.raises(ValueError, match="interpret_outcrop"):
        outcrop_to_model(None, height_m=10)


def test_grid_geometry_follows_aspect_and_targets():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, num_traces=51, nz_target=200)
    assert m["nx"] == 51 and m["x"].shape == (51,)
    assert m["width_m"] == pytest.approx(40.0)            # 400x200 image -> aspect 2
    assert m["dx"] == pytest.approx(40.0 / 50)
    assert m["dz"] == pytest.approx(0.1)                  # 20 m / 200 rows
    assert m["facies"].shape == m["vp"].shape == (m["nz"], 51)


def test_dz_floor_is_10cm():
    m = outcrop_to_model(_interp([BAND]), height_m=2.0, nz_target=400)
    assert m["dz"] == pytest.approx(0.1)


def test_band_rasterizes_at_expected_depth_and_pads_with_background():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, nz_target=200, pad_m=5.0)
    top = m["image_top_m"]
    assert top == pytest.approx(5.0)
    # rows inside the band (y 0.2-0.4 of 20 m -> 4-8 m below image top)
    z = m["z"]
    inside = (z > top + 4.05) & (z < top + 7.95)
    outside_above = z < top + 3.95
    assert np.all(m["facies"][inside, :] == 1)
    assert np.all(m["facies"][outside_above, :] == 0)
    assert np.all(m["facies"][:, 0] == m["facies"][:, -1])  # full-width band


def test_polygon_lens_is_laterally_bounded_and_later_region_wins():
    m = outcrop_to_model(_interp([BAND, LENS]), height_m=20.0, nz_target=200, pad_m=2.0)
    z, x = m["z"], m["x"]
    top = m["image_top_m"]
    zi = np.argmin(np.abs(z - (top + 0.75 * 20)))     # centre of lens in depth
    row = m["facies"][zi]
    assert row[np.argmin(np.abs(x - 0.5 * m["width_m"]))] == 2
    assert row[0] == 0 and row[-1] == 0
    # overlap test: a second band over the first -> later wins
    over = dict(BAND, id=7, label="silt", lithology="siltstone")
    m2 = outcrop_to_model(_interp([BAND, over]), height_m=20.0, nz_target=200, pad_m=2.0)
    assert 1 not in np.unique(m2["facies"]) and 7 in np.unique(m2["facies"])


def test_cover_is_background_and_not_in_grid():
    m = outcrop_to_model(_interp([COVER, BAND]), height_m=20.0)
    assert set(np.unique(m["facies"])) == {0, 1}
    prov = {r["id"]: r for r in m["regions"]}
    assert prov[3]["route"] == "background" and prov[3]["n_cells"] == 0


def test_elastic_grids_match_resolve_lithology():
    m = outcrop_to_model(_interp([BAND, LENS]), height_m=20.0)
    sand = resolve_lithology("sandstone")
    shale = resolve_lithology("shale")
    assert m["vp"][m["facies"] == 1].min() == pytest.approx(sand["vp"])
    assert m["vp"][m["facies"] == 0].max() == pytest.approx(shale["vp"])
    assert m["vp"][m["facies"] == 2].max() == pytest.approx(5000.0)
    assert np.all(m["vs"] < m["vp"]) and np.all(m["rho"] > 0)


def test_default_pad_is_1p5_wavelengths_of_background():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, wavelet_freq=25.0)
    v_bg = resolve_lithology("shale")["vp"]
    assert m["pad_m"] == pytest.approx(1.5 * v_bg / 25.0)
    assert m["z"][0] == pytest.approx(m["dz"] / 2)


def test_overrides_by_id_and_label():
    regs = _interp([BAND, LENS])["regions"]
    out = apply_overrides(regs, {"1": {"fluid": "gas"}, "lime lens": {"lithology": "dolomite"}})
    assert out[0]["fluid"] == "gas" and out[1]["lithology"] == "dolomite"
    assert "fluid" not in regs[0]                      # input untouched


def test_overrides_unknown_key_or_field_rejected():
    regs = _interp([BAND])["regions"]
    with pytest.raises(ValueError, match="no region"):
        apply_overrides(regs, {"99": {"fluid": "gas"}})
    with pytest.raises(ValueError, match="unknown override"):
        apply_overrides(regs, {"1": {"colour": "red"}})


def test_gas_override_changes_grid_and_direct_route_override_errors():
    brine = outcrop_to_model(_interp([BAND]), height_m=20.0)
    gas = outcrop_to_model(_interp([BAND]), height_m=20.0, overrides={1: {"fluid": "gas"}})
    assert gas["vp"][gas["facies"] == 1].max() < brine["vp"][brine["facies"] == 1].min()
    assert gas["regions"][0]["fluid"] == "gas"
    with pytest.raises(ValueError, match="limestone"):
        outcrop_to_model(_interp([LENS]), height_m=20.0, overrides={2: {"fluid": "gas"}})


def test_background_override():
    m = outcrop_to_model(_interp([BAND]), height_m=20.0, background_lithology="limestone")
    assert m["legend"][0]["lithology"] == "limestone"
    assert m["vp"][m["facies"] == 0].max() == pytest.approx(5000.0)


def test_missing_image_size_warns_and_uses_default_aspect():
    d = _interp([BAND]); d.pop("image_size")
    with pytest.warns(UserWarning, match="aspect"):
        m = outcrop_to_model(d, height_m=20.0)
    assert m["width_m"] == pytest.approx(30.0)


def test_bad_geometry_params_rejected():
    with pytest.raises(ValueError):
        outcrop_to_model(_interp([BAND]), height_m=-5)
    with pytest.raises(ValueError):
        outcrop_to_model(_interp([BAND]), height_m=20, num_traces=1)
    with pytest.raises(ValueError):
        outcrop_to_model(_interp([BAND]), height_m=20, pad_m=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_outcrop_model.py -q`
Expected: FAIL — `ImportError: cannot import name 'outcrop_to_model'`

- [ ] **Step 3: Append `apply_overrides` and `outcrop_to_model`**

Append to `tools/outcrop_tools.py`:

```python
# ---------------------------------------------------------------------------
# Earth model
# ---------------------------------------------------------------------------

_OVERRIDE_FIELDS = ("lithology", "fluid", "porosity", "vclay")
DEFAULT_ASPECT = 1.5


def apply_overrides(regions: List[Dict[str, Any]],
                    overrides: Optional[Dict[Any, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Return a copy of `regions` with per-region user corrections applied.

    Keys: region id (int or digit string) or label (case-insensitive).
    Fields: lithology, fluid, porosity, vclay. Anything else -> ValueError.
    """
    out = [dict(r) for r in regions]
    if not overrides:
        return out
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be an object keyed by region id or label")
    by_id = {r["id"]: r for r in out}
    by_label = {r["label"].lower(): r for r in out}
    for key, fields in overrides.items():
        target = None
        skey = str(key).strip()
        if skey.lstrip("-").isdigit() and int(skey) in by_id:
            target = by_id[int(skey)]
        elif skey.lower() in by_label:
            target = by_label[skey.lower()]
        if target is None:
            raise ValueError(f"no region with id or label {key!r}; regions are "
                             f"{[(r['id'], r['label']) for r in out]}")
        if not isinstance(fields, dict):
            raise ValueError(f"override for {key!r} must be an object of fields")
        for f, v in fields.items():
            if f not in _OVERRIDE_FIELDS:
                raise ValueError(f"unknown override field {f!r}; use one of {_OVERRIDE_FIELDS}")
            if f == "lithology":
                target["lithology"] = _norm_lithology(v)
            elif f == "fluid":
                target["fluid"] = str(v).lower()
            else:
                target[f] = _opt_fraction(v, f, target["id"])
    return out


def outcrop_to_model(interpretation: Optional[Dict[str, Any]] = None,
                     height_m: Optional[float] = None,
                     overrides: Optional[Dict[Any, Dict[str, Any]]] = None,
                     background_lithology: Optional[str] = None,
                     num_traces: int = 101, wavelet_freq: float = 30.0,
                     pad_m: Optional[float] = None, nz_target: int = 400) -> Dict[str, Any]:
    """Interpretation + scale + lithology table -> EarthModel2D dict (see plan).

    Deterministic and offline: re-run freely after the user corrects the scale
    ("the cliff is 40 m") or a region ("make #2 gas-filled").
    """
    if interpretation is None:
        raise ValueError("Interpret an outcrop photo first (interpret_outcrop) — "
                         "there is no interpretation to build a model from.")
    interp = validate_interpretation(interpretation)

    # --- scale policy: user > vision > ask
    if height_m is not None:
        height = float(height_m)
        scale_source, scale_conf = "user", "high"
    elif interp["scale"]["estimated_height_m"] is not None:
        height = float(interp["scale"]["estimated_height_m"])
        scale_source, scale_conf = "vision", interp["scale"]["confidence"]
    else:
        raise ValueError("I need the outcrop height in metres: no scale reference was found "
                         "in the photo. Tell me e.g. 'the exposure is 30 m high'.")
    if height <= 0:
        raise ValueError(f"height_m must be positive (got {height})")
    num_traces = int(num_traces)
    if num_traces < 2:
        raise ValueError("num_traces must be >= 2")
    if nz_target < 2:
        raise ValueError("nz_target must be >= 2")
    if wavelet_freq <= 0:
        raise ValueError("wavelet_freq must be positive")

    size = interp.get("image_size")
    if size and len(size) == 2 and size[1] > 0:
        aspect = float(size[0]) / float(size[1])
    else:
        warnings.warn(f"no image size on the interpretation; assuming aspect ratio "
                      f"{DEFAULT_ASPECT} (width/height)", stacklevel=2)
        aspect = DEFAULT_ASPECT
    width = height * aspect

    # --- lithologies
    background = _norm_lithology(background_lithology or interp["background_lithology"])
    bg = resolve_lithology(background)
    regions = apply_overrides(interp["regions"], overrides)
    props = {0: bg}
    legend = {0: {"lithology": background, "label": "background"}}
    provenance = []
    for r in regions:
        if r["lithology"] == "cover":
            provenance.append({"id": r["id"], "label": r["label"], "lithology": "cover",
                               "route": "background", "phit": None, "vclay": None,
                               "fluid": None, "vp": None, "vs": None, "rho": None, "n_cells": 0})
            continue
        try:
            p = resolve_lithology(r["lithology"], porosity=r.get("porosity"),
                                  vclay=r.get("vclay"), fluid=r.get("fluid"))
        except ValueError as exc:
            raise ValueError(f"region #{r['id']} ({r['label']}): {exc}")
        props[r["id"]] = p
        legend[r["id"]] = {"lithology": r["lithology"], "label": r["label"]}
        provenance.append({"id": r["id"], "label": r["label"], "lithology": r["lithology"],
                           "route": p["route"], "phit": p["phit"], "vclay": p["vclay"],
                           "fluid": p["fluid"], "vp": p["vp"], "vs": p["vs"], "rho": p["rho"],
                           "n_cells": 0})

    # --- grid
    dz = max(height / float(nz_target), 0.1)
    nz_img = max(1, int(round(height / dz)))
    if pad_m is None:
        pad_m = 1.5 * bg["vp"] / float(wavelet_freq)
    pad_m = float(pad_m)
    if pad_m <= 0:
        raise ValueError("pad_m must be positive")
    npad = max(1, int(np.ceil(pad_m / dz)))
    nz = nz_img + 2 * npad
    dx = width / float(num_traces - 1)
    x = np.arange(num_traces) * dx
    z = (np.arange(nz) + 0.5) * dz
    image_top = npad * dz

    facies = np.zeros((nz, num_traces), dtype=int)
    # cell centres of the image part in normalized coordinates
    xn = x / width if width > 0 else np.zeros_like(x)
    yn = ((np.arange(nz_img) + 0.5) * dz) / height
    XN, YN = np.meshgrid(xn, yn)
    # Nudge edge cells (x=0, x=1) just inside the unit square: Path.contains_points is
    # undefined exactly on a polygon edge, and full-width bands/polygons touch the edges.
    eps = 1e-6
    query = np.column_stack([np.clip(XN.ravel(), eps, 1.0 - eps),
                             np.clip(YN.ravel(), eps, 1.0 - eps)])
    img_facies = np.zeros((nz_img, num_traces), dtype=int)
    for r in regions:
        if r["lithology"] == "cover":
            continue
        mask = MplPath(np.asarray(r["points"], dtype=float)).contains_points(query)
        mask = mask.reshape(nz_img, num_traces)
        img_facies[mask] = r["id"]
    facies[npad:npad + nz_img, :] = img_facies
    for row in provenance:
        if row["route"] != "background":
            row["n_cells"] = int(np.count_nonzero(facies == row["id"]))

    max_id = max(props)
    lut_vp = np.zeros(max_id + 1); lut_vs = np.zeros(max_id + 1); lut_rho = np.zeros(max_id + 1)
    for fid, p in props.items():
        lut_vp[fid], lut_vs[fid], lut_rho[fid] = p["vp"], p["vs"], p["rho"]
    vp = lut_vp[facies]; vs = lut_vs[facies]; rho = lut_rho[facies]

    return {
        "facies": facies, "legend": legend,
        "vp": vp, "vs": vs, "rho": rho,
        "z": z, "x": x, "dz": float(dz), "dx": float(dx),
        "nz": int(nz), "nx": int(num_traces),
        "height_m": height, "width_m": float(width), "pad_m": pad_m,
        "image_top_m": float(image_top),
        "scale_source": scale_source, "scale_confidence": scale_conf,
        "background_lithology": background,
        "regions": provenance,
        "image_path": interp.get("image_path"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_outcrop_model.py -q`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add tools/outcrop_tools.py tests/test_outcrop_model.py
git commit -m "feat(outcrop): outcrop_to_model — scale policy, overrides, polygon rasterization, padded elastic grids"
```

---

### Task 6: Generic 2-D convolutional model (`tools/section_tools.py`)

**Files:**
- Create: `tools/section_tools.py`
- Test: `tests/test_section_tools.py`

**Interfaces:**
- Consumes: `tools.wedge_tools.gen_wavelet(dt, wv_type, ricker_freq, ormsby_freq, wavelet_str, wavelet_fname, phase_rot, wavelet_length=500) -> (t, wavelet, label)`; `tools.avo_tools.shuey_reflectivity` / `zoeppritz_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, angles) -> ndarray`; `tools.synthetic_tools._ormsby_corners(ormsby_freq) -> (f1,f2,f3,f4)`; `physics_guards.angles_error`, `warn_if_aliased`.
- Produces:
  - `validate_section_inputs(vp, vs, rho, dz, dx, angle, method, wv_type, ormsby_freq, dt, pad_time, wavelet_freq, domain) -> None` (raises `ValueError`).
  - `create_synthetic_section(vp, vs, rho, dz, dx, wavelet_freq=30.0, wv_type="ricker", ormsby_freq=None, phase_rot=0.0, angle=0.0, method="shuey", dt=1.0, pad_time=50.0, domain="time") -> (axis, section, parameters)`; time domain: `axis` = TWT ms (nt), `section` nt×nx. `parameters` keys: `nt, dt, nx, dx, nz, dz, pad_time, angle, method, wavelet_freq, wavelet_label, domain, n_interfaces, max_abs_amplitude, n_postcritical_zeroed`.
  - `depth_convert(section, time_array, vp, dz, pad_time) -> ndarray (nz×nx)` (Task 7 wires `domain="depth"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_section_tools.py`:

```python
"""create_synthetic_section: oracle vs the 1-D synthetic, guards, angle paths."""
import warnings

import numpy as np
import pytest

from tools.section_tools import create_synthetic_section, validate_section_inputs
from tools.synthetic_tools import create_synthetic_seismogram

VP = [3000.0, 2500.0, 3200.0]
RHO = [2.4, 2.2, 2.5]
VS = [1500.0, 1250.0, 1600.0]
TH = [50.0, 50.0]


def _layer_cake(nx=5, dz=1.0, bottom=60.0):
    """Horizontal 3-layer grid: 50 m / 50 m / bottom m basal layer."""
    rows = [int(TH[0] / dz), int(TH[1] / dz), int(bottom / dz)]
    vp = np.concatenate([np.full(r, v) for r, v in zip(rows, VP)])
    vs = np.concatenate([np.full(r, v) for r, v in zip(rows, VS)])
    rho = np.concatenate([np.full(r, v) for r, v in zip(rows, RHO)])
    tile = lambda a: np.tile(a[:, None], (1, nx))
    return tile(vp), tile(vs), tile(rho)


def test_oracle_every_column_matches_1d_synthetic():
    vp, vs, rho = _layer_cake()
    t, sec, par = create_synthetic_section(vp, vs, rho, dz=1.0, dx=10.0, dt=1.0, pad_time=50.0)
    t1, trace, p1 = create_synthetic_seismogram(TH, VP, RHO, vs=VS, dt=1.0, pad_time=50.0)
    n = min(len(t1), len(t))
    for j in range(sec.shape[1]):
        np.testing.assert_allclose(sec[:n, j], trace[:n], rtol=1e-6, atol=1e-9)
    assert par["n_interfaces"] == 2 * vp.shape[1]
    assert par["max_abs_amplitude"] == pytest.approx(np.max(np.abs(trace)), rel=1e-6)
    assert par["wavelet_label"] == p1["wavelet_label"]


def test_oracle_at_angle_shuey_and_zoeppritz():
    vp, vs, rho = _layer_cake(nx=2)
    for method in ("shuey", "zoeppritz"):
        _, sec, _ = create_synthetic_section(vp, vs, rho, 1.0, 10.0, dt=1.0, angle=20.0, method=method)
        _, trace, _ = create_synthetic_seismogram(TH, VP, RHO, vs=VS, dt=1.0, angle=20.0, method=method)
        n = min(len(trace), sec.shape[0])
        np.testing.assert_allclose(sec[:n, 0], trace[:n], rtol=1e-6, atol=1e-9)


def test_tiny_angle_zoeppritz_agrees_with_acoustic():
    """Exact Zoeppritz at theta -> 0 is the acoustic RC (Shuey's R0 is only linearized)."""
    vp, vs, rho = _layer_cake(nx=2)
    _, a, _ = create_synthetic_section(vp, vs, rho, 1.0, 10.0, angle=0.0)
    _, z, _ = create_synthetic_section(vp, vs, rho, 1.0, 10.0, angle=1e-6, method="zoeppritz")
    np.testing.assert_allclose(a, z, atol=1e-6)


def test_uniform_grid_gives_zero_section():
    vp = np.full((100, 4), 3000.0); vs = vp / 2; rho = np.full((100, 4), 2.4)
    _, sec, par = create_synthetic_section(vp, vs, rho, 1.0, 5.0)
    assert np.all(sec == 0) and par["n_interfaces"] == 0 and par["max_abs_amplitude"] == 0


def test_lateral_variation_is_column_independent():
    vp, vs, rho = _layer_cake(nx=3)
    vp2 = vp.copy(); vp2[50:100, 2] = 3000.0   # remove the contrast in column 2 only
    rho2 = rho.copy(); rho2[50:100, 2] = 2.4
    vs2 = vs.copy(); vs2[50:100, 2] = 1500.0
    _, sec, _ = create_synthetic_section(vp2, vs2, rho2, 1.0, 5.0)
    np.testing.assert_allclose(sec[:, 0], sec[:, 1])
    assert not np.allclose(sec[:, 0], sec[:, 2])


def test_thin_layers_superpose():
    vp = np.full((100, 1), 3000.0); rho = np.full((100, 1), 2.4); vs = vp / 2
    vp[50] = 2000.0; rho[50] = 2.0            # one-cell layer: two interfaces in one sample at dt=1
    _, sec, par = create_synthetic_section(vp, vs, rho, dz=0.5, dx=1.0, dt=1.0)
    assert par["n_interfaces"] == 2
    assert np.max(np.abs(sec)) < 0.05          # near-cancelling RCs superpose (+=), not overwrite


def test_ormsby_wavelet_path():
    vp, vs, rho = _layer_cake(nx=2)
    _, sec, par = create_synthetic_section(vp, vs, rho, 1.0, 10.0, wv_type="ormsby",
                                           ormsby_freq="5,10,40,60")
    assert par["wavelet_freq"] == pytest.approx(25.0) and "Ormsby" in par["wavelet_label"]
    assert np.max(np.abs(sec)) > 0


def test_postcritical_zoeppritz_zeroed_with_warning():
    vp = np.full((40, 1), 1500.0); vp[20:] = 4500.0
    vs = vp / 2; rho = np.full((40, 1), 2.2)
    with pytest.warns(UserWarning, match="post-critical"):
        _, sec, par = create_synthetic_section(vp, vs, rho, 1.0, 1.0, angle=40.0, method="zoeppritz")
    assert np.isfinite(sec).all() and par["n_postcritical_zeroed"] == 1


def test_guards_reject_bad_inputs():
    vp, vs, rho = _layer_cake(nx=2)
    with pytest.raises(ValueError, match="shape"):
        validate_section_inputs(vp, vs[:-1], rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="2-D"):
        validate_section_inputs(vp[:, 0], vs[:, 0], rho[:, 0], 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    bad_vs = vs.copy(); bad_vs[0, 0] = 9999.0
    with pytest.raises(ValueError, match="vs"):
        validate_section_inputs(vp, bad_vs, rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="dz"):
        validate_section_inputs(vp, vs, rho, 0.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="angle"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 95.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="method"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 10.0, "magic", "ricker", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="domain"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "sideways")
    with pytest.raises(ValueError, match="ormsby"):
        validate_section_inputs(vp, vs, rho, 1.0, 1.0, 0.0, "shuey", "ormsby", None, 1.0, 50.0, 30.0, "time")
    with pytest.raises(ValueError, match="finite"):
        nan_vp = vp.copy(); nan_vp[0, 0] = np.nan
        validate_section_inputs(nan_vp, vs, rho, 1.0, 1.0, 0.0, "shuey", "ricker", None, 1.0, 50.0, 30.0, "time")


def test_aliasing_warns():
    vp, vs, rho = _layer_cake(nx=1)
    with pytest.warns(UserWarning, match="Nyquist"):
        create_synthetic_section(vp, vs, rho, 1.0, 1.0, wavelet_freq=200.0, dt=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_section_tools.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.section_tools'`

- [ ] **Step 3: Implement `tools/section_tools.py`**

```python
"""Generic 2-D convolutional synthetic over an elastic grid.

Input: vp / vs / rho grids (nz x nx) on a regular (dz, dx) mesh — any gridded
earth model (outcrop rasterization, hand-built, future imports). Per column:
depth -> TWT through that column's velocities, reflectivity at every property
change (acoustic at normal incidence; Shuey or exact Zoeppritz at an angle),
interfaces rounded onto the dt grid with superposition, then convolution with
a Ricker/Ormsby wavelet (tools/wedge_tools.gen_wavelet — same as the 1-D tool).

This module knows nothing about outcrops or photos.
"""
import os
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from tools.wedge_tools import gen_wavelet, plot_vawig
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity
from tools.synthetic_tools import _ormsby_corners
from tools.physics_guards import angles_error, warn_if_aliased

WAVELET_LENGTH_MS = 256.0   # identical to create_synthetic_seismogram


def validate_section_inputs(vp, vs, rho, dz, dx, angle, method, wv_type, ormsby_freq,
                            dt, pad_time, wavelet_freq, domain) -> None:
    """REJECT tier for the 2-D section (raises ValueError)."""
    vp = np.asarray(vp, dtype=float); vs = np.asarray(vs, dtype=float); rho = np.asarray(rho, dtype=float)
    if vp.ndim != 2:
        raise ValueError(f"vp/vs/rho must be 2-D (nz x nx) grids; got {vp.ndim}-D")
    if not (vp.shape == vs.shape == rho.shape):
        raise ValueError(f"vp, vs and rho must share one shape; got {vp.shape}, {vs.shape}, {rho.shape}")
    if vp.shape[0] < 2 or vp.shape[1] < 1:
        raise ValueError(f"grid needs at least 2 rows and 1 column; got shape {vp.shape}")
    for name, arr in (("vp", vp), ("vs", vs), ("rho", rho)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} grid must be finite everywhere")
    if np.any(vp <= 0):
        raise ValueError("vp must be positive everywhere")
    if np.any(rho <= 0):
        raise ValueError("rho must be positive everywhere")
    if np.any(vs <= 0) or np.any(vs >= vp):
        raise ValueError("vs must satisfy 0 < vs < vp everywhere")
    for name, val in (("dz", dz), ("dx", dx), ("dt", dt), ("wavelet_freq", wavelet_freq)):
        if not (isinstance(val, (int, float)) and np.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be a positive number (got {val!r})")
    if not (isinstance(pad_time, (int, float)) and pad_time >= 0):
        raise ValueError(f"pad_time must be >= 0 ms (got {pad_time!r})")
    err = angles_error(np.atleast_1d(float(angle)))
    if err:
        raise ValueError(f"angle: {err}")
    if method not in ("shuey", "zoeppritz"):
        raise ValueError("method must be 'shuey' or 'zoeppritz'")
    if wv_type not in ("ricker", "ormsby"):
        raise ValueError("wv_type must be 'ricker' or 'ormsby'")
    if wv_type == "ormsby":
        if not ormsby_freq:
            raise ValueError("ormsby_freq ('f1,f2,f3,f4') is required when wv_type='ormsby'")
        _ormsby_corners(ormsby_freq)   # raises on malformed corners
    if domain not in ("time", "depth"):
        raise ValueError("domain must be 'time' or 'depth'")


def _interface_rc(vp1, vs1, rho1, vp2, vs2, rho2, angle, method, cache):
    key = (vp1, vs1, rho1, vp2, vs2, rho2)
    if key in cache:
        return cache[key]
    if angle == 0:
        z1, z2 = vp1 * rho1, vp2 * rho2
        rc = (z2 - z1) / (z2 + z1)
    else:
        fn = shuey_reflectivity if method == "shuey" else zoeppritz_reflectivity
        rc = float(np.asarray(fn(vp1=vp1, vs1=vs1, rho1=rho1, vp2=vp2, vs2=vs2,
                                 rho2=rho2, angles=[angle])).ravel()[0])
    cache[key] = rc
    return rc


def create_synthetic_section(vp, vs, rho, dz, dx, wavelet_freq=30.0, wv_type="ricker",
                             ormsby_freq=None, phase_rot=0.0, angle=0.0, method="shuey",
                             dt=1.0, pad_time=50.0, domain="time"
                             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """2-D convolutional synthetic. Returns (axis, section, parameters).

    domain='time'  -> axis is TWT in ms (nt), section is nt x nx.
    domain='depth' -> axis is depth in m (nz), section is the depth-converted
                      nt x nx result (computed in time, then mapped column by
                      column through that column's own t(z)).
    """
    validate_section_inputs(vp, vs, rho, dz, dx, angle, method, wv_type, ormsby_freq,
                            dt, pad_time, wavelet_freq, domain)
    vp = np.asarray(vp, dtype=float); vs = np.asarray(vs, dtype=float); rho = np.asarray(rho, dtype=float)
    nz, nx = vp.shape
    angle = float(angle)

    if wv_type == "ormsby":
        corners = _ormsby_corners(ormsby_freq)
        content_hz = corners[3]
        dominant_freq = (corners[1] + corners[2]) / 2.0
    else:
        content_hz = 3.0 * wavelet_freq
        dominant_freq = float(wavelet_freq)
    warn_if_aliased(content_hz, dt / 1000.0, "section wavelet")

    # TWT at the BOTTOM of every cell, per column (ms); top of grid at pad_time.
    twt_bottom = pad_time + np.cumsum(2000.0 * dz / vp, axis=0)
    total_twt = twt_bottom[-1, :].max()

    _, wavelet, wavelet_label = gen_wavelet(dt, wv_type, wavelet_freq, ormsby_freq, "", "",
                                            phase_rot, wavelet_length=WAVELET_LENGTH_MS)
    nt = int(round((total_twt + pad_time) / dt)) + 1
    nt = max(nt, wavelet.size)
    time_array = np.arange(nt) * dt

    rc_series = np.zeros((nt, nx))
    cache: Dict[tuple, float] = {}
    n_interfaces = 0
    n_postcritical = 0
    for j in range(nx):
        col_vp, col_vs, col_rho = vp[:, j], vs[:, j], rho[:, j]
        change = np.where((col_vp[1:] != col_vp[:-1]) | (col_vs[1:] != col_vs[:-1])
                          | (col_rho[1:] != col_rho[:-1]))[0]
        for k in change:
            rc = _interface_rc(col_vp[k], col_vs[k], col_rho[k],
                               col_vp[k + 1], col_vs[k + 1], col_rho[k + 1],
                               angle, method, cache)
            n_interfaces += 1
            if not np.isfinite(rc):
                n_postcritical += 1
                rc = 0.0
            idx = int(round(twt_bottom[k, j] / dt))
            if 0 <= idx < nt:
                rc_series[idx, j] += rc   # superpose thin layers (same as the 1-D tool)
    if n_postcritical:
        warnings.warn(f"{n_postcritical} post-critical Zoeppritz interface(s) at {angle:g} deg "
                      f"were set to zero reflectivity", stacklevel=2)

    section = scipy.signal.convolve(rc_series, wavelet[:, None], mode="same")

    parameters = {
        "nt": int(nt), "dt": float(dt), "nx": int(nx), "dx": float(dx),
        "nz": int(nz), "dz": float(dz), "pad_time": float(pad_time),
        "angle": angle, "method": method,
        "wavelet_freq": float(dominant_freq), "wavelet_label": wavelet_label,
        "domain": domain, "n_interfaces": int(n_interfaces),
        "max_abs_amplitude": float(np.max(np.abs(section))) if section.size else 0.0,
        "n_postcritical_zeroed": int(n_postcritical),
    }
    if domain == "depth":
        z_axis = (np.arange(nz) + 0.5) * dz
        return z_axis, depth_convert(section, time_array, vp, dz, pad_time), parameters
    return time_array, section, parameters


def depth_convert(section, time_array, vp, dz, pad_time) -> np.ndarray:
    """Map a time section (nt x nx) onto the model's depth cells (nz x nx).

    Each column is interpolated at the TWT of its own cell centres, so the
    result registers with the elastic grid (and the photo) column by column.
    """
    section = np.asarray(section, dtype=float); vp = np.asarray(vp, dtype=float)
    nz, nx = vp.shape
    out = np.zeros((nz, nx))
    for j in range(nx):
        cell_twt = 2000.0 * dz / vp[:, j]
        t_center = pad_time + np.cumsum(cell_twt) - 0.5 * cell_twt
        out[:, j] = np.interp(t_center, time_array, section[:, j], left=0.0, right=0.0)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_section_tools.py -q`
Expected: 11 passed. If `test_oracle_every_column_matches_1d_synthetic` fails on a single sample, check that the interface sample index in both tools is `round(twt/dt)` with identical `pad_time` and `dt` — do **not** loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add tools/section_tools.py tests/test_section_tools.py
git commit -m "feat(section): generic 2-D convolutional synthetic over an elastic grid (oracle-tested vs 1-D)"
```

---

### Task 7: Depth domain, model adapter, and `plot_seismic_section` (`tools/section_tools.py`, part 2)

**Files:**
- Modify: `tools/section_tools.py` (append)
- Test: `tests/test_section_plot.py`

**Interfaces:**
- Consumes: `create_synthetic_section`, `depth_convert` (Task 6); EarthModel2D dict keys `vp, vs, rho, dz, dx, facies, legend, z, x` (Task 5); `tools.wedge_tools.plot_vawig(ax, data, t, z_min, dz, excursion)` (data is ntraces × nsamples, normalized inside).
- Produces:
  - `synthetic_section_from_model(model=None, wavelet_freq=30.0, wv_type="ricker", ormsby_freq=None, phase_rot=0.0, angle=0.0, method="shuey", dt=1.0, pad_time=50.0, domain="time") -> (axis, section, parameters)` — the registry-facing entry; `ValueError("Build an earth model first ...")` when `model` is None.
  - `plot_seismic_section(section, parameters, axis=None, model=None, display="image", output_path=None) -> str` — `display ∈ {"image","wiggle","both"}`; left panel = acoustic impedance of `model` when given; wiggle decimated to ≤ `MAX_WIGGLE_TRACES = 80`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_section_plot.py`:

```python
"""Depth conversion, the model adapter, and plot_seismic_section."""
import os

import numpy as np
import pytest

from tools.section_tools import (create_synthetic_section, synthetic_section_from_model,
                                 plot_seismic_section, MAX_WIGGLE_TRACES)


def _single_interface(nx=3, nz=200, dz=0.5, z_int=60.0):
    vp = np.full((nz, nx), 2500.0); vp[int(z_int / dz):] = 3500.0
    vs = vp / 2; rho = np.full((nz, nx), 2.3); rho[int(z_int / dz):] = 2.6
    return vp, vs, rho


def _model(nx=3, nz=200, dz=0.5):
    vp, vs, rho = _single_interface(nx, nz, dz)
    facies = np.zeros((nz, nx), int); facies[int(60.0 / dz):] = 1
    return {"vp": vp, "vs": vs, "rho": rho, "dz": dz, "dx": 2.0, "facies": facies,
            "legend": {0: {"lithology": "shale", "label": "background"},
                       1: {"lithology": "limestone", "label": "lime"}},
            "z": (np.arange(nz) + 0.5) * dz, "x": np.arange(nx) * 2.0}


def test_depth_domain_peak_sits_at_interface_depth():
    vp, vs, rho = _single_interface()
    z, sec, par = create_synthetic_section(vp, vs, rho, 0.5, 2.0, domain="depth")
    assert par["domain"] == "depth" and sec.shape == vp.shape and z.shape == (vp.shape[0],)
    peak_z = z[np.argmax(np.abs(sec[:, 0]))]
    assert abs(peak_z - 60.0) <= 0.5 + 1e-9          # within one cell (zero-phase wavelet)


def test_time_domain_peak_sits_at_interface_time():
    vp, vs, rho = _single_interface()
    t, sec, par = create_synthetic_section(vp, vs, rho, 0.5, 2.0, pad_time=50.0)
    expected = 50.0 + 2000.0 * 60.0 / 2500.0
    assert abs(t[np.argmax(np.abs(sec[:, 0]))] - expected) <= par["dt"]


def test_model_adapter_matches_direct_call():
    m = _model()
    a1, s1, p1 = synthetic_section_from_model(m, wavelet_freq=25.0)
    a2, s2, p2 = create_synthetic_section(m["vp"], m["vs"], m["rho"], m["dz"], m["dx"], wavelet_freq=25.0)
    np.testing.assert_allclose(s1, s2); np.testing.assert_allclose(a1, a2)
    assert p1 == p2


def test_model_adapter_requires_model():
    with pytest.raises(ValueError, match="earth model first"):
        synthetic_section_from_model(None)


@pytest.mark.parametrize("display", ["image", "wiggle", "both"])
def test_plot_modes_write_png(display):
    m = _model()
    axis, sec, par = synthetic_section_from_model(m)
    png = plot_seismic_section(sec, par, axis=axis, model=m, display=display)
    try:
        assert png.endswith(".png") and os.path.getsize(png) > 0
    finally:
        os.remove(png)


def test_plot_without_model_and_without_axis():
    m = _model()
    axis, sec, par = synthetic_section_from_model(m, domain="depth")
    png = plot_seismic_section(sec, par)      # axis reconstructed from parameters
    try:
        assert os.path.getsize(png) > 0
    finally:
        os.remove(png)


def test_plot_bad_display_rejected():
    m = _model()
    axis, sec, par = synthetic_section_from_model(m)
    with pytest.raises(ValueError, match="display"):
        plot_seismic_section(sec, par, display="hologram")


def test_wiggle_decimation_step():
    from tools.section_tools import _wiggle_step
    assert _wiggle_step(50) == 1 and _wiggle_step(MAX_WIGGLE_TRACES) == 1
    assert _wiggle_step(MAX_WIGGLE_TRACES + 1) == 2 and _wiggle_step(401) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_section_plot.py -q`
Expected: FAIL — `ImportError: cannot import name 'synthetic_section_from_model'`

- [ ] **Step 3: Append the adapter and plot**

Append to `tools/section_tools.py`:

```python
# ---------------------------------------------------------------------------
# Registry-facing adapter + plot
# ---------------------------------------------------------------------------

MAX_WIGGLE_TRACES = 80


def synthetic_section_from_model(model: Optional[Dict[str, Any]] = None, wavelet_freq=30.0,
                                 wv_type="ricker", ormsby_freq=None, phase_rot=0.0,
                                 angle=0.0, method="shuey", dt=1.0, pad_time=50.0,
                                 domain="time"):
    """Run create_synthetic_section on an EarthModel2D dict (vp, vs, rho, dz, dx).

    `model` is filled by the chatbot from the last outcrop_to_model result.
    """
    if model is None:
        raise ValueError("Build an earth model first (outcrop_to_model) — there is no "
                         "elastic grid to convolve.")
    for key in ("vp", "vs", "rho", "dz", "dx"):
        if key not in model:
            raise ValueError(f"model is missing {key!r}; expected an outcrop_to_model result")
    return create_synthetic_section(model["vp"], model["vs"], model["rho"], model["dz"], model["dx"],
                                    wavelet_freq=wavelet_freq, wv_type=wv_type,
                                    ormsby_freq=ormsby_freq, phase_rot=phase_rot, angle=angle,
                                    method=method, dt=dt, pad_time=pad_time, domain=domain)


def _wiggle_step(nx: int) -> int:
    return max(1, int(np.ceil(nx / float(MAX_WIGGLE_TRACES))))


def _axis_from_parameters(parameters: Dict[str, Any]) -> np.ndarray:
    if parameters.get("domain") == "depth":
        return (np.arange(int(parameters["nz"])) + 0.5) * float(parameters["dz"])
    return np.arange(int(parameters["nt"])) * float(parameters["dt"])


def plot_seismic_section(section, parameters, axis=None, model=None, display="image",
                         output_path=None) -> str:
    """Model (AI, depth) | section as variable-density image, wiggle, or both."""
    if display not in ("image", "wiggle", "both"):
        raise ValueError("display must be 'image', 'wiggle' or 'both'")
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    section = np.asarray(section, dtype=float)
    nsamp, nx = section.shape
    axis = np.asarray(axis, dtype=float) if axis is not None else _axis_from_parameters(parameters)
    dx = float(parameters["dx"])
    x = np.arange(nx) * dx
    domain = parameters.get("domain", "time")
    ylabel = "Depth (m)" if domain == "depth" else "TWT (ms)"
    amax = float(np.max(np.abs(section))) or 1.0

    panels = (["model"] if model is not None else []) + (["image", "wiggle"] if display == "both" else [display])
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 7), squeeze=False)
    axes = axes[0]
    for ax, kind in zip(axes, panels):
        if kind == "model":
            ai = np.asarray(model["vp"]) * np.asarray(model["rho"])
            z = np.asarray(model["z"]); xm = np.asarray(model["x"])
            im = ax.imshow(ai, aspect="auto", cmap="viridis",
                           extent=[xm[0], xm[-1], z[-1], z[0]])
            fig.colorbar(im, ax=ax, label="AI (m/s·g/cc)")
            ax.set_ylabel("Depth (m)"); ax.set_xlabel("Distance (m)")
            ax.set_title("Earth model (acoustic impedance)")
        elif kind == "image":
            ax.imshow(section, aspect="auto", cmap="seismic", vmin=-amax, vmax=amax,
                      extent=[x[0], x[-1], axis[-1], axis[0]])
            ax.set_ylabel(ylabel); ax.set_xlabel("Distance (m)")
            ax.set_title("Synthetic section")
        else:  # wiggle
            step = _wiggle_step(nx)
            data = section[:, ::step].T                 # ntraces x nsamp
            spacing = dx * step
            plot_vawig(ax, data, axis, x[0], spacing, 0.9 * spacing)
            ax.set_xlim(x[0] - spacing, x[::step][-1] + spacing)
            ax.set_ylim(axis[-1], axis[0])
            ax.set_ylabel(ylabel); ax.set_xlabel("Distance (m)")
            ax.set_title(f"Synthetic section (wiggle, every {step} trace(s))" if step > 1
                         else "Synthetic section (wiggle)")
    title = f"{parameters.get('wavelet_label', '')}"
    if parameters.get("angle", 0):
        title += f" — {parameters['angle']:g}°, {parameters['method']}"
    fig.suptitle(title.strip(" —"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_section_plot.py tests/test_section_tools.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add tools/section_tools.py tests/test_section_plot.py
git commit -m "feat(section): depth conversion, model adapter, plot_seismic_section (image/wiggle/both)"
```

---

### Task 8: Register the five tools (`core/tool_registry.py`)

**Files:**
- Modify: `core/tool_registry.py` (imports + five `ToolSpec`s appended before the `_WORKFLOW_TOOL_SPECS` block), `tests/test_tool_registry.py:6`
- Test: `tests/test_outcrop_registry.py`

**Interfaces:**
- Consumes: `interpret_outcrop`, `plot_outcrop_interpretation`, `outcrop_to_model` (Tasks 4–5); `synthetic_section_from_model`, `plot_seismic_section` (Task 7).
- Produces: registry names `interpret_outcrop`, `plot_outcrop_interpretation`, `outcrop_to_model`, `synthetic_section`, `plot_seismic_section`; `AUTO_PLOT["interpret_outcrop"] == "plot_outcrop_interpretation"`, `AUTO_PLOT["synthetic_section"] == "plot_seismic_section"`. Context-filled params (never sent by the LLM): `interpret_outcrop.image_path`, `outcrop_to_model.interpretation`, `synthetic_section.model`, all defaulting to `None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_outcrop_registry.py`:

```python
"""Registry contract for the outcrop tools."""
from core import tool_registry as reg
from core.tool_manager import ToolManager

NEW = ["interpret_outcrop", "plot_outcrop_interpretation", "outcrop_to_model",
       "synthetic_section", "plot_seismic_section"]


def test_tools_registered_with_functions():
    for name in NEW:
        assert name in reg.REGISTRY_BY_NAME, name
        assert callable(reg.TOOL_FUNCTIONS[name])


def test_auto_plot_chain():
    assert reg.AUTO_PLOT["interpret_outcrop"] == "plot_outcrop_interpretation"
    assert reg.AUTO_PLOT["synthetic_section"] == "plot_seismic_section"
    assert "outcrop_to_model" not in reg.AUTO_PLOT


def test_context_filled_params_are_optional_with_none_default():
    spec = reg.REGISTRY_BY_NAME
    assert "image_path" not in spec["interpret_outcrop"].required
    assert spec["interpret_outcrop"].defaults["image_path"] is None
    assert "interpretation" not in spec["outcrop_to_model"].required
    assert spec["outcrop_to_model"].defaults["interpretation"] is None
    assert "model" not in spec["synthetic_section"].required
    assert spec["synthetic_section"].defaults["model"] is None


def test_schema_descriptions_tell_llm_not_to_pass_context_params():
    schemas = {s["name"]: s for s in reg.TOOL_SCHEMAS}
    assert "automatically" in schemas["interpret_outcrop"]["parameters"]["properties"]["image_path"]["description"].lower()
    assert "automatically" in schemas["outcrop_to_model"]["parameters"]["properties"]["interpretation"]["description"].lower()
    assert "automatically" in schemas["synthetic_section"]["parameters"]["properties"]["model"]["description"].lower()


def test_tool_manager_surfaces_clear_errors_without_context():
    tm = ToolManager()
    import pytest
    with pytest.raises(ValueError, match="upload an outcrop photo"):
        tm.process_tool_call("interpret_outcrop", {})
    with pytest.raises(ValueError, match="interpret_outcrop"):
        tm.process_tool_call("outcrop_to_model", {"height_m": 10})
    with pytest.raises(ValueError, match="earth model first"):
        tm.process_tool_call("synthetic_section", {})


def test_synthetic_section_defaults():
    d = reg.REGISTRY_BY_NAME["synthetic_section"].defaults
    assert d["dt"] == 1.0 and d["domain"] == "time" and d["method"] == "shuey"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_outcrop_registry.py -q`
Expected: FAIL — `KeyError: 'interpret_outcrop'`

- [ ] **Step 3: Add imports and specs**

In `core/tool_registry.py`, after the `from tools.synthetic_tools import ...` line add:

```python
from tools.outcrop_tools import interpret_outcrop, plot_outcrop_interpretation, outcrop_to_model
from tools.section_tools import synthetic_section_from_model, plot_seismic_section
```

Insert these five specs into `REGISTRY` immediately after the `predict_elastic_layer` spec (before the closing `]`):

```python
    ToolSpec(
        name="interpret_outcrop",
        fn=interpret_outcrop,
        description=(
            "Interprets the user's uploaded outcrop PHOTO with a vision model: outlines "
            "rock bodies/beds as regions with a lithology each (sandstone, shale, limestone, "
            "coal, ...), estimates the exposure height in metres from any scale reference "
            "(hammer, person, scale bar) with a confidence, and returns the interpretation "
            "plus an overlay plot. Call this first when a message says '[image attached'. "
            "Follow with outcrop_to_model, then synthetic_section."
        ),
        params={
            "image_path": {"type": "string",
                           "description": "Leave empty — supplied automatically from the uploaded photo."},
        },
        required=[],
        defaults={"image_path": None},
        auto_plot="plot_outcrop_interpretation",
    ),
    ToolSpec(
        name="plot_outcrop_interpretation",
        fn=plot_outcrop_interpretation,
        description="Draws the interpreted facies polygons over the outcrop photo with a legend and the scale estimate.",
        params={
            "interpretation": {"type": "object",
                               "description": "Interpretation dict returned by interpret_outcrop."},
        },
        required=["interpretation"],
        defaults={},
    ),
    ToolSpec(
        name="outcrop_to_model",
        fn=outcrop_to_model,
        description=(
            "Turns the latest outcrop interpretation into a 2-D elastic earth model (Vp/Vs/density "
            "grids on a shale background): resolves the scale (your height_m overrides the photo's "
            "estimate; if neither exists it asks), maps each region's lithology to properties "
            "(Han 1986/Gassmann for clastics, literature values for carbonates/coal/salt), and pads "
            "the model above and below. Re-run it to apply corrections such as a new height or "
            "overrides like {\"2\": {\"fluid\": \"gas\"}} or {\"sand lens\": {\"lithology\": \"siltstone\"}}."
        ),
        params={
            "interpretation": {"type": "object",
                               "description": "Leave empty — supplied automatically from the last interpret_outcrop result."},
            "height_m": {"type": "number",
                         "description": "Total height of the photographed exposure in metres. Overrides the vision estimate; required if the photo had no scale reference."},
            "overrides": {"type": "object",
                          "description": "Per-region corrections keyed by region id or label; fields: lithology, fluid (water/brine/oil/gas), porosity, vclay. Fluid/porosity/vclay apply to clastic lithologies only."},
            "background_lithology": {"type": "string",
                                     "description": "Lithology filling everything not outlined (default: the interpretation's, normally 'shale')."},
            "num_traces": {"type": "integer",
                           "description": "Number of traces (columns) across the outcrop width (default 101)."},
            "wavelet_freq": {"type": "number",
                             "description": "Intended wavelet frequency in Hz, used to size the background padding (default 30)."},
            "pad_m": {"type": "number",
                      "description": "Background padding above and below the outcrop in metres (default 1.5 wavelengths)."},
        },
        required=[],
        defaults={"interpretation": None, "height_m": None, "overrides": None,
                  "background_lithology": None, "num_traces": 101,
                  "wavelet_freq": 30.0, "pad_m": None},
    ),
    ToolSpec(
        name="synthetic_section",
        fn=synthetic_section_from_model,
        description=(
            "Convolves the latest 2-D earth model (from outcrop_to_model) into a synthetic seismic "
            "section: per-trace depth-to-time, reflectivity at every property change (acoustic at "
            "normal incidence, Shuey/Zoeppritz at an angle), Ricker or Ormsby wavelet. Returns the "
            "section in time (default) or depth-converted, and plots it as a variable-density image, "
            "wiggle traces, or both."
        ),
        params={
            "model": {"type": "object",
                      "description": "Leave empty — supplied automatically from the last outcrop_to_model result."},
            "wavelet_freq": {"type": "number", "description": "Ricker dominant frequency in Hz (default 30)."},
            "wv_type": {"type": "string", "description": "'ricker' (default) or 'ormsby'."},
            "ormsby_freq": {"type": "string", "description": "Four increasing Ormsby corners 'f1,f2,f3,f4'; required when wv_type='ormsby'."},
            "phase_rot": {"type": "number", "description": "Wavelet phase rotation in degrees (default 0)."},
            "angle": {"type": "number", "description": "Incidence angle in degrees, 0 <= angle < 90 (default 0)."},
            "method": {"type": "string", "description": "'shuey' (default) or 'zoeppritz' when angle > 0."},
            "dt": {"type": "number", "description": "Time sampling in ms (default 1)."},
            "pad_time": {"type": "number", "description": "Quiet time in ms above the model top and below its base (default 50)."},
            "domain": {"type": "string", "description": "'time' (default, TWT) or 'depth' (depth-converted so it registers with the photo)."},
        },
        required=[],
        defaults={"model": None, "wavelet_freq": 30.0, "wv_type": "ricker", "ormsby_freq": None,
                  "phase_rot": 0.0, "angle": 0.0, "method": "shuey", "dt": 1.0,
                  "pad_time": 50.0, "domain": "time"},
        auto_plot="plot_seismic_section",
    ),
    ToolSpec(
        name="plot_seismic_section",
        fn=plot_seismic_section,
        description="Plots a synthetic seismic section (earth-model impedance panel plus image and/or wiggle display).",
        params={
            "section": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Section samples (nt x ntraces)."},
            "parameters": {"type": "object", "description": "Parameters dict returned by synthetic_section."},
            "axis": {"type": "array", "items": {"type": "number"}, "description": "Vertical axis returned by synthetic_section."},
            "model": {"type": "object", "description": "Earth model dict for the impedance panel (optional)."},
            "display": {"type": "string", "description": "'image' (default), 'wiggle', or 'both'."},
        },
        required=["section", "parameters"],
        defaults={"axis": None, "model": None, "display": "image"},
    ),
```

Then in `tests/test_tool_registry.py` change `assert len(reg.REGISTRY) == 33` to `assert len(reg.REGISTRY) == 38`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_outcrop_registry.py tests/test_tool_registry.py tests/test_tool_manager.py tests/test_no_dead_code.py -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add core/tool_registry.py tests/test_outcrop_registry.py tests/test_tool_registry.py
git commit -m "feat(registry): interpret_outcrop, outcrop_to_model, synthetic_section + plot tools with auto-plot chain"
```

---

### Task 9: `outcrop_to_seismic` workflow recipe

**Files:**
- Create: `workflows/recipes/outcrop_to_seismic.py`
- Modify: `workflows/engine.py` (import + `WorkflowSpec`), `tests/test_tool_registry.py:6` (38 → 39)
- Test: `tests/test_outcrop_to_seismic.py`

**Interfaces:**
- Consumes: `interpret_outcrop(image_path, vision_client=None)`, `plot_outcrop_interpretation`, `outcrop_to_model(...)` (Tasks 4–5); `synthetic_section_from_model`, `plot_seismic_section` (Task 7); `workflows.sweep.run_sweep(recipe, grid, metric, fixed=None)`.
- Produces: `outcrop_to_seismic(image_path=None, height_m=None, overrides=None, background_lithology=None, wavelet_freq=30.0, angle=0.0, method="shuey", domain="time", display="image", num_traces=101, vision_client=None) -> dict` with keys `interpretation`, `model`, `section` (`{"axis","section","parameters"}`), `regions` (provenance), `scale` (`{height_m, source, confidence}`), `grid_shape`, `n_regions`, `n_interfaces`, `max_abs_amplitude`, `wavelet_freq`, `angle`, `domain`, `image_path` (section PNG), `extra_image_paths` (`[overlay PNG]`). Registered as workflow `outcrop_to_seismic`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_outcrop_to_seismic.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_outcrop_to_seismic.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflows.recipes.outcrop_to_seismic'`

- [ ] **Step 3: Implement the recipe**

Create `workflows/recipes/outcrop_to_seismic.py`:

```python
"""outcrop_to_seismic: outcrop photo -> interpretation -> 2-D model -> seismic section.

One-shot chain of the staged tools (interpret_outcrop -> outcrop_to_model ->
synthetic_section -> plots). The chatbot stores the intermediate results so
follow-up corrections can re-run only the offline steps.
"""
import os
import tempfile

import numpy as np

from tools.outcrop_tools import (interpret_outcrop, plot_outcrop_interpretation,
                                 outcrop_to_model)
from tools.section_tools import synthetic_section_from_model, plot_seismic_section


def outcrop_to_seismic(image_path=None, height_m=None, overrides=None,
                       background_lithology=None, wavelet_freq=30.0, angle=0.0,
                       method="shuey", domain="time", display="image",
                       num_traces=101, vision_client=None):
    """Returns a JSON-friendly dict (see tests). Only interpret_outcrop calls the VLM."""
    interp = interpret_outcrop(image_path, vision_client=vision_client)
    overlay = plot_outcrop_interpretation(interp)
    model = outcrop_to_model(interp, height_m=height_m, overrides=overrides,
                             background_lithology=background_lithology,
                             num_traces=num_traces, wavelet_freq=wavelet_freq)
    axis, section, parameters = synthetic_section_from_model(
        model, wavelet_freq=wavelet_freq, angle=angle, method=method, domain=domain)
    png = plot_seismic_section(section, parameters, axis=axis, model=model, display=display)
    return {
        "interpretation": interp,
        "model": model,
        "section": {"axis": axis, "section": section, "parameters": parameters},
        "regions": model["regions"],
        "scale": {"height_m": model["height_m"], "source": model["scale_source"],
                  "confidence": model["scale_confidence"]},
        "grid_shape": [int(model["nz"]), int(model["nx"])],
        "n_regions": sum(1 for r in model["regions"] if r["route"] != "background"),
        "n_interfaces": parameters["n_interfaces"],
        "max_abs_amplitude": parameters["max_abs_amplitude"],
        "wavelet_freq": float(wavelet_freq),
        "angle": float(angle),
        "domain": domain,
        "image_path": png,
        "extra_image_paths": [overlay],
    }
```

- [ ] **Step 4: Register the workflow**

In `workflows/engine.py` add after the `petro_to_synthetic` import:

```python
from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic
```

Append to `WORKFLOW_REGISTRY` (after the `petro_to_synthetic` spec):

```python
    WorkflowSpec(
        name="outcrop_to_seismic",
        fn=outcrop_to_seismic,
        description=(
            "One-shot outcrop photo to synthetic seismic section: interprets the uploaded "
            "photo with a vision model (facies regions + scale), builds a 2-D elastic model "
            "on a shale background (Han 1986/Gassmann for clastics), and convolves it into a "
            "seismic section (image or wiggle, time or depth). Use when the user uploads a "
            "photo and asks directly for the seismic response; use the staged tools "
            "(interpret_outcrop / outcrop_to_model / synthetic_section) when they want to "
            "check or correct the interpretation first."
        ),
        params={
            "image_path": {"type": "string",
                           "description": "Leave empty — supplied automatically from the uploaded photo."},
            "height_m": {"type": "number",
                         "description": "Exposure height in metres (overrides the photo's scale estimate)."},
            "overrides": {"type": "object",
                          "description": "Per-region corrections keyed by id or label; fields lithology, fluid, porosity, vclay."},
            "background_lithology": {"type": "string", "description": "Background lithology (default shale)."},
            "wavelet_freq": {"type": "number", "description": "Ricker dominant frequency in Hz (default 30)."},
            "angle": {"type": "number", "description": "Incidence angle in degrees (default 0)."},
            "method": {"type": "string", "description": "'shuey' (default) or 'zoeppritz'."},
            "domain": {"type": "string", "description": "'time' (default) or 'depth'."},
            "display": {"type": "string", "description": "'image' (default), 'wiggle', or 'both'."},
            "num_traces": {"type": "integer", "description": "Traces across the outcrop width (default 101)."},
        },
        required=[],
        defaults={"image_path": None, "height_m": None, "overrides": None,
                  "background_lithology": None, "wavelet_freq": 30.0, "angle": 0.0,
                  "method": "shuey", "domain": "time", "display": "image", "num_traces": 101},
        auto_plot=None,
    ),
```

Change `tests/test_tool_registry.py` to `assert len(reg.REGISTRY) == 39`.

In `workflows/sweep.py::run_sweep`, where a cell's `image_path` PNG is removed, also remove the recipe's overlay plots so sweeps don't leak temp files:

```python
        for extra in result.get("extra_image_paths") or []:
            if isinstance(extra, str) and os.path.exists(extra):
                os.remove(extra)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_outcrop_to_seismic.py tests/test_tool_registry.py tests/test_workflow_engine.py tests/test_workflow_meta_tool.py -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add workflows/recipes/outcrop_to_seismic.py workflows/engine.py workflows/sweep.py tests/test_outcrop_to_seismic.py tests/test_tool_registry.py
git commit -m "feat(workflows): outcrop_to_seismic recipe — one-shot photo to section, run_sweep compatible"
```

---

### Task 10: Chatbot wiring (`core/chatbot_tool_use.py`)

**Files:**
- Modify: `core/chatbot_tool_use.py` — `__init__` (session id), `_create_system_prompt`, `_is_knowledge_question`, `_handle_tool_request`, `_harvest_images`, `_handle_automatic_chaining`, `_update_context`; add `attach_image`, `_inject_context_inputs`
- Test: `tests/test_chatbot_outcrop.py`

**Interfaces:**
- Consumes: registry names from Tasks 8–9; context keys `last_image`, `last_outcrop`, `last_earth_model`, `last_section`.
- Produces:
  - `SeismicChatBotToolUse.session_id: str` (uuid hex, fresh per instance; `new_session()` already builds a new instance).
  - `attach_image(path: str) -> None` — sets `last_image`.
  - `_inject_context_inputs(tool_name, tool_input) -> dict` — copies context into omitted params: `interpret_outcrop.image_path` / `outcrop_to_seismic.image_path` ← `last_image`; `outcrop_to_model.interpretation` ← `last_outcrop`; `synthetic_section.model` ← `last_earth_model`.
  - `_harvest_images` also collects every `.png` in a top-level `extra_image_paths` list.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chatbot_outcrop.py`:

```python
"""Chatbot wiring for the outcrop pipeline: context injection, storage, chains, prompt."""
import json
import os

import numpy as np
import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse
from tools import outcrop_tools as ot
from tools.section_tools import synthetic_section_from_model

INTERP = {"regions": [{"id": 1, "label": "sand", "lithology": "sandstone",
                       "geometry": {"type": "band", "y_top": 0.3, "y_bottom": 0.5}}],
          "scale": {"estimated_height_m": 20, "reference": "hammer", "confidence": "medium"},
          "background_lithology": "shale", "mode": "polygons"}


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "SEISMIC_UPLOAD_DIR", str(tmp_path))


@pytest.fixture
def bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]), knowledge_base=object())


@pytest.fixture
def interp(outcrop_image, fake_vision_factory):
    return ot.interpret_outcrop(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]))


def _rm(path):
    if path and os.path.exists(path):
        os.remove(path)


def test_session_id_unique_per_session(bot):
    a, b = bot.new_session(), bot.new_session()
    assert a.session_id != b.session_id and len(a.session_id) == 32


def test_attach_image_sets_last_image_per_session(bot, outcrop_image):
    a, b = bot.new_session(), bot.new_session()
    a.attach_image(outcrop_image)
    assert a.context_manager.get_context("last_image") == outcrop_image
    assert b.context_manager.get_context("last_image") is None


def test_inject_image_path_for_interpret_and_recipe(bot, outcrop_image):
    bot.attach_image(outcrop_image)
    assert bot._inject_context_inputs("interpret_outcrop", {}) == {"image_path": outcrop_image}
    assert bot._inject_context_inputs("outcrop_to_seismic", {"height_m": 5}) == {
        "height_m": 5, "image_path": outcrop_image}
    # explicit value wins; unrelated tools untouched
    assert bot._inject_context_inputs("interpret_outcrop", {"image_path": "x.png"}) == {"image_path": "x.png"}
    assert bot._inject_context_inputs("make_ricker", {"frequency": 30}) == {"frequency": 30}


def test_inject_interpretation_and_model(bot, interp):
    bot._update_context("interpret_outcrop", {}, interp)
    assert bot.context_manager.get_context("last_outcrop") is interp
    filled = bot._inject_context_inputs("outcrop_to_model", {"height_m": 30})
    assert filled["interpretation"] is interp
    model = ot.outcrop_to_model(interp, height_m=30, num_traces=11)
    bot._update_context("outcrop_to_model", filled, model)
    assert bot.context_manager.get_context("last_earth_model") is model
    assert bot._inject_context_inputs("synthetic_section", {})["model"] is model


def test_inject_without_context_leaves_param_absent(bot):
    assert bot._inject_context_inputs("outcrop_to_model", {}) == {}


def test_auto_chain_interpret_to_overlay(bot, interp):
    bot._update_context("interpret_outcrop", {}, interp)
    chained = bot._handle_automatic_chaining("interpret_outcrop", {}, interp)
    try:
        assert chained and chained["image_path"].endswith(".png")
    finally:
        _rm((chained or {}).get("image_path"))


def test_auto_chain_section_to_plot_uses_model_from_context(bot, interp):
    model = ot.outcrop_to_model(interp, height_m=20, num_traces=11)
    bot._update_context("outcrop_to_model", {}, model)
    result = synthetic_section_from_model(model)
    bot._update_context("synthetic_section", {"wavelet_freq": 30}, result)
    stored = bot.context_manager.get_context("last_section")
    assert stored["parameters"]["nt"] == result[2]["nt"] and stored["input_params"] == {"wavelet_freq": 30}
    chained = bot._handle_automatic_chaining("synthetic_section", {}, result)
    try:
        assert chained and os.path.getsize(chained["image_path"]) > 0
    finally:
        _rm((chained or {}).get("image_path"))


def test_auto_chain_without_context_returns_none(bot, interp):
    assert bot._handle_automatic_chaining("interpret_outcrop", {}, interp) is None


def test_recipe_result_populates_staged_context(bot, outcrop_image, fake_vision_factory):
    from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic
    res = outcrop_to_seismic(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]),
                             num_traces=11)
    try:
        bot._update_context("outcrop_to_seismic", {}, res)
        cm = bot.context_manager
        assert cm.get_context("last_outcrop") is res["interpretation"]
        assert cm.get_context("last_earth_model") is res["model"]
        assert cm.get_context("last_section")["parameters"] is res["section"]["parameters"]
        assert cm.get_context("last_workflow_result") is res
        images = []
        bot._harvest_images(res, images)
        assert images == [res["image_path"]] + res["extra_image_paths"]
    finally:
        _rm(res["image_path"]); _rm(res["extra_image_paths"][0])


def test_compaction_keeps_tool_result_small(bot, outcrop_image, fake_vision_factory):
    from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic
    res = outcrop_to_seismic(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]),
                             num_traces=101)
    try:
        text = bot._compact_tool_result(res)
        assert len(text) < 6000
        assert "<plot generated" in text and "facies" in text
    finally:
        _rm(res["image_path"]); _rm(res["extra_image_paths"][0])


def test_image_attached_marker_routes_to_tools(bot):
    assert bot._is_knowledge_question("[image attached: a.png] what is this?") is False


def test_tool_loop_injects_image_path(fake_llm_factory, outcrop_image, fake_vision_factory, monkeypatch):
    """End-to-end through _handle_tool_request with a scripted LLM and vision model."""
    class _Func:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

    class FakeToolCall:   # tests/ is not a package, so mirror conftest's shape locally
        def __init__(self, name, arguments, call_id="call_1"):
            self.id, self.function = call_id, _Func(name, arguments)

    monkeypatch.setattr("core.vision_client.build_vision_client",
                        lambda: fake_vision_factory([json.dumps(INTERP)]))
    llm = fake_llm_factory([
        {"content": "", "tool_calls": [FakeToolCall("interpret_outcrop", "{}")], "usage": None},
        {"content": "<reply>Found one sandstone bed, ~20 m high.</reply>", "tool_calls": None, "usage": None},
    ])
    bot = SeismicChatBotToolUse(llm_client=llm, knowledge_base=object())
    bot.attach_image(outcrop_image)
    out = bot._handle_tool_request("[image attached: outcrop.png] interpret this outcrop")
    try:
        assert "sandstone" in out["reply"]
        assert len(out["images"]) == 1 and out["images"][0].endswith(".png")
        assert bot.context_manager.get_context("last_outcrop")["regions"][0]["lithology"] == "sandstone"
        tool_msg = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"][0]
        assert "sandstone" in tool_msg["content"]
    finally:
        for p in out["images"]:
            _rm(p)


def test_system_prompt_lists_outcrop_tools(bot):
    prompt = bot._create_system_prompt()
    for name in ("interpret_outcrop", "outcrop_to_model", "synthetic_section", "outcrop_to_seismic"):
        assert f"- {name}:" in prompt
    assert "[image attached" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chatbot_outcrop.py -q`
Expected: FAIL — `AttributeError: 'SeismicChatBotToolUse' object has no attribute 'session_id'`

- [ ] **Step 3: Session id + `attach_image`**

In `core/chatbot_tool_use.py` add `import uuid` at the top. In `__init__`, after `self.context_manager = ContextManager()` add:

```python
        self.session_id = uuid.uuid4().hex  # names this session's upload sandbox subdir
```

Add these methods right after `new_session`:

```python
    def attach_image(self, path: str) -> None:
        """Remember the user's uploaded photo (per session) for the outcrop tools."""
        self.context_manager.set_context("last_image", path)

    # Tools whose heavy inputs live in per-session context rather than in the
    # LLM's arguments: (tool name, parameter name, context key).
    _CONTEXT_INPUTS = (
        ("interpret_outcrop", "image_path", "last_image"),
        ("outcrop_to_seismic", "image_path", "last_image"),
        ("outcrop_to_model", "interpretation", "last_outcrop"),
        ("synthetic_section", "model", "last_earth_model"),
    )

    def _inject_context_inputs(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Fill omitted context-resident parameters from the session context."""
        filled = dict(tool_input)
        for name, param, key in self._CONTEXT_INPUTS:
            if name == tool_name and filled.get(param) is None:
                value = self.context_manager.get_context(key)
                if value is not None:
                    filled[param] = value
                else:
                    filled.pop(param, None)
        return filled
```

- [ ] **Step 4: Use the injection in the tool loop and widen image harvesting**

In `_handle_tool_request`, replace

```python
                tool_input = self._parse_tool_input(tool_input_str)
                tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
```

with

```python
                tool_input = self._inject_context_inputs(
                    tool_name, self._parse_tool_input(tool_input_str))
                tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
```

In `_harvest_images`, replace the body after the docstring with:

```python
        paths = []
        if isinstance(tool_result, str) and tool_result.endswith(".png"):
            paths.append(tool_result)
        elif isinstance(tool_result, dict):
            p = tool_result.get("image_path")
            if isinstance(p, str) and p.endswith(".png"):
                paths.append(p)
            for extra in tool_result.get("extra_image_paths") or []:
                if isinstance(extra, str) and extra.endswith(".png"):
                    paths.append(extra)
        for path in paths:
            if path not in collected:
                collected.append(path)
```

and extend its docstring with: `A top-level "extra_image_paths" list (outcrop_to_seismic's overlay) is collected too.`

- [ ] **Step 5: Context storage and auto-chain branches**

In `_update_context`, before the `elif tool_name in WORKFLOW_NAMES:` branch, add:

```python
            elif tool_name == "interpret_outcrop":
                if isinstance(tool_result, dict) and "regions" in tool_result:
                    self.context_manager.set_context("last_outcrop", tool_result)

            elif tool_name == "outcrop_to_model":
                if isinstance(tool_result, dict) and "facies" in tool_result:
                    self.context_manager.set_context("last_earth_model", tool_result)

            elif tool_name == "synthetic_section":
                if isinstance(tool_result, tuple) and len(tool_result) == 3:
                    axis, section, parameters = tool_result
                    self.context_manager.set_context("last_section", {
                        "axis": axis,
                        "section": section,
                        "parameters": parameters,
                        "input_params": tool_input
                    })

            elif tool_name == "outcrop_to_seismic":
                if isinstance(tool_result, dict):
                    self.context_manager.set_context("last_workflow_result", tool_result)
                    if tool_result.get("interpretation") is not None:
                        self.context_manager.set_context("last_outcrop", tool_result["interpretation"])
                    if tool_result.get("model") is not None:
                        self.context_manager.set_context("last_earth_model", tool_result["model"])
                    sec = tool_result.get("section")
                    if isinstance(sec, dict):
                        self.context_manager.set_context("last_section", {
                            "axis": sec.get("axis"),
                            "section": sec.get("section"),
                            "parameters": sec.get("parameters"),
                            "input_params": tool_input
                        })
```

(`outcrop_to_seismic` is in `WORKFLOW_NAMES`, so this branch must sit **before** the generic workflow branch.)

In `_handle_automatic_chaining`, before the final `else: return None`, add:

```python
            elif tool_name == "interpret_outcrop":
                last = self.context_manager.get_context("last_outcrop")
                if not last:
                    return None
                plot_input = {"interpretation": last}
            elif tool_name == "synthetic_section":
                last = self.context_manager.get_context("last_section")
                if not (last and "section" in last and "parameters" in last):
                    return None
                plot_input = {
                    "section": last["section"],
                    "parameters": last["parameters"],
                    "axis": last.get("axis"),
                    "model": self.context_manager.get_context("last_earth_model"),
                    "display": (last.get("input_params") or {}).get("display", "image"),
                }
```

- [ ] **Step 6: Intent short-circuit and system prompt**

In `_is_knowledge_question`, add as the first statement of the method body (before the `try`):

```python
        if user_input.lstrip().startswith("[image attached"):
            return False  # an uploaded photo is always a tool request
```

In `_create_system_prompt`, after the `- petro_to_synthetic:` bullet add:

```
- interpret_outcrop: Interprets the user's uploaded outcrop photo with a vision model into facies regions (lithology each) plus a scale estimate with confidence, and shows an overlay plot. Use it when a message starts with "[image attached".
- outcrop_to_model: Builds a 2-D elastic earth model from the latest outcrop interpretation on a shale background; takes height_m (overrides the photo's scale; required if none was found) and per-region overrides (lithology / fluid / porosity / vclay keyed by region id or label). Re-run it for corrections — no vision call needed.
- synthetic_section: Convolves the latest 2-D earth model into a synthetic seismic section (wavelet frequency, angle, Shuey/Zoeppritz, time or depth domain) and plots it as an image, wiggle, or both.
- outcrop_to_seismic: One-shot photo → interpretation → 2-D model → seismic section (with both plots). Use when the user uploads a photo and asks directly for the seismic image; use the staged tools when they want to check or correct the interpretation first.
```

And after guideline 5 add:

```
6. A user message beginning "[image attached: ...]" means a photo was uploaded this turn: call interpret_outcrop (or outcrop_to_seismic if they ask directly for the seismic response). Never pass image_path, interpretation or model arguments yourself — they are supplied automatically.
7. After interpret_outcrop, report the regions and the scale estimate WITH its confidence, and ask the user to confirm or correct the height before building the model if the confidence is low or no scale was found.
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_chatbot_outcrop.py tests/test_chatbot_synthetic.py tests/test_chatbot_workflow.py tests/test_chatbot_narration.py tests/test_result_compaction.py tests/test_session_isolation.py -q`
Expected: all passed

- [ ] **Step 8: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_outcrop.py
git commit -m "feat(chatbot): outcrop pipeline wiring — context injection, staged context, auto-plot chains, prompt"
```

---

### Task 11: Gradio photo upload (`interfaces/gradio_interface.py`)

**Files:**
- Modify: `interfaces/gradio_interface.py` (imports, new `prepare_turn`, `respond` signature, layout, event wiring)
- Test: `tests/test_gradio_upload.py`

**Interfaces:**
- Consumes: `stage_upload(src, base_dir, session_id, max_mb)` (Task 1); `SeismicChatBotToolUse.session_id`, `attach_image` (Task 10); settings `SEISMIC_UPLOAD_DIR`, `MAX_IMAGE_MB`.
- Produces: `prepare_turn(message: str, image_path: Optional[str], session_bot, upload_dir: str, max_mb: float) -> str` — stages the image (if any), attaches it to the session, and returns the message text to send (prefixed `[image attached: <basename>] ` when an image was uploaded this turn). Raises `ValueError` from the sandbox unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gradio_upload.py`:

```python
"""Gradio upload path: staging into the sandbox + attaching to the session."""
import os

import pytest

from interfaces.gradio_interface import prepare_turn
from core.chatbot_tool_use import SeismicChatBotToolUse


def _bot():
    return SeismicChatBotToolUse(llm_client=object(), knowledge_base=object())


def test_no_image_passes_message_through(tmp_path):
    bot = _bot()
    assert prepare_turn("hello", None, bot, str(tmp_path), 10) == "hello"
    assert bot.context_manager.get_context("last_image") is None


def test_image_is_staged_attached_and_marked(outcrop_image, tmp_path):
    bot = _bot()
    base = str(tmp_path / "uploads")
    text = prepare_turn("what is this?", outcrop_image, bot, base, 10)
    staged = bot.context_manager.get_context("last_image")
    assert staged and staged.startswith(os.path.join(base, bot.session_id))
    assert text == f"[image attached: {os.path.basename(staged)}] what is this?"
    assert os.path.getsize(staged) == os.path.getsize(outcrop_image)


def test_image_with_empty_message_gets_default_request(outcrop_image, tmp_path):
    bot = _bot()
    text = prepare_turn("", outcrop_image, bot, str(tmp_path), 10)
    assert text.startswith("[image attached:") and "interpret" in text.lower()


def test_bad_upload_raises_value_error(tmp_path):
    bad = tmp_path / "x.gif"
    bad.write_bytes(b"GIF89a")
    with pytest.raises(ValueError, match="extension"):
        prepare_turn("hi", str(bad), _bot(), str(tmp_path / "u"), 10)


def test_two_sessions_do_not_share_last_image(outcrop_image, tmp_path):
    base = _bot()
    a, b = base.new_session(), base.new_session()
    prepare_turn("x", outcrop_image, a, str(tmp_path), 10)
    assert a.context_manager.get_context("last_image")
    assert b.context_manager.get_context("last_image") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gradio_upload.py -q`
Expected: FAIL — `ImportError: cannot import name 'prepare_turn'`

- [ ] **Step 3: Implement `prepare_turn` and wire the upload**

In `interfaces/gradio_interface.py`, extend the imports:

```python
import os
from typing import Optional

import gradio as gr
from core.chatbot_tool_use import SeismicChatBotToolUse
from config.example_prompts import EXAMPLE_PROMPTS, search_prompts, get_random_prompts
from config.settings import SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB
from tools.image_safety import stage_upload
```

Add after `append_bot_response`:

```python
DEFAULT_IMAGE_REQUEST = "Interpret this outcrop photo."


def prepare_turn(message: str, image_path: Optional[str], session_bot,
                 upload_dir: str, max_mb: float) -> str:
    """Stage an uploaded photo into the session sandbox and mark the message.

    The staged path is stored on the session (context key ``last_image``) so
    the outcrop tools can pick it up without the LLM ever handling a path.
    Returns the text to send to the chatbot.
    """
    message = (message or "").strip()
    if not image_path:
        return message
    staged = stage_upload(image_path, upload_dir, session_bot.session_id, max_mb)
    session_bot.attach_image(staged)
    if not message:
        message = DEFAULT_IMAGE_REQUEST
    return f"[image attached: {os.path.basename(staged)}] {message}"
```

Replace the `respond` function with:

```python
    def respond(message, image_path, chat_history, session_bot):
        """Process a user message (+ optional photo) using a per-session chatbot."""
        if session_bot is None:
            session_bot = base_bot.new_session()

        chat_history = chat_history or []
        shown = message or ("(photo uploaded)" if image_path else "")
        chat_history.append([shown, None])
        try:
            text = prepare_turn(message, image_path, session_bot, SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB)
            response = session_bot.process_single_input(text)
            chat_history = append_bot_response(chat_history, response)

            # Per-session token usage for display
            token_usage = session_bot.context_manager.get_token_usage()
            token_str = f"Prompt: {token_usage['prompt_tokens']} | Completion: {token_usage['completion_tokens']} | Total: {token_usage['total_tokens']}"
            return "", None, chat_history, token_str, session_bot

        except Exception as e:
            chat_history[-1][1] = f"Error processing request: {str(e)}"
            return "", None, chat_history, "", session_bot
```

In the layout, directly above the `with gr.Row():` that holds `msg`, add:

```python
                photo = gr.Image(type="filepath", label="Outcrop photo (optional — jpg/png/webp)",
                                 height=160)
```

Update the welcome Markdown bullet list with:

```
        - Interpreting an uploaded **outcrop photo** into a 2-D earth model and synthetic seismic section
```

Replace the two event bindings at the bottom with:

```python
        submit.click(respond, [msg, photo, chat_display, session_state],
                     [msg, photo, chat_display, token_usage_display, session_state])
        msg.submit(respond, [msg, photo, chat_display, session_state],
                   [msg, photo, chat_display, token_usage_display, session_state])
```

(Returning `None` for `photo` clears the upload widget after each send; the staged file stays attached to the session as `last_image` for follow-ups.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gradio_upload.py tests/test_gradio_response_format.py -q`
Expected: all passed

- [ ] **Step 5: Smoke-launch the UI**

Run (needs a `.env` with DeepSeek credentials): `python -c "from interfaces.gradio_interface import create_chat_interface; create_chat_interface()"`
Expected: returns without error (the Blocks object is built; no launch). If it raises about `gr.Image(height=...)`, drop the `height` kwarg.

- [ ] **Step 6: Commit**

```bash
git add interfaces/gradio_interface.py tests/test_gradio_upload.py
git commit -m "feat(ui): outcrop photo upload — staged into the session sandbox and attached to the chat turn"
```

---

### Task 12: Docs, example prompts, smoke script, full suite

**Files:**
- Modify: `CLAUDE.md`, `config/example_prompts.py`, `interfaces/web_interface.html`, `docs/superpowers/specs/2026-06-15-scientific-completeness-roadmap.md`
- Create: `test_outcrop_vision.py` (package root; standalone, not in the suite)

**Interfaces:** none new.

- [ ] **Step 1: Example prompts (keep `.py` and `.html` in sync)**

Add a new category to `config/example_prompts.py` after `"Workflows & Advanced Analysis"` (before `"Agentic Flows"`):

```python
    "Outcrop to Seismic": [
        {
            "title": "Interpret an outcrop photo",
            "prompt": "Interpret this outcrop photo: outline the beds and bodies, tell me the lithologies and how tall the exposure looks",
            "description": "Upload a photo first — interpret_outcrop returns facies regions, a scale estimate with confidence, and an overlay plot"
        },
        {
            "title": "Correct the scale and a facies",
            "prompt": "The cliff is 35 m high and region 2 is a gas-filled sandstone — rebuild the earth model",
            "description": "outcrop_to_model re-runs offline with height_m and per-region overrides (no new vision call)"
        },
        {
            "title": "Seismic section from the model",
            "prompt": "Generate the synthetic seismic section with a 40 Hz Ricker wavelet as wiggle traces in depth",
            "description": "synthetic_section convolves the 2-D model; image or wiggle display, time or depth domain"
        },
        {
            "title": "One-shot photo to seismic",
            "prompt": "Turn this outcrop photo straight into a seismic image with a 30 Hz wavelet; the exposure is about 50 m high",
            "description": "outcrop_to_seismic workflow: interpretation, 2-D shale-background model and section in one call"
        },
    ],
```

Add the same four entries, in the same category name and order, to the JSON object in `interfaces/web_interface.html` after the `"Workflows & Advanced Analysis"` array (before `"Agentic Flows"`), formatted like its neighbours.

Verify sync: `python - <<'PY'
import json, re
from config.example_prompts import EXAMPLE_PROMPTS
html = open("interfaces/web_interface.html").read()
m = re.search(r'"Outcrop to Seismic":\s*(\[.*?\])\s*,\s*"Agentic Flows"', html, re.S)
assert m, "html category missing"
assert json.loads(m.group(1)) == EXAMPLE_PROMPTS["Outcrop to Seismic"]
print("prompts in sync")
PY`

- [ ] **Step 2: Roadmap tick**

In `docs/superpowers/specs/2026-06-15-scientific-completeness-roadmap.md`, under "Build order summary" after item 3 add:

```
3b. Outcrop photo → 2-D earth model → seismic section (`interpret_outcrop` / `outcrop_to_model` / `synthetic_section` / `outcrop_to_seismic`). **DONE 2026-08-23** (see `2026-08-22-outcrop-to-seismic-design.md`). Adds the first 2-D gridded model (`tools/section_tools.py`) and the first vision capability.
```

- [ ] **Step 3: CLAUDE.md**

Add a section after "## N-layer synthetic seismogram":

```markdown
## Outcrop photo → seismic section

Spec: `docs/superpowers/specs/2026-08-22-outcrop-to-seismic-design.md`. Four staged registry
tools hand results through `ContextManager` so only the first touches a network:

1. `interpret_outcrop` (`tools/outcrop_tools.py`) — the uploaded photo → validated
   `OutcropInterpretation` (regions with lithology + normalized polygon/band geometry,
   scale estimate with confidence, background lithology) via `core/vision_client.py`
   (`AnthropicVisionClient` or `OpenAIVisionClient`, picked by `VISION_PROVIDER` /
   `ANTHROPIC_API_KEY` / `VISION_API_KEY`+`VISION_BASE_URL`; `VISION_MODEL` optional).
   One retry on invalid JSON, then a clear `ValueError`. Auto-plots
   `plot_outcrop_interpretation`. Stored as `last_outcrop`.
2. `outcrop_to_model` — scale policy **user `height_m` > vision estimate > ask**; per-region
   `overrides` (lithology / fluid / porosity / vclay, keyed by id or label);
   `LITHOLOGY_TABLE` routes clastics through `predict_layer` (Han 1986 + Gassmann) and
   carbonates/coal/salt/basalt through fixed literature Vp/Vs/ρ (petro overrides on those
   raise). Rasterizes polygons with `matplotlib.path.Path` onto an nz≈400 × `num_traces`
   grid, pads 1.5 background wavelengths above/below. Stored as `last_earth_model`.
3. `synthetic_section` (`tools/section_tools.py::synthetic_section_from_model`) — generic
   2-D convolutional model over **any** `(vp, vs, rho, dz, dx)` grid: per-column
   depth→TWT, RC at every property change (acoustic / Shuey / Zoeppritz; post-critical
   NaNs → 0 with a warning), superposition onto the `dt` grid (default **1 ms**), Ricker or
   Ormsby. `domain="depth"` returns a column-wise depth-converted section. Oracle-tested
   per column against `create_synthetic_seismogram`. Auto-plots `plot_seismic_section`
   (`display` = image / wiggle / both; wiggle decimated to ≤ 80 traces). Stored as
   `last_section`.
4. `outcrop_to_seismic` (`workflows/recipes/`) — one-shot chain; its result also populates
   the three staged context keys, so corrections after a one-shot run re-use steps 2–3.

The chatbot fills `image_path` / `interpretation` / `model` from context
(`_inject_context_inputs`) — the LLM never passes them. A message starting with
`[image attached: …]` (added by the Gradio upload via `prepare_turn`) is always routed to
tools. Uploads are staged into `SEISMIC_UPLOAD_DIR/<session_id>/` by
`tools/image_safety.py` (`.jpg/.jpeg/.png/.webp`, `MAX_IMAGE_MB`, traversal rejected) and
downscaled to ≤ 1568 px for the vision call. Vision credentials are optional: without
them `interpret_outcrop` raises at call time and everything else works. Tests:
`tests/test_image_safety.py`, `test_vision_client.py`, `test_outcrop_*.py`,
`test_section_*.py`, `test_chatbot_outcrop.py`, `test_gradio_upload.py`; real-VLM smoke:
`python test_outcrop_vision.py <photo>` (credential-gated, not in the suite).
```

Also add rows to the **Environment variables** tables: `VISION_PROVIDER`, `ANTHROPIC_API_KEY`, `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL` (LLM provider table, marked optional) and `SEISMIC_UPLOAD_DIR`, `MAX_IMAGE_MB` (security containment table), with the defaults from the spec.

- [ ] **Step 4: Real-VLM smoke script**

Create `test_outcrop_vision.py` at the package root:

```python
"""Manual smoke test against a real vision provider (NOT part of the pytest suite).

Usage:  python test_outcrop_vision.py path/to/outcrop.jpg [height_m]
Needs ANTHROPIC_API_KEY or VISION_API_KEY + VISION_BASE_URL in .env.
"""
import os
import shutil
import sys

from config.settings import SEISMIC_UPLOAD_DIR
from core.vision_client import build_vision_client
from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    try:
        client = build_vision_client()
    except RuntimeError as exc:
        print(f"skipped: {exc}"); return 0
    os.makedirs(SEISMIC_UPLOAD_DIR, exist_ok=True)
    staged = os.path.join(SEISMIC_UPLOAD_DIR, os.path.basename(sys.argv[1]))
    shutil.copyfile(sys.argv[1], staged)
    height = float(sys.argv[2]) if len(sys.argv) > 2 else None
    res = outcrop_to_seismic(staged, height_m=height, vision_client=client, display="both")
    print(res["interpretation"]["summary"])
    for r in res["regions"]:
        print(f"  #{r['id']} {r['label']:<20} {r['lithology']:<16} route={r['route']:<10} "
              f"vp={r['vp']} cells={r['n_cells']}")
    print("scale:", res["scale"])
    print("section:", res["grid_shape"], "max|amp| =", round(res["max_abs_amplitude"], 4))
    print("plots:", res["image_path"], res["extra_image_paths"][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Full suite**

Run: `pytest -q`
Expected: all tests pass (the pre-existing `test_tool_use_pattern` stdin-capture failure, if present on this branch, is unrelated — note it in the task report rather than fixing it here).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md config/example_prompts.py interfaces/web_interface.html docs/superpowers/specs/2026-06-15-scientific-completeness-roadmap.md test_outcrop_vision.py
git commit -m "docs(outcrop): CLAUDE.md section + env vars, example prompts (py+html sync), roadmap tick, VLM smoke script"
```
