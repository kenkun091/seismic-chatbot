# Outcrop Web App — Server API (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the seismic chatbot a session-based REST API (`/sessions/...`) through which a browser client can upload an outcrop photo, run the interpretation, push hand-drawn region geometry back, build the model and depth-domain section, fetch plot PNGs, and chat on the same session — all behind the existing `X-API-Key` gate.

**Architecture:** A `SessionStore` (`interfaces/sessions.py`) keeps one `SeismicChatBotToolUse` session per id with a lock, an allow-list of servable files, a version counter and an idle TTL. An `APIRouter` (`interfaces/outcrop_api.py`) exposes the routes and runs every registry tool through `ToolLoopRunner.execute_call` (the same per-call path as chat turns and skill replay) with a new `auto_plot=False` opt-out. `interfaces/serialize.py` turns numpy-bearing tool results into JSON payloads. The router is mounted into the existing FastAPI app, which also serves the (later) client bundle at `/app`.

**Tech Stack:** Python 3, FastAPI + Starlette `TestClient`, `python-multipart` (upload), numpy, the existing tool registry / tool loop / image sandbox.

**Spec:** `docs/superpowers/specs/2026-09-01-outcrop-webapp-design.md` (sections "Architecture", "API surface", "Security and limits", "Testing", "Repository layout", phase 1 of "Delivery phases"). The client (phases 2–4) is a separate plan.

## Global Constraints

- Run everything from inside `geo-mcp/seismic_chatbot/` (absolute top-level imports: `from core... import`, `from interfaces... import`). Commit with `git` from this directory (it is its own repo).
- Tests are hermetic: no network, no LLM/vision credentials. Use `fake_llm_factory` / `fake_vision_factory` / `outcrop_image` from `tests/conftest.py`; fake the vision backend with `monkeypatch.setattr("core.vision_client.build_vision_client", lambda: fake)`.
- Importing `interfaces.api_interface` builds the real `SeismicChatBotToolUse()` (needs credentials). Tests that touch it must stub `core.chatbot_tool_use.SeismicChatBotToolUse` **before** `importlib.reload(api_module)` (pattern in `tests/test_api_chat_contract.py`). Router-level tests instead build their own `FastAPI()` app with `build_router(...)`.
- Every tool route calls `entry.bot._tool_loop.execute_call(name, args, images, auto_plot=False)`. Never call `tools.*` functions directly from a route (except `validate_interpretation` for `PUT /interpretation`, which is a validator, not a tool).
- All `/sessions*` routes take the auth dependency passed to `build_router` — in production `enforce_chat_policy` (fail-closed `X-API-Key`, per-client rate limit).
- The file route serves **only** paths registered in `entry.allowed_files`; never build a path from the URL.
- Interpretation caps (verbatim from the spec): ≤ 200 regions, ≤ 2000 points per region, ≤ 1 MB body → `413`.
- Error mapping (verbatim): tool/validator `ValueError` → `400 {"error": msg}`; missing vision credentials → `503`; unknown session → `404`; session busy → `409`; body caps → `413`.
- Env vars added: `SESSION_TTL_SECONDS` (default `7200`), `MAX_SESSIONS` (default `50`).
- Numeric arrays in responses are rounded to 4 significant digits; `traces` is `nx` columns of `len(z)` floats (the section is stored `(nz × nx)`, so `traces = section.T`).
- `POST /chat` (legacy) must keep passing `tests/test_api_chat_contract.py` unchanged.
- The existing `ContextManager` is **not** modified; the store detects changes by comparing object identity of `last_outcrop` / `last_earth_model` / `last_section` before and after a request.
- `.gitignore` ignores `*.json` and `*.txt` (except `requirements.txt`) — do not add JSON fixtures; build test payloads in Python.

---

## File structure

| File | Responsibility |
|---|---|
| `core/tool_loop.py` (modify `execute_call`, lines 219–277) | add keyword-only `auto_plot: bool = True`; when `False`, skip `handle_automatic_chaining` and emit no `auto_plot` event |
| `interfaces/serialize.py` (create) | `to_jsonable(value, sig=4)`; `model_summary(model)`; `section_payload(last_section)`; `interpretation_caps(data)` |
| `interfaces/sessions.py` (create) | `SessionEntry`, `SessionStore` (create/get/delete/sweep, `acquire()` context manager: lock → busy, `last_used`, version bump), `SessionNotFound`, `SessionBusy`, `SessionLimit` |
| `interfaces/outcrop_api.py` (create) | `build_router(store, auth_dependency, upload_dir, max_image_mb) -> APIRouter` with all `/sessions` routes and the error mapping |
| `interfaces/api_interface.py` (modify) | build the store from env, `app.include_router(...)`, mount `webapp/dist` at `/app` when present |
| `requirements.txt` (modify) | add `fastapi`, `uvicorn`, `python-multipart` |
| `CLAUDE.md` (modify) | "Outcrop web app API" section + env-var rows |
| `tests/test_execute_call.py` (modify) | auto-plot opt-out tests |
| `tests/test_serialize.py` (create) | serializer tests |
| `tests/test_sessions.py` (create) | store tests |
| `tests/test_outcrop_api.py` (create) | route tests (lifecycle, upload, files, interpret, PUT interpretation, model, section oracle, plot, chat, auth) |
| `tests/test_api_mount.py` (create) | router is mounted in the real app; `/app` mount behavior |

---

### Task 1: `execute_call(auto_plot=False)` opt-out

**Files:**
- Modify: `core/tool_loop.py:219-277`
- Test: `tests/test_execute_call.py`

**Interfaces:**
- Consumes: `ToolLoopRunner.execute_call(tool_name, raw_input, collected_images)` (existing).
- Produces: `ToolLoopRunner.execute_call(tool_name: str, raw_input: dict, collected_images: list[str], *, auto_plot: bool = True) -> Any`. With `auto_plot=False` no plot tool runs, `collected_images` gains only images the tool itself returned, and no `auto_plot` trace event is emitted. Everything else (context injection, warning capture, `tool_call` event, `update_context`, recording) is unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_execute_call.py`)

```python
def test_execute_call_auto_plot_opt_out_skips_chaining():
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("make a 30 Hz ricker")
    runner = ToolLoopRunner(None, ToolManager(), cm)
    images = []
    result = runner.execute_call("make_ricker", {"frequency": 30}, images, auto_plot=False)
    assert isinstance(result, tuple) and len(result) == 2
    assert images == []                                   # no plot rendered
    assert cm.get_context("last_ricker_wavelet") is not None  # context still updated
    kinds = [e["t"] for e in cm.trace.events]
    assert "tool_call" in kinds and "auto_plot" not in kinds


def test_execute_call_auto_plot_default_still_chains():
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("make a 30 Hz ricker")
    runner = ToolLoopRunner(None, ToolManager(), cm)
    images = []
    runner.execute_call("make_ricker", {"frequency": 30}, images)
    assert images and images[0].endswith(".png")
    assert any(e["t"] == "auto_plot" and e.get("fired") for e in cm.trace.events)
    for p in images:
        import os
        os.remove(p)
```

- [ ] **Step 2: Run them to verify the first fails**

Run: `pytest tests/test_execute_call.py -q -k auto_plot`
Expected: `test_execute_call_auto_plot_opt_out_skips_chaining` FAILS with `TypeError: execute_call() got an unexpected keyword argument 'auto_plot'`; the default test passes.

- [ ] **Step 3: Implement**

In `core/tool_loop.py`, change the signature and the chaining block:

```python
    def execute_call(self, tool_name: str, raw_input: Dict[str, Any],
                     collected_images: List[str], *, auto_plot: bool = True) -> Any:
        """Run ONE tool with everything a live turn does around it: context
        injection, warning capture, tool_call event, context update, image
        harvest + provenance sidecar, auto-plot chaining, and the in-memory
        current_turn_calls recording used by save_skill. Shared by run(),
        skill replay and the session API. Raises on tool failure; returns the
        raw result. ``auto_plot=False`` skips the plot chain entirely (no plot
        tool runs, no auto_plot event) — used by API routes whose client renders
        the result itself."""
```

and replace the block from `chained_result = self.handle_automatic_chaining(...)` through the `elif AUTO_PLOT.get(tool_name):` branch with:

```python
        if auto_plot:
            chained_result = self.handle_automatic_chaining(tool_name, tool_input, tool_result)
            if chained_result:
                before_chained = len(collected_images)
                self.harvest_images(chained_result, collected_images)
                self._write_provenance(collected_images[before_chained:],
                                       AUTO_PLOT.get(tool_name) or "auto_plot",
                                       {}, compute_tool=tool_name,
                                       compute_input=public_input)
                emit_event(self.context_manager, "auto_plot", compute=tool_name,
                           plot=AUTO_PLOT.get(tool_name), fired=True)
            elif AUTO_PLOT.get(tool_name):
                logger.warning(
                    f"auto-plot {AUTO_PLOT[tool_name]} did not run after "
                    f"{tool_name} (missing context or plot error)")
                emit_event(self.context_manager, "auto_plot", compute=tool_name,
                           plot=AUTO_PLOT[tool_name], fired=False)
        return tool_result
```

- [ ] **Step 4: Run the whole file and the loop/skill suites**

Run: `pytest tests/test_execute_call.py tests/test_tool_loop_trace.py tests/test_skill_execution.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add core/tool_loop.py tests/test_execute_call.py
git commit -m "feat(tool_loop): execute_call(auto_plot=False) opt-out for API callers"
```

---

### Task 2: JSON serialization helpers

**Files:**
- Create: `interfaces/serialize.py`
- Test: `tests/test_serialize.py`

**Interfaces:**
- Produces:
  - `to_jsonable(value: Any, sig: int = 4) -> Any` — numpy arrays → nested lists, numpy scalars → Python scalars, floats rounded to `sig` significant digits, `NaN`/`inf` → `None`, tuples → lists, dicts recursed with keys stringified, other objects → `str(obj)`.
  - `model_summary(model: dict) -> dict` — the scalar/legend subset of an `outcrop_to_model` result: `height_m, width_m, image_top_m, dz, dx, nz, nx, pad_m, scale_source, scale_confidence, background_lithology, legend, regions` (no `facies`/`vp`/`vs`/`rho`/`z`/`x`/`image_path`). `legend` keys become strings.
  - `section_payload(last_section: dict, model: dict | None) -> dict` — `{"z": [...], "traces": [[...], ...], "domain", "dx", "nx", "wavelet_freq", "angle", "method", "max_abs_amplitude", "image_top_m", "height_m", "width_m"}` where `traces = section.T` and the three photo keys come from `model` (or `None` when absent). Raises `ValueError` if `last_section["parameters"]["domain"] != "depth"`.
  - `interpretation_caps(data: Any, max_regions: int = 200, max_points: int = 2000) -> None` — raises `ValueError` when `data["regions"]` (if a list) exceeds `max_regions` or any region's `points` list exceeds `max_points`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_serialize.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_serialize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'interfaces.serialize'`.

- [ ] **Step 3: Implement**

```python
# interfaces/serialize.py
"""JSON payload builders for the session API.

Tool results carry numpy arrays and server paths; the browser client needs
plain JSON with bounded precision. Values only — never paths."""
import math
from typing import Any, Dict, Optional

import numpy as np

_MODEL_SUMMARY_KEYS = ("height_m", "width_m", "image_top_m", "dz", "dx", "nz", "nx",
                       "pad_m", "scale_source", "scale_confidence",
                       "background_lithology", "legend", "regions")
_PHOTO_KEYS = ("image_top_m", "height_m", "width_m")


def _round_sig(x: float, sig: int) -> Optional[float]:
    if math.isnan(x) or math.isinf(x):
        return None
    if x == 0.0:
        return 0.0
    return float(f"{x:.{sig}g}")


def to_jsonable(value: Any, sig: int = 4) -> Any:
    """Recursively convert numpy/tuples/floats into JSON-native values."""
    if isinstance(value, np.ndarray):
        return [to_jsonable(v, sig) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, sig) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v, sig) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _round_sig(float(value), sig)
    return str(value)


def model_summary(model: Dict[str, Any]) -> Dict[str, Any]:
    """Scalars + legend + region provenance of an outcrop_to_model result; no grids."""
    return {k: to_jsonable(model.get(k)) for k in _MODEL_SUMMARY_KEYS}


def section_payload(last_section: Dict[str, Any],
                    model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Depth-domain section as columns plus the photo extent for overlay registration."""
    params = last_section["parameters"]
    if params.get("domain") != "depth":
        raise ValueError("section must be computed in the depth domain for the overlay")
    section = np.asarray(last_section["section"], dtype=float)
    out: Dict[str, Any] = {
        "z": to_jsonable(np.asarray(last_section["axis"], dtype=float)),
        "traces": to_jsonable(section.T),
        "domain": "depth",
    }
    for k in ("dx", "nx", "wavelet_freq", "angle", "method", "max_abs_amplitude"):
        out[k] = to_jsonable(params.get(k))
    for k in _PHOTO_KEYS:
        out[k] = to_jsonable(model.get(k)) if isinstance(model, dict) else None
    return out


def interpretation_caps(data: Any, max_regions: int = 200, max_points: int = 2000) -> None:
    """Bound the rasterization cost of a client-supplied interpretation."""
    if not isinstance(data, dict):
        return
    regions = data.get("regions")
    if isinstance(regions, list):
        if len(regions) > max_regions:
            raise ValueError(f"too many regions ({len(regions)} > {max_regions})")
        for r in regions:
            pts = r.get("points") if isinstance(r, dict) else None
            if isinstance(pts, list) and len(pts) > max_points:
                raise ValueError(f"too many points in a region ({len(pts)} > {max_points})")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_serialize.py -q`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add interfaces/serialize.py tests/test_serialize.py
git commit -m "feat(api): JSON serializers for model summary, depth section and interpretation caps"
```

---

### Task 3: `SessionStore`

**Files:**
- Create: `interfaces/sessions.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- Consumes: `base_bot.new_session()` returning an object with `.session_id: str`, `.context_manager` (with `get_context(key)`), `.attach_image(path)`, `._tool_loop`, `.process_single_input(msg)`.
- Produces:
  - `class SessionNotFound(KeyError)`, `class SessionBusy(RuntimeError)`, `class SessionLimit(RuntimeError)`.
  - `@dataclass SessionEntry`: `bot`, `lock: threading.Lock`, `created: float`, `last_used: float`, `allowed_files: dict[str, str]` (basename → absolute path), `version: int`, `plot_files: list[str]`.
  - `class SessionStore(base_bot, ttl_seconds=7200.0, max_sessions=50, upload_dir=None, clock=time.time)`:
    - `create() -> SessionEntry` (sweeps first; raises `SessionLimit` at cap)
    - `get(session_id) -> SessionEntry` (raises `SessionNotFound`)
    - `delete(session_id) -> None` (removes the session's upload subdir `upload_dir/<session_id>` and every path in `plot_files`; missing files ignored; unknown id → `SessionNotFound`)
    - `sweep() -> list[str]` (deletes entries idle longer than `ttl_seconds`; returns their ids)
    - `acquire(session_id)` — context manager yielding the entry; raises `SessionBusy` if the lock is already held; on exit sets `last_used` and bumps `version` when `identity_snapshot` changed.
    - `identity_snapshot(entry) -> tuple[int, int, int]` — `id()` of `last_outcrop`, `last_earth_model`, `last_section` from the bot's context.
    - `register_file(entry, path) -> str` — records `path` under its basename in `allowed_files` and returns the basename; also appends to `plot_files` when it ends with `.png`.
    - `__len__`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sessions.py
import os
import threading

import pytest

from interfaces.sessions import (SessionStore, SessionNotFound, SessionBusy, SessionLimit)


class _Ctx:
    def __init__(self):
        self.d = {}

    def get_context(self, k, default=None):
        return self.d.get(k, default)

    def set_context(self, k, v):
        self.d[k] = v


class _Session:
    _n = 0

    def __init__(self):
        _Session._n += 1
        self.session_id = f"sid{_Session._n}"
        self.context_manager = _Ctx()


class _Base:
    def new_session(self):
        return _Session()


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def store(tmp_path):
    return SessionStore(_Base(), ttl_seconds=100, max_sessions=2,
                        upload_dir=str(tmp_path), clock=_Clock())


def test_create_get_delete(store):
    e = store.create()
    assert store.get(e.bot.session_id) is e and len(store) == 1
    store.delete(e.bot.session_id)
    assert len(store) == 0
    with pytest.raises(SessionNotFound):
        store.get(e.bot.session_id)
    with pytest.raises(SessionNotFound):
        store.delete(e.bot.session_id)


def test_cap_and_sweep(store):
    a = store.create(); b = store.create()
    with pytest.raises(SessionLimit):
        store.create()
    store._clock.t += 101            # both idle past ttl
    assert sorted(store.sweep()) == sorted([a.bot.session_id, b.bot.session_id])
    assert len(store) == 0
    store.create()                   # room again after sweep


def test_delete_removes_upload_dir_and_plots(store, tmp_path):
    e = store.create()
    sub = tmp_path / e.bot.session_id
    sub.mkdir()
    (sub / "photo.png").write_bytes(b"x")
    plot = tmp_path / "plot.png"
    plot.write_bytes(b"y")
    assert store.register_file(e, str(plot)) == "plot.png"
    store.delete(e.bot.session_id)
    assert not sub.exists() and not plot.exists()


def test_acquire_updates_last_used_and_version_on_context_change(store):
    e = store.create()
    v0 = e.version
    store._clock.t += 5
    with store.acquire(e.bot.session_id) as entry:
        assert entry is e
    assert e.last_used == store._clock.t and e.version == v0   # nothing changed
    with store.acquire(e.bot.session_id) as entry:
        entry.bot.context_manager.set_context("last_outcrop", {"regions": []})
    assert e.version == v0 + 1
    with store.acquire(e.bot.session_id):
        pass
    assert e.version == v0 + 1                                # same object → no bump


def test_acquire_is_exclusive(store):
    e = store.create()
    with store.acquire(e.bot.session_id):
        with pytest.raises(SessionBusy):
            with store.acquire(e.bot.session_id):
                pass
    with store.acquire(e.bot.session_id):     # released after the block
        pass


def test_register_file_allowlists_by_basename(store, tmp_path):
    e = store.create()
    p = tmp_path / "a.png"; p.write_bytes(b"z")
    name = store.register_file(e, str(p))
    assert e.allowed_files[name] == str(p)
    assert str(p) in e.plot_files
    q = tmp_path / "photo.jpg"; q.write_bytes(b"z")
    store.register_file(e, str(q))
    assert str(q) not in e.plot_files       # only .png plots are cleanup targets
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sessions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'interfaces.sessions'`.

- [ ] **Step 3: Implement**

```python
# interfaces/sessions.py
"""Server-owned chat sessions for the browser client.

One SeismicChatBotToolUse session per id (shared heavy components, fresh
ContextManager), a per-session lock (the tool loop is not concurrency-safe),
an allow-list of files the file route may serve, a version counter that
tracks changes to the outcrop context keys, and an idle TTL."""
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

_TRACKED_KEYS = ("last_outcrop", "last_earth_model", "last_section")


class SessionNotFound(KeyError):
    pass


class SessionBusy(RuntimeError):
    pass


class SessionLimit(RuntimeError):
    pass


@dataclass
class SessionEntry:
    bot: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    created: float = 0.0
    last_used: float = 0.0
    allowed_files: Dict[str, str] = field(default_factory=dict)
    plot_files: List[str] = field(default_factory=list)
    version: int = 0


class SessionStore:
    def __init__(self, base_bot: Any, ttl_seconds: float = 7200.0, max_sessions: int = 50,
                 upload_dir: Optional[str] = None, clock: Callable[[], float] = time.time):
        self._base = base_bot
        self._ttl = float(ttl_seconds)
        self._max = int(max_sessions)
        self._upload_dir = upload_dir
        self._clock = clock
        self._entries: Dict[str, SessionEntry] = {}
        self._guard = threading.Lock()   # protects _entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- lifecycle ---------------------------------------------------------
    def create(self) -> SessionEntry:
        self.sweep()
        with self._guard:
            if len(self._entries) >= self._max:
                raise SessionLimit(f"session limit reached ({self._max})")
            bot = self._base.new_session()
            now = self._clock()
            entry = SessionEntry(bot=bot, created=now, last_used=now)
            self._entries[bot.session_id] = entry
            return entry

    def get(self, session_id: str) -> SessionEntry:
        try:
            return self._entries[session_id]
        except KeyError:
            raise SessionNotFound(session_id)

    def delete(self, session_id: str) -> None:
        with self._guard:
            entry = self._entries.pop(session_id, None)
        if entry is None:
            raise SessionNotFound(session_id)
        self._cleanup(session_id, entry)

    def sweep(self) -> List[str]:
        now = self._clock()
        expired: List[Tuple[str, SessionEntry]] = []
        with self._guard:
            for sid, entry in list(self._entries.items()):
                if now - entry.last_used > self._ttl and not entry.lock.locked():
                    expired.append((sid, self._entries.pop(sid)))
        for sid, entry in expired:
            self._cleanup(sid, entry)
        return [sid for sid, _ in expired]

    def _cleanup(self, session_id: str, entry: SessionEntry) -> None:
        if self._upload_dir:
            shutil.rmtree(os.path.join(self._upload_dir, session_id), ignore_errors=True)
        for path in entry.plot_files:
            try:
                os.remove(path)
            except OSError:
                pass

    # -- per-request access ------------------------------------------------
    @staticmethod
    def identity_snapshot(entry: SessionEntry) -> Tuple[int, ...]:
        cm = entry.bot.context_manager
        return tuple(id(cm.get_context(k)) for k in _TRACKED_KEYS)

    @contextmanager
    def acquire(self, session_id: str) -> Iterator[SessionEntry]:
        entry = self.get(session_id)
        if not entry.lock.acquire(blocking=False):
            raise SessionBusy(session_id)
        before = self.identity_snapshot(entry)
        try:
            yield entry
        finally:
            if self.identity_snapshot(entry) != before:
                entry.version += 1
            entry.last_used = self._clock()
            entry.lock.release()

    @staticmethod
    def register_file(entry: SessionEntry, path: str) -> str:
        name = os.path.basename(path)
        entry.allowed_files[name] = path
        if path.endswith(".png") and path not in entry.plot_files:
            entry.plot_files.append(path)
        return name
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sessions.py -q`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add interfaces/sessions.py tests/test_sessions.py
git commit -m "feat(api): SessionStore with lock, allow-listed files, version counter and idle TTL"
```

---

### Task 4: Router skeleton — session lifecycle, upload, file serving, state, auth

**Files:**
- Create: `interfaces/outcrop_api.py`
- Test: `tests/test_outcrop_api.py`

**Interfaces:**
- Consumes: `SessionStore` (Task 3), `tools.image_safety.stage_upload(src_path, base_dir, session_id, max_mb) -> str`, `tools.image_safety.image_size(path) -> (w, h)`, `entry.bot.attach_image(path)`.
- Produces: `build_router(store: SessionStore, auth_dependency: Callable, upload_dir: str, max_image_mb: float = 10.0) -> APIRouter` with routes:
  - `POST /sessions` → `201 {"session_id"}`; `503 {"error"}` at cap.
  - `DELETE /sessions/{sid}` → `204`; `404`.
  - `POST /sessions/{sid}/image` (multipart field `file`) → `{"width", "height", "url": "/sessions/{sid}/files/<name>"}`; `400` on a rejected upload.
  - `GET /sessions/{sid}/files/{name}` → the file (only if registered); `404` otherwise.
  - `GET /sessions/{sid}/state` → `{"version", "image", "interpretation", "model_summary", "section_meta"}` (this task returns `image` and `version`; the other three come in Tasks 5–6 and are `None` here).
  - Error mapping helpers `_http(status, msg)`; `_session_errors` handling `SessionNotFound → 404`, `SessionBusy → 409`, `SessionLimit → 503`, `ValueError → 400`.
  - Every session route holds `store.acquire(sid)` for the duration of the request.
  - `entry.image_path` is stored as `entry.allowed_files` entry plus a plain attribute on the bot's context: the staged path is `entry.bot.context_manager.get_context("last_image")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_outcrop_api.py
"""Session API routes, exercised on a private FastAPI app so the real
interfaces.api_interface (which builds a credentialed chatbot at import) is
never imported here."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException, Header
from fastapi.testclient import TestClient

from core.chatbot_tool_use import SeismicChatBotToolUse
from interfaces.outcrop_api import build_router
from interfaces.sessions import SessionStore
from tools import outcrop_tools as ot

INTERP = {"regions": [{"id": 1, "label": "sand", "lithology": "sandstone",
                       "geometry": {"type": "band", "y_top": 0.3, "y_bottom": 0.5}}],
          "scale": {"estimated_height_m": 20, "reference": "hammer", "confidence": "medium"},
          "background_lithology": "shale", "mode": "polygons"}


def _no_auth():
    return None


def _key_auth(x_api_key: str = Header(default=None, alias="X-API-Key")):
    if x_api_key != "sekret":
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    d = str(tmp_path / "uploads")
    os.makedirs(d)
    monkeypatch.setattr(ot, "SEISMIC_UPLOAD_DIR", d)
    return d


@pytest.fixture
def base_bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]), knowledge_base=object())


@pytest.fixture
def store(base_bot, upload_dir):
    return SessionStore(base_bot, ttl_seconds=100, max_sessions=3, upload_dir=upload_dir)


@pytest.fixture
def client(store, upload_dir):
    app = FastAPI()
    app.include_router(build_router(store, _no_auth, upload_dir, max_image_mb=10))
    return TestClient(app)


@pytest.fixture
def sid(client):
    r = client.post("/sessions")
    assert r.status_code == 201
    return r.json()["session_id"]


def _upload(client, sid, path):
    with open(path, "rb") as f:
        return client.post(f"/sessions/{sid}/image",
                           files={"file": (os.path.basename(path), f, "image/png")})


# ---- lifecycle -------------------------------------------------------------

def test_create_and_delete_session(client, store):
    sid = client.post("/sessions").json()["session_id"]
    assert len(store) == 1
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert len(store) == 0
    assert client.delete(f"/sessions/{sid}").status_code == 404
    assert client.get(f"/sessions/{sid}/state").status_code == 404


def test_session_cap_returns_503(client):
    for _ in range(3):
        assert client.post("/sessions").status_code == 201
    r = client.post("/sessions")
    assert r.status_code == 503 and "limit" in r.json()["error"]


def test_auth_dependency_applies_to_every_session_route(store, upload_dir):
    app = FastAPI()
    app.include_router(build_router(store, _key_auth, upload_dir))
    c = TestClient(app)
    assert c.post("/sessions").status_code == 401
    sid = c.post("/sessions", headers={"X-API-Key": "sekret"}).json()["session_id"]
    assert c.get(f"/sessions/{sid}/state").status_code == 401
    assert c.get(f"/sessions/{sid}/state", headers={"X-API-Key": "sekret"}).status_code == 200


# ---- upload + files --------------------------------------------------------

def test_upload_stages_attaches_and_serves(client, sid, store, outcrop_image, upload_dir):
    r = _upload(client, sid, outcrop_image)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 400 and body["height"] == 200
    assert body["url"].startswith(f"/sessions/{sid}/files/")
    staged = store.get(sid).bot.context_manager.get_context("last_image")
    assert staged.startswith(os.path.join(upload_dir, sid))
    f = client.get(body["url"])
    assert f.status_code == 200 and f.headers["content-type"] == "image/png"
    assert len(f.content) == os.path.getsize(outcrop_image)
    state = client.get(f"/sessions/{sid}/state").json()
    assert state["image"] == {"width": 400, "height": 200, "url": body["url"]}
    assert state["interpretation"] is None and state["model_summary"] is None


def test_upload_rejects_bad_file(client, sid, tmp_path):
    bad = tmp_path / "x.gif"
    bad.write_bytes(b"GIF89a")
    with open(bad, "rb") as f:
        r = client.post(f"/sessions/{sid}/image", files={"file": ("x.gif", f, "image/gif")})
    assert r.status_code == 400 and "extension" in r.json()["error"]


def test_files_route_refuses_unregistered_names(client, sid, upload_dir):
    secret = os.path.join(upload_dir, "secret.png")
    with open(secret, "wb") as f:
        f.write(b"x")
    assert client.get(f"/sessions/{sid}/files/secret.png").status_code == 404
    assert client.get(f"/sessions/{sid}/files/..%2Fsecret.png").status_code == 404


def test_state_on_fresh_session(client, sid):
    state = client.get(f"/sessions/{sid}/state").json()
    assert state == {"version": 0, "image": None, "interpretation": None,
                     "model_summary": None, "section_meta": None}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_outcrop_api.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'interfaces.outcrop_api'`.

- [ ] **Step 3: Implement the router skeleton**

```python
# interfaces/outcrop_api.py
"""Session-scoped REST API for the outcrop → seismic web client.

Every tool route runs the registry tool through ToolLoopRunner.execute_call —
the same per-call path as a chat turn — so validators, physics guards,
sandboxes, trace events and provenance apply. Files are served only when
registered on the session."""
import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from interfaces.serialize import (interpretation_caps, model_summary, section_payload,
                                  to_jsonable)
from interfaces.sessions import (SessionBusy, SessionEntry, SessionLimit,
                                 SessionNotFound, SessionStore)
from tools.image_safety import MIME_BY_EXTENSION, image_size, stage_upload

logger = logging.getLogger(__name__)

MAX_INTERPRETATION_BYTES = 1_000_000


def _http(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": message})


def _image_info(entry: SessionEntry, sid: str) -> Optional[Dict[str, Any]]:
    path = entry.bot.context_manager.get_context("last_image")
    if not path:
        return None
    w, h = image_size(path)
    name = os.path.basename(path)
    return {"width": w, "height": h, "url": f"/sessions/{sid}/files/{name}"}


def build_router(store: SessionStore, auth_dependency: Callable, upload_dir: str,
                 max_image_mb: float = 10.0) -> APIRouter:
    router = APIRouter(prefix="/sessions", dependencies=[Depends(auth_dependency)])

    @contextmanager
    def session(sid: str) -> Iterator[SessionEntry]:
        try:
            with store.acquire(sid) as entry:
                yield entry
        except SessionNotFound:
            raise _http(404, f"unknown session {sid}")
        except SessionBusy:
            raise _http(409, "session is busy with another request")

    # -- lifecycle ---------------------------------------------------------
    @router.post("", status_code=201)
    def create_session():
        try:
            entry = store.create()
        except SessionLimit as e:
            raise _http(503, str(e))
        return {"session_id": entry.bot.session_id}

    @router.delete("/{sid}", status_code=204)
    def delete_session(sid: str):
        try:
            store.delete(sid)
        except SessionNotFound:
            raise _http(404, f"unknown session {sid}")
        return Response(status_code=204)

    # -- upload + files ----------------------------------------------------
    @router.post("/{sid}/image")
    def upload_image(sid: str, file: UploadFile = File(...)):
        with session(sid) as entry:
            suffix = os.path.splitext(file.filename or "")[1].lower()
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(file.file.read())
                try:
                    staged = stage_upload(tmp, upload_dir, entry.bot.session_id, max_image_mb)
                except ValueError as e:
                    raise _http(400, str(e))
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            entry.bot.attach_image(staged)
            store.register_file(entry, staged)
            return _image_info(entry, sid)

    @router.get("/{sid}/files/{name}")
    def get_file(sid: str, name: str):
        with session(sid) as entry:
            path = entry.allowed_files.get(name)
            if path is None or os.path.basename(name) != name or not os.path.isfile(path):
                raise _http(404, "no such file on this session")
            ext = os.path.splitext(path)[1].lower()
            return FileResponse(path, media_type=MIME_BY_EXTENSION.get(ext, "image/png"))

    # -- state -------------------------------------------------------------
    @router.get("/{sid}/state")
    def get_state(sid: str):
        with session(sid) as entry:
            return _state(entry, sid)

    def _state(entry: SessionEntry, sid: str) -> Dict[str, Any]:
        return {"version": entry.version,
                "image": _image_info(entry, sid),
                "interpretation": None,
                "model_summary": None,
                "section_meta": None}

    router.state_builder = _state   # used by later routes to return state with results
    return router


def install_error_handlers(app) -> None:
    """Render HTTPException details of the form {"error": ...} as that JSON body."""
    from fastapi.exceptions import HTTPException as _HTTPException

    @app.exception_handler(_HTTPException)
    async def _handler(request: Request, exc: _HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)
```

Note for the tests: the test app must call `install_error_handlers(app)` so `{"error": ...}` bodies render. Update the `client` fixture and the auth test:

```python
from interfaces.outcrop_api import build_router, install_error_handlers
...
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(build_router(store, _no_auth, upload_dir, max_image_mb=10))
```

(and the same two lines in `test_auth_dependency_applies_to_every_session_route`).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_outcrop_api.py -q`
Expected: 7 PASS. (If `python-multipart` is missing, FastAPI raises at route definition: `pip install python-multipart`; it is added to `requirements.txt` in Task 8.)

- [ ] **Step 5: Commit**

```bash
git add interfaces/outcrop_api.py tests/test_outcrop_api.py
git commit -m "feat(api): /sessions router — lifecycle, sandboxed upload, allow-listed file serving, state"
```

---

### Task 5: Interpretation routes — `POST /interpret`, `PUT /interpretation`

**Files:**
- Modify: `interfaces/outcrop_api.py`
- Test: `tests/test_outcrop_api.py`

**Interfaces:**
- Consumes: `entry.bot._tool_loop.execute_call("interpret_outcrop", {}, images, auto_plot=False)` (context injects `image_path` from `last_image`; stores `last_outcrop`); `tools.outcrop_tools.validate_interpretation(data) -> dict`; `interpretation_caps` (Task 2).
- Produces:
  - `POST /sessions/{sid}/interpret` → `{"interpretation": <normalized>, "warnings": [str], "version": int}`; `400` when no image (`ValueError` from the tool: "Please upload an outcrop photo first."); `503` when `core.vision_client.build_vision_client` raises `RuntimeError` (no credentials).
  - `PUT /sessions/{sid}/interpretation` (JSON body = interpretation) → same shape; `413` on caps or body > 1 MB; `400` on validation errors. The normalized dict is stored as `last_outcrop` via `entry.bot.context_manager.set_context`. The stored copy keeps `image_path` = the session's `last_image` and `image_size` from it (so `outcrop_to_model` can compute `width_m`).
  - `GET /state` now returns `interpretation` = `to_jsonable(last_outcrop)` minus `image_path`.
  - Helper `_run_tool(entry, name, args) -> (result, warnings)` — wraps `execute_call(..., auto_plot=False)` and collects `physics_warning` event messages emitted during the call (compare `len(trace.events)` before/after). Maps `ValueError` → `400`, `RuntimeError` whose message mentions vision/credentials → `503`.
  - Each tool route brackets the call with `trace.begin_turn(f"api:{name}")` / `trace.end_turn()` so events persist like a chat turn, and calls `entry.bot.context_manager.begin_turn_recording(f"api:{name}")` first (keeps `current_turn_calls` a list, as `execute_call` expects).

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---- interpret + PUT interpretation ---------------------------------------

@pytest.fixture
def fake_vision(monkeypatch, fake_vision_factory):
    fake = fake_vision_factory([json.dumps(INTERP)])
    monkeypatch.setattr("core.vision_client.build_vision_client", lambda: fake)
    return fake


def test_interpret_requires_image(client, sid):
    r = client.post(f"/sessions/{sid}/interpret")
    assert r.status_code == 400 and "upload" in r.json()["error"].lower()


def test_interpret_runs_vlm_and_stores_context(client, sid, store, outcrop_image, fake_vision):
    _upload(client, sid, outcrop_image)
    r = client.post(f"/sessions/{sid}/interpret")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["interpretation"]["regions"][0]["lithology"] == "sandstone"
    assert body["interpretation"]["regions"][0]["geometry_type"] == "band"
    assert "image_path" not in body["interpretation"]
    assert body["version"] == 1 and body["warnings"] == []
    assert len(fake_vision.calls) == 1
    stored = store.get(sid).bot.context_manager.get_context("last_outcrop")
    assert stored["regions"][0]["points"] == [[0.0, 0.3], [1.0, 0.3], [1.0, 0.5], [0.0, 0.5]]
    state = client.get(f"/sessions/{sid}/state").json()
    assert state["version"] == 1 and state["interpretation"]["regions"][0]["label"] == "sand"


def test_interpret_without_vision_credentials_is_503(client, sid, outcrop_image, monkeypatch):
    def _boom():
        raise RuntimeError("no vision credentials configured")
    monkeypatch.setattr("core.vision_client.build_vision_client", _boom)
    _upload(client, sid, outcrop_image)
    r = client.post(f"/sessions/{sid}/interpret")
    assert r.status_code == 503 and "vision" in r.json()["error"]


def test_put_interpretation_round_trips_and_bumps_version(client, sid, store, outcrop_image):
    _upload(client, sid, outcrop_image)
    drawn = {"regions": [{"id": 1, "label": "channel", "lithology": "sandstone",
                          "geometry": {"type": "polygon",
                                       "points": [[0.1, 0.2], [0.6, 0.25], [0.5, 0.6]]}}],
             "scale": {"estimated_height_m": None, "reference": None, "confidence": "low"},
             "background_lithology": "shale", "mode": "polygons"}
    r = client.put(f"/sessions/{sid}/interpretation", json=drawn)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 1
    first = body["interpretation"]
    assert first["regions"][0]["geometry_type"] == "polygon"
    # idempotent: sending the normalized form back yields the same thing
    r2 = client.put(f"/sessions/{sid}/interpretation", json=first)
    assert r2.status_code == 200 and r2.json()["interpretation"] == first
    stored = store.get(sid).bot.context_manager.get_context("last_outcrop")
    assert stored["image_path"] == store.get(sid).bot.context_manager.get_context("last_image")
    assert stored["image_size"] == [400, 200]


def test_put_interpretation_validation_and_caps(client, sid, outcrop_image):
    _upload(client, sid, outcrop_image)
    bad = {"regions": [{"id": 1, "label": "x", "lithology": "sandstone",
                        "geometry": {"type": "polygon", "points": [[0, 0], [1, 1]]}}],
           "scale": {}, "background_lithology": "shale", "mode": "polygons"}
    r = client.put(f"/sessions/{sid}/interpretation", json=bad)
    assert r.status_code == 400 and "3 points" in r.json()["error"]
    many = {"regions": [{"id": i, "label": "r", "lithology": "shale",
                         "geometry": {"type": "band", "y_top": 0.1, "y_bottom": 0.2}}
                        for i in range(1, 202)],
            "scale": {}, "background_lithology": "shale", "mode": "bands"}
    r = client.put(f"/sessions/{sid}/interpretation", json=many)
    assert r.status_code == 413
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_outcrop_api.py -q -k "interpret"`
Expected: the new tests FAIL with `405`/`404` responses (routes don't exist).

- [ ] **Step 3: Implement**

Add to `interfaces/outcrop_api.py` (imports at top):

```python
from fastapi import Body
from tools.outcrop_tools import validate_interpretation
from tools.image_safety import image_size
```

Inside `build_router`, before the `# -- state` block, add the tool runner and the two routes:

```python
    # -- tool execution ----------------------------------------------------
    def _run_tool(entry: SessionEntry, name: str, args: Dict[str, Any]):
        """execute_call with the per-turn bookkeeping a chat turn does; returns
        (result, warnings). Maps tool errors to HTTP statuses."""
        cm = entry.bot.context_manager
        cm.trace.begin_turn(f"api:{name}")
        cm.begin_turn_recording(f"api:{name}")
        images: list = []
        try:
            result = entry.bot._tool_loop.execute_call(name, args, images, auto_plot=False)
        except ValueError as e:
            raise _http(400, str(e))
        except RuntimeError as e:
            msg = str(e)
            if "vision" in msg.lower() or "credential" in msg.lower():
                raise _http(503, msg)
            raise _http(500, msg)
        finally:
            events = list(cm.trace.events)
            cm.trace.end_turn()
        warnings = [e.get("message", "") for e in events if e.get("t") == "physics_warning"]
        for p in images:
            store.register_file(entry, p)
        return result, warnings

    def _interp_public(interp: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(interp, dict):
            return None
        return to_jsonable({k: v for k, v in interp.items() if k != "image_path"})

    # -- interpretation ----------------------------------------------------
    @router.post("/{sid}/interpret")
    def interpret(sid: str):
        with session(sid) as entry:
            result, warnings = _run_tool(entry, "interpret_outcrop", {})
        entry = store.get(sid)
        return {"interpretation": _interp_public(result), "warnings": warnings,
                "version": entry.version}

    @router.put("/{sid}/interpretation")
    async def put_interpretation(sid: str, request: Request):
        raw = await request.body()
        if len(raw) > MAX_INTERPRETATION_BYTES:
            raise _http(413, "interpretation body exceeds 1 MB")
        try:
            data = json.loads(raw)
        except ValueError:
            raise _http(400, "body is not valid JSON")
        try:
            interpretation_caps(data)
        except ValueError as e:
            raise _http(413, str(e))
        with session(sid) as entry:
            cm = entry.bot.context_manager
            try:
                normalized = validate_interpretation(data)
            except ValueError as e:
                raise _http(400, str(e))
            image = cm.get_context("last_image")
            if image:
                normalized["image_path"] = image
                normalized["image_size"] = list(image_size(image))
            cm.set_context("last_outcrop", normalized)
        entry = store.get(sid)
        return {"interpretation": _interp_public(normalized), "warnings": [],
                "version": entry.version}
```

Add `import json` at the top. Update `_state` to fill the interpretation:

```python
    def _state(entry: SessionEntry, sid: str) -> Dict[str, Any]:
        cm = entry.bot.context_manager
        return {"version": entry.version,
                "image": _image_info(entry, sid),
                "interpretation": _interp_public(cm.get_context("last_outcrop")),
                "model_summary": None,
                "section_meta": None}
```

Note: `version` is read **after** leaving the `session(...)` block because the store bumps it on exit; `store.get(sid)` outside the lock is a plain dict read.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_outcrop_api.py -q`
Expected: 12 PASS.

- [ ] **Step 5: Commit**

```bash
git add interfaces/outcrop_api.py tests/test_outcrop_api.py
git commit -m "feat(api): interpret route (VLM via execute_call) and PUT interpretation round-trip"
```

---

### Task 6: Model, section and plot routes (with oracle test)

**Files:**
- Modify: `interfaces/outcrop_api.py`
- Test: `tests/test_outcrop_api.py`

**Interfaces:**
- Consumes: `_run_tool` (Task 5); `model_summary`, `section_payload` (Task 2); `tools.section_tools.plot_seismic_section(section, parameters, axis=None, model=None, display="image", output_path=None) -> str`; `tools.section_tools.synthetic_section_from_model(model, ..., domain="depth")` (oracle).
- Produces:
  - `POST /sessions/{sid}/model` body `{height_m?, overrides?, background_lithology?, num_traces?, wavelet_freq?, pad_m?}` → `{"model": model_summary, "warnings", "version"}`; `400` when no interpretation ("Run interpret_outcrop first" comes from the tool as a `ValueError`).
  - `POST /sessions/{sid}/section` body `{wavelet_freq?, wv_type?, ormsby_freq?, phase_rot?, angle?, method?, dt?, pad_time?}` → `section_payload(...) + {"warnings", "version"}`. The route always forces `domain="depth"` and `display="overlay"`; a client-supplied `domain` is ignored.
  - `GET /sessions/{sid}/plot.png?display=overlay|image|wiggle|both|overlay_image` → PNG (`image/png`), rendered with `plot_seismic_section(last_section["section"], last_section["parameters"], axis=last_section["axis"], model=last_earth_model, display=display)`, file registered on the session; `400` without a section; `400` on an unknown `display`.
  - `GET /state` fills `model_summary` and `section_meta` (the section payload without `traces`).

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---- model + section + plot -------------------------------------------------

@pytest.fixture
def interpreted(client, sid, outcrop_image, fake_vision):
    _upload(client, sid, outcrop_image)
    assert client.post(f"/sessions/{sid}/interpret").status_code == 200
    return sid


def test_model_requires_interpretation(client, sid, outcrop_image):
    _upload(client, sid, outcrop_image)
    r = client.post(f"/sessions/{sid}/model", json={})
    assert r.status_code == 400


def test_model_returns_summary_without_grids(client, interpreted, store):
    sid = interpreted
    r = client.post(f"/sessions/{sid}/model", json={"num_traces": 21, "height_m": 25})
    assert r.status_code == 200, r.text
    body = r.json()
    m = body["model"]
    assert m["height_m"] == 25 and m["nx"] == 21 and m["width_m"] == 50.0
    assert "facies" not in m and "vp" not in m
    assert m["legend"]["1"]["lithology"] == "sandstone"
    assert body["version"] == 2
    assert store.get(sid).bot.context_manager.get_context("last_earth_model")["nx"] == 21


def test_section_matches_direct_tool_call(client, interpreted, store):
    from tools.section_tools import synthetic_section_from_model
    sid = interpreted
    client.post(f"/sessions/{sid}/model", json={"num_traces": 11, "height_m": 20})
    r = client.post(f"/sessions/{sid}/section", json={"wavelet_freq": 40, "domain": "time"})
    assert r.status_code == 200, r.text
    body = r.json()
    model = store.get(sid).bot.context_manager.get_context("last_earth_model")
    z, sec, params = synthetic_section_from_model(model, wavelet_freq=40, domain="depth")
    assert body["domain"] == "depth"                      # client 'time' ignored
    assert len(body["traces"]) == 11 and len(body["traces"][0]) == len(z)
    np.testing.assert_allclose(np.array(body["traces"]).T, sec, rtol=2e-3, atol=1e-6 * params["max_abs_amplitude"] + 1e-12)
    np.testing.assert_allclose(body["z"], z, rtol=2e-3)
    assert body["image_top_m"] == model["image_top_m"] and body["height_m"] == 20
    assert body["max_abs_amplitude"] == pytest.approx(params["max_abs_amplitude"], rel=2e-3)
    assert body["version"] == 3
    state = client.get(f"/sessions/{sid}/state").json()
    assert state["model_summary"]["nx"] == 11
    assert "traces" not in state["section_meta"] and state["section_meta"]["nx"] == 11


def test_section_requires_model(client, interpreted):
    r = client.post(f"/sessions/{interpreted}/section", json={})
    assert r.status_code == 400


def test_plot_png_each_display(client, interpreted, store):
    sid = interpreted
    client.post(f"/sessions/{sid}/model", json={"num_traces": 11, "height_m": 20})
    client.post(f"/sessions/{sid}/section", json={})
    for display in ("overlay", "image", "wiggle", "both", "overlay_image"):
        r = client.get(f"/sessions/{sid}/plot.png", params={"display": display})
        assert r.status_code == 200, (display, r.text)
        assert r.headers["content-type"] == "image/png" and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/sessions/{sid}/plot.png", params={"display": "nope"}).status_code == 400
    plots = list(store.get(sid).plot_files)
    assert plots                                        # registered for cleanup
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert not any(os.path.exists(p) for p in plots)   # swept with the session


def test_plot_requires_section(client, interpreted):
    assert client.get(f"/sessions/{interpreted}/plot.png").status_code == 400
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_outcrop_api.py -q -k "model or section or plot"`
Expected: FAIL with 404/405 responses.

- [ ] **Step 3: Implement**

Add imports:

```python
from tools.section_tools import plot_seismic_section
from fastapi import Query
```

Add the routes inside `build_router` after the interpretation routes:

```python
    _MODEL_KEYS = ("height_m", "overrides", "background_lithology", "num_traces",
                   "wavelet_freq", "pad_m")
    _SECTION_KEYS = ("wavelet_freq", "wv_type", "ormsby_freq", "phase_rot", "angle",
                     "method", "dt", "pad_time")
    _DISPLAYS = ("overlay", "image", "wiggle", "both", "overlay_image")

    @router.post("/{sid}/model")
    def build_model(sid: str, body: Dict[str, Any] = Body(default={})):
        args = {k: body[k] for k in _MODEL_KEYS if k in body and body[k] is not None}
        with session(sid) as entry:
            result, warnings = _run_tool(entry, "outcrop_to_model", args)
        entry = store.get(sid)
        return {"model": model_summary(result), "warnings": warnings, "version": entry.version}

    @router.post("/{sid}/section")
    def build_section(sid: str, body: Dict[str, Any] = Body(default={})):
        args = {k: body[k] for k in _SECTION_KEYS if k in body and body[k] is not None}
        args["domain"] = "depth"
        args["display"] = "overlay"
        with session(sid) as entry:
            _, warnings = _run_tool(entry, "synthetic_section", args)
            cm = entry.bot.context_manager
            payload = section_payload(cm.get_context("last_section"),
                                      cm.get_context("last_earth_model"))
        entry = store.get(sid)
        payload.update({"warnings": warnings, "version": entry.version})
        return payload

    @router.get("/{sid}/plot.png")
    def plot_png(sid: str, display: str = Query(default="overlay")):
        if display not in _DISPLAYS:
            raise _http(400, f"display must be one of {list(_DISPLAYS)}")
        with session(sid) as entry:
            cm = entry.bot.context_manager
            last = cm.get_context("last_section")
            if not last:
                raise _http(400, "Build a section first (POST /section).")
            try:
                path = plot_seismic_section(last["section"], last["parameters"],
                                            axis=last.get("axis"),
                                            model=cm.get_context("last_earth_model"),
                                            display=display)
            except ValueError as e:
                raise _http(400, str(e))
            store.register_file(entry, path)
            return FileResponse(path, media_type="image/png")
```

Update `_state`:

```python
    def _state(entry: SessionEntry, sid: str) -> Dict[str, Any]:
        cm = entry.bot.context_manager
        model = cm.get_context("last_earth_model")
        last = cm.get_context("last_section")
        meta = None
        if last:
            meta = section_payload(last, model)
            meta.pop("traces", None)
        return {"version": entry.version,
                "image": _image_info(entry, sid),
                "interpretation": _interp_public(cm.get_context("last_outcrop")),
                "model_summary": model_summary(model) if isinstance(model, dict) else None,
                "section_meta": meta}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_outcrop_api.py -q`
Expected: 18 PASS. If the oracle comparison fails only on rounding, the 4-significant-digit rounding is the cause; keep `rtol=2e-3` (4 sig. digits ⇒ ≤ 5e-4 relative) and do not loosen further.

- [ ] **Step 5: Commit**

```bash
git add interfaces/outcrop_api.py tests/test_outcrop_api.py
git commit -m "feat(api): model/section/plot routes with depth-domain section payload (oracle-tested)"
```

---

### Task 7: Chat on the session

**Files:**
- Modify: `interfaces/outcrop_api.py`
- Test: `tests/test_outcrop_api.py`

**Interfaces:**
- Consumes: `entry.bot.process_single_input(message) -> {"reply", "images", "trace"}`; `store.register_file`.
- Produces: `POST /sessions/{sid}/chat` body `{"message": str}` → `{"reply": str, "images": ["/sessions/{sid}/files/<name>", ...], "trace": dict|None, "version": int}`; `400` on an empty message. A `ValueError` from the turn is already converted to a reply by the chatbot; the route never raises for tool errors.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---- chat on the same session ---------------------------------------------

class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


class _ScriptedLLM:
    """No get_simple_completion → keyword fallback routes 'make/build ...' to tools."""
    def __init__(self, responses):
        self._responses = list(responses)

    def get_completion(self, *a, **k):
        return self._responses.pop(0)


def test_chat_edits_shared_context_and_bumps_version(store, upload_dir, outcrop_image, fake_vision):
    from core.tool_manager import ToolManager
    llm = _ScriptedLLM([
        {"content": "", "usage": None,
         "tool_calls": [_FakeToolCall("outcrop_to_model",
                                      '{"overrides": {"sand": {"fluid": "gas"}}, "num_traces": 11}')]},
        {"content": "<reply>sand is now gas-filled</reply>", "tool_calls": None, "usage": None},
    ])
    base = SeismicChatBotToolUse(llm_client=llm, tool_manager=ToolManager(), knowledge_base=object())
    store = SessionStore(base, ttl_seconds=100, max_sessions=3, upload_dir=upload_dir)
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(build_router(store, _no_auth, upload_dir))
    c = TestClient(app)
    sid = c.post("/sessions").json()["session_id"]
    _upload(c, sid, outcrop_image)
    c.post(f"/sessions/{sid}/interpret")
    c.post(f"/sessions/{sid}/model", json={"num_traces": 11, "height_m": 20})
    before = store.get(sid).bot.context_manager.get_context("last_earth_model")
    v_before = c.get(f"/sessions/{sid}/state").json()["version"]

    r = c.post(f"/sessions/{sid}/chat", json={"message": "make the sand gas-bearing"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == "sand is now gas-filled"
    assert body["version"] == v_before + 1
    assert body["trace"]["tools_used"] == ["outcrop_to_model"]
    after = store.get(sid).bot.context_manager.get_context("last_earth_model")
    assert after is not before
    # gas lowers Vp inside the sandstone region (facies id 1) relative to the water case
    mask = after["facies"] == 1
    assert mask.any()
    assert after["vp"][mask].mean() < before["vp"][mask].mean()
    state = c.get(f"/sessions/{sid}/state").json()
    assert state["version"] == body["version"] and state["model_summary"]["nx"] == 11


def test_chat_images_are_served_via_files_route(store, upload_dir, monkeypatch):
    from core.tool_manager import ToolManager
    llm = _ScriptedLLM([
        {"content": "", "usage": None,
         "tool_calls": [_FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>here</reply>", "tool_calls": None, "usage": None},
    ])
    base = SeismicChatBotToolUse(llm_client=llm, tool_manager=ToolManager(), knowledge_base=object())
    store = SessionStore(base, ttl_seconds=100, max_sessions=3, upload_dir=upload_dir)
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(build_router(store, _no_auth, upload_dir))
    c = TestClient(app)
    sid = c.post("/sessions").json()["session_id"]
    r = c.post(f"/sessions/{sid}/chat", json={"message": "make a 30 Hz ricker wavelet"})
    body = r.json()
    assert len(body["images"]) == 1 and body["images"][0].startswith(f"/sessions/{sid}/files/")
    img = c.get(body["images"][0])
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    assert body["version"] == 0                     # wavelet is not an outcrop key
    c.delete(f"/sessions/{sid}")


def test_chat_rejects_empty_message(client, sid):
    assert client.post(f"/sessions/{sid}/chat", json={"message": "  "}).status_code == 400
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_outcrop_api.py -q -k chat`
Expected: FAIL with 404/405.

- [ ] **Step 3: Implement** (inside `build_router`, after the plot route)

```python
    @router.post("/{sid}/chat")
    def chat(sid: str, body: Dict[str, Any] = Body(...)):
        message = str(body.get("message") or "").strip()
        if not message:
            raise _http(400, "message must not be empty")
        with session(sid) as entry:
            result = entry.bot.process_single_input(message)
            if isinstance(result, dict) and "reply" in result:
                reply = str(result["reply"])
                paths = [str(p) for p in result.get("images") or []]
                trace = result.get("trace")
            else:
                reply, paths, trace = str(result), [], None
            urls = [f"/sessions/{sid}/files/{store.register_file(entry, p)}" for p in paths]
        entry = store.get(sid)
        return {"reply": reply, "images": urls, "trace": trace, "version": entry.version}
```

- [ ] **Step 4: Run the file**

Run: `pytest tests/test_outcrop_api.py -q`
Expected: 21 PASS.

- [ ] **Step 5: Commit**

```bash
git add interfaces/outcrop_api.py tests/test_outcrop_api.py
git commit -m "feat(api): session-scoped chat route sharing the outcrop context"
```

---

### Task 8: Mount into the app, dependencies, docs

**Files:**
- Modify: `interfaces/api_interface.py` (after `_chat_rate_limiter` definition and after the `/examples/*` routes)
- Modify: `requirements.txt`
- Modify: `CLAUDE.md` (Environment variables table + a new section after "Outcrop photo → seismic section")
- Test: `tests/test_api_mount.py`

**Interfaces:**
- Consumes: `build_router`, `install_error_handlers` (Task 4), `SessionStore` (Task 3), `enforce_chat_policy` (existing), `config.settings.SEISMIC_UPLOAD_DIR`, `MAX_IMAGE_MB`.
- Produces: module attributes `session_store: SessionStore`, `WEBAPP_DIST = <package dir>/webapp/dist`; the router mounted on `app`; `StaticFiles(directory=WEBAPP_DIST, html=True)` at `/app` when that directory exists at import time.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_mount.py
"""The /sessions router and the /app static mount are wired into the real app.
Stubs the chatbot before reloading interfaces.api_interface (no credentials)."""
import importlib
import os

import pytest

pytest.importorskip("fastapi")


class _Ctx:
    def __init__(self):
        self.d = {}

    def get_context(self, k, default=None):
        return self.d.get(k, default)

    def set_context(self, k, v):
        self.d[k] = v


class _StubSession:
    def __init__(self):
        import uuid
        self.session_id = uuid.uuid4().hex
        self.context_manager = _Ctx()

    def process_single_input(self, message):
        return {"reply": "ok", "images": []}


class _StubBot:
    def __init__(self, *a, **k):
        pass

    def new_session(self):
        return _StubSession()


@pytest.fixture
def api(monkeypatch, tmp_path):
    import core.chatbot_tool_use as bot_module
    monkeypatch.setattr(bot_module, "SeismicChatBotToolUse", _StubBot)
    monkeypatch.setenv("SESSION_TTL_SECONDS", "123")
    monkeypatch.setenv("MAX_SESSIONS", "7")
    import interfaces.api_interface as api_module
    api_module = importlib.reload(api_module)
    monkeypatch.setattr(api_module, "API_AUTH_KEY", "sekret")
    return api_module


def test_sessions_router_is_mounted_behind_the_key(api):
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    assert c.post("/sessions").status_code == 401       # enforce_chat_policy reads the patched module key
    r = c.post("/sessions", headers={"X-API-Key": "sekret"})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert c.get(f"/sessions/{sid}/state", headers={"X-API-Key": "sekret"}).json()["version"] == 0


def test_store_reads_env(api):
    assert api.session_store._ttl == 123.0 and api.session_store._max == 7


def test_legacy_chat_unchanged(api):
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    r = c.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "sekret"})
    assert r.status_code == 200 and r.json()["response"] == "ok"


def test_app_mount_only_when_dist_exists(api):
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    mounted = os.path.isdir(api.WEBAPP_DIST)
    r = c.get("/app/")
    assert (r.status_code == 200) if mounted else (r.status_code == 404)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_mount.py -q`
Expected: FAIL — `/sessions` returns `404`, `session_store` attribute missing.

- [ ] **Step 3: Implement**

In `interfaces/api_interface.py`, after the `enforce_chat_policy` function:

```python
# --- Session API for the outcrop web client ---------------------------------
from config.settings import SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB
from interfaces.outcrop_api import build_router, install_error_handlers
from interfaces.sessions import SessionStore

session_store = SessionStore(
    base_chatbot,
    ttl_seconds=float(os.environ.get("SESSION_TTL_SECONDS", "7200")),
    max_sessions=int(os.environ.get("MAX_SESSIONS", "50")),
    upload_dir=SEISMIC_UPLOAD_DIR,
)
install_error_handlers(app)
app.include_router(build_router(session_store, enforce_chat_policy,
                                SEISMIC_UPLOAD_DIR, MAX_IMAGE_MB))

# Static client bundle (webapp/dist) — mounted only when it has been built.
WEBAPP_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "webapp", "dist")
if os.path.isdir(WEBAPP_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory=WEBAPP_DIST, html=True), name="webapp")
```

`install_error_handlers` registers a handler for `HTTPException`; the legacy `/chat` route never raises `HTTPException` with a non-dict detail except from `enforce_chat_policy` (`503/401/429` string details), which the handler wraps as `{"error": "..."}` — `tests/test_api_chat_contract.py` only asserts on `200` responses, so it is unaffected. Check `tests/test_security.py` for any assertion on the `detail` key of those error bodies; if there is one, keep the handler from touching string details:

```python
        if not isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
```

`requirements.txt` — append under "Additional utilities":

```
# HTTP API (interfaces/api_interface.py, interfaces/outcrop_api.py)
fastapi>=0.110.0
uvicorn>=0.27.0
python-multipart>=0.0.9
```

`CLAUDE.md` — add rows to the "Security containment" env table:

```
| `SESSION_TTL_SECONDS` | `7200` | Idle lifetime of a `/sessions/{id}` web-client session before its files are swept (`interfaces/sessions.py`). |
| `MAX_SESSIONS` | `50` | Cap on live web-client sessions; `POST /sessions` returns `503` beyond it. |
```

and a new section after "Outcrop photo → seismic section":

```markdown
## Outcrop web app API (session-scoped)

Spec: `docs/superpowers/specs/2026-09-01-outcrop-webapp-design.md`. `interfaces/outcrop_api.py`
mounts a `/sessions` router into the FastAPI app for the browser/iPad client:
`POST /sessions` → id; `POST /sessions/{id}/image` (multipart, `image_safety` sandbox);
`POST .../interpret` (the one VLM hop); `PUT .../interpretation` (client-drawn regions →
`validate_interpretation` → `last_outcrop`; caps 200 regions / 2000 points / 1 MB → 413);
`POST .../model`, `POST .../section` (always depth domain; returns `z` + `traces` columns +
photo extent for the client overlay); `GET .../plot.png?display=`; `POST .../chat` (same
session context); `GET .../state` (`version` bumps whenever `last_outcrop` /
`last_earth_model` / `last_section` identity changes); `GET .../files/{name}` serves only
files registered on that session. Every tool route runs through
`ToolLoopRunner.execute_call(..., auto_plot=False)`. `interfaces/sessions.py::SessionStore`
holds one `SeismicChatBotToolUse` session per id with a per-request lock (`409` when busy),
idle TTL and cap. All routes use the `/chat` key gate. The client bundle (`webapp/dist`,
built with `npm run build`) is served at `/app` when present. Errors are `{"error": msg}`
with 400 (tool/validator), 404 (session), 409 (busy), 413 (caps), 503 (no vision creds /
session cap). Tests: `tests/test_sessions.py`, `test_serialize.py`, `test_outcrop_api.py`,
`test_api_mount.py`.
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all PASS (including `tests/test_api_chat_contract.py`, `tests/test_security.py`).

- [ ] **Step 5: Smoke the server manually (no credentials needed for lifecycle)**

Run: `API_AUTH_KEY=dev uvicorn interfaces.api_interface:app --port 8000` in one shell (requires LLM credentials in `.env` because the real chatbot is built at import), then:

```bash
curl -s -X POST -H 'X-API-Key: dev' localhost:8000/sessions
```
Expected: `{"session_id":"<32 hex>"}`. Stop the server.

- [ ] **Step 6: Commit**

```bash
git add interfaces/api_interface.py requirements.txt CLAUDE.md tests/test_api_mount.py
git commit -m "feat(api): mount /sessions router and /app static bundle; deps and docs"
```

---

## Self-review against the spec

- **Architecture / SessionStore** (TTL, cap, lock → 409, allowed_files, version by identity, cleanup) → Task 3; mounted → Task 8.
- **`execute_call(auto_plot=False)`** → Task 1.
- **`interfaces/serialize.py`** → Task 2.
- **API surface:** `POST/DELETE /sessions`, `POST /image`, `GET /files/{name}`, `GET /state` → Task 4; `POST /interpret`, `PUT /interpretation` (+ caps 200/2000/1 MB → 413) → Task 5; `POST /model`, `POST /section` (depth-forced, `z`/`traces`/extent), `GET /plot.png` → Task 6; `POST /chat` → Task 7. Legacy `/chat` untouched (Task 8 test).
- **Error mapping** 400/404/409/413/503 → Tasks 4–6; warnings returned on interpret/model/section → `_run_tool` (Task 5).
- **Security:** key gate on every route (Task 4 test + Task 8), per-session allow-list (Task 4), upload sandbox reuse (Task 4), same-origin static mount (Task 8), `uuid4` ids (existing `session_id`).
- **Testing list in the spec:** lifecycle/TTL/cap/busy (Tasks 3–4), upload (4), interpret + 503 (5), PUT round-trip/validation/caps (5), model/section shapes + oracle (6), plot displays (6), chat shares context + version + files (7), file route refusal (4), auto-plot opt-out (1), legacy contract (8).
- **Not in this plan (by design):** the `webapp/` client — phases 2–4 get their own plan once these routes exist. Note for that plan: `.gitignore` excludes `*.json`, so `package.json`/`tsconfig.json` will need explicit `!webapp/*.json` entries.
- **Type consistency check:** `build_router(store, auth_dependency, upload_dir, max_image_mb)` is called identically in Tasks 4, 7, 8; `store.register_file(entry, path) -> basename` used in Tasks 4–7; `_run_tool(entry, name, args) -> (result, warnings)` used in Tasks 5–6; `section_payload(last_section, model)` used in Task 6 and `_state`; `entry.version` read after the `session()` block everywhere.
