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
from interfaces.outcrop_api import build_router, install_error_handlers
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
    install_error_handlers(app)
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
    install_error_handlers(app)
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


def test_upload_over_cap_rejected_with_413(store, upload_dir, outcrop_image):
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(build_router(store, _no_auth, upload_dir, max_image_mb=0.001))
    c = TestClient(app)
    sid = c.post("/sessions").json()["session_id"]
    r = _upload(c, sid, outcrop_image)
    assert r.status_code == 413
    assert "exceeds" in r.json()["error"]


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


def test_put_interpretation_ignores_client_supplied_image_path(client, sid, store):
    # No image uploaded on this session: a client-supplied image_path/image_size
    # must never be trusted through to storage or the response.
    drawn = {"regions": [{"id": 1, "label": "channel", "lithology": "sandstone",
                          "geometry": {"type": "polygon",
                                       "points": [[0.1, 0.2], [0.6, 0.25], [0.5, 0.6]]}}],
             "scale": {"estimated_height_m": None, "reference": None, "confidence": "low"},
             "background_lithology": "shale", "mode": "polygons",
             "image_path": "/etc/passwd", "image_size": [1, 1]}
    r = client.put(f"/sessions/{sid}/interpretation", json=drawn)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "image_path" not in body["interpretation"]
    stored = store.get(sid).bot.context_manager.get_context("last_outcrop")
    assert stored.get("image_path") is None
    assert stored.get("image_size") is None


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
                                      '{"overrides": {"sand": {"fluid": "gas", "porosity": 0.25}}, "num_traces": 11}')]},
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
    # override sets gas AND porosity 0.25: default sandstone porosity (0.20) sits near
    # the brine/gas Vp crossover, so fluid alone would be indeterminate; at phit=0.25
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
