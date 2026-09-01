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
