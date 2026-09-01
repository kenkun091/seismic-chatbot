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
