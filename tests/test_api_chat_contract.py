"""Contract test for POST /chat: the JSON body carries the narrated reply plus
plot paths. Importing interfaces.api_interface builds the real chatbot (heavy,
needs LLM credentials), so the fixture stubs SeismicChatBotToolUse before a
module reload — hermetic, no network, no credentials."""
import importlib

import pytest

pytest.importorskip("fastapi")


class _StubSession:
    def process_single_input(self, message):
        return {"reply": "Tuning is 12.5 m.", "images": ["/tmp/a.png", "/tmp/b.png"]}


class _StubBot:
    def __init__(self, *a, **k):
        pass

    def new_session(self):
        return _StubSession()


@pytest.fixture
def api(monkeypatch):
    import core.chatbot_tool_use as bot_module
    monkeypatch.setattr(bot_module, "SeismicChatBotToolUse", _StubBot)
    import interfaces.api_interface as api_module
    api_module = importlib.reload(api_module)
    monkeypatch.setattr(api_module, "API_AUTH_KEY", "sekret")
    return api_module


def test_chat_response_includes_reply_and_images(api):
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.post("/chat", json={"message": "run tuning"},
                    headers={"X-API-Key": "sekret"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["response"] == "Tuning is 12.5 m."
    assert body["images"] == ["/tmp/a.png", "/tmp/b.png"]


def test_chat_plain_string_response_has_empty_images(api, monkeypatch):
    from fastapi.testclient import TestClient

    class _LegacySession:
        def process_single_input(self, message):
            return "plain text"

    class _LegacyBot:
        def new_session(self):
            return _LegacySession()

    monkeypatch.setattr(api, "base_chatbot", _LegacyBot())
    client = TestClient(api.app)
    r = client.post("/chat", json={"message": "hi"},
                    headers={"X-API-Key": "sekret"})
    body = r.json()
    assert body["response"] == "plain text"
    assert body["images"] == []
