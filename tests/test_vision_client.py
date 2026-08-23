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
