"""Mode wiring: --mode agentic builds the orchestrator; injection bypasses the default bot."""
import pytest
import interfaces.gradio_interface as gi


class DummyBot:
    session_id = "dummy"

    def new_session(self):
        return self

    def process_single_input(self, text):
        return {"reply": "ok", "images": []}


def test_create_chat_interface_uses_injected_bot(monkeypatch):
    def boom():
        raise AssertionError("default bot must not be constructed when base_bot is given")
    monkeypatch.setattr(gi, "SeismicChatBotToolUse", boom)
    demo = gi.create_chat_interface(base_bot=DummyBot())
    assert demo is not None


def test_build_interface_agentic_uses_orchestrator(monkeypatch):
    import main
    built = {}

    def fake_orchestrator():
        built["orchestrator"] = True
        return DummyBot()

    def fake_create(base_bot=None):
        built["base_bot"] = base_bot
        return "demo"

    monkeypatch.setattr(main, "create_chat_interface", fake_create)
    monkeypatch.setattr("core.orchestrator.SeismicOrchestrator", fake_orchestrator)
    demo = main.build_interface("agentic")
    assert demo == "demo"
    assert built["orchestrator"] is True
    assert isinstance(built["base_bot"], DummyBot)


def test_build_interface_default_mode_passes_no_bot(monkeypatch):
    import main
    seen = {}

    def fake_create(base_bot=None):
        seen["base_bot"] = base_bot
        return "demo"

    monkeypatch.setattr(main, "create_chat_interface", fake_create)
    assert main.build_interface("tool-use") == "demo"
    assert seen["base_bot"] is None
