import numpy as np
from core.chatbot_tool_use import SeismicChatBotToolUse


class _FakeToolManager:
    def __init__(self):
        self.calls = []

    def process_tool_call(self, name, params):
        self.calls.append((name, params))
        return "/tmp/fake_plot.png"


class _FakeContext:
    def __init__(self, store):
        self.store = store

    def get_context(self, key):
        return self.store.get(key)


def _bare_bot():
    # Build an instance WITHOUT running __init__ (avoids loading RAG/LLM deps).
    return object.__new__(SeismicChatBotToolUse)


def test_chaining_make_ricker_calls_plot_ricker():
    bot = _bare_bot()
    bot.tool_manager = _FakeToolManager()
    bot.context_manager = _FakeContext({
        "last_ricker_wavelet": {"wavelet": [0.0, 1.0, 0.0], "time_array": [-1.0, 0.0, 1.0]}
    })
    out = bot._handle_automatic_chaining("make_ricker", {"frequency": 30}, (None, None))
    assert out == {"image_path": "/tmp/fake_plot.png"}
    assert bot.tool_manager.calls[0][0] == "plot_ricker"


def test_chaining_avo_calls_plot_avo():
    bot = _bare_bot()
    bot.tool_manager = _FakeToolManager()
    bot.context_manager = _FakeContext({})
    rc = np.array([0.1, 0.12, 0.15])
    out = bot._handle_automatic_chaining("zoeppritz_reflectivity", {"angles": [0, 10, 20]}, rc)
    assert out == {"image_path": "/tmp/fake_plot.png"}
    assert bot.tool_manager.calls[0][0] == "plot_avo_reflectivity"


def test_no_chaining_for_plain_tool():
    bot = _bare_bot()
    bot.tool_manager = _FakeToolManager()
    bot.context_manager = _FakeContext({})
    out = bot._handle_automatic_chaining("calculate_rock_properties", {}, (1, 2))
    assert out is None
