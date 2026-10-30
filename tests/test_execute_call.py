from core.context_manager import ContextManager
from core.tool_loop import ToolLoopRunner
from core.tool_manager import ToolManager


class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = FakeFunc(name, arguments)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def get_completion(self, *a, **k):
        return self._responses.pop(0)


def _cm():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    return cm


def test_execute_call_runs_tool_with_full_surroundings():
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("make a 30 Hz ricker")
    runner = ToolLoopRunner(None, ToolManager(), cm)
    images = []
    result = runner.execute_call("make_ricker", {"frequency": 30}, images)
    assert isinstance(result, tuple) and len(result) == 2
    assert images and images[0].endswith(".png")  # auto-plot harvested
    kinds = [e["t"] for e in cm.trace.events]
    assert "tool_call" in kinds and "auto_plot" in kinds
    calls = cm.get_context("current_turn_calls")
    assert calls == [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]


def test_execute_call_raises_on_failure_and_records_nothing():
    import pytest
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    runner = ToolLoopRunner(None, ToolManager(), cm)
    with pytest.raises(ValueError):
        runner.execute_call("no_such_tool", {}, [])
    assert cm.get_context("current_turn_calls") == []


def test_begin_turn_recording_rotates():
    cm = _cm()
    cm.begin_turn_recording("first")
    cm.get_context("current_turn_calls").append({"tool": "a", "args": {}, "ok": True})
    cm.begin_turn_recording("second")
    assert cm.get_context("last_turn_calls") == [{"tool": "a", "args": {}, "ok": True}]
    assert cm.get_context("last_turn_input") == "first"
    assert cm.get_context("current_turn_calls") == []
    assert cm.get_context("current_turn_input") == "second"


def test_run_loop_still_returns_contract_and_records_calls():
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 25}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]), ToolManager(), cm)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert set(out) == {"reply", "images", "tools_used"}
    assert out["tools_used"] == ["make_ricker"]
    assert cm.get_context("current_turn_calls")[0]["args"] == {"frequency": 25}


def test_process_single_input_rotates_recording():
    from core.chatbot_tool_use import SeismicChatBotToolUse
    bot = SeismicChatBotToolUse(llm_client=FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>ok</reply>", "tool_calls": None, "usage": None},
        {"content": "<reply>second</reply>", "tool_calls": None, "usage": None},
    ]), tool_manager=ToolManager(), knowledge_base=object())
    bot.context_manager.trace.persist_dir = ""
    bot.process_single_input("make a 30 Hz ricker wavelet")
    assert bot.context_manager.get_context("current_turn_calls")[0]["tool"] == "make_ricker"
    bot.process_single_input("make another wavelet please")
    assert bot.context_manager.get_context("last_turn_calls")[0]["tool"] == "make_ricker"
    assert bot.context_manager.get_context("last_turn_input") == "make a 30 Hz ricker wavelet"


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
