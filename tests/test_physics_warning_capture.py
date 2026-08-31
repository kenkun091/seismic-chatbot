import warnings

from core.context_manager import ContextManager
from core.tool_loop import ToolLoopRunner


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


class WarningToolManager:
    """Stub manager whose tool issues a physics-style warning."""
    specs = {}

    def process_tool_call(self, name, tool_input):
        warnings.warn("vp 9000.0 m/s outside typical range 300-8000")
        warnings.warn("possible aliasing above Nyquist")
        return {"value": 1}


def test_tool_warnings_become_physics_warning_events():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("test")
    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("warn_tool", '{}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]), WarningToolManager(), cm)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an escaped warning would fail the test
        out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["reply"] == "done"
    physics = [e for e in cm.trace.events if e["t"] == "physics_warning"]
    assert len(physics) == 2
    assert physics[0]["tool"] == "warn_tool"
    assert physics[0]["category"] == "UserWarning"
    assert "9000.0" in physics[0]["message"]
    assert "aliasing" in physics[1]["message"]


def test_message_truncated_to_300_chars():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("test")

    class LongWarnManager:
        specs = {}

        def process_tool_call(self, name, tool_input):
            warnings.warn("x" * 500)
            return {}

    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("t", '{}')], "usage": None},
        {"content": "<reply>ok</reply>", "tool_calls": None, "usage": None},
    ]), LongWarnManager(), cm)
    runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    physics = [e for e in cm.trace.events if e["t"] == "physics_warning"]
    assert len(physics[0]["message"]) == 300
