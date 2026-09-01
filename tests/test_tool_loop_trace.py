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


def _runner(responses, max_rounds=5):
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("test")
    return ToolLoopRunner(FakeLLM(responses), ToolManager(), cm,
                          max_tool_rounds=max_rounds), cm


def test_successful_tool_call_and_auto_plot_are_traced():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses)
    out = runner.run("sys", [{"role": "user", "content": "30 Hz ricker"}], tools=[])
    assert set(out) == {"reply", "images", "tools_used"}  # contract unchanged
    events = cm.trace.events
    tool_evts = [e for e in events if e["t"] == "tool_call"]
    assert tool_evts[0]["tool"] == "make_ricker" and tool_evts[0]["ok"] is True
    assert isinstance(tool_evts[0]["defaults_filled"], list)
    assert tool_evts[0]["injected"] == [] and tool_evts[0]["overridden"] == []
    auto = [e for e in events if e["t"] == "auto_plot"][0]
    assert auto == {**auto, "compute": "make_ricker", "plot": "plot_ricker", "fired": True}
    llm_evts = [e for e in events if e["t"] == "llm"]
    assert len(llm_evts) == 2 and llm_evts[0]["tool_call"] is True
    assert llm_evts[0]["total_tokens"] == 7
    assert cm.get_token_usage()["total_tokens"] == 7


def test_failed_tool_call_is_traced_with_error():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("no_such_tool", '{}')], "usage": None},
        {"content": "<reply>sorry</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["tools_used"] == []
    evt = [e for e in cm.trace.events if e["t"] == "tool_call"][0]
    assert evt["ok"] is False and "Unknown tool" in evt["error"]


def test_parallel_tool_calls_dropped_event():
    responses = [
        {"content": "", "tool_calls": [
            FakeToolCall("make_ricker", '{"frequency": 30}', call_id="c1"),
            FakeToolCall("wedge_model", '{}', call_id="c2")], "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses)
    runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    dropped = [e for e in cm.trace.events if e["t"] == "parallel_calls_dropped"][0]
    assert dropped["dropped"] == ["wedge_model"]


def test_budget_exhaustion_is_traced():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>forced</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses, max_rounds=1)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["reply"] == "forced"
    budget = [e for e in cm.trace.events if e["t"] == "budget_exhausted"]
    assert budget and budget[0]["rounds"] == 1
    assert budget[0]["scope"] == "tool_loop"
