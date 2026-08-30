from core.context_manager import ContextManager  # noqa: F401  (parity import)
from core.orchestrator import SeismicOrchestrator
from core.tool_index import ToolCard
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
    # no get_simple_completion → intent classification keyword-falls-back (existing contract)


class FakeIndex:
    def search(self, q, top_k=5, threshold=0.2):
        return [ToolCard(name="make_ricker", card="make_ricker: Ricker wavelet",
                         required=("frequency",), score=0.9)]


def _orchestrator(responses):
    orch = SeismicOrchestrator(llm_client=FakeLLM(responses),
                               tool_manager=ToolManager(),
                               knowledge_base=object(),
                               tool_index=FakeIndex())
    orch.context_manager.trace.persist_dir = ""
    return orch


def test_agentic_turn_returns_full_trace():
    responses = [
        {"content": "", "tool_calls": [
            FakeToolCall("discover_tools", '{"task_description": "make a ricker"}')]},
        {"content": "", "tool_calls": [
            FakeToolCall("run_task",
                         '{"brief": "make a 30 Hz ricker", "tool_names": ["make_ricker"]}')]},
        # executor's inner loop:
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>made it</reply>", "tool_calls": None},
        # orchestrator's final answer:
        {"content": "<reply>All done</reply>", "tool_calls": None},
    ]
    orch = _orchestrator(responses)
    out = orch.process_single_input("make a 30 Hz ricker wavelet")
    assert out["reply"] == "All done"
    trace = out["trace"]
    assert trace["session"] == orch.session_id
    kinds = [e["t"] for e in trace["events"]]
    for expected in ("turn_start", "intent", "discover", "tool_call", "run_task"):
        assert expected in kinds, f"missing {expected} in {kinds}"
    discover = [e for e in trace["events"] if e["t"] == "discover"][0]
    assert discover["hits"] == [["make_ricker", 0.9]]
    run_task = [e for e in trace["events"] if e["t"] == "run_task"][0]
    assert run_task["tools_used"] == ["make_ricker"]
    assert run_task["error"] is None
    assert trace["tools_used"] == ["make_ricker"]


def test_agentic_error_turn_still_returns_trace():
    orch = _orchestrator([])  # first meta-loop completion raises IndexError
    out = orch.process_single_input("make a 30 Hz ricker wavelet")
    assert "error" in out["reply"].lower()
    assert out["trace"]["turn"] == 1
    assert any(e["t"] == "turn_error" for e in out["trace"]["events"])
