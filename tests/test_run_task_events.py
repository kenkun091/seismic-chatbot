from core.context_manager import ContextManager  # noqa: F401
from core.orchestrator import SeismicOrchestrator
from core.tool_manager import ToolManager


def _orchestrator():
    orch = SeismicOrchestrator(llm_client=object(), tool_manager=ToolManager(),
                               knowledge_base=object(), tool_index=object())
    orch.context_manager.trace.persist_dir = ""
    orch.context_manager.trace.begin_turn("test")
    return orch


def test_empty_tool_names_emits_run_task_error():
    orch = _orchestrator()
    out = orch._run_task("do something", [], [])
    assert "tool_names is empty" in out
    evt = [e for e in orch.context_manager.trace.events if e["t"] == "run_task"][0]
    assert evt["error"] == "tool_names empty"
    assert evt["tools_used"] == [] and evt["n_images"] == 0


def test_unknown_tool_names_emit_run_task_error():
    orch = _orchestrator()
    out = orch._run_task("do something", ["no_such_tool"], [])
    assert "Unknown tool name(s)" in out
    evt = [e for e in orch.context_manager.trace.events if e["t"] == "run_task"][0]
    assert evt["error"] == "unknown tools: no_such_tool"
    assert evt["tool_names"] == ["no_such_tool"]
