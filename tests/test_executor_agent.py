import pytest
from core.executor_agent import ExecutorAgent, TaskResult
from core.tool_manager import ToolManager
from core.context_manager import ContextManager


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


def make_executor(fake_llm_factory, responses):
    llm = fake_llm_factory(responses)
    return ExecutorAgent(llm, ToolManager(), ContextManager()), llm


def test_scoped_schemas_only(fake_llm_factory):
    ex, llm = make_executor(fake_llm_factory, [
        {"content": "<reply>done</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    ex.run("make a 30 Hz ricker", ["make_ricker", "analyze_wedge"])
    sent = llm.calls[0]["tools"]
    assert {t["function"]["name"] for t in sent} == {"make_ricker", "analyze_wedge"}


def test_system_prompt_carries_assigned_cards_only(fake_llm_factory):
    ex, llm = make_executor(fake_llm_factory, [
        {"content": "<reply>done</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    ex.run("brief", ["make_ricker"])
    sys = llm.calls[0]["system_prompt"]
    assert "make_ricker:" in sys
    assert "wedge_model:" not in sys


def test_run_executes_and_returns_task_result(fake_llm_factory):
    ex, llm = make_executor(fake_llm_factory, [
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>30 Hz Ricker built.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    result = ex.run("make a 30 Hz ricker", ["make_ricker"])
    assert isinstance(result, TaskResult)
    assert result.summary == "30 Hz Ricker built."
    assert result.tools_used == ["make_ricker"]
    assert result.error is None
    assert len(result.images) == 1 and result.images[0].endswith(".png")


def test_unknown_tool_name_is_error_not_exception(fake_llm_factory):
    ex, _ = make_executor(fake_llm_factory, [])
    result = ex.run("brief", ["make_ricker", "nonsense_tool"])
    assert result.error is not None and "nonsense_tool" in result.error
    assert result.summary == "" and result.images == []


def test_shared_context_manager_receives_results(fake_llm_factory):
    cm = ContextManager()
    llm = fake_llm_factory([
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>done</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    ExecutorAgent(llm, ToolManager(), cm).run("brief", ["make_ricker"])
    assert cm.get_context("last_ricker_wavelet") is not None  # visible to the NEXT executor
