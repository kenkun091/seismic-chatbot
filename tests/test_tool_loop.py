"""The extracted shared tool loop: same behavior the classic bot pinned, callable standalone."""
import numpy as np
import pytest
from core.tool_loop import ToolLoopRunner, extract_reply
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


def make_runner(fake_llm_factory, responses):
    llm = fake_llm_factory(responses)
    return ToolLoopRunner(llm, ToolManager(), ContextManager()), llm


def test_extract_reply():
    assert extract_reply("x <reply> hi </reply> y") == "hi"
    assert extract_reply("no tags") is None


def test_run_no_tool_call_returns_reply(fake_llm_factory):
    runner, llm = make_runner(fake_llm_factory, [
        {"content": "<reply>Just words</reply>", "tool_calls": None, "stop_reason": "stop", "usage": None},
    ])
    out = runner.run("SYS", [{"role": "user", "content": "hi"}], tools=[])
    assert out == {"reply": "Just words", "images": [], "tools_used": []}
    assert llm.calls[0]["system_prompt"] == "SYS"


def test_run_executes_tool_and_harvests_plot(fake_llm_factory):
    runner, llm = make_runner(fake_llm_factory, [
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>Made a 30 Hz Ricker.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    out = runner.run("SYS", [{"role": "user", "content": "30 Hz ricker"}],
                     tools=ToolManager().get_tool_schemas())
    assert out["reply"] == "Made a 30 Hz Ricker."
    assert out["tools_used"] == ["make_ricker"]
    # auto-chained plot_ricker produced a png
    assert len(out["images"]) == 1 and out["images"][0].endswith(".png")
    # context stored for follow-ups
    assert runner.context_manager.get_context("last_ricker_wavelet") is not None


def test_run_tool_error_becomes_tool_message_and_loop_continues(fake_llm_factory):
    runner, llm = make_runner(fake_llm_factory, [
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": -5}')]},
        {"content": "<reply>That frequency is invalid.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    out = runner.run("SYS", [{"role": "user", "content": "bad"}],
                     tools=ToolManager().get_tool_schemas())
    assert out["tools_used"] == []
    # the error was surfaced to the model as a tool message
    sent = llm.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "failed" in m["content"] for m in sent)


def test_run_round_budget_forces_tool_free_completion(fake_llm_factory):
    tool_turn = {"content": "", "stop_reason": "tool_calls", "usage": None,
                 "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]}
    responses = [dict(tool_turn) for _ in range(5)] + [
        {"content": "<reply>Stopping here.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ]
    runner, llm = make_runner(fake_llm_factory, responses)
    out = runner.run("SYS", [{"role": "user", "content": "loop"}],
                     tools=ToolManager().get_tool_schemas())
    assert out["reply"] == "Stopping here."
    assert llm.calls[-1]["tools"] is None  # forced tool-free final call


def test_inject_context_inputs_last_image_always_wins(fake_llm_factory):
    runner, _ = make_runner(fake_llm_factory, [])
    runner.context_manager.set_context("last_image", "/sandbox/session/photo.png")
    filled = runner.inject_context_inputs(
        "interpret_outcrop", {"image_path": "/evil/elsewhere.png"})
    assert filled["image_path"] == "/sandbox/session/photo.png"
