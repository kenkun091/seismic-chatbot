"""Tests for the rock-properties plot tool, its auto-plot wiring, and the
multi-round tool loop in _handle_tool_request.

Covers the two defects behind the broken "Calculate and plot rock properties"
example prompt:
  A) there was no plot capability for calculate_rock_properties; and
  B) _handle_tool_request executed only a single tool round, silently dropping
     any follow-up tool call (returning the model's dangling preamble).
"""
import os

import pytest

from core.tool_registry import REGISTRY_BY_NAME, TOOL_FUNCTIONS, AUTO_PLOT
from core.chatbot_tool_use import SeismicChatBotToolUse
from tools.rock_physics_tools import calculate_rock_properties, plot_rock_properties


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


def _completion(tool_calls=None, content="", stop_reason=None):
    return {
        "content": content,
        "tool_calls": tool_calls,
        "stop_reason": stop_reason or ("tool_calls" if tool_calls else "stop"),
        "usage": None,
    }


@pytest.fixture
def bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


# --- Defect A: the plot tool itself ----------------------------------------

def test_plot_rock_properties_returns_png_for_arrays():
    phit = [0.10, 0.20, 0.30]
    vclay = [0.10, 0.30, 0.50]
    vp, vs, rhob, vpvs, ai, si = calculate_rock_properties(phit, vclay, print_results=False)
    path = plot_rock_properties(phit, vclay, vp, vs, rhob, vpvs, ai, si, fluid_type="water")
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.exists(path)


def test_plot_rock_properties_handles_scalar_inputs():
    vp, vs, rhob, vpvs, ai, si = calculate_rock_properties(0.2, 0.1, print_results=False)
    path = plot_rock_properties(0.2, 0.1, vp, vs, rhob, vpvs, ai, si)
    assert path.endswith(".png") and os.path.exists(path)


# --- Defect A: registry wiring ---------------------------------------------

def test_calculate_rock_properties_auto_plots():
    assert REGISTRY_BY_NAME["calculate_rock_properties"].auto_plot == "plot_rock_properties"
    assert AUTO_PLOT.get("calculate_rock_properties") == "plot_rock_properties"


def test_plot_rock_properties_is_registered():
    assert "plot_rock_properties" in REGISTRY_BY_NAME
    assert "plot_rock_properties" in TOOL_FUNCTIONS


def test_automatic_chaining_produces_image(bot):
    phit = [0.10, 0.20, 0.30]
    vclay = [0.10, 0.30, 0.50]
    tool_input = {"phit": phit, "vclay": vclay, "fluid_type": "water"}
    result = calculate_rock_properties(phit, vclay, print_results=False)
    bot._update_context("calculate_rock_properties", tool_input, result)

    chained = bot._handle_automatic_chaining("calculate_rock_properties", tool_input, result)
    assert isinstance(chained, dict) and "image_path" in chained
    assert chained["image_path"].endswith(".png")
    assert os.path.exists(chained["image_path"])


# --- Defect B: the multi-round tool loop -----------------------------------

def test_tool_loop_executes_follow_up_tool_call(bot, fake_llm_factory):
    """A tool with no auto-plot followed by a second tool call must NOT drop the
    follow-up: the model's preamble is not the final answer."""
    # predict_elastic_layer has auto_plot=None and needs no network.
    tc1 = _FakeToolCall("predict_elastic_layer", '{"phit": [0.2], "vclay": [0.1]}', "c1")
    tc2 = _FakeToolCall("predict_elastic_layer", '{"phit": [0.25], "vclay": [0.15]}', "c2")

    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1], content=""),
        _completion(tool_calls=[tc2], content="Let me also look up some rock physics context."),
        _completion(tool_calls=None, content="<reply>Vp is about 4000 m/s.</reply>"),
    ])

    result = bot._handle_tool_request("predict the sand layer and explain it")
    assert "look up some rock physics context" not in result["reply"]
    assert result["reply"] == "Vp is about 4000 m/s."
    assert result["images"] == []


def test_tool_loop_single_round_still_returns_text(bot, fake_llm_factory):
    tc1 = _FakeToolCall("predict_elastic_layer", '{"phit": [0.2], "vclay": [0.1]}', "c1")
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1], content=""),
        _completion(tool_calls=None, content="<reply>Done.</reply>"),
    ])
    result = bot._handle_tool_request("predict the layer")
    assert result["reply"] == "Done."
    assert result["images"] == []
