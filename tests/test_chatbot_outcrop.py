"""Chatbot wiring for the outcrop pipeline: context injection, storage, chains, prompt."""
import json
import os

import numpy as np
import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse
from tools import outcrop_tools as ot
from tools.section_tools import synthetic_section_from_model

INTERP = {"regions": [{"id": 1, "label": "sand", "lithology": "sandstone",
                       "geometry": {"type": "band", "y_top": 0.3, "y_bottom": 0.5}}],
          "scale": {"estimated_height_m": 20, "reference": "hammer", "confidence": "medium"},
          "background_lithology": "shale", "mode": "polygons"}


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "SEISMIC_UPLOAD_DIR", str(tmp_path))


@pytest.fixture
def bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]), knowledge_base=object())


@pytest.fixture
def interp(outcrop_image, fake_vision_factory):
    return ot.interpret_outcrop(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]))


def _rm(path):
    if path and os.path.exists(path):
        os.remove(path)


def test_session_id_unique_per_session(bot):
    a, b = bot.new_session(), bot.new_session()
    assert a.session_id != b.session_id and len(a.session_id) == 32


def test_attach_image_sets_last_image_per_session(bot, outcrop_image):
    a, b = bot.new_session(), bot.new_session()
    a.attach_image(outcrop_image)
    assert a.context_manager.get_context("last_image") == outcrop_image
    assert b.context_manager.get_context("last_image") is None


def test_inject_image_path_for_interpret_and_recipe(bot, outcrop_image):
    bot.attach_image(outcrop_image)
    assert bot._inject_context_inputs("interpret_outcrop", {}) == {"image_path": outcrop_image}
    assert bot._inject_context_inputs("outcrop_to_seismic", {"height_m": 5}) == {
        "height_m": 5, "image_path": outcrop_image}
    # the session's attached photo always wins over an LLM-supplied image_path
    # (the LLM cannot know the staged sandbox filename); unrelated tools untouched
    assert bot._inject_context_inputs("interpret_outcrop", {"image_path": "x.png"}) == {
        "image_path": outcrop_image}
    assert bot._inject_context_inputs("make_ricker", {"frequency": 30}) == {"frequency": 30}


def test_inject_image_path_without_last_image_leaves_explicit_value(bot):
    # no attach_image call -> no last_image in context; an explicit LLM value is left as-is.
    assert bot._inject_context_inputs("interpret_outcrop", {"image_path": "x.png"}) == {
        "image_path": "x.png"}


def test_inject_interpretation_and_model(bot, interp):
    bot._update_context("interpret_outcrop", {}, interp)
    assert bot.context_manager.get_context("last_outcrop") is interp
    filled = bot._inject_context_inputs("outcrop_to_model", {"height_m": 30})
    assert filled["interpretation"] is interp
    model = ot.outcrop_to_model(interp, height_m=30, num_traces=11)
    bot._update_context("outcrop_to_model", filled, model)
    assert bot.context_manager.get_context("last_earth_model") is model
    assert bot._inject_context_inputs("synthetic_section", {})["model"] is model


def test_inject_without_context_leaves_param_absent(bot):
    assert bot._inject_context_inputs("outcrop_to_model", {}) == {}


def test_auto_chain_interpret_to_overlay(bot, interp):
    bot._update_context("interpret_outcrop", {}, interp)
    chained = bot._handle_automatic_chaining("interpret_outcrop", {}, interp)
    try:
        assert chained and chained["image_path"].endswith(".png")
    finally:
        _rm((chained or {}).get("image_path"))


def test_auto_chain_section_to_plot_uses_model_from_context(bot, interp):
    model = ot.outcrop_to_model(interp, height_m=20, num_traces=11)
    bot._update_context("outcrop_to_model", {}, model)
    result = synthetic_section_from_model(model)
    bot._update_context("synthetic_section", {"wavelet_freq": 30}, result)
    stored = bot.context_manager.get_context("last_section")
    assert stored["parameters"]["nt"] == result[2]["nt"] and stored["input_params"] == {"wavelet_freq": 30}
    chained = bot._handle_automatic_chaining("synthetic_section", {}, result)
    try:
        assert chained and os.path.getsize(chained["image_path"]) > 0
    finally:
        _rm((chained or {}).get("image_path"))


def test_auto_chain_without_context_returns_none(bot, interp):
    assert bot._handle_automatic_chaining("interpret_outcrop", {}, interp) is None


def test_auto_chain_section_passes_display_from_stored_parameters(bot, interp, monkeypatch):
    """display lives on the stored parameters (set by synthetic_section_from_model),
    not on the raw tool_input the LLM sent — the chain must read it from there."""
    model = ot.outcrop_to_model(interp, height_m=20, num_traces=11)
    result = synthetic_section_from_model(model, display="wiggle")
    bot._update_context("synthetic_section", {}, result)

    recorded = {}

    def _fake_process_tool_call(tool_name, tool_input):
        recorded["tool_name"] = tool_name
        recorded["tool_input"] = tool_input
        return "x.png"

    monkeypatch.setattr(bot.tool_manager, "process_tool_call", _fake_process_tool_call)
    chained = bot._handle_automatic_chaining("synthetic_section", {}, result)
    assert chained == {"image_path": "x.png"}
    assert recorded["tool_name"] == "plot_seismic_section"
    assert recorded["tool_input"]["display"] == "wiggle"


def test_recipe_result_populates_staged_context(bot, outcrop_image, fake_vision_factory):
    from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic
    res = outcrop_to_seismic(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]),
                             num_traces=11)
    try:
        bot._update_context("outcrop_to_seismic", {}, res)
        cm = bot.context_manager
        assert cm.get_context("last_outcrop") is res["interpretation"]
        assert cm.get_context("last_earth_model") is res["model"]
        assert cm.get_context("last_section")["parameters"] is res["section"]["parameters"]
        assert cm.get_context("last_workflow_result") is res
        images = []
        bot._harvest_images(res, images)
        assert images == [res["image_path"]] + res["extra_image_paths"]
    finally:
        _rm(res["image_path"]); _rm(res["extra_image_paths"][0])


def test_compaction_keeps_tool_result_small(bot, outcrop_image, fake_vision_factory):
    from workflows.recipes.outcrop_to_seismic import outcrop_to_seismic
    res = outcrop_to_seismic(outcrop_image, vision_client=fake_vision_factory([json.dumps(INTERP)]),
                             num_traces=101)
    try:
        text = bot._compact_tool_result(res)
        assert len(text) < 6000
        assert "<plot generated" in text and "facies" in text
    finally:
        _rm(res["image_path"]); _rm(res["extra_image_paths"][0])


def test_image_attached_marker_routes_to_tools(bot):
    assert bot._is_knowledge_question("[image attached: a.png] what is this?") is False


def test_tool_loop_injects_image_path(fake_llm_factory, outcrop_image, fake_vision_factory, monkeypatch):
    """End-to-end through _handle_tool_request with a scripted LLM and vision model."""
    class _Func:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

    class FakeToolCall:   # tests/ is not a package, so mirror conftest's shape locally
        def __init__(self, name, arguments, call_id="call_1"):
            self.id, self.function = call_id, _Func(name, arguments)

    monkeypatch.setattr("core.vision_client.build_vision_client",
                        lambda: fake_vision_factory([json.dumps(INTERP)]))
    llm = fake_llm_factory([
        {"content": "", "tool_calls": [FakeToolCall("interpret_outcrop", "{}")], "usage": None},
        {"content": "<reply>Found one sandstone bed, ~20 m high.</reply>", "tool_calls": None, "usage": None},
    ])
    bot = SeismicChatBotToolUse(llm_client=llm, knowledge_base=object())
    bot.attach_image(outcrop_image)
    out = bot._handle_tool_request("[image attached: outcrop.png] interpret this outcrop")
    try:
        assert "sandstone" in out["reply"]
        assert len(out["images"]) == 1 and out["images"][0].endswith(".png")
        assert bot.context_manager.get_context("last_outcrop")["regions"][0]["lithology"] == "sandstone"
        tool_msg = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"][0]
        assert "sandstone" in tool_msg["content"]
    finally:
        for p in out["images"]:
            _rm(p)


def test_system_prompt_lists_outcrop_tools(bot):
    prompt = bot._create_system_prompt()
    for name in ("interpret_outcrop", "outcrop_to_model", "synthetic_section", "outcrop_to_seismic"):
        assert f"- {name}:" in prompt
    assert "[image attached" in prompt


def test_auto_chain_display_fallback_is_overlay(bot, interp):
    """Stored parameters without a display key (hand-built context) chain to overlay."""
    model = ot.outcrop_to_model(interp, height_m=20, num_traces=11)
    bot._update_context("outcrop_to_model", {}, model)
    axis, section, parameters = synthetic_section_from_model(model)
    parameters = dict(parameters); parameters.pop("display")
    bot._update_context("synthetic_section", {}, (axis, section, parameters))
    seen = {}
    bot.tool_manager.process_tool_call = lambda name, inp: seen.update(inp) or "x.png"
    bot._handle_automatic_chaining("synthetic_section", {}, (axis, section, parameters))
    assert seen["display"] == "overlay"
