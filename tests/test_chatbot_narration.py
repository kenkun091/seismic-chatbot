"""End-to-end tests for the narrated response contract of the agentic tool
loop: _handle_tool_request / process_single_input return
{"reply": str, "images": list[str]}. No network — scripted FakeLLMClient
(tests/conftest.py) plus a scripted fake tool manager."""
import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


class _ScriptedToolManager:
    """Returns a scripted result per tool name; a scripted Exception raises."""
    def __init__(self, results):
        self._results = dict(results)
        self.calls = []

    def process_tool_call(self, name, params):
        self.calls.append((name, params))
        result = self._results[name]
        if isinstance(result, Exception):
            raise result
        return result


def _completion(tool_calls=None, content=""):
    return {
        "content": content,
        "tool_calls": tool_calls,
        "stop_reason": "tool_calls" if tool_calls else "stop",
        "usage": None,
    }


@pytest.fixture
def bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


def test_workflow_result_is_narrated_with_image(bot, fake_llm_factory):
    tc = _FakeToolCall("tuning", '{"phit_sand": 0.25}')
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc]),
        _completion(content="<reply>Tuning thickness is 12.5 m at 30 Hz.</reply>"),
    ])
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/t.png"},
    })
    out = bot._handle_tool_request("tuning analysis for a 25% porosity sand")
    assert out == {"reply": "Tuning thickness is 12.5 m at 30 Hz.",
                   "images": ["/tmp/t.png"]}


def test_multiple_rounds_collect_all_images_in_order(bot, fake_llm_factory):
    tc1 = _FakeToolCall("tuning", '{"phit_sand": 0.25}', "c1")
    tc2 = _FakeToolCall("fluid_scenario", '{"phit_sand": 0.25}', "c2")
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1]),
        _completion(tool_calls=[tc2]),
        _completion(content="<reply>Both analyses are done.</reply>"),
    ])
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/a.png"},
        "fluid_scenario": {"cases": {}, "image_path": "/tmp/b.png"},
    })
    out = bot._handle_tool_request("tuning then fluid scenarios")
    assert out["reply"] == "Both analyses are done."
    assert out["images"] == ["/tmp/a.png", "/tmp/b.png"]


def test_repeated_image_path_deduped(bot, fake_llm_factory):
    tc1 = _FakeToolCall("tuning", '{"phit_sand": 0.25}', "c1")
    tc2 = _FakeToolCall("tuning", '{"phit_sand": 0.30}', "c2")
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1]),
        _completion(tool_calls=[tc2]),
        _completion(content="<reply>Done.</reply>"),
    ])
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/same.png"},
    })
    out = bot._handle_tool_request("two tuning runs")
    assert out["images"] == ["/tmp/same.png"]


def test_tool_message_content_is_compacted(bot, fake_llm_factory):
    tc = _FakeToolCall("tuning", '{"phit_sand": 0.25}')
    llm = fake_llm_factory([
        _completion(tool_calls=[tc]),
        _completion(content="<reply>Done.</reply>"),
    ])
    bot.llm_client = llm
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"curve": list(range(61)), "image_path": "/tmp/t.png"},
    })
    bot._handle_tool_request("tuning")
    final_messages = llm.calls[-1]["messages"]
    tool_msgs = [m for m in final_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "<61 values" in tool_msgs[0]["content"]
    assert "/tmp/t.png" not in tool_msgs[0]["content"]
    assert "plot generated and shown to the user" in tool_msgs[0]["content"]


def test_round_exhaustion_returns_reply_and_images(bot, fake_llm_factory):
    calls = [_FakeToolCall("tuning", '{"phit_sand": 0.25}', f"c{i}") for i in range(5)]
    bot.llm_client = fake_llm_factory(
        [_completion(tool_calls=[c]) for c in calls]
        + [_completion(content="<reply>Stopping here; tuning is 12.5 m.</reply>")]
    )
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/t.png"},
    })
    out = bot._handle_tool_request("keep going")
    assert out == {"reply": "Stopping here; tuning is 12.5 m.",
                   "images": ["/tmp/t.png"]}


def test_tool_error_recovers_and_narrates(bot, fake_llm_factory):
    tc1 = _FakeToolCall("tuning", '{"phit_sand": 0.25}', "c1")
    tc2 = _FakeToolCall("fluid_scenario", '{"phit_sand": 0.25}', "c2")
    llm = fake_llm_factory([
        _completion(tool_calls=[tc1]),
        _completion(tool_calls=[tc2]),
        _completion(content="<reply>Fluid scenario failed; tuning is 12.5 m.</reply>"),
    ])
    bot.llm_client = llm
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/t.png"},
        "fluid_scenario": ValueError("bad fluids"),
    })
    out = bot._handle_tool_request("tuning then fluids")
    assert out == {"reply": "Fluid scenario failed; tuning is 12.5 m.",
                   "images": ["/tmp/t.png"]}
    # The model saw the error as a tool message it can react to.
    final_messages = llm.calls[-1]["messages"]
    assert any(m.get("role") == "tool" and "Tool execution failed" in m["content"]
               for m in final_messages)


def test_persistent_tool_errors_exhaust_rounds_and_still_answer(bot, fake_llm_factory):
    calls = [_FakeToolCall("fluid_scenario", '{"phit_sand": 0.25}', f"c{i}") for i in range(5)]
    bot.llm_client = fake_llm_factory(
        [_completion(tool_calls=[c]) for c in calls]
        + [_completion(content="<reply>I could not run the fluid scenario.</reply>")]
    )
    bot.tool_manager = _ScriptedToolManager({"fluid_scenario": ValueError("bad fluids")})
    out = bot._handle_tool_request("fluids")
    assert out == {"reply": "I could not run the fluid scenario.", "images": []}


def test_process_single_input_passes_through_tool_dict(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: False)
    monkeypatch.setattr(bot, "_handle_tool_request",
                        lambda text: {"reply": "hi", "images": ["/tmp/a.png"]})
    assert bot.process_single_input("x") == {"reply": "hi", "images": ["/tmp/a.png"]}


def test_process_single_input_wraps_knowledge_string(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: True)
    monkeypatch.setattr(bot, "_handle_knowledge_question",
                        lambda text: "A Ricker wavelet is a zero-phase pulse.")
    assert bot.process_single_input("what is a ricker?") == {
        "reply": "A Ricker wavelet is a zero-phase pulse.", "images": []}


def test_process_single_input_error_returns_dict(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: False)

    def _boom(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(bot, "_handle_tool_request", _boom)
    out = bot.process_single_input("x")
    assert out["reply"].startswith("I encountered an error:")
    assert out["images"] == []


def test_process_single_input_none_reply_gets_fallback_text(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: True)
    monkeypatch.setattr(bot, "_handle_knowledge_question", lambda text: None)
    out = bot.process_single_input("x")
    assert out == {"reply": "I didn't get a response. Please try again.",
                   "images": []}


def test_process_single_input_empty_reply_with_images_gets_caption(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: False)
    monkeypatch.setattr(bot, "_handle_tool_request",
                        lambda text: {"reply": "", "images": ["/tmp/a.png"]})
    assert bot.process_single_input("x") == {"reply": "Here are the results.",
                                             "images": ["/tmp/a.png"]}
