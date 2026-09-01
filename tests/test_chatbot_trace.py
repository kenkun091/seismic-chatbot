from core.chatbot_tool_use import SeismicChatBotToolUse
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
    # no get_simple_completion → keyword fallback routes "make a ..." to tools


def _bot(responses):
    bot = SeismicChatBotToolUse(llm_client=FakeLLM(responses),
                                tool_manager=ToolManager(),
                                knowledge_base=object())
    bot.context_manager.trace.persist_dir = ""
    return bot


def test_classic_turn_returns_trace_with_tools_used():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    bot = _bot(responses)
    out = bot.process_single_input("make a 30 Hz ricker wavelet")
    assert set(out) == {"reply", "images", "trace"}
    assert out["reply"] == "done"
    assert out["trace"]["session"] == bot.session_id
    assert out["trace"]["tools_used"] == ["make_ricker"]
    kinds = [e["t"] for e in out["trace"]["events"]]
    assert "intent" in kinds and "tool_call" in kinds and "auto_plot" in kinds


def test_classic_handle_tool_request_threads_tools_used():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    bot = _bot(responses)
    result = bot._handle_tool_request("make a 30 Hz ricker wavelet")
    assert result["tools_used"] == ["make_ricker"]


def test_classic_error_turn_still_returns_trace():
    bot = _bot([])  # empty script → IndexError inside the loop
    out = bot.process_single_input("make a 30 Hz ricker wavelet")
    assert "error" in out["reply"].lower()
    assert any(e["t"] == "turn_error" for e in out["trace"]["events"])
