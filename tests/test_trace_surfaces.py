def test_chat_response_carries_trace():
    from interfaces.api_interface import ChatResponse
    r = ChatResponse(response="x", success=True,
                     trace={"turn": 1, "tools_used": ["make_ricker"], "events": []})
    assert r.trace["turn"] == 1
    assert ChatResponse(response="x", success=True).trace is None


def test_format_status_appends_tools():
    from interfaces.gradio_interface import format_status
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert format_status(usage) == "Prompt: 10 | Completion: 5 | Total: 15"
    out = format_status(usage, {"tools_used": ["make_ricker", "analyze_wedge"]})
    assert out.endswith("| Tools: make_ricker → analyze_wedge")
    assert format_status(usage, {"tools_used": []}) == "Prompt: 10 | Completion: 5 | Total: 15"
