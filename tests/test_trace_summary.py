from core.trace_summary import format_trace_markdown, summarize_trace


def _record(events, tools_used=None):
    return {"session": "s1", "turn": 3, "tools_used": tools_used or [],
            "events": events}


_FULL = [
    {"t": "turn_start", "ts": 100.0, "input": "make a ricker"},
    {"t": "intent", "ts": 100.4, "verdict": "TOOL", "via": "llm"},
    {"t": "llm", "ts": 101.5, "model": "deepseek-chat", "latency_ms": 1100.0,
     "tool_call": True, "prompt_tokens": 20, "completion_tokens": 10,
     "total_tokens": 30},
    {"t": "tool_call", "ts": 101.9, "tool": "make_ricker", "ok": True, "ms": 350.0,
     "injected": [], "overridden": [], "defaults_filled": ["time_length", "dt"]},
    {"t": "auto_plot", "ts": 102.2, "compute": "make_ricker",
     "plot": "plot_ricker", "fired": True},
    {"t": "llm", "ts": 103.0, "model": "deepseek-chat", "latency_ms": 700.0,
     "tool_call": False, "prompt_tokens": 25, "completion_tokens": 15,
     "total_tokens": 40},
]


def test_headline_covers_routing_tools_plots_and_cost():
    s = summarize_trace(_record(_FULL, tools_used=["make_ricker"]))
    assert "Routed to tools (intent via llm)" in s["headline"]
    assert "ran make_ricker" in s["headline"]
    assert "1 plot(s) auto-generated" in s["headline"]
    assert "2 LLM call(s), 70 tokens, 3.0s" in s["headline"]


def test_defaults_filled_produces_info_flag():
    s = summarize_trace(_record(_FULL, tools_used=["make_ricker"]))
    assert "ℹ️ make_ricker: defaults used for time_length, dt" in s["flags"]


def test_high_stakes_flags_verbatim():
    events = [
        {"t": "turn_start", "ts": 1.0, "input": "x"},
        {"t": "physics_warning", "ts": 1.1, "tool": "wedge_model",
         "category": "UserWarning", "message": "vp 9000.0 outside 300-8000 m/s"},
        {"t": "tool_call", "ts": 1.2, "tool": "bad", "ok": False,
         "error": "Unknown tool: bad"},
        {"t": "budget_exhausted", "ts": 1.3, "rounds": 5, "scope": "tool_loop"},
        {"t": "auto_plot", "ts": 1.4, "compute": "wedge_model",
         "plot": "plot_wedge_model", "fired": False},
        {"t": "turn_error", "ts": 1.5, "error": "boom"},
    ]
    flags = summarize_trace(_record(events))["flags"]
    assert flags == [
        "⚠️ Physics: vp 9000.0 outside 300-8000 m/s",
        "⚠️ Tool failed: bad — Unknown tool: bad",
        "⚠️ Reasoning budget exhausted — the answer was completed without "
        "further tool use",
        "⚠️ Expected plot plot_wedge_model was not generated after wedge_model",
        "⚠️ Turn failed: boom",
    ]


def test_knowledge_route_headline():
    events = [
        {"t": "turn_start", "ts": 1.0, "input": "what is tuning?"},
        {"t": "intent", "ts": 1.1, "verdict": "KNOWLEDGE", "via": "keyword_fallback"},
        {"t": "rag", "ts": 1.2, "rag_type": "retrieve_and_generate",
         "retrieved": 2, "scores": [0.8, 0.5]},
    ]
    s = summarize_trace(_record(events))
    assert s["headline"].startswith(
        "Answered from knowledge base (intent via keyword_fallback)")
    assert s["flags"] == []
    assert any(line.startswith("rag: 2 doc(s)") for line in s["detail_lines"])


def test_detail_lines_skip_turn_start_and_cover_all_events():
    s = summarize_trace(_record(_FULL))
    assert len(s["detail_lines"]) == len(_FULL) - 1
    assert s["detail_lines"][0] == "intent: TOOL (via llm)"
    assert any(line.startswith("tool: make_ricker, 350.0 ms") for line in s["detail_lines"])
    assert any("defaults: time_length, dt" in line for line in s["detail_lines"])
    assert any(line == "plot: plot_ricker auto-generated after make_ricker"
               for line in s["detail_lines"])


def test_unknown_event_gets_fallback_line():
    s = summarize_trace(_record([
        {"t": "turn_start", "ts": 1.0, "input": "x"},
        {"t": "mystery", "ts": 1.1, "foo": "bar"},
    ]))
    assert s["detail_lines"] == ["mystery: foo=bar"]


def test_format_trace_markdown():
    assert format_trace_markdown(None) == "_No decision trace for this turn._"
    assert format_trace_markdown({"events": []}) == "_No decision trace for this turn._"
    md = format_trace_markdown(_record(_FULL, tools_used=["make_ricker"]))
    lines = md.split("\n")
    assert lines[0].startswith("**Routed to tools")
    assert any(line.startswith("ℹ️ make_ricker") for line in lines)
    assert any(line.startswith("- tool: make_ricker") for line in lines)
