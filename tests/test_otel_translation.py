from core.otel_export import spans_from_record


def _record(events, session="s1", turn=1, tools_used=None):
    return {"session": session, "turn": turn,
            "tools_used": tools_used or [], "events": events}


def test_empty_record_yields_no_spans():
    assert spans_from_record(_record([])) == []


def test_root_span_covers_turn_and_carries_session():
    events = [
        {"t": "turn_start", "ts": 100.0, "input": "make a ricker"},
        {"t": "intent", "ts": 100.5, "verdict": "TOOL", "via": "llm"},
        {"t": "llm", "ts": 102.0, "model": "deepseek-chat", "latency_ms": 1500.0,
         "tool_call": True, "prompt_tokens": 10, "completion_tokens": 5,
         "total_tokens": 15},
        {"t": "tool_call", "ts": 102.4, "tool": "make_ricker", "ok": True, "ms": 350.0,
         "injected": [], "overridden": [], "defaults_filled": ["time_length"]},
    ]
    spans = spans_from_record(_record(events, tools_used=["make_ricker"]))
    root = spans[0]
    assert root["name"] == "invoke_agent seismic-chatbot"
    assert root["attributes"]["gen_ai.operation.name"] == "invoke_agent"
    assert root["attributes"]["session.id"] == "s1"
    assert root["attributes"]["langfuse.session.id"] == "s1"
    assert root["attributes"]["seismic.turn"] == 1
    assert root["attributes"]["seismic.tools_used"] == "make_ricker"
    assert root["start_ns"] == 100_000_000_000
    assert root["end_ns"] >= 102_400_000_000
    assert "gen_ai.input.messages" not in root["attributes"]  # content off by default
    assert [e["name"] for e in root["events"]] == ["intent"]
    assert root["status_error"] is None


def test_llm_event_becomes_chat_span_with_reconstructed_timing():
    spans = spans_from_record(_record([
        {"t": "turn_start", "ts": 100.0, "input": "x"},
        {"t": "llm", "ts": 102.0, "model": "m", "latency_ms": 1500.0,
         "prompt_tokens": 10, "completion_tokens": 5},
    ]))
    chat = [s for s in spans if s["name"] == "chat m"][0]
    assert chat["start_ns"] == 100_500_000_000
    assert chat["end_ns"] == 102_000_000_000
    assert chat["attributes"]["gen_ai.operation.name"] == "chat"
    assert chat["attributes"]["gen_ai.request.model"] == "m"
    assert chat["attributes"]["gen_ai.usage.input_tokens"] == 10
    assert chat["attributes"]["gen_ai.usage.output_tokens"] == 5


def test_tool_call_span_and_error_status():
    spans = spans_from_record(_record([
        {"t": "turn_start", "ts": 1.0, "input": "x"},
        {"t": "tool_call", "ts": 1.5, "tool": "make_ricker", "ok": True, "ms": 200.0,
         "injected": ["image_path"], "overridden": [], "defaults_filled": ["dt"]},
        {"t": "tool_call", "ts": 1.9, "tool": "bad", "ok": False,
         "error": "Unknown tool"},
    ]))
    ok = [s for s in spans if s["name"] == "execute_tool make_ricker"][0]
    assert ok["attributes"]["gen_ai.tool.name"] == "make_ricker"
    assert ok["attributes"]["seismic.injected"] == ["image_path"]
    assert ok["start_ns"] == 1_300_000_000 and ok["end_ns"] == 1_500_000_000
    assert ok["status_error"] is None
    bad = [s for s in spans if s["name"] == "execute_tool bad"][0]
    assert bad["status_error"] == "Unknown tool"


def test_turn_error_marks_root_and_lists_as_event():
    spans = spans_from_record(_record([
        {"t": "turn_start", "ts": 1.0, "input": "x"},
        {"t": "turn_error", "ts": 1.1, "error": "boom"},
    ]))
    root = spans[0]
    assert root["status_error"] == "boom"
    assert root["events"][0]["name"] == "turn_error"


def test_capture_content_flag_attaches_input():
    record = _record([{"t": "turn_start", "ts": 1.0, "input": "hello"}])
    assert "gen_ai.input.messages" not in spans_from_record(record)[0]["attributes"]
    spans = spans_from_record(record, capture_content=True)
    assert spans[0]["attributes"]["gen_ai.input.messages"] == "hello"


def test_attribute_coercion_is_otel_legal():
    spans = spans_from_record(_record([
        {"t": "turn_start", "ts": 1.0, "input": "x"},
        {"t": "discover", "ts": 1.1, "query": "q", "hits": [["make_ricker", 0.9]]},
        {"t": "rag", "ts": 1.2, "rag_type": None, "retrieved": 0, "scores": [0.5]},
    ]))
    root = spans[0]
    discover = [e for e in root["events"] if e["name"] == "discover"][0]
    assert isinstance(discover["attributes"]["hits"], str)  # nested list -> json
    rag = [e for e in root["events"] if e["name"] == "rag"][0]
    assert "rag_type" not in rag["attributes"]  # None dropped
    assert rag["attributes"]["scores"] == [0.5]  # scalar list kept
    for span in spans:
        for v in span["attributes"].values():
            assert isinstance(v, (str, bool, int, float, list))


def test_root_clamps_to_earliest_child_start():
    spans = spans_from_record(_record([
        {"t": "turn_start", "ts": 100.0, "input": "x"},
        {"t": "llm", "ts": 100.05, "latency_ms": 200.0},
    ]))
    chat = spans[1]
    assert spans[0]["start_ns"] <= chat["start_ns"]
    assert spans[0]["end_ns"] >= chat["end_ns"]
