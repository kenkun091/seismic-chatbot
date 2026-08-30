from types import SimpleNamespace

from core.turn_trace import TraceRecorder, emit_event, usage_dict


def test_recorder_accumulates_and_flushes(tmp_path):
    rec = TraceRecorder(session_id="abc123", persist_dir=str(tmp_path))
    rec.begin_turn("make a 30 Hz ricker wavelet " + "x" * 300)
    rec.emit("tool_call", tool="make_ricker", ok=True, ms=1.2)
    rec.emit("tool_call", tool="bad_tool", ok=False, error="boom")
    record = rec.end_turn()
    assert record["session"] == "abc123"
    assert record["turn"] == 1
    assert record["tools_used"] == ["make_ricker"]  # ok=False excluded
    assert record["events"][0]["t"] == "turn_start"
    assert len(record["events"][0]["input"]) <= 200  # truncated
    # one JSONL line per turn, named by session
    lines = (tmp_path / "abc123.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    import json
    assert json.loads(lines[0])["turn"] == 1


def test_second_turn_resets_events(tmp_path):
    rec = TraceRecorder(session_id="s", persist_dir=str(tmp_path))
    rec.begin_turn("one")
    rec.emit("intent", verdict="TOOL", via="llm")
    rec.end_turn()
    rec.begin_turn("two")
    record = rec.end_turn()
    assert record["turn"] == 2
    assert all(e["t"] != "intent" for e in record["events"])
    assert len((tmp_path / "s.jsonl").read_text().strip().splitlines()) == 2


def test_persist_failure_is_swallowed():
    rec = TraceRecorder(session_id="s", persist_dir="/dev/null/not-a-dir")
    rec.begin_turn("hello")
    record = rec.end_turn()  # must not raise
    assert record["turn"] == 1


def test_no_persist_dir_disables_writes():
    rec = TraceRecorder(session_id="s", persist_dir="")
    rec.begin_turn("hello")
    assert rec.end_turn()["turn"] == 1


def test_emit_event_is_safe_without_trace():
    emit_event(None, "intent", verdict="TOOL")           # no-op, no raise
    emit_event(object(), "intent", verdict="TOOL")       # no trace attr: no-op
    cm = SimpleNamespace(trace=TraceRecorder(session_id="s", persist_dir=""))
    emit_event(cm, "intent", verdict="TOOL", via="llm")
    assert cm.trace.events[-1] == {**cm.trace.events[-1]}  # exists
    assert cm.trace.events[-1]["verdict"] == "TOOL"


def test_usage_dict_tolerates_shapes():
    assert usage_dict(None) == {}
    assert usage_dict({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}) == {
        "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    obj = SimpleNamespace(prompt_tokens=5, completion_tokens=6, total_tokens=11)
    assert usage_dict(obj)["total_tokens"] == 11


def test_settings_expose_trace_dir():
    from config.settings import SEISMIC_TRACE_DIR
    assert isinstance(SEISMIC_TRACE_DIR, str) and SEISMIC_TRACE_DIR


def test_log_level_env_override(monkeypatch):
    import importlib
    import config.settings as settings
    try:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        importlib.reload(settings)
        assert settings.LOG_LEVEL == "DEBUG"
    finally:
        monkeypatch.undo()
        importlib.reload(settings)
    assert settings.LOG_LEVEL == "INFO"
