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
        monkeypatch.delenv("LOG_LEVEL")
        importlib.reload(settings)
        assert settings.LOG_LEVEL == "INFO"
    finally:
        monkeypatch.undo()
        importlib.reload(settings)


def test_context_manager_owns_a_recorder():
    from core.context_manager import ContextManager
    cm = ContextManager()
    assert isinstance(cm.trace, TraceRecorder)
    cm2 = ContextManager()
    assert cm.trace is not cm2.trace


def _bare_llm_client(content="hi", usage=None):
    """Real LLMClient minus credential resolution, with a stubbed transport."""
    from core.llm_client import LLMClient
    client = object.__new__(LLMClient)
    client.model, client.temperature, client.max_tokens = "test-model", 0.1, 100

    class _Msg:
        pass
    msg = _Msg()
    msg.content, msg.tool_calls = content, None

    class _Choice:
        pass
    choice = _Choice()
    choice.message = msg

    class _Resp:
        pass
    resp = _Resp()
    resp.choices, resp.usage = [choice], usage

    class _Completions:
        def create(self, **kw):
            return resp

    class _Chat:
        pass
    chat = _Chat()
    chat.completions = _Completions()

    class _Client:
        pass
    inner = _Client()
    inner.chat = chat
    client.client = inner
    return client


def test_get_completion_reports_model_and_latency():
    client = _bare_llm_client()
    res = client.get_completion("s", "u")
    assert res["model"] == "test-model"
    assert isinstance(res["latency_ms"], float)


def test_get_simple_completion_accounts_tokens_and_traces():
    from core.context_manager import ContextManager
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    client = _bare_llm_client(content="  KNOWLEDGE  ", usage=usage)
    cm = ContextManager()
    cm.trace.persist_dir = ""  # keep the test filesystem-clean
    out = client.get_simple_completion("s", "u", context_manager=cm)
    assert out == "KNOWLEDGE"
    assert cm.get_token_usage()["total_tokens"] == 10
    llm_events = [e for e in cm.trace.events if e["t"] == "llm"]
    assert llm_events and llm_events[0]["total_tokens"] == 10
    assert llm_events[0]["model"] == "test-model"


def test_get_simple_completion_without_context_manager_unchanged():
    client = _bare_llm_client(content="plain")
    assert client.get_simple_completion("s", "u") == "plain"


def test_trace_dir_off_disables_persistence(monkeypatch):
    import importlib
    import config.settings as settings
    try:
        monkeypatch.setenv("SEISMIC_TRACE_DIR", "off")
        importlib.reload(settings)
        assert settings.SEISMIC_TRACE_DIR == ""
    finally:
        monkeypatch.undo()
        importlib.reload(settings)


def test_exporter_hook_receives_record_and_errors_are_swallowed():
    from core.turn_trace import (register_trace_exporter, clear_trace_exporters)
    calls = []

    def good(record):
        calls.append(record["turn"])

    def bad(record):
        raise RuntimeError("exporter boom")

    register_trace_exporter(bad)
    register_trace_exporter(good)
    register_trace_exporter(good)  # duplicate registration is a no-op
    try:
        rec = TraceRecorder(session_id="s", persist_dir="")
        rec.begin_turn("x")
        record = rec.end_turn()  # bad exporter must not raise out of end_turn
        assert calls == [1]
        assert record["turn"] == 1
    finally:
        clear_trace_exporters()


def test_unregister_and_clear_exporters():
    from core.turn_trace import (register_trace_exporter, unregister_trace_exporter,
                                 clear_trace_exporters, _TRACE_EXPORTERS)

    def f(record):
        pass

    register_trace_exporter(f)
    unregister_trace_exporter(f)
    unregister_trace_exporter(f)  # absent: tolerated
    register_trace_exporter(f)
    clear_trace_exporters()
    assert _TRACE_EXPORTERS == []
