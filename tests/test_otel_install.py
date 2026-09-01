import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from core import otel_export
from core.turn_trace import TraceRecorder


@pytest.fixture
def otel(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_GENAI_CAPTURE_CONTENT", raising=False)
    exporter = InMemorySpanExporter()
    assert otel_export.install(span_exporter=exporter) is True
    yield exporter
    otel_export.uninstall()


def test_install_noop_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert otel_export.install() is False
    assert not otel_export._STATE.get("installed")


def test_turn_exports_parented_spans(otel):
    rec = TraceRecorder(session_id="sess-otel", persist_dir="")
    rec.begin_turn("make a ricker")
    rec.emit("intent", verdict="TOOL", via="llm")
    rec.emit("llm", model="m", latency_ms=10.0, prompt_tokens=3,
             completion_tokens=2, total_tokens=5, tool_call=True)
    rec.emit("tool_call", tool="make_ricker", ok=True, ms=5.0,
             injected=[], overridden=[], defaults_filled=[])
    rec.end_turn()
    spans = otel.get_finished_spans()
    assert sorted(s.name for s in spans) == [
        "chat m", "execute_tool make_ricker", "invoke_agent seismic-chatbot"]
    root = [s for s in spans if s.name.startswith("invoke_agent")][0]
    for child in [s for s in spans if not s.name.startswith("invoke_agent")]:
        assert child.parent is not None
        assert child.parent.span_id == root.context.span_id
    assert root.attributes["session.id"] == "sess-otel"
    assert [e.name for e in root.events] == ["intent"]


def test_failed_tool_marks_child_error(otel):
    rec = TraceRecorder(session_id="s", persist_dir="")
    rec.begin_turn("x")
    rec.emit("tool_call", tool="bad", ok=False, error="Unknown tool")
    rec.end_turn()
    child = [s for s in otel.get_finished_spans()
             if s.name == "execute_tool bad"][0]
    assert child.status.status_code == StatusCode.ERROR


def test_install_is_idempotent(otel):
    assert otel_export.install() is True  # second call: already installed


def test_uninstall_stops_export(otel):
    otel_export.uninstall()
    rec = TraceRecorder(session_id="s", persist_dir="")
    rec.begin_turn("x")
    rec.end_turn()
    assert otel.get_finished_spans() == ()
    assert otel_export.install(span_exporter=otel) is True  # fixture uninstalls again
