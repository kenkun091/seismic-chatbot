# OTel Trace Export + RAG Token Accounting (Tier 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export each finished TurnTrace record as GenAI-semconv OpenTelemetry spans over OTLP (viewable in self-hosted Langfuse/Phoenix), and close the RAG-generation token-accounting gap.

**Architecture:** A callback-list exporter hook on `TraceRecorder.end_turn` (stdlib-only) feeds `core/otel_export.py`, which is split into a pure record→span-dict translation (`spans_from_record`, testable with no OTel) and a guarded, env-gated OTel emitter (`install`/`uninstall`, module-local `TracerProvider`, no global mutation). RAG accounting threads the shared accounted `LLMClient` and the session `context_manager` through `KnowledgeBase` → `RAGSystem` → `get_simple_completion`.

**Tech Stack:** Python 3.9.7; optional `opentelemetry-sdk` + `opentelemetry-exporter-otlp-proto-http` (installed in this venv at 1.36.0 — SDK tests use `InMemorySpanExporter` behind `pytest.importorskip`); pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-otel-trace-export-spec.md` (read it first).

## Global Constraints

- Python 3.9.7 — `from __future__ import annotations` in new/edited modules; `typing.Optional[X]`, never `X | None`.
- `core/turn_trace.py` stays stdlib-only (the hook is a plain callback list; OTel names never appear in it).
- No hard new runtime dependency: `core/otel_export.py` imports OTel only inside `install()`, failures log a WARNING and return False; `requirements.txt` unchanged; OTel goes in a pyproject `otel` extra.
- An exporter callback may never raise out of `end_turn` — each call individually try/excepted to a WARNING.
- `install()` is a no-op returning False unless `OTEL_EXPORTER_OTLP_ENDPOINT`/`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set or a `span_exporter` is injected; never calls `opentelemetry.trace.set_tracer_provider` (no globals).
- Span/event attributes must be OTel-legal (str/bool/int/float or lists thereof); `None` values dropped; content (the `turn_start.input` snippet) attached only when `OTEL_GENAI_CAPTURE_CONTENT` is truthy.
- Legacy-fake compatibility: every new optional kwarg call gets a `TypeError` fallback to the old call shape (same pattern as Tier 1's `_simple`); existing fake KBs/LLMs in tests must keep working unmodified.
- Working dir: `/Users/kumono/Desktop/devs/dash-apps/wedge-model/geo-mcp/seismic_chatbot` (its own git repo; commit from inside it).
- Branch: `otel-export`, created from `agent-decision-trace` (Task 1 Step 0). Do NOT run the full suite until the final task (~220s); per-task runs use the named test files only.

---

### Task 1: Exporter hook on TraceRecorder

**Files:**
- Modify: `core/turn_trace.py`
- Test: `tests/test_turn_trace.py` (append)

**Interfaces:**
- Consumes: existing `TraceRecorder.end_turn`.
- Produces (later tasks rely on exact names): module functions in `core.turn_trace` — `register_trace_exporter(fn) -> None` (idempotent), `unregister_trace_exporter(fn) -> None` (tolerates absent), `clear_trace_exporters() -> None`; `end_turn()` invokes each registered callback with the record after `_persist`, swallowing exceptions to a WARNING.

- [ ] **Step 0: Create the working branch**

```bash
cd /Users/kumono/Desktop/devs/dash-apps/wedge-model/geo-mcp/seismic_chatbot
git checkout agent-decision-trace && git checkout -b otel-export
```

- [ ] **Step 1: Write the failing tests** — append to `tests/test_turn_trace.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_turn_trace.py -q`
Expected: 2 new tests FAIL with `ImportError: cannot import name 'register_trace_exporter'`.

- [ ] **Step 3: Implement** in `core/turn_trace.py`

Add after the `emit_event` function:

```python
# Registered callbacks receive each finished turn record (e.g. the OTel span
# exporter in core/otel_export.py). Callbacks may never break a turn: each is
# wrapped in its own try/except in end_turn.
_TRACE_EXPORTERS: List[Any] = []


def register_trace_exporter(fn: Any) -> None:
    if fn not in _TRACE_EXPORTERS:
        _TRACE_EXPORTERS.append(fn)


def unregister_trace_exporter(fn: Any) -> None:
    try:
        _TRACE_EXPORTERS.remove(fn)
    except ValueError:
        pass


def clear_trace_exporters() -> None:
    del _TRACE_EXPORTERS[:]
```

In `end_turn`, after `self._persist(record)` and before `return record`:

```python
        for exporter in list(_TRACE_EXPORTERS):
            try:
                exporter(record)
            except Exception as e:
                logger.warning(f"trace exporter failed: {e}")
```

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/test_turn_trace.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/turn_trace.py tests/test_turn_trace.py
git commit -m "feat(otel): exporter callback hook on TraceRecorder.end_turn (stdlib-only)"
```

---

### Task 2: Pure record→span translation (`spans_from_record`)

**Files:**
- Create: `core/otel_export.py` (translation half only)
- Test: Create `tests/test_otel_translation.py`

**Interfaces:**
- Consumes: turn-record shape from Tier 1 (`{"session", "turn", "tools_used", "events"}`; events carry `t`, `ts`, and per-type fields).
- Produces: `spans_from_record(record, capture_content=False, agent_name="seismic-chatbot") -> List[Dict[str, Any]]` — root span dict first, then children; each dict has exactly `name`, `start_ns`, `end_ns`, `attributes` (no None values), `events` (root only carries entries: `{"name", "ts_ns", "attributes"}`), `status_error` (Optional[str]).

- [ ] **Step 1: Write the failing tests** — create `tests/test_otel_translation.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_otel_translation.py -q`
Expected: `ModuleNotFoundError: No module named 'core.otel_export'`.

- [ ] **Step 3: Implement** — create `core/otel_export.py` with the translation half:

```python
"""Tier-2 OTel export: turn records -> GenAI-semconv spans.

Two halves, deliberately separate:
- spans_from_record: PURE translation of a TurnTrace record into plain span
  dicts (no OTel imports) — the tested core.
- install()/uninstall(): guarded, env-gated OTel emission over OTLP (added in
  a later task).

Timing is reconstructed from the events: `llm`/`tool_call` events carry an
end timestamp (`ts`) and a duration (`latency_ms`/`ms`), so child spans are
`[ts - duration, ts]`; everything else becomes a span event on the root.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_NAME = "seismic-chatbot"


def _ns(ts: float) -> int:
    return int(round(float(ts) * 1e9))


def _attr_value(value: Any) -> Any:
    """Coerce one field to an OTel-legal attribute value; None means 'drop'."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if all(isinstance(v, (str, bool, int, float)) and v is not None for v in seq):
            return seq
        return json.dumps(seq, default=str)
    return json.dumps(value, default=str)


def _clean_attrs(fields: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        av = _attr_value(v)
        if av is not None:
            out[k] = av
    return out


def spans_from_record(record: Dict[str, Any], capture_content: bool = False,
                      agent_name: str = _DEFAULT_AGENT_NAME) -> List[Dict[str, Any]]:
    """Translate one turn record into span dicts (root first, then children)."""
    events = record.get("events") or []
    if not events:
        return []
    times = [e["ts"] for e in events if isinstance(e.get("ts"), (int, float))]
    start_ts = min(times) if times else 0.0
    end_ts = max(times) if times else start_ts

    root_attrs = _clean_attrs({
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": agent_name,
        "session.id": record.get("session"),
        "langfuse.session.id": record.get("session"),
        "seismic.turn": record.get("turn"),
        "seismic.tools_used": ",".join(record.get("tools_used") or []),
    })
    root: Dict[str, Any] = {"name": f"invoke_agent {agent_name}",
                            "start_ns": _ns(start_ts), "end_ns": _ns(end_ts),
                            "attributes": root_attrs, "events": [],
                            "status_error": None}
    spans: List[Dict[str, Any]] = [root]

    for e in events:
        t = e.get("t")
        ts = e.get("ts", start_ts)
        if t == "turn_start":
            if capture_content and e.get("input"):
                root_attrs["gen_ai.input.messages"] = str(e["input"])
            continue
        if t == "llm":
            dur_s = float(e.get("latency_ms") or 0.0) / 1000.0
            model = e.get("model") or "unknown"
            attrs = _clean_attrs({
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": e.get("model"),
                "gen_ai.usage.input_tokens": e.get("prompt_tokens"),
                "gen_ai.usage.output_tokens": e.get("completion_tokens"),
                "seismic.requested_tool_call": e.get("tool_call"),
            })
            spans.append({"name": f"chat {model}",
                          "start_ns": _ns(ts - dur_s), "end_ns": _ns(ts),
                          "attributes": attrs, "events": [], "status_error": None})
        elif t == "tool_call":
            dur_s = float(e.get("ms") or 0.0) / 1000.0
            tool = e.get("tool") or "unknown"
            attrs = _clean_attrs({
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool,
                "seismic.injected": e.get("injected"),
                "seismic.overridden": e.get("overridden"),
                "seismic.defaults_filled": e.get("defaults_filled"),
            })
            error = None if e.get("ok") else str(e.get("error") or "tool failed")
            spans.append({"name": f"execute_tool {tool}",
                          "start_ns": _ns(ts - dur_s), "end_ns": _ns(ts),
                          "attributes": attrs, "events": [], "status_error": error})
        else:
            attrs = _clean_attrs({k: v for k, v in e.items() if k not in ("t", "ts")})
            root["events"].append({"name": t or "event", "ts_ns": _ns(ts),
                                   "attributes": attrs})
            if t == "turn_error":
                root["status_error"] = str(e.get("error") or "turn error")

    root["start_ns"] = min(s["start_ns"] for s in spans)
    root["end_ns"] = max(s["end_ns"] for s in spans)
    return spans
```

(The `os` import is unused until the next task adds `install()`; keep it.)

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/test_otel_translation.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add core/otel_export.py tests/test_otel_translation.py
git commit -m "feat(otel): pure TurnTrace-record -> GenAI-semconv span translation"
```

---

### Task 3: OTel emitter (`install`/`uninstall`), entry-point wiring, packaging

**Files:**
- Modify: `core/otel_export.py` (emission half)
- Modify: `main.py`, `interfaces/api_interface.py`, `pyproject.toml`
- Test: Create `tests/test_otel_install.py`

**Interfaces:**
- Consumes: Task 1's `register_trace_exporter`/`unregister_trace_exporter`; Task 2's `spans_from_record`.
- Produces: `core.otel_export.install(span_exporter=None) -> bool` (idempotent; env-gated unless an exporter is injected; guarded imports); `core.otel_export.uninstall() -> None`; `_STATE: Dict[str, Any]` module dict (tests may inspect `_STATE.get("installed")`).

- [ ] **Step 1: Write the failing tests** — create `tests/test_otel_install.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_otel_install.py -q`
Expected: FAIL/ERROR — `otel_export` has no `install`.

- [ ] **Step 3: Implement**

Append to `core/otel_export.py`:

```python
# ---------------------------------------------------------------------------
# Emission half: env-gated, guarded OTel wiring. No module-level OTel imports
# and no global tracer-provider mutation — everything lives in _STATE.
# ---------------------------------------------------------------------------
from core.turn_trace import register_trace_exporter, unregister_trace_exporter

_STATE: Dict[str, Any] = {}


def install(span_exporter: Any = None) -> bool:
    """Enable OTel export of turn traces. Returns True when installed.

    Without an injected span_exporter this is env-gated: it only activates
    when OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is
    set, and quietly returns False (WARNING on missing SDK) otherwise — so
    the package carries no hard OTel dependency.
    """
    if _STATE.get("installed"):
        return True
    endpoint_set = bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
    if span_exporter is None and not endpoint_set:
        return False
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (BatchSpanProcessor,
                                                    SimpleSpanProcessor)
        from opentelemetry.trace import Status, StatusCode
    except ImportError as e:
        logger.warning(f"OTel trace export requested but the SDK is missing: {e}. "
                       f"pip install 'seismic-chatbot[otel]'")
        return False
    if span_exporter is None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter)
        except ImportError as e:
            logger.warning(f"OTLP exporter missing: {e}. "
                           f"pip install 'seismic-chatbot[otel]'")
            return False
        processor = BatchSpanProcessor(OTLPSpanExporter())
    else:
        processor = SimpleSpanProcessor(span_exporter)
    provider = TracerProvider(resource=Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", _DEFAULT_AGENT_NAME)}))
    provider.add_span_processor(processor)
    _STATE.update(
        installed=True,
        provider=provider,
        tracer=provider.get_tracer("seismic_chatbot.turn_trace"),
        trace_api=trace_api,
        status_cls=Status,
        status_code_cls=StatusCode,
        capture_content=(os.environ.get("OTEL_GENAI_CAPTURE_CONTENT", "")
                         .strip().lower() in ("1", "true", "yes")),
    )
    register_trace_exporter(_export_record)
    logger.info("OTel turn-trace export installed")
    return True


def uninstall() -> None:
    if not _STATE.get("installed"):
        return
    unregister_trace_exporter(_export_record)
    provider = _STATE.get("provider")
    if provider is not None:
        try:
            provider.shutdown()
        except Exception as e:
            logger.warning(f"OTel provider shutdown failed: {e}")
    _STATE.clear()


def _export_record(record: Dict[str, Any]) -> None:
    if not _STATE.get("installed"):
        return
    spans = spans_from_record(record, capture_content=_STATE["capture_content"])
    if not spans:
        return
    tracer, trace_api = _STATE["tracer"], _STATE["trace_api"]
    status_cls, error_code = _STATE["status_cls"], _STATE["status_code_cls"].ERROR
    root_spec, child_specs = spans[0], spans[1:]
    root = tracer.start_span(root_spec["name"], start_time=root_spec["start_ns"],
                             attributes=root_spec["attributes"])
    for ev in root_spec["events"]:
        root.add_event(ev["name"], attributes=ev["attributes"],
                       timestamp=ev["ts_ns"])
    ctx = trace_api.set_span_in_context(root)
    for spec in child_specs:
        child = tracer.start_span(spec["name"], context=ctx,
                                  start_time=spec["start_ns"],
                                  attributes=spec["attributes"])
        if spec["status_error"]:
            child.set_status(status_cls(error_code, spec["status_error"]))
        child.end(end_time=spec["end_ns"])
    if root_spec["status_error"]:
        root.set_status(status_cls(error_code, root_spec["status_error"]))
    root.end(end_time=root_spec["end_ns"])
```

Wire the entry points:

- `main.py` — inside `main()`, right after the `logging.basicConfig(...)` block and `logger = logging.getLogger(__name__)`:

```python
    # Optional OTel export of decision traces (no-op unless the OTLP endpoint
    # env vars are set; see core/otel_export.py).
    from core.otel_export import install as install_otel
    if install_otel():
        logger.info("OTel trace export enabled")
```

- `interfaces/api_interface.py` — right after its `logging.basicConfig(...)` line:

```python
from core.otel_export import install as _install_otel

_install_otel()  # no-op unless OTLP endpoint env vars are set
```

- `pyproject.toml` — add after the `[project.scripts]` table:

```toml
[project.optional-dependencies]
# OTel span export of decision traces (core/otel_export.py). Vendor-neutral
# OTLP: point OTEL_EXPORTER_OTLP_ENDPOINT at Langfuse/Phoenix/Jaeger/etc.
otel = [
    "opentelemetry-sdk>=1.30",
    "opentelemetry-exporter-otlp-proto-http>=1.30",
]
```

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/test_otel_install.py tests/test_otel_translation.py tests/test_turn_trace.py tests/test_api_chat_contract.py -q` (the last file confirms the api_interface import-time `_install_otel()` no-op is harmless; if that filename doesn't exist, substitute the tests that import `interfaces.api_interface` — `pytest tests -k "api" -q`).
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/otel_export.py main.py interfaces/api_interface.py pyproject.toml tests/test_otel_install.py
git commit -m "feat(otel): env-gated OTLP emitter (install/uninstall), entry-point wiring, otel extra"
```

---

### Task 4: RAG generation token/trace accounting

**Files:**
- Modify: `knowledge/knowledge_base.py` (`__init__`, `query_knowledge`)
- Modify: `knowledge/rag_system.py` (`retrieve_and_generate`, `_generate_response`)
- Modify: `core/knowledge_router.py` (`handle_knowledge_question`)
- Modify: `core/chatbot_tool_use.py:33`, `core/orchestrator.py:80` (KnowledgeBase default construction)
- Test: Create `tests/test_rag_accounting.py`

**Interfaces:**
- Consumes: Tier 1's `get_simple_completion(..., context_manager=)`; `KnowledgeRouter.context_manager`.
- Produces: `KnowledgeBase.__init__(self, llm_client=None)`; `KnowledgeBase.query_knowledge(self, query, domain=None, context_manager=None)`; `RAGSystem.retrieve_and_generate(..., context_manager=None)`; `RAGSystem._generate_response(self, query, retrieved_docs, context_manager=None)`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_rag_accounting.py`:

```python
from core.context_manager import ContextManager
from core.knowledge_router import KnowledgeRouter
from knowledge.rag_system import RAGSystem


def _cm():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    return cm


class AccountingFake:
    """Modern client: accepts context_manager and accounts tokens like the real one."""

    def __init__(self, reply="generated"):
        self.calls = []
        self.reply = reply

    def get_simple_completion(self, system_prompt, user_prompt, context_manager=None):
        self.calls.append(context_manager)
        if context_manager is not None:
            context_manager.update_token_usage(
                {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        return self.reply


_DOCS = [{"document": "Ricker wavelets are zero-phase.", "score": 0.8,
          "metadata": {"topic": "ricker", "subtopic": "overview"}}]


def test_generate_response_threads_context_manager():
    fake = AccountingFake()
    rag = object.__new__(RAGSystem)  # skip heavy __init__ (chroma vector db)
    rag.llm_client = fake
    cm = _cm()
    out = rag._generate_response("what is a ricker?", _DOCS, context_manager=cm)
    assert out == "generated"
    assert fake.calls == [cm]
    assert cm.get_token_usage()["total_tokens"] == 6


def test_generate_response_tolerates_legacy_two_arg_client():
    class Legacy:
        def get_simple_completion(self, s, u):
            return "legacy"

    rag = object.__new__(RAGSystem)
    rag.llm_client = Legacy()
    assert rag._generate_response("q", _DOCS, context_manager=_cm()) == "legacy"


def test_knowledge_base_injects_llm_client(monkeypatch):
    created = {}

    class FakeRAG:
        def __init__(self, llm_client=None):
            created["llm_client"] = llm_client

        def populate_knowledge_base(self, topics):
            pass

    monkeypatch.setattr("knowledge.knowledge_base.RAGSystem", FakeRAG)
    from knowledge.knowledge_base import KnowledgeBase
    sentinel = object()
    KnowledgeBase(llm_client=sentinel)
    assert created["llm_client"] is sentinel


def test_knowledge_base_query_passes_context_manager(monkeypatch):
    seen = {}

    class FakeRAG:
        def __init__(self, llm_client=None):
            pass

        def populate_knowledge_base(self, topics):
            pass

        def retrieve_and_generate(self, query, domain=None, context_manager=None):
            seen["cm"] = context_manager
            return {"rag_type": "no_results", "generated_response": "",
                    "retrieved_documents": [], "total_retrieved": 0}

    monkeypatch.setattr("knowledge.knowledge_base.RAGSystem", FakeRAG)
    from knowledge.knowledge_base import KnowledgeBase
    cm = _cm()
    KnowledgeBase().query_knowledge("q", context_manager=cm)
    assert seen["cm"] is cm


def test_router_passes_context_manager_with_legacy_fallback():
    cm = _cm()

    class ModernKB:
        def query_knowledge(self, q, domain=None, context_manager=None):
            self.cm = context_manager
            return {"rag_type": "retrieve_and_generate", "generated_response": "ans",
                    "total_retrieved": 1, "retrieved_documents": [{"score": 0.5}]}

    kb = ModernKB()
    router = KnowledgeRouter(None, kb, context_manager=cm)
    assert "ans" in router.handle_knowledge_question("what is tuning?")
    assert kb.cm is cm

    class LegacyKB:
        def query_knowledge(self, q):
            return {"rag_type": "retrieve_and_generate", "generated_response": "old",
                    "total_retrieved": 0, "retrieved_documents": []}

    router2 = KnowledgeRouter(None, LegacyKB(), context_manager=_cm())
    assert "old" in router2.handle_knowledge_question("what is tuning?")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_rag_accounting.py -q`
Expected: failures — `_generate_response` rejects `context_manager`, `KnowledgeBase.__init__` rejects `llm_client`, router never passes the kwarg (ModernKB.cm stays unset → AttributeError or None).

- [ ] **Step 3: Implement**

`knowledge/knowledge_base.py`:
- `def __init__(self, llm_client=None):` with docstring line "llm_client: shared, token/trace-accounted LLM client; RAGSystem builds its own when None." and `self.rag_system = RAGSystem(llm_client=llm_client)`.
- `query_knowledge` becomes:

```python
    def query_knowledge(self, query: str, domain: Optional[str] = None,
                        context_manager: Any = None) -> Dict[str, Any]:
        """RAG query. context_manager (optional) receives token/trace
        accounting for the generation LLM call."""
        return self.rag_system.retrieve_and_generate(
            query, domain, context_manager=context_manager)
```

`knowledge/rag_system.py`:
- `retrieve_and_generate(self, query, domain=None, top_k=None, similarity_threshold=None, context_manager=None)`; find its internal `self._generate_response(query, retrieved_docs)` call and pass `context_manager=context_manager`.
- `_generate_response(self, query, retrieved_docs, context_manager=None)`; replace the line `response = self.llm_client.get_simple_completion(system_prompt, user_prompt)` with:

```python
            try:
                response = self.llm_client.get_simple_completion(
                    system_prompt, user_prompt, context_manager=context_manager)
            except TypeError:
                response = self.llm_client.get_simple_completion(
                    system_prompt, user_prompt)
```

`core/knowledge_router.py` — in `handle_knowledge_question`, replace `rag_response = self.knowledge_base.query_knowledge(user_input)` with:

```python
            try:
                rag_response = self.knowledge_base.query_knowledge(
                    user_input, context_manager=self.context_manager)
            except TypeError:
                rag_response = self.knowledge_base.query_knowledge(user_input)
```

(The `rag` emit_event lines immediately after stay exactly where they are.)

Bot wiring — one line each:
- `core/chatbot_tool_use.py:33`: `self.knowledge_base = knowledge_base or KnowledgeBase(llm_client=self.llm_client)`
- `core/orchestrator.py:80`: `self.knowledge_base = knowledge_base or KnowledgeBase(llm_client=self.llm_client)`

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/test_rag_accounting.py tests/test_knowledge_router.py tests/test_rag_no_results.py tests/test_knowledge_base.py tests/test_chatbot.py -q`
Expected: all pass (the TypeError fallbacks keep every legacy fake working; if `tests/test_knowledge_base.py` doesn't exist under that exact name, run `pytest tests -k "knowledge" -q`).

- [ ] **Step 5: Commit**

```bash
git add knowledge/knowledge_base.py knowledge/rag_system.py core/knowledge_router.py core/chatbot_tool_use.py core/orchestrator.py tests/test_rag_accounting.py
git commit -m "feat(trace): RAG generation tokens accounted — shared LLM client + context threading through KnowledgeBase/RAGSystem"
```

---

### Task 5: Full suite + docs

**Files:**
- Modify: `CLAUDE.md` ("Decision trace" section)
- Test: full suite

- [ ] **Step 1: Full suite**

Run: `pytest -q` with a ≥ 420s timeout.
Expected: everything green except the one pre-existing `test_tool_use.py::test_tool_use_pattern` stdin failure (602 passed at Tier-1 head; this branch adds ~17 tests). Investigate and fix any NEW failure; leave the pre-existing one.

- [ ] **Step 2: Update CLAUDE.md** — in the "Decision trace (agent observability, Tier 0+1)" section: change the heading to `## Decision trace (agent observability, Tier 0-2)`, replace the sentence beginning "`LLMClient.get_simple_completion(..., context_manager=)` accounts router-side tokens — but `knowledge/rag_system.py` still builds its own `LLMClient`..." with:

```markdown
`LLMClient.get_simple_completion(..., context_manager=)` accounts router-side AND
RAG-generation tokens (`KnowledgeBase(llm_client=...)` shares the bots' client;
`query_knowledge(..., context_manager=)` threads the session through — closed Tier-2 gap).
```

and append to the end of the section:

```markdown
**OTel export (Tier 2):** `core/turn_trace.py` exposes `register_trace_exporter(fn)`;
`core/otel_export.py` translates each turn record into GenAI-semconv spans
(`invoke_agent` root with `session.id`/`langfuse.session.id`; `chat` and `execute_tool`
children with timings reconstructed from `latency_ms`/`ms`; other events become root span
events) and ships them over OTLP. Enable by installing the extra
(`pip install -e ".[otel]"`) and setting `OTEL_EXPORTER_OTLP_ENDPOINT` (+
`OTEL_EXPORTER_OTLP_HEADERS` for auth; `OTEL_SERVICE_NAME` defaults to seismic-chatbot) —
e.g. a self-hosted Langfuse's `/api/public/otel` endpoint with a Basic-auth header, or
Phoenix/Jaeger. Unset ⇒ complete no-op with no OTel import. Prompt content stays out of
spans unless `OTEL_GENAI_CAPTURE_CONTENT=1`. `install()` builds a module-local
TracerProvider (never mutates the global). Tests: `tests/test_otel_translation.py` (pure,
no SDK), `test_otel_install.py` (importorskip-gated, InMemorySpanExporter).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: Tier-2 OTel export + RAG token accounting in decision-trace section"
```

- [ ] **Step 4: Report** — suite tally, then hand off per superpowers:finishing-a-development-branch.
