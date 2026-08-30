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

# Event fields that carry request-derived text (agentic-mode task briefs and
# discovery queries). Exported only when capture_content is enabled; error
# strings are deliberately NOT gated (standard OTel error data).
_CONTENT_EVENT_KEYS = ("brief", "query")


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
        ts = e.get("ts")
        if not isinstance(ts, (int, float)):
            ts = start_ts
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
            fields = {k: v for k, v in e.items() if k not in ("t", "ts")}
            if not capture_content:
                for key in _CONTENT_EVENT_KEYS:
                    fields.pop(key, None)
            attrs = _clean_attrs(fields)
            root["events"].append({"name": t or "event", "ts_ns": _ns(ts),
                                   "attributes": attrs})
            if t == "turn_error":
                root["status_error"] = str(e.get("error") or "turn error")

    root["start_ns"] = min(s["start_ns"] for s in spans)
    root["end_ns"] = max(s["end_ns"] for s in spans)
    return spans


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
    # SDK default shutdown_on_exit=True registers an atexit hook that calls
    # provider.shutdown(), draining the batch queue on normal interpreter
    # exit — do not "fix" the apparent missing flush, and do not disable it.
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
