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
