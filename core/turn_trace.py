"""Per-turn decision-trace recorder (Tier 1 of the observability roadmap).

One TraceRecorder per session, hanging off ContextManager. Events are plain
JSON-safe dicts recording *decisions* (intent verdicts, tool selection, arg
provenance, auto-plot outcomes, failures, budgets, discovery scores, per-call
tokens) — never full prompts or parameter values. end_turn() appends one JSONL
line per turn to <persist_dir>/<session_id>.jsonl and returns the record,
which process_single_input surfaces as the additive "trace" key.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_INPUT_TRUNCATE = 200


def usage_dict(usage: Any) -> Dict[str, int]:
    """Tolerant token extraction from a dict or a CompletionUsage-like object."""
    if not usage:
        return {}
    out: Dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(usage, "get"):
            value = usage.get(key, None)
        else:
            value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


def emit_event(context_manager: Any, t: str, **fields: Any) -> None:
    """Emit onto context_manager.trace when present; safe no-op otherwise."""
    recorder = getattr(context_manager, "trace", None)
    if recorder is not None:
        recorder.emit(t, **fields)


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


class TraceRecorder:
    def __init__(self, session_id: Optional[str] = None,
                 persist_dir: Optional[str] = None) -> None:
        if persist_dir is None:
            from config.settings import SEISMIC_TRACE_DIR
            persist_dir = SEISMIC_TRACE_DIR
        self.session_id = session_id or uuid.uuid4().hex
        self.persist_dir = persist_dir
        self.turn = 0
        self.events: List[Dict[str, Any]] = []

    def begin_turn(self, user_input: str) -> None:
        self.turn += 1
        self.events = []
        self.emit("turn_start", input=str(user_input)[:_INPUT_TRUNCATE])

    def emit(self, t: str, **fields: Any) -> None:
        event: Dict[str, Any] = {"t": t, "ts": round(time.time(), 3)}
        event.update(fields)
        self.events.append(event)

    def end_turn(self) -> Dict[str, Any]:
        record = {
            "session": self.session_id,
            "turn": self.turn,
            "tools_used": [e["tool"] for e in self.events
                           if e.get("t") == "tool_call" and e.get("ok")],
            "events": self.events,
        }
        self._persist(record)
        for exporter in list(_TRACE_EXPORTERS):
            try:
                exporter(record)
            except Exception as e:
                logger.warning(f"trace exporter failed: {e}")
        return record

    def _persist(self, record: Dict[str, Any]) -> None:
        if not self.persist_dir:
            return
        try:
            os.makedirs(self.persist_dir, mode=0o700, exist_ok=True)
            path = os.path.join(self.persist_dir, f"{self.session_id}.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"trace persist failed: {e}")
