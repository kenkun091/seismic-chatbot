"""Human-readable turn summaries from TurnTrace records (Tier 3).

Pure and stdlib-only. Consumed by the Gradio layer (headline + drill-down
panel; high-stakes flags appended to the chat bubble) and usable by any API
client holding a ChatResponse.trace. Curated summary over raw event dump:
progressive disclosure, never chain-of-thought.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _detail_line(e: Dict[str, Any]) -> str:
    t = e.get("t")
    if t == "intent":
        return f"intent: {e.get('verdict')} (via {e.get('via')})"
    if t == "rag":
        return f"rag: {e.get('retrieved')} doc(s), scores {e.get('scores')}"
    if t == "discover":
        hits = e.get("hits") or []
        listed = ", ".join(f"{h[0]} ({h[1]})" for h in hits
                           if isinstance(h, (list, tuple)) and len(h) == 2)
        return f"discover: {listed or 'no hits'}"
    if t == "run_task":
        line = f"run_task: tools {e.get('tool_names')} → used {e.get('tools_used')}"
        if e.get("error"):
            line += f" — ERROR: {e['error']}"
        return line
    if t == "llm":
        model = e.get("model") or "llm"
        return (f"llm: {model}, {e.get('total_tokens') or 0} tokens, "
                f"{e.get('latency_ms') or 0} ms")
    if t == "tool_call":
        if not e.get("ok", True):
            return f"tool FAILED: {e.get('tool')} — {e.get('error')}"
        extras = []
        if e.get("defaults_filled"):
            extras.append(f"defaults: {', '.join(e['defaults_filled'])}")
        if e.get("injected"):
            extras.append(f"from session: {', '.join(e['injected'])}")
        if e.get("overridden"):
            extras.append(f"overridden: {', '.join(e['overridden'])}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        return f"tool: {e.get('tool')}, {e.get('ms') or 0} ms{suffix}"
    if t == "auto_plot":
        if e.get("fired"):
            return f"plot: {e.get('plot')} auto-generated after {e.get('compute')}"
        return f"plot: {e.get('plot')} SKIPPED after {e.get('compute')}"
    if t == "parallel_calls_dropped":
        return f"dropped parallel tool calls: {', '.join(e.get('dropped') or [])}"
    if t == "budget_exhausted":
        return (f"budget exhausted ({e.get('scope') or 'tool loop'}, "
                f"{e.get('rounds')} rounds)")
    if t == "physics_warning":
        return f"physics warning [{e.get('tool')}]: {e.get('message')}"
    if t == "turn_error":
        return f"turn error: {e.get('error')}"
    fields = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "ts"))
    return f"{t}: {fields}"


def summarize_trace(record: Dict[str, Any]) -> Dict[str, Any]:
    """Curated view of one turn record: headline, high-stakes flags, details."""
    events = record.get("events") or []
    tools = record.get("tools_used") or []
    intent = next((e for e in events if e.get("t") == "intent"), None)
    llm_events = [e for e in events if e.get("t") == "llm"]
    tokens = sum(e.get("total_tokens") or 0 for e in llm_events)
    ts_values = [e["ts"] for e in events if isinstance(e.get("ts"), (int, float))]
    duration_s = (round(max(ts_values) - min(ts_values), 1)
                  if len(ts_values) >= 2 else 0.0)
    fired_plots = [e for e in events
                   if e.get("t") == "auto_plot" and e.get("fired")]

    parts: List[str] = []
    if intent is not None:
        route = ("Answered from knowledge base"
                 if intent.get("verdict") == "KNOWLEDGE" else "Routed to tools")
        parts.append(f"{route} (intent via {intent.get('via')})")
    if tools:
        parts.append("ran " + " → ".join(tools))
    if fired_plots:
        parts.append(f"{len(fired_plots)} plot(s) auto-generated")
    parts.append(f"{len(llm_events)} LLM call(s), {tokens} tokens, {duration_s}s")

    flags: List[str] = []
    for e in events:
        t = e.get("t")
        if t == "physics_warning":
            flags.append(f"⚠️ Physics: {e.get('message')}")
        elif t == "tool_call" and not e.get("ok", True):
            flags.append(f"⚠️ Tool failed: {e.get('tool')} — {e.get('error')}")
        elif t == "budget_exhausted":
            flags.append("⚠️ Reasoning budget exhausted — the answer was "
                         "completed without further tool use")
        elif t == "auto_plot" and not e.get("fired", True):
            flags.append(f"⚠️ Expected plot {e.get('plot')} was not generated "
                         f"after {e.get('compute')}")
        elif t == "turn_error":
            flags.append(f"⚠️ Turn failed: {e.get('error')}")
        elif t == "tool_call" and e.get("defaults_filled"):
            flags.append(f"ℹ️ {e.get('tool')}: defaults used for "
                         f"{', '.join(e['defaults_filled'])}")

    return {"headline": " · ".join(parts), "flags": flags,
            "detail_lines": [_detail_line(e) for e in events
                             if e.get("t") != "turn_start"]}


def format_trace_markdown(record: Any) -> str:
    """Markdown block for the UI drill-down panel."""
    if not isinstance(record, dict) or not record.get("events"):
        return "_No decision trace for this turn._"
    s = summarize_trace(record)
    lines = [f"**{s['headline']}**"]
    if s["flags"]:
        lines.append("")
        lines.extend(s["flags"])
    if s["detail_lines"]:
        lines.append("")
        lines.extend(f"- {line}" for line in s["detail_lines"])
    return "\n".join(lines)
