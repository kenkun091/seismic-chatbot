# Turn Transparency + Plot Provenance (Tier 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Curated per-turn summaries with drill-down in the Gradio UI, high-stakes flags (physics warnings, defaults, failures, budget) surfaced in the chat, physics-warning capture, and provenance sidecars next to every generated plot.

**Architecture:** Two new stdlib-only pure modules (`core/trace_summary.py`, `core/provenance.py`) consume the existing TurnTrace record; `ToolLoopRunner` gains warning capture and sidecar wiring at already-instrumented points; the Gradio layer renders summary/flags from the `"trace"` key it already receives. Bot contracts unchanged.

**Tech Stack:** Python 3.9.7, stdlib only; pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-turn-transparency-spec.md` (read first — it defines flag texts and the values-vs-names ruling).

## Global Constraints

- Python 3.9.7 — `from __future__ import annotations`; `typing.Optional[X]`, never `X | None`.
- `core/trace_summary.py` and `core/provenance.py` import stdlib only.
- `process_single_input` keeps returning exactly `{"reply", "images", "trace"}`; `ToolLoopRunner.run` keeps `{"reply", "images", "tools_used"}`; reply text is modified only in the Gradio layer.
- Sidecar writes and summary rendering may never raise out of a turn (guarded, WARNING-logged).
- Flag strings must match the spec verbatim (they are user-facing copy).
- Working dir: `/Users/kumono/Desktop/devs/dash-apps/wedge-model/geo-mcp/seismic_chatbot` (own git repo). Branch: `turn-transparency`, created from `otel-export` (Task 1 Step 0).
- Do NOT run the full suite until the final task; per-task runs use only the named files.

---

### Task 1: `core/trace_summary.py` — summarize + markdown

**Files:**
- Create: `core/trace_summary.py`
- Test: Create `tests/test_trace_summary.py`

**Interfaces:**
- Consumes: TurnTrace record shape (`{"session","turn","tools_used","events"}`).
- Produces: `summarize_trace(record) -> Dict` with exactly keys `headline` (str), `flags` (List[str]), `detail_lines` (List[str]); `format_trace_markdown(record) -> str` returning `"_No decision trace for this turn._"` for None/non-dict/empty-events input.

- [ ] **Step 0: Create the working branch**

```bash
cd /Users/kumono/Desktop/devs/dash-apps/wedge-model/geo-mcp/seismic_chatbot
git checkout otel-export && git checkout -b turn-transparency
```

- [ ] **Step 1: Write the failing tests** — create `tests/test_trace_summary.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_trace_summary.py -q`
Expected: `ModuleNotFoundError: No module named 'core.trace_summary'`.

- [ ] **Step 3: Implement** — create `core/trace_summary.py`:

```python
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
```

- [ ] **Step 4: Run to verify green** — `pytest tests/test_trace_summary.py -q`; expected 7 passed.

- [ ] **Step 5: Commit**

```bash
git add core/trace_summary.py tests/test_trace_summary.py
git commit -m "feat(transparency): trace_summary — curated headline, high-stakes flags, drill-down lines"
```

---

### Task 2: physics-warning capture in the tool loop

**Files:**
- Modify: `core/tool_loop.py` (imports + the timed `process_tool_call` call inside `run()`'s try block, currently around lines 483-484)
- Test: Create `tests/test_physics_warning_capture.py`

**Interfaces:**
- Consumes: `emit_event` (already imported in tool_loop).
- Produces: `physics_warning` events `{tool, category, message}` (message truncated to 300 chars); captured warnings re-logged via `logger.warning`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_physics_warning_capture.py`:

```python
import warnings

from core.context_manager import ContextManager
from core.tool_loop import ToolLoopRunner


class FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = FakeFunc(name, arguments)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def get_completion(self, *a, **k):
        return self._responses.pop(0)


class WarningToolManager:
    """Stub manager whose tool issues a physics-style warning."""
    specs = {}

    def process_tool_call(self, name, tool_input):
        warnings.warn("vp 9000.0 m/s outside typical range 300-8000")
        warnings.warn("possible aliasing above Nyquist")
        return {"value": 1}


def test_tool_warnings_become_physics_warning_events():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("test")
    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("warn_tool", '{}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]), WarningToolManager(), cm)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an escaped warning would fail the test
        out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["reply"] == "done"
    physics = [e for e in cm.trace.events if e["t"] == "physics_warning"]
    assert len(physics) == 2
    assert physics[0]["tool"] == "warn_tool"
    assert physics[0]["category"] == "UserWarning"
    assert "9000.0" in physics[0]["message"]
    assert "aliasing" in physics[1]["message"]


def test_message_truncated_to_300_chars():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("test")

    class LongWarnManager:
        specs = {}

        def process_tool_call(self, name, tool_input):
            warnings.warn("x" * 500)
            return {}

    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("t", '{}')], "usage": None},
        {"content": "<reply>ok</reply>", "tool_calls": None, "usage": None},
    ]), LongWarnManager(), cm)
    runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    physics = [e for e in cm.trace.events if e["t"] == "physics_warning"]
    assert len(physics[0]["message"]) == 300
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_physics_warning_capture.py -q`; expected: FAIL (first test errors because `simplefilter("error")` turns the escaped warning into an exception inside the loop's try → tool marked failed → reply "done" still returned but no physics events; the `len(physics) == 2` assert fails).

- [ ] **Step 3: Implement** in `core/tool_loop.py`

Add `import warnings` to the module imports (alphabetical, after `import time` if present, else with the stdlib block).

Replace (inside `run()`'s try block):

```python
                started = time.perf_counter()
                tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
```

with:

```python
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
                for w in caught:
                    message = str(w.message)[:300]
                    logger.warning(f"{tool_name}: {w.category.__name__}: {message}")
                    emit_event(self.context_manager, "physics_warning",
                               tool=tool_name, category=w.category.__name__,
                               message=message)
```

(Everything after — the `tool_call` emit, `tools_used.append`, etc. — stays untouched.)

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/test_physics_warning_capture.py tests/test_tool_loop_trace.py tests/test_wedge_correctness.py -q`
Expected: all pass (`test_wedge_correctness` exercises real warning-emitting paths — confirms capture doesn't break code that *asserts* warnings via `pytest.warns` outside the loop; those tests call tools directly, not through the loop).

- [ ] **Step 5: Commit**

```bash
git add core/tool_loop.py tests/test_physics_warning_capture.py
git commit -m "feat(transparency): capture tool warnings as physics_warning events (re-logged, 300-char cap)"
```

---

### Task 3: budget scope + run_task early-return events

**Files:**
- Modify: `core/tool_loop.py` (the `budget_exhausted` emit, currently ~line 531), `core/orchestrator.py` (the `budget_exhausted` emit ~line 152; `_run_task` early returns ~lines 193-199)
- Test: `tests/test_tool_loop_trace.py` (one assert), create `tests/test_run_task_events.py`

**Interfaces:**
- Produces: `budget_exhausted` events carry `scope="tool_loop"` / `scope="meta_loop"`; `_run_task` early returns emit `run_task` with `error`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tool_loop_trace.py::test_budget_exhaustion_is_traced`, extend the final assert block with:

```python
    assert budget[0]["scope"] == "tool_loop"
```

Create `tests/test_run_task_events.py`:

```python
from core.context_manager import ContextManager  # noqa: F401
from core.orchestrator import SeismicOrchestrator
from core.tool_manager import ToolManager


def _orchestrator():
    orch = SeismicOrchestrator(llm_client=object(), tool_manager=ToolManager(),
                               knowledge_base=object(), tool_index=object())
    orch.context_manager.trace.persist_dir = ""
    orch.context_manager.trace.begin_turn("test")
    return orch


def test_empty_tool_names_emits_run_task_error():
    orch = _orchestrator()
    out = orch._run_task("do something", [], [])
    assert "tool_names is empty" in out
    evt = [e for e in orch.context_manager.trace.events if e["t"] == "run_task"][0]
    assert evt["error"] == "tool_names empty"
    assert evt["tools_used"] == [] and evt["n_images"] == 0


def test_unknown_tool_names_emit_run_task_error():
    orch = _orchestrator()
    out = orch._run_task("do something", ["no_such_tool"], [])
    assert "Unknown tool name(s)" in out
    evt = [e for e in orch.context_manager.trace.events if e["t"] == "run_task"][0]
    assert evt["error"] == "unknown tools: no_such_tool"
    assert evt["tool_names"] == ["no_such_tool"]
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_run_task_events.py tests/test_tool_loop_trace.py::test_budget_exhaustion_is_traced -q`; expected: 3 failures (no scope field, no events on early returns).

- [ ] **Step 3: Implement**

`core/tool_loop.py`: the budget emit becomes
`emit_event(self.context_manager, "budget_exhausted", rounds=self.max_tool_rounds, scope="tool_loop")`.

`core/orchestrator.py`: the meta-loop budget emit becomes
`emit_event(self.context_manager, "budget_exhausted", rounds=MAX_ORCH_ROUNDS, scope="meta_loop")`,
and `_run_task`'s two early returns become:

```python
        if not isinstance(tool_names, list) or not tool_names:
            emit_event(self.context_manager, "run_task", brief=brief[:200],
                       tool_names=[], tools_used=[],
                       error="tool_names empty", n_images=0)
            return "tool_names is empty — call discover_tools first."
        unknown = [n for n in tool_names if n not in REGISTRY_BY_NAME]
        if unknown:
            emit_event(self.context_manager, "run_task", brief=brief[:200],
                       tool_names=list(tool_names), tools_used=[],
                       error=f"unknown tools: {', '.join(unknown)}", n_images=0)
            return (f"Unknown tool name(s): {', '.join(unknown)}. "
                    f"Use names exactly as returned by discover_tools.")
```

- [ ] **Step 4: Run to verify green** — `pytest tests/test_run_task_events.py tests/test_tool_loop_trace.py tests/test_orchestrator_trace.py tests/test_orchestrator.py -q`; expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/tool_loop.py core/orchestrator.py tests/test_tool_loop_trace.py tests/test_run_task_events.py
git commit -m "feat(trace): budget_exhausted scope field; run_task early-return events"
```

---

### Task 4: provenance sidecars

**Files:**
- Create: `core/provenance.py`
- Modify: `core/tool_loop.py` (import; new `_write_provenance` method; the two `harvest_images` call sites in `run()`)
- Test: Create `tests/test_provenance.py`

**Interfaces:**
- Consumes: `ToolLoopRunner.compact_value`, `self.context_manager.trace` (getattr-guarded), `AUTO_PLOT`.
- Produces: `core.provenance.write_plot_provenance(image_path: str, payload: Dict) -> Optional[str]` (sidecar path or None); sidecar at `<image_path>.prov.json` with keys `artifact`, `generator`, `created` + payload (`session`, `turn`, `tool`, `parameters`, optional `compute_tool`/`compute_parameters`).

- [ ] **Step 1: Write the failing tests** — create `tests/test_provenance.py`:

```python
import json

from core.provenance import write_plot_provenance


def test_write_sidecar_next_to_artifact(tmp_path):
    png = tmp_path / "plot.png"
    png.write_bytes(b"fake")
    sidecar = write_plot_provenance(str(png), {"tool": "plot_ricker",
                                               "session": "s1", "turn": 2,
                                               "parameters": {"frequency": 30}})
    assert sidecar == str(png) + ".prov.json"
    data = json.loads((tmp_path / "plot.png.prov.json").read_text())
    assert data["artifact"] == "plot.png"
    assert data["generator"] == "seismic-chatbot"
    assert data["tool"] == "plot_ricker"
    assert data["parameters"] == {"frequency": 30}
    assert "created" in data


def test_write_failure_is_swallowed():
    assert write_plot_provenance("/dev/null/nope/plot.png", {"tool": "x"}) is None


def test_loop_writes_sidecar_for_auto_plotted_ricker():
    from core.context_manager import ContextManager
    from core.tool_loop import ToolLoopRunner
    from core.tool_manager import ToolManager

    class FakeFunc:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, name, arguments, call_id="call_1"):
            self.id = call_id
            self.function = FakeFunc(name, arguments)

    class FakeLLM:
        def __init__(self, responses):
            self._responses = list(responses)

        def get_completion(self, *a, **k):
            return self._responses.pop(0)

    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.session_id = "prov-session"
    cm.trace.begin_turn("30 Hz ricker")
    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker",
                                                    '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]), ToolManager(), cm)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["images"], "auto-plot should have produced a png"
    sidecar_path = out["images"][0] + ".prov.json"
    data = json.loads(open(sidecar_path).read())
    assert data["session"] == "prov-session"
    assert data["turn"] == 1
    assert data["compute_tool"] == "make_ricker"
    assert data["compute_parameters"]["frequency"] == 30
    assert data["tool"] == "plot_ricker"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_provenance.py -q`; expected `ModuleNotFoundError: No module named 'core.provenance'`.

- [ ] **Step 3: Implement**

Create `core/provenance.py`:

```python
"""Plot provenance sidecars (Tier 3): <plot>.png.prov.json next to each
generated figure — session/turn, the producing tool, its (compacted)
parameter values, and the compute tool behind an auto-chained plot.

Local reproducibility metadata: unlike trace events (names-not-values, may be
exported), sidecars deliberately carry parameter VALUES; they live next to
the artifact and are never exported anywhere.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GENERATOR = "seismic-chatbot"


def write_plot_provenance(image_path: str, payload: Dict[str, Any]) -> Optional[str]:
    """Write <image_path>.prov.json; returns the sidecar path, or None on failure."""
    sidecar = f"{image_path}.prov.json"
    record: Dict[str, Any] = {
        "artifact": os.path.basename(image_path),
        "generator": GENERATOR,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    record.update(payload)
    try:
        with open(sidecar, "w") as f:
            json.dump(record, f, default=str, indent=2)
        return sidecar
    except Exception as e:
        logger.warning(f"provenance sidecar failed for {image_path}: {e}")
        return None
```

In `core/tool_loop.py`: add `from core.provenance import write_plot_provenance` to the imports; add the method after `harvest_images`:

```python
    def _write_provenance(self, paths: List[str], tool_name: str,
                          tool_input: Dict[str, Any],
                          compute_tool: Optional[str] = None,
                          compute_input: Optional[Dict[str, Any]] = None) -> None:
        """Sidecar every newly harvested plot with what produced it."""
        if not paths:
            return
        trace = getattr(self.context_manager, "trace", None)
        payload: Dict[str, Any] = {
            "session": getattr(trace, "session_id", None),
            "turn": getattr(trace, "turn", None),
            "tool": tool_name,
            "parameters": self.compact_value(tool_input),
        }
        if compute_tool:
            payload["compute_tool"] = compute_tool
            payload["compute_parameters"] = self.compact_value(compute_input or {})
        for path in paths:
            write_plot_provenance(path, payload)
```

In `run()`'s try block, replace

```python
                self.harvest_images(tool_result, collected_images)
```

with

```python
                before_direct = len(collected_images)
                self.harvest_images(tool_result, collected_images)
                self._write_provenance(collected_images[before_direct:],
                                       tool_name, tool_input)
```

and replace

```python
                if chained_result:
                    self.harvest_images(chained_result, collected_images)
```

with

```python
                if chained_result:
                    before_chained = len(collected_images)
                    self.harvest_images(chained_result, collected_images)
                    self._write_provenance(collected_images[before_chained:],
                                           AUTO_PLOT.get(tool_name) or "auto_plot",
                                           {}, compute_tool=tool_name,
                                           compute_input=tool_input)
```

(the `emit_event(..., "auto_plot", ...)` lines stay exactly where they are).

- [ ] **Step 4: Run to verify green** — `pytest tests/test_provenance.py tests/test_tool_loop_trace.py tests/test_chatbot_synthetic.py -q`; expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/provenance.py core/tool_loop.py tests/test_provenance.py
git commit -m "feat(provenance): .prov.json sidecars for every harvested plot (session/turn/tool/params, compute chain)"
```

---

### Task 5: Gradio surfacing — flags in the bubble, drill-down panel

**Files:**
- Modify: `interfaces/gradio_interface.py` (`append_bot_response`, `respond`, layout, event-binding outputs)
- Test: Create `tests/test_gradio_trace_panel.py`; authorized updates to any test unpacking `respond`'s 5-tuple

**Interfaces:**
- Consumes: Task 1's `summarize_trace` / `format_trace_markdown`.
- Produces: `respond` returns 6 outputs `(msg, photo, chat_history, token_str, trace_md, session_bot)`; a `gr.Accordion("🔍 Decision trace (last turn)")` holding `trace_display = gr.Markdown(...)`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_gradio_trace_panel.py`:

```python
from interfaces.gradio_interface import append_bot_response

_TRACE = {"session": "s", "turn": 1, "tools_used": ["make_ricker"], "events": [
    {"t": "turn_start", "ts": 1.0, "input": "x"},
    {"t": "physics_warning", "ts": 1.1, "tool": "make_ricker",
     "category": "UserWarning", "message": "vp outside typical range"},
]}


def test_flags_are_appended_to_the_bubble():
    history = append_bot_response([["hi", None]],
                                  {"reply": "done", "images": [], "trace": _TRACE})
    assert history[-1][1].startswith("done")
    assert "⚠️ Physics: vp outside typical range" in history[-1][1]


def test_no_flags_leaves_reply_untouched():
    quiet = {"session": "s", "turn": 1, "tools_used": [], "events": [
        {"t": "turn_start", "ts": 1.0, "input": "x"},
        {"t": "intent", "ts": 1.1, "verdict": "KNOWLEDGE", "via": "llm"},
    ]}
    history = append_bot_response([["hi", None]],
                                  {"reply": "done", "images": [], "trace": quiet})
    assert history[-1][1] == "done"


def test_traceless_response_unchanged():
    history = append_bot_response([["hi", None]], {"reply": "done", "images": []})
    assert history[-1][1] == "done"


def test_plain_string_response_still_renders():
    history = append_bot_response([["hi", None]], "plain")
    assert history[-1][1] == "plain"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_gradio_trace_panel.py -q`; expected: first test FAILS (no flags appended).

- [ ] **Step 3: Implement** in `interfaces/gradio_interface.py`

Add to imports: `from core.trace_summary import format_trace_markdown, summarize_trace`.

In `append_bot_response`, inside the dict branch, replace:

```python
    if isinstance(response, dict) and "reply" in response:
        chat_history[-1][1] = response.get("reply") or ""
        for path in response.get("images") or []:
            chat_history.append([None, (path,)])
```

with:

```python
    if isinstance(response, dict) and "reply" in response:
        reply = response.get("reply") or ""
        trace = response.get("trace")
        if isinstance(trace, dict) and trace.get("events"):
            flags = summarize_trace(trace)["flags"]
            if flags:
                reply = (reply + "\n\n" + "\n".join(flags)).strip()
        chat_history[-1][1] = reply
        for path in response.get("images") or []:
            chat_history.append([None, (path,)])
```

In `respond`'s success path, after `token_str = format_status(token_usage, trace)` add:

```python
            trace_md = format_trace_markdown(trace)
            return "", None, chat_history, token_str, trace_md, session_bot
```

(replacing the old 5-tuple return), and the error path becomes:

```python
        except Exception as e:
            chat_history[-1][1] = f"Error processing request: {str(e)}"
            token_str = format_status(session_bot.context_manager.get_token_usage())
            return "", None, chat_history, token_str, format_trace_markdown(None), session_bot
```

In the layout, right after the token-usage `gr.HTML` style block, add:

```python
                with gr.Accordion("🔍 Decision trace (last turn)", open=False):
                    trace_display = gr.Markdown("_No decision trace yet._")
```

Update both event bindings to:

```python
        submit.click(respond, [msg, photo, chat_display, session_state],
                     [msg, photo, chat_display, token_usage_display, trace_display, session_state])
        msg.submit(respond, [msg, photo, chat_display, session_state],
                   [msg, photo, chat_display, token_usage_display, trace_display, session_state])
```

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/test_gradio_trace_panel.py tests/test_gradio_response_format.py tests/test_gradio_upload.py tests/test_main_modes.py -q`
Expected: all pass. If any existing test unpacks `respond`'s return as a 5-tuple, update that unpacking to the 6-tuple (`*_, trace_md, session_bot` style or explicit) — the one authorized test-contract change; do not weaken any assertion.

- [ ] **Step 5: Commit**

```bash
git add interfaces/gradio_interface.py tests/test_gradio_trace_panel.py
git commit -m "feat(transparency): decision-trace panel + high-stakes flags in the chat bubble"
```

(add any updated existing test files to the same commit.)

---

### Task 6: Full suite + docs

**Files:**
- Modify: `CLAUDE.md`
- Test: full suite

- [ ] **Step 1: Full suite** — `pytest -q` with ≥ 420s timeout. Expected: green except the pre-existing `test_tool_use.py::test_tool_use_pattern` stdin failure (624 passed at the Tier-2 head; this branch adds ~15 tests). Fix any NEW failure that traces to this branch; leave the pre-existing one.

- [ ] **Step 2: Update CLAUDE.md** — change the section heading `## Decision trace (agent observability, Tier 0-2)` to `## Decision trace (agent observability, Tier 0-3)` and append to the end of that section:

```markdown
**Turn transparency (Tier 3):** `core/trace_summary.py::summarize_trace` renders a turn
record into a curated headline + high-stakes flags (physics warnings, failed tools,
budget-forced completions, missing auto-plots, auto-filled defaults) + drill-down lines;
the Gradio UI appends flags to the chat bubble and shows
`format_trace_markdown` in a collapsed "Decision trace (last turn)" accordion (`respond`
returns 6 outputs now). Tool warnings are captured in the loop
(`warnings.catch_warnings(record=True)` around `process_tool_call`) as `physics_warning`
events — re-logged at WARNING, message capped at 300 chars, exported in OTel spans
ungated (diagnostic text, same ruling as error strings). `budget_exhausted` carries
`scope` (tool_loop/meta_loop); `_run_task` early returns emit `run_task` events with
`error`. Every harvested plot gets a `<plot>.png.prov.json` sidecar
(`core/provenance.py`: session/turn, producing tool, compacted parameter VALUES, and the
compute tool behind an auto-chained plot — local reproducibility metadata, deliberately
values-not-names, never exported; `run_sweep`'s PNG cleanup leaves sidecars behind in the
tmpdir, and the wedge CSV export gets no sidecar — known limitations). Tests:
`tests/test_trace_summary.py`, `test_physics_warning_capture.py`,
`test_run_task_events.py`, `test_provenance.py`, `test_gradio_trace_panel.py`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: Tier-3 turn transparency + plot provenance in decision-trace section"
```

- [ ] **Step 4: Report** — suite tally; hand off per superpowers:finishing-a-development-branch.
