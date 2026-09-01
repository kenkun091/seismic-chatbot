# Turn Transparency + Plot Provenance (Tier 3) Spec

Source: "Two Kinds of Trace" report §3 Tier 3
(https://claude.ai/code/artifact/67545329-a132-4b32-941b-920fad689e67), building on Tier 0-2
(PRs #3/#4; specs `2026-08-30-agent-decision-trace-spec.md`, `2026-08-30-otel-trace-export-spec.md`).
Out of scope: replay/evals (Tier 4), CSV-export sidecars (the wedge `export_path` CSV is
written inside the tool where the loop can't see it — noted as open).

## Goal

1. **User-facing decision transparency** (progressive disclosure, per the agentic-UX
   research): a curated one-line turn summary with drill-down details in the Gradio UI,
   and the four high-stakes classes ALWAYS surfaced in the chat itself — physics warnings,
   auto-filled defaults, tool failures, budget-forced completions.
2. **Physics warnings finally reach the user**: `tools/physics_guards.py` warnings
   currently die on stderr (acknowledged gap since Tier 0 planning). Capture them during
   tool execution as `physics_warning` events.
3. **Plot provenance sidecars**: every generated figure gets `<plot>.png.prov.json`
   recording session/turn, the producing tool and its (compacted) parameter **values**,
   and the compute tool behind an auto-chained plot — lightweight PROV for scientific
   defensibility.
4. Two parked event enrichments the drill-down needs: `budget_exhausted` gains
   `scope` ("tool_loop"/"meta_loop"); orchestrator `_run_task` early returns (empty /
   unknown `tool_names`) emit a `run_task` event with an `error` before returning.

## New events / event changes

- `physics_warning` — emitted by `ToolLoopRunner.run`: tool execution
  (`process_tool_call` only, not the auto-plot chain) runs inside
  `warnings.catch_warnings(record=True)` + `simplefilter("always")`; each captured
  warning is re-logged at WARNING (capture suppresses normal propagation) and emitted as
  `{tool, category, message[:300]}`. Any warning category is captured (physics_guards is
  the dominant source; numpy/matplotlib warnings are equally worth surfacing). Warning
  messages are diagnostic text (may embed parameter values) and are exported in OTel
  spans ungated — same ruling as error strings.
- `budget_exhausted` — adds `scope="tool_loop"` (tool_loop.py) / `scope="meta_loop"`
  (orchestrator.py).
- `run_task` — the two early-return branches in `_run_task` emit the event with
  `tools_used=[]`, `n_images=0`, and `error` ("tool_names empty" / "unknown tools: …").

## `core/trace_summary.py` (new, stdlib-only, pure)

- `summarize_trace(record) -> {"headline": str, "flags": List[str], "detail_lines": List[str]}`
  - **headline**: routing (from `intent`: "Answered from knowledge base"/"Routed to
    tools" + via), tool chain (`tools_used` joined with →), fired auto-plot count, and
    "{n} LLM call(s), {tokens} tokens, {duration}s" (tokens summed over `llm` events,
    duration = max−min event ts).
  - **flags** (the always-surfaced classes, in event order): `physics_warning` → "⚠️
    Physics: {message}"; failed `tool_call` → "⚠️ Tool failed: {tool} — {error}";
    `budget_exhausted` → "⚠️ Reasoning budget exhausted — the answer was completed
    without further tool use"; unfired `auto_plot` → "⚠️ Expected plot {plot} was not
    generated after {compute}"; `turn_error` → "⚠️ Turn failed: {error}"; successful
    `tool_call` with `defaults_filled` → "ℹ️ {tool}: defaults used for {names}".
  - **detail_lines**: one human-readable line per event (skip `turn_start`), with an
    unknown-event fallback line.
- `format_trace_markdown(record) -> str` — bold headline, flags block, `- ` bulleted
  details; `"_No decision trace for this turn._"` for None/empty.

## UI surfacing (`interfaces/gradio_interface.py`)

- `append_bot_response`: when the response dict carries a trace with events and
  `summarize_trace` yields flags, append `"\n\n" + "\n".join(flags)` to the reply bubble
  (chat contract untouched otherwise; responses without traces render exactly as before).
- New collapsed accordion "🔍 Decision trace (last turn)" holding a `gr.Markdown`
  (`trace_display`), below the token-usage display. `respond` returns a sixth output
  (`format_trace_markdown(trace)`; the error path returns `format_trace_markdown(None)`),
  and both event bindings add `trace_display` to their outputs (position 5, before
  `session_state`).
- API: unchanged — `ChatResponse.trace` already carries the record; clients can call the
  same pure functions.

## Provenance sidecars (`core/provenance.py` new + `core/tool_loop.py` wiring)

- `write_plot_provenance(image_path, payload) -> Optional[str]` writes
  `<image_path>.prov.json` containing `{artifact: basename, generator:
  "seismic-chatbot", created: iso-utc}` merged with `payload`; `json.dump(...,
  default=str, indent=2)`; failures swallowed to a WARNING, returns sidecar path or None.
- `ToolLoopRunner` gains `_write_provenance(paths, tool_name, tool_input,
  compute_tool=None, compute_input=None)` building the payload: `session`/`turn` from
  `getattr(self.context_manager, "trace", None)` (getattr-guarded), `tool`, `parameters =
  self.compact_value(tool_input)`, plus `compute_tool`/`compute_parameters` when chained.
- In `run()`: newly harvested paths are diffed by list length before/after each
  `harvest_images` call — direct tool images get `(tool_name, tool_input)`; chained plot
  images get `(AUTO_PLOT[tool_name], {}, compute_tool=tool_name, compute_input=tool_input)`
  (the plot tool's own inputs are context-built inside the chain; the compute parameters
  are the scientifically meaningful ones).
- **Values-not-names is deliberate here**: sidecars are local reproducibility metadata
  living next to the artifact, never exported; parameters are compacted via the loop's
  existing `compact_value` (arrays → summary strings) so files stay small and JSON-safe.
- Known limitation (documented): run_sweep's per-cell plots never get sidecars (recipes
  run via the WorkflowEngine, not the tool loop). CSV exports get no sidecar
  (written inside the tool).

## Constraints

- Python 3.9.7; `from __future__ import annotations`; `typing.Optional`, never `X | None`.
- `core/trace_summary.py` and `core/provenance.py` are stdlib-only.
- No change to `process_single_input`'s `{"reply", "images", "trace"}` contract or to
  `ToolLoopRunner.run`'s return keys; the reply text is modified only in the Gradio
  layer (flags), never by the bots.
- Sidecar/summary failures may never break a turn (guarded, logged).
- Existing tests asserting event-kind inclusion keep passing (physics_warning events may
  now additionally appear in real-tool traces); tests unpacking `respond`'s 5-tuple are
  updated to the 6-tuple — the one authorized test-contract change.
