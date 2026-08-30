# Agent Decision Trace — Tier 0 + Tier 1 Spec

Source research: "Two Kinds of Trace" report (artifact
https://claude.ai/code/artifact/67545329-a132-4b32-941b-920fad689e67), §2 audit + §3 roadmap.
This spec covers **Tier 0 (stop destroying evidence)** and **Tier 1 (structured TurnTrace)** only.
OTel export, UI accordion, provenance sidecars, and replay are explicitly out of scope.

## Problem

Both request flows (classic `SeismicChatBotToolUse` and agentic `SeismicOrchestrator`) make
~29 decisions per turn that are invisible: intent verdicts, RAG/discovery similarity scores,
which tool args were LLM-supplied vs default-filled vs context-injected, silent auto-plot
bailouts, swallowed tool failures, dropped parallel tool calls, budget-forced completions,
and untracked router LLM calls. Much of the evidence is already computed and discarded
(`TaskResult`, `tools_used`, `ToolCard.score`, per-doc RAG scores).

## Tier 0 — logging hardening

1. `LOG_LEVEL` becomes env-overridable (`os.environ.get("LOG_LEVEL", "INFO")`).
2. `interfaces/gradio_interface.py` and `interfaces/api_interface.py` get module-level
   `logging.basicConfig` (no-op when root already configured by `main.py`).
3. Promote to INFO: resolved-params line (`tool_manager.py:54`), retrieval-count line
   (`vector_db.py:207`).
4. `exc_info=True` on the top-level error logs: `tool_loop.py` tool-failure and chaining
   catch, `chatbot_tool_use.py` / `orchestrator.py` `process_single_input` catch,
   `executor_agent.py` executor catch.
5. New WARNING logs: parallel tool calls dropped, auto-plot skipped after a compute tool
   with a registered partner, tool-round / orchestrator-round budget exhausted.
6. Intent routing logged at INFO **in both modes** with which branch decided
   (llm / keyword_fallback / image_shortcut).

## Tier 1 — TurnTrace

New stdlib-only module `core/turn_trace.py`:

- `TraceRecorder(session_id=None, persist_dir=None)` — per-session event accumulator.
  `begin_turn(user_input)` / `emit(t, **fields)` / `end_turn() -> dict`. `end_turn`
  returns `{"session", "turn", "tools_used", "events"}` and appends it as one JSONL line
  to `<persist_dir>/<session_id>.jsonl` (write failures are swallowed with a WARNING).
  `tools_used` = tools from `tool_call` events with `ok: true`, in order.
- `emit_event(context_manager, t, **fields)` — emits onto `context_manager.trace` when
  present (safe no-op otherwise).
- `usage_dict(usage)` — tolerant `{prompt_tokens, completion_tokens, total_tokens}`
  extraction from dict or `CompletionUsage`.
- `SEISMIC_TRACE_DIR` setting (default `<tmpdir>/seismic_traces`), same convention as
  `SEISMIC_EXPORT_DIR`.

`ContextManager` gains `self.trace = TraceRecorder()`; both bots stamp
`self.context_manager.trace.session_id = self.session_id`.

### Event vocabulary (all JSON-serializable; user input truncated to 200 chars)

| t | emitted from | fields |
|---|---|---|
| `turn_start` | `begin_turn` | `input` (truncated) |
| `intent` | `KnowledgeRouter.classify` | `verdict` ("KNOWLEDGE"/"TOOL"), `via` ("llm"/"keyword_fallback"/"image_shortcut") |
| `rag` | `handle_knowledge_question` | `rag_type`, `retrieved`, `scores` (per-doc, rounded) |
| `llm` | tool loop, meta loop, `get_simple_completion` | `model`, `latency_ms`, `tool_call` (bool, loop only), `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `tool_call` | `ToolLoopRunner.run` | `tool`, `ok`, `ms`, `injected`, `overridden`, `defaults_filled`; on failure `ok: false`, `error` |
| `parallel_calls_dropped` | `ToolLoopRunner.run` | `dropped` (names) |
| `auto_plot` | `ToolLoopRunner.run` | `compute`, `plot`, `fired` (bool) |
| `budget_exhausted` | tool loop / meta loop | `rounds` |
| `discover` | `SeismicOrchestrator._discover` | `query`, `hits` (`[[name, score], ...]`) |
| `run_task` | `SeismicOrchestrator._run_task` | `brief` (truncated), `tool_names`, `tools_used`, `error`, `n_images` |
| `turn_error` | `process_single_input` catch | `error` |

### Plumbing fixes bundled in

- `LLMClient.get_completion` result gains `"model"` and `"latency_ms"`.
- `LLMClient.get_simple_completion` gains optional `context_manager=None`; when given it
  updates token usage and emits an `llm` event — closing the KnowledgeRouter token blind
  spot **without changing the FakeLLMClient contract** (fakes lack `get_simple_completion`,
  so classification still falls back to keywords in tests; the router tolerates legacy
  2-arg signatures via `TypeError` fallback).
- `KnowledgeRouter` gains optional `context_manager=None`; `classify()` returns
  `{"is_knowledge", "via"}` and `is_knowledge_question` delegates to it.
- Classic mode threads `tools_used` through `_handle_tool_request` instead of dropping it.
- `process_single_input` (both bots) returns `{"reply", "images", "trace"}` — additive;
  both existing consumers key-guard on `"reply"` and ignore unknown keys.
- API `ChatResponse` gains `trace: Optional[dict] = None`.
- Gradio status line appends `| Tools: a → b` when tools ran, and the error path keeps
  showing session token totals instead of blanking.

### Known-remaining gaps (documented, not fixed here)

`knowledge/rag_system.py` builds its own independent `LLMClient`, so RAG *generation*
tokens remain untracked (router-level calls are now tracked). The legacy `chat()` REPL and
`core/chatbot.py` are not instrumented. No OTel, no UI drill-down — Tiers 2–3.

## Constraints

- Python 3.9.7: `from __future__ import annotations`; no `X | None` syntax.
- No new runtime dependencies.
- `ToolLoopRunner.run` keeps returning exactly `{"reply", "images", "tools_used"}`.
- All new reads of LLM response dicts use `.get()` (scripted fakes are plain dicts).
- Do not persist full prompts/results in events (OTel privacy posture): inputs truncated,
  tool args recorded as **key provenance only** (names of injected/overridden/defaulted
  params), never values.
