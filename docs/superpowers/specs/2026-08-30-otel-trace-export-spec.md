# OTel Trace Export + RAG Token Accounting (Tier 2) Spec

Source: "Two Kinds of Trace" report §3 Tier 2 (https://claude.ai/code/artifact/67545329-a132-4b32-941b-920fad689e67),
building on the shipped Tier 0+1 (`docs/superpowers/specs/2026-08-30-agent-decision-trace-spec.md`, PR #3).
Out of scope: UI drill-down, provenance sidecars, replay/evals (Tiers 3-4).

## Goal

1. Export each turn's TurnTrace record as OpenTelemetry spans following the GenAI semantic
   conventions, over vendor-neutral OTLP — so a self-hosted Langfuse (recommended) or
   Phoenix/Jaeger/anything-OTLP renders session-grouped waterfalls with zero code coupling
   to any vendor.
2. Close the documented Tier-2 gap: RAG *generation* LLM calls (`knowledge/rag_system.py`)
   are currently unaccounted (own `LLMClient`, no token/trace attribution).

## Design decision: translate the record, don't instrument the hot paths

The report sketched live spans at three seams. Tier 1 already made those seams emit
structured events with timestamps and durations, so Tier 2 instead converts the finished
turn record into spans in one place — an exporter hook on `TraceRecorder.end_turn`.
Rationale: single integration point; the request path is already proven un-breakable by
the emission layer; spans and JSONL derive from the same source of truth; tests need no
network and no OTel for the translation logic. Span timing is reconstructed:
`chat`/`execute_tool` spans use `start = ts − duration`, `end = ts` from the events'
`latency_ms`/`ms` fields.

## Exporter hook (`core/turn_trace.py` — stays stdlib-only)

- Module-level `_TRACE_EXPORTERS: List` with `register_trace_exporter(fn)` (idempotent),
  `unregister_trace_exporter(fn)` (tolerates absent), `clear_trace_exporters()`.
- `end_turn()` calls each registered exporter with the record after `_persist`, each in
  its own try/except that logs a WARNING — an exporter can never break a turn.

## Translation (`core/otel_export.py::spans_from_record`) — pure, no OTel imports

`spans_from_record(record, capture_content=False, agent_name="seismic-chatbot") -> List[dict]`
returns plain dicts (`name`, `start_ns`, `end_ns`, `attributes`, `events`, `status_error`),
root span first:

- **Root** `invoke_agent seismic-chatbot`: `gen_ai.operation.name=invoke_agent`,
  `gen_ai.agent.name`, `session.id` + `langfuse.session.id` (Langfuse session grouping),
  `seismic.turn`, `seismic.tools_used` (comma-joined). Span time = min event ts → max
  child/event end (clamped so children never precede the root). `turn_error` sets the
  root's error status.
- **`llm` events** → child `chat {model|unknown}`: `gen_ai.operation.name=chat`,
  `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens` (from
  prompt/completion tokens), `seismic.requested_tool_call`; duration from `latency_ms`.
- **`tool_call` events** → child `execute_tool {tool}`: `gen_ai.operation.name=execute_tool`,
  `gen_ai.tool.name`, `seismic.injected`/`overridden`/`defaults_filled`; duration from
  `ms`; `ok: false` → error status with the event's `error` text.
- **All other events** (`intent`, `rag`, `discover`, `run_task`, `parallel_calls_dropped`,
  `auto_plot`, `budget_exhausted`, `turn_error`) → span events on the root, attributes =
  the event's fields coerced to OTel-legal values (scalars kept; scalar lists kept;
  anything else `json.dumps`'d; `None` dropped).
- **Content stays out by default** (semconv posture): the `turn_start.input` snippet is
  attached (as `gen_ai.input.messages`) only when `capture_content=True`; agentic-mode
  `run_task.brief` and `discover.query` are likewise gated behind `capture_content`; error
  strings are exported (standard OTel error data); everything else is names-not-values
  (held since Tier 1).

## Emission (`core/otel_export.py::install/uninstall`) — guarded, env-gated, no globals

- `install(span_exporter=None) -> bool`. No-op returning False unless
  `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set (or a
  `span_exporter` is injected — the test path, which uses a `SimpleSpanProcessor`).
  OTel imports happen inside `install` and failures log a WARNING and return False —
  the package keeps zero hard OTel dependency.
- Builds a module-local `TracerProvider` (never calls `trace.set_tracer_provider` — no
  global mutation) with `Resource(service.name = OTEL_SERVICE_NAME | "seismic-chatbot")`
  and a `BatchSpanProcessor(OTLPSpanExporter())` (the SDK reads endpoint/headers/proto
  from standard env vars). Registers `_export_record` on the turn_trace hook. Idempotent.
- `OTEL_GENAI_CAPTURE_CONTENT` truthy (1/true/yes) → `capture_content=True`. Default off.
- `_export_record(record)`: root span started with explicit `start_time`, span events
  added with explicit timestamps, children created under the root context with explicit
  start/end times, error statuses set; root ended last.
- `uninstall()`: unregister the hook, `provider.shutdown()`, reset state (test hygiene).
- Wiring: `main.py` calls `install()` after `basicConfig` (INFO log when enabled);
  `interfaces/api_interface.py` calls it at module level. Gradio runs via `main.py`.
- Packaging: `[project.optional-dependencies] otel = ["opentelemetry-sdk>=1.30",
  "opentelemetry-exporter-otlp-proto-http>=1.30"]` in pyproject.toml. requirements.txt
  unchanged.

## RAG token accounting

- `KnowledgeBase.__init__(self, llm_client=None)` → `RAGSystem(llm_client=llm_client)`
  (RAGSystem already accepts injection; only KnowledgeBase hardcoded it).
- Both bots construct `KnowledgeBase(llm_client=self.llm_client)` when none injected, so
  RAG generation reuses the shared, accounted client.
- Context threading: `KnowledgeBase.query_knowledge(query, domain=None, context_manager=None)`
  → `RAGSystem.retrieve_and_generate(..., context_manager=None)` → `_generate_response`
  passes it to `get_simple_completion(system, user, context_manager=cm)` (the Tier-1
  accounting seam — tokens counted + `llm` event emitted).
- `KnowledgeRouter.handle_knowledge_question` calls
  `query_knowledge(user_input, context_manager=self.context_manager)` with a `TypeError`
  fallback to the legacy 1-arg call — same compatibility pattern as `_simple`, so fake
  KBs in existing tests keep working unchanged.

## Constraints

- Python 3.9.7; `from __future__ import annotations`; `typing.Optional`, never `X | None`.
- `core/turn_trace.py` remains stdlib-only (hook is a plain callback list).
- No hard new runtime dependency; OTel is optional (guarded imports, env-gated).
- No behavior change to any request path; exporters may never raise out of `end_turn`.
- The venv here has opentelemetry-sdk 1.36 — SDK-touching tests use
  `opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter` and are
  `pytest.importorskip`-gated so the suite passes without OTel installed.
- Translation attributes must be OTel-legal types (str/bool/int/float or lists thereof).

## Known-remaining after this tier

`classify_intent_detailed` still calls `get_simple_completion` raw (unused in production).
UI drill-down, provenance sidecars, replay/evals stay Tier 3-4.
