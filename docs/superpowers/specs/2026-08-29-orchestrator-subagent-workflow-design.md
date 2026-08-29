# Orchestrator + Subagent Workflow (Agentic Mode)

**Date:** 2026-08-29
**Status:** Approved design, pre-implementation
**Branch:** `stabilize-tool-layer` (or a feature branch off it)

## Problem

The chatbot's single-agent loop (`core/chatbot_tool_use.py::SeismicChatBotToolUse`)
ships **every** tool schema from `core/tool_registry.py` (32 tools + workflow-derived
specs) to the LLM on every call, and hand-lists the tools again in prose inside the
system prompt. Both grow linearly with the registry, the prose list is already
drifting from the registry, and multi-step requests are served only by the LLM
chaining calls inside one context with all schemas loaded.

**Goals (in priority order):** multi-step autonomy (decompose → delegate → compose)
and architecture scalability (adding tool domain N+1 touches only the registry).
Token cost per turn improves as a side effect but is not the driver.

## Decision summary

| Decision | Choice |
|---|---|
| Delegation granularity | Subagent owns a **domain task** (brief + scoped toolset + own mini tool-loop) |
| Tool discovery | **Semantic**: embedding search over registry-derived tool cards (no static domain tags) |
| Orchestrator style | **Meta-tool loop** (approach B): `discover_tools` + `run_task` are its only tools |
| Rollout | **New mode beside** the existing loop (`python main.py --mode agentic`); classic `tool-use` stays the default |
| Subagent scheduling | Sequential only (no parallelism — shared-context races for no win at this scale) |
| Knowledge/RAG path | Unchanged; intent split happens before the orchestrator |

## Architecture

```
user turn
  │
  ├─ knowledge_router: intent split (LLM + keyword fallback)  [unchanged logic, extracted]
  │     knowledge question ──► RAG path (unchanged)
  │
  └─ tool request ──► SeismicOrchestrator (core/orchestrator.py)
        meta-tool loop, MAX_ORCH_ROUNDS = 8
        tools = [discover_tools, run_task] ONLY — never real tool schemas
          │
          ├─ discover_tools(task_description)
          │     └─ core/tool_index.py::search()  (pure embedding search, no LLM)
          │           → ranked tool cards (name, one-liner, required params)
          │
          └─ run_task(brief, tool_names)
                └─ core/executor_agent.py::ExecutorAgent
                      scoped tool-use loop, MAX_TOOL_ROUNDS = 5
                      (context injection, AUTO_PLOT chaining, compaction,
                       image harvest — via extracted core/tool_loop.py)
                      → TaskResult {summary, images, tools_used, error}
        │
        └─ composes final <reply>; images from all TaskResults aggregated
```

### New modules

- `core/tool_index.py` — semantic tool index derived from `REGISTRY`.
- `core/executor_agent.py` — `ExecutorAgent` + `TaskResult`.
- `core/orchestrator.py` — `SeismicOrchestrator`.
- `core/tool_loop.py` — **extraction** of the shared tool-loop plumbing currently
  inside `SeismicChatBotToolUse`: `_inject_context_inputs` (incl. the
  last_image-always-wins override), `_parse_tool_input`, `_compact_tool_result` /
  `_compact_value`, `_harvest_images`, `_handle_automatic_chaining`, and the
  bounded tool-round loop itself.
- `core/knowledge_router.py` — **extraction** of the intent split + knowledge path:
  `_is_knowledge_question`, `_classify_intent_with_llm`,
  `_is_knowledge_question_keywords`, `_handle_knowledge_question`,
  `_handle_no_rag_results`, `_fallback_knowledge_response`.

The two extractions are the only changes to the existing bot:
`SeismicChatBotToolUse` keeps its public API and delegates to the shared modules.
Behavior is identical; existing tests must keep passing unmodified (any test that
pins a private method may switch to the shared module, but assertions don't change).

## Component specs

### 1. Tool index (`core/tool_index.py`)

**Card rendering.** At import, each `ToolSpec` in `REGISTRY` renders to a card:

```
{name}: {description} Params: {p1} ({type}, required), {p2} ({type}), ...
```

Workflow-derived tools are included automatically (they are already in `REGISTRY`).

**Plot-tool exclusion.** Any tool that appears as some spec's `auto_plot` target is
excluded from the index. The LLM must never call `plot_*` directly (chaining does
it); discovery therefore only ever surfaces compute tools.

**Storage.** A dedicated ChromaDB collection `tool_index` (cosine space) in the
existing `chroma_db/` directory, reusing `knowledge/vector_db.py`'s
content-derived-ID + upsert idempotency. Embedding model: the existing
`RAG_EMBEDDING_MODEL` (`all-MiniLM-L6-v2`).

**Stale cleanup (improvement over the knowledge store).** On startup population,
stored IDs not present in the current registry-derived ID set are **deleted**, so
renamed/removed tools cannot linger. (The knowledge store's known
stale-chunk limitation is out of scope here.)

**API.**

```python
class ToolIndex:
    def __init__(self, persist_directory: str | None = None): ...
    def search(self, task_description: str, top_k: int = 5) -> list[ToolCard]:
        """Ranked cards. Cosine threshold ~0.2, EXCEPT the top-3 hits are always
        returned regardless of score (the index always holds >=3 tools), so an
        on-topic request never gets an empty discovery. Results beyond the top 3
        must clear the threshold."""

@dataclass(frozen=True)
class ToolCard:
    name: str
    card: str          # rendered card text
    required: list[str]
    score: float
```

Shared across sessions (stateless after build), injected into the orchestrator.

### 2. Executor agent (`core/executor_agent.py`)

```python
@dataclass
class TaskResult:
    summary: str               # executor's final prose (compacted numbers, no paths)
    images: list[str]          # harvested .png paths, ordered, deduped
    tools_used: list[str]
    error: str | None = None

class ExecutorAgent:
    def __init__(self, llm_client, tool_manager, context_manager): ...
    def run(self, brief: str, tool_names: list[str]) -> TaskResult: ...
```

- `run()` builds the scoped schema list from `REGISTRY_BY_NAME` (schemas via the
  registry's `to_openai_schema`), a small generated system prompt (role + the
  existing "Tool results and plots" rules + the assigned tools' cards), and runs
  the shared `core/tool_loop.py` loop with `MAX_TOOL_ROUNDS = 5`.
- Shares the **session's** `ContextManager`, so context injection
  (`last_image` / `last_outcrop` / `last_earth_model`) and `last_*` result storage
  behave exactly as today, and task N+1 reads task N's outputs with no
  inter-agent protocol.
- Token usage from executor calls flows into the same session token counter.
- Exceptions inside the loop follow today's pattern (error text as tool message,
  loop continues); an exception escaping the loop becomes `TaskResult.error`.

### 3. Orchestrator (`core/orchestrator.py`)

```python
class SeismicOrchestrator:
    def __init__(self, llm_client=None, tool_manager=None,
                 knowledge_base=None, tool_index=None): ...
    def new_session(self) -> "SeismicOrchestrator": ...
    def attach_image(self, path: str) -> None: ...
    def process_single_input(self, user_input: str) -> dict:  # {"reply", "images"}
```

Public surface mirrors `SeismicChatBotToolUse` so interfaces host either bot.
Shared components: `llm_client`, `tool_manager`, `knowledge_base`, `tool_index`.
Per-session: fresh `ContextManager`, `session_id`.

**Meta-tool schemas** (the only `tools=` payload the orchestrator LLM ever sees):

- `discover_tools(task_description: str)` → tool message listing ranked cards.
  Empty result → `"No tools matched; rephrase the task or answer directly."`
- `run_task(brief: str, tool_names: list[str])` → validates names against
  `REGISTRY_BY_NAME`; unknown names return an error tool-message (no executor
  spawned). Otherwise spawns `ExecutorAgent.run()` and returns a compacted
  rendering of `TaskResult` (summary + tools_used + error; **not** image paths —
  images aggregate out-of-band).

**System prompt** (static except one line): role ("decompose the request into task
briefs; discover tools for each; delegate via run_task; compose one final answer"),
the preserved behavioral rules from today's prompt — the `[image attached: ...]`
convention routes to outcrop tooling, never pass `image_path`/`interpretation`/
`model` arguments, report scale-estimate confidence after `interpret_outcrop`,
state key quantitative results, `<reply></reply>` wrapping — plus one generated
line summarizing which session context keys are currently populated (e.g.
`Session context: last_wedge_model, last_image`), so follow-ups reuse state.
**No per-tool documentation appears in the prompt** — the prompt must not grow
with the registry.

**Loop.** `MAX_ORCH_ROUNDS = 8` meta-calls; on exhaustion, forced tool-free
completion (today's pattern). Reply extraction via the same `<reply>` parsing.
Images: concatenation of all `TaskResult.images` in execution order, deduped.

**Flow per turn:** `knowledge_router` intent split first (identical to classic
mode); `[image attached` prefix always routes to the tool path.

**Cost note.** A simple request ≈ 4 LLM calls (orchestrator → executor×2 →
composition) vs. 2 today; accepted as the price of the architecture.

### 4. Mode wiring

- `main.py`: `--mode` gains `agentic` (choices: `tool-use`, `agentic`, `legacy`;
  default stays `tool-use`). Agentic mode builds a `SeismicOrchestrator` base bot.
- `interfaces/gradio_interface.py::create_chat_interface` accepts an optional
  base-bot factory/instance (default: classic bot) — the `gr.State` +
  `new_session()` lazy-derivation pattern is unchanged.
- FastAPI (`interfaces/api_interface.py`) stays on the classic loop (YAGNI).
- Swapping the Gradio default to agentic, and deleting the classic loop, are
  explicitly **out of scope** — separate future tasks after the mode is proven.

## Error handling

| Failure | Behavior |
|---|---|
| Discovery returns nothing | Informative tool message; orchestrator rephrases or answers directly |
| `run_task` with unknown tool name | Error tool message naming the bad names; no executor spawned |
| Tool raises inside executor loop | Today's pattern: error text as tool message, executor continues |
| Executor loop exhausts rounds | Forced tool-free completion → that text is the `TaskResult.summary` |
| Exception escapes executor | `TaskResult.error` set; orchestrator told; it may retry differently or explain |
| Orchestrator exhausts rounds | Forced tool-free completion; user always gets prose |
| Top-level exception | `process_single_input` catches → `{"reply": "I encountered an error: ...", "images": []}` (today's contract) |

## Out of scope (YAGNI)

- Parallel executor subagents.
- Static domain tags on `ToolSpec` / hybrid catalog+search discovery.
- FastAPI agentic mode; changing the default Gradio mode; deleting the classic loop.
- Streaming intermediate orchestrator progress to the UI.
- Cross-provider models per role (everything stays on the configured
  DeepSeek/Databricks client).

## Testing

All LLM behavior is tested with `tests/conftest.py::fake_llm_factory` scripted
completions (no network). New files:

- `tests/test_tool_index.py` — card rendering (params/required), plot-tool
  exclusion, idempotent re-population, stale-ID deletion on registry change
  (monkeypatched registry), `search` returns ranked cards and never fewer than 3
  for an on-topic query against the real registry.
- `tests/test_executor_agent.py` — scoped schemas only (asserts the `tools=`
  payload passed to the fake LLM), context injection (last_image override),
  AUTO_PLOT chaining + image harvest into `TaskResult`, error paths, round budget.
- `tests/test_orchestrator.py` — end-to-end fake-LLM flows: single-task request;
  multi-task decomposition (two `run_task`s sharing context through
  `ContextManager`); unknown tool name; empty discovery; round budget; `<reply>`
  extraction and image aggregation; meta-schemas are the only tools sent.
- `tests/test_knowledge_router.py` / `tests/test_tool_loop.py` — the extracted
  modules keep the pinned behaviors (much of this is relocation of existing
  coverage; existing chatbot tests must pass unmodified).
- `tests/test_session_isolation.py` — extended to cover
  `SeismicOrchestrator.new_session()` (shared components reused, contexts isolated).
- `tests/test_main_modes.py` — `--mode agentic` wires the orchestrator into the
  Gradio factory.

TDD throughout (RED before GREEN per task).

## Implementation order (for the plan)

1. Extract `core/tool_loop.py` + `core/knowledge_router.py`; classic bot delegates;
   full suite green (pure refactor).
2. `core/tool_index.py` + tests.
3. `core/executor_agent.py` + tests.
4. `core/orchestrator.py` + tests.
5. Mode wiring (`main.py`, Gradio factory) + tests; docs (CLAUDE.md section).
