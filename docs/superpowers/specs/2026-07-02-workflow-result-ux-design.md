# Workflow Result UX: Narrated Replies + Multi-Image Plumbing

**Date:** 2026-07-02
**Branch:** `stabilize-tool-layer`
**Status:** Approved design, pending implementation plan

## Problem

Two UX defects in how tool/workflow results reach the user:

1. **Results are dropped, not narrated.** When a workflow recipe (or any plotting
   tool) returns, the agentic loop in `core/chatbot_tool_use.py` short-circuits
   and returns `{"image_path": ...}` alone. The rich result dict — AVO class,
   intercept/gradient, tuning thickness, sweep stats and coverage — never reaches
   the user. The UI renders a bare image with no text.
2. **The contract is single-image by construction.** `process_single_input`
   returns either a `str` or `{"image_path": str}`. If the agentic loop
   (MAX_TOOL_ROUNDS=5, added in `6a4323c`) runs several plotting tools, only the
   first image survives.

These are the two deferred feature-scope items from the Task 14 backlog
(multi-image workflow output; LLM narration of workflow result dicts).

## Decisions (user-confirmed)

- **Narration:** the LLM narrates. No short-circuit after a tool: a *compacted*
  result is fed back into the agentic loop and the model writes the final prose.
- **Narration scope:** ALL tools, not just workflow recipes. Simple plot requests
  also get a one-line prose answer plus the image. One consistent path; the cost
  is one extra LLM round per tool request.
- **Multi-image scope:** plumbing only. The response contract and UIs carry a
  list of image paths; recipes keep their single composite plot (no science-code
  changes this stage).

## Design

### 1. Response contract

`process_single_input` **always** returns a dict:

```python
{"reply": str, "images": list[str]}   # images may be empty
```

- Tool path: final loop prose + every plot harvested along the way.
- Knowledge/RAG path: `{"reply": rag_answer, "images": []}`.
- Error paths: `{"reply": "I encountered an error: ...", "images": [<collected before failure>]}`.
- The existing bool/None safety checks fold into building this dict.
- The `chat()` REPL method is unchanged: it is a console-only loop that never
  surfaced images and does not use `_handle_tool_request`.

### 2. Agentic loop (`core/chatbot_tool_use.py::_handle_tool_request`)

- Add a `collected_images: list[str]` accumulator at the top of the tool loop.
- **Delete all three short-circuits**: the `_is_image_output(...)` return, the
  `_workflow_image_output(...)` return, and returning the auto-chain result.
- New helper `_harvest_images(result) -> list[str]`: pulls `.png` paths out of
  any tool result — a plain string path, or an `image_path` key in a dict.
  Appended to the accumulator, deduped preserving order. No `os.path.exists`
  check (matches current behavior; keeps fake-path tests simple).
- **Auto-chaining stays.** `AUTO_PLOT` still deterministically runs the partner
  plot tool after a compute tool; its plot path goes into the accumulator
  instead of ending the turn.
- The `role: "tool"` message content becomes `_compact_tool_result(tool_result)`
  instead of `str(tool_result)`:
  - numeric lists/ndarrays longer than 12 values → `"<n=61 values, min=…, max=…>"`
  - nested dicts/lists recursed
  - scalars and strings kept verbatim
  - `image_path` values masked as `"<plot generated and shown to the user>"`
    so the model never echoes temp file paths.
- Loop continues until the model answers without a tool call →
  `{"reply": extracted, "images": collected_images}`.
- The round-exhaustion fallback (forced tool-free completion after
  `MAX_TOOL_ROUNDS`) also returns the collected images.
- A tool-execution exception is fed back to the model as a `role:"tool"` error
  message and the loop continues (bounded by `MAX_TOOL_ROUNDS`), so the model
  can retry or narrate what it has. (Amended post-review: the original abort
  behavior turned recoverable model mistakes — e.g. a malformed self-initiated
  plot call — into raw error replies.)

### 3. System prompt

Add one short paragraph: tool results arrive compacted (large arrays
summarized); any plots are displayed to the user automatically — do not print
file paths; after tools finish, summarize the key quantitative results
(tuning thickness, AVO class, intercept/gradient, sweep stats) in the
`<reply>`.

### 4. Interfaces

- `interfaces/gradio_interface.py` (`respond`): render `reply` as the assistant
  text, then append one chat-history row per image (Gradio's tuple format is
  one file per message).
- `interfaces/gradio_interface_legacy.py`: **out of scope** — it consumes the
  legacy `core/chatbot.py::SeismicChatBot.process_input`, whose contract this
  design does not change (discovered during planning; the spec originally
  assumed it consumed the tool-use bot).
- `interfaces/api_interface.py`: `ChatResponse` gains `images: List[str] = []`;
  the `response` field carries the reply text (name kept for client compat —
  it previously received `str(dict)` for image results, which no client could
  have parsed anyway).
- `main.py --test` example flows print reply + image paths.
- `interfaces/web_interface.html` is unaffected (static prompt browser, no chat).

### 5. Explicitly unchanged

Recipes (`workflows/recipes/*`), `run_sweep`, `WorkflowSpec` / tool registry,
`ToolManager`, `ContextManager`, RAG internals, `_extract_reply`. Recipes keep
returning `image_path` inside their result dicts — that is now the harvest
source rather than a UI escape hatch. `_is_image_output` and
`_workflow_image_output` are deleted (absorbed into `_harvest_images`).

### 6. Edge cases

- LLM re-calls tools after seeing a compacted result: bounded by
  `MAX_TOOL_ROUNDS=5`; exhaustion path still returns text + images.
- Duplicate image paths (e.g. auto-chain plus workflow dict): deduped in order.
- Narration token cost: bounded by compaction; one extra LLM round per tool
  request is the accepted trade-off.

### 7. Testing

Update the contract-pinned tests:
- `tests/test_chatbot.py` — NO changes needed (discovered during planning):
  its `{"image_path": ...}` asserts pin `_handle_automatic_chaining`, whose
  return contract is intentionally preserved.
- `tests/test_chatbot_workflow.py` — short-circuit tests become harvest tests.
- `tests/test_rock_properties_plot.py` — chaining assert → images list.

New coverage (all via `tests/conftest.py::fake_llm_factory`, no network):
- Narration flow: scripted tool call → scripted final reply → returns
  `{"reply", "images"}` with the harvested image.
- Multi-image: two tool rounds each producing a plot → both paths present,
  order preserved, deduped.
- `_compact_tool_result` unit tests: long numeric list, nested dict,
  `image_path` masking, scalar/string passthrough.
- Round exhaustion carrying images.
- Mid-loop tool error returning collected images.
- API `/chat` response includes the `images` field.
