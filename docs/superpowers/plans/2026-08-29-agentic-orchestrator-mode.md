# Agentic Orchestrator Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an orchestrator + subagent chat mode (`--mode agentic`) where a meta-tool LLM loop discovers tools by semantic search and delegates domain tasks to scoped executor subagents, beside the untouched classic loop.

**Architecture:** Two behavior-preserving extractions (`core/tool_loop.py`, `core/knowledge_router.py`) let the classic bot and the new agents share one tested tool-loop/knowledge implementation. A `ToolIndex` (ChromaDB collection over registry-derived tool cards) powers `discover_tools`; `run_task` spawns an `ExecutorAgent` that runs the shared tool loop with only its assigned schemas against the session's shared `ContextManager`.

**Tech Stack:** Python, chromadb + sentence-transformers (already in use), OpenAI-compatible DeepSeek client, Gradio, pytest with `fake_llm_factory` scripted completions.

**Spec:** `docs/superpowers/specs/2026-08-29-orchestrator-subagent-workflow-design.md`

## Global Constraints

- Working directory is the package root (`geo-mcp/seismic_chatbot/`); imports are top-level absolute (`from core.tool_loop import ...`).
- Commit with plain `git` from inside this directory (this package is its own repo; the outer repos do not track it).
- Every commit: full suite green — `pytest` from the package root. Existing tests must pass **unmodified** through Tasks 1–2 (pure refactors).
- TDD: every task RED before GREEN (write failing test, watch it fail, implement, watch it pass).
- Scripted tool calls in tests: define a local `_FakeToolCall`/`_FakeFunc` pair per test file (the established pattern — see `tests/test_chatbot_narration.py:10-19`); `tests/conftest.py`'s classes are not importable as a module.
- `FakeLLMClient` records a **reference** to the mutated `messages` list, so every recorded call shows the list's FINAL state. Assert on message history by filtering roles from `llm.calls[0]["messages"]`, never by positional `[-1]` indexing on intermediate calls.
- No network in tests: all LLM behavior via `tests/conftest.py::fake_llm_factory`; `ToolIndex` tests use `tmp_path` persist dirs, never the real `chroma_db/`.
- Bounds are spec-fixed: `MAX_TOOL_ROUNDS = 5` (executor/classic), `MAX_ORCH_ROUNDS = 8` (orchestrator). Discovery: cosine threshold `0.2`, top-3 always returned regardless of score.
- The `last_image`-always-wins override in context injection is a security behavior — it must survive extraction byte-for-byte.

---

### Task 1: Extract `core/tool_loop.py` (pure refactor)

**Files:**
- Create: `core/tool_loop.py`
- Create: `tests/test_tool_loop.py`
- Modify: `core/chatbot_tool_use.py` (replace bodies with delegation; no behavior change)

**Interfaces:**
- Consumes: `ToolManager.process_tool_call(name, input) -> Any`, `ContextManager.get_context/set_context/update_token_usage`, `llm_client.get_completion(system_prompt, user_prompt, tools, messages) -> {"content", "tool_calls", "stop_reason", "usage"}`, `core.tool_registry.AUTO_PLOT`, `workflows.engine.WORKFLOW_NAMES`.
- Produces (later tasks rely on these exact names):
  - `core.tool_loop.extract_reply(text: str) -> Optional[str]` (module function)
  - `class ToolLoopRunner` with `__init__(self, llm_client, tool_manager, context_manager, max_tool_rounds: int = 5)` and:
    - `run(self, system_prompt: str, messages: list, tools: list) -> dict` returning `{"reply": str, "images": list[str], "tools_used": list[str]}`
    - public helpers `parse_tool_input`, `compact_tool_result`, `compact_value`, `inject_context_inputs`, `harvest_images`, `handle_automatic_chaining`, `update_context` — same signatures as today's `_`-prefixed bot methods.

The moved code is the **verbatim body** of today's `SeismicChatBotToolUse` members: `_MAX_ARRAY_PREVIEW`, `_CONTEXT_INPUTS`, `_parse_tool_input`, `_compact_tool_result`, `_compact_value`, `_inject_context_inputs`, `_harvest_images`, `_handle_automatic_chaining`, `_update_context`, `_extract_reply`, and the loop inside `_handle_tool_request` (`core/chatbot_tool_use.py:66-102, 168-227, 726-938, 957-1093`). Only the spelling changes: `self.llm_client/tool_manager/context_manager` stay (they're runner attributes now), `self._parse_tool_input` → `self.parse_tool_input`, etc. `run()` additionally appends each successfully executed tool name to `tools_used` (right after `process_tool_call` returns) and includes it in the return dict.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_loop.py
"""The extracted shared tool loop: same behavior the classic bot pinned, callable standalone."""
import numpy as np
import pytest
from core.tool_loop import ToolLoopRunner, extract_reply
from core.tool_manager import ToolManager
from core.context_manager import ContextManager


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


def make_runner(fake_llm_factory, responses):
    llm = fake_llm_factory(responses)
    return ToolLoopRunner(llm, ToolManager(), ContextManager()), llm


def test_extract_reply():
    assert extract_reply("x <reply> hi </reply> y") == "hi"
    assert extract_reply("no tags") is None


def test_run_no_tool_call_returns_reply(fake_llm_factory):
    runner, llm = make_runner(fake_llm_factory, [
        {"content": "<reply>Just words</reply>", "tool_calls": None, "stop_reason": "stop", "usage": None},
    ])
    out = runner.run("SYS", [{"role": "user", "content": "hi"}], tools=[])
    assert out == {"reply": "Just words", "images": [], "tools_used": []}
    assert llm.calls[0]["system_prompt"] == "SYS"


def test_run_executes_tool_and_harvests_plot(fake_llm_factory):
    runner, llm = make_runner(fake_llm_factory, [
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>Made a 30 Hz Ricker.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    out = runner.run("SYS", [{"role": "user", "content": "30 Hz ricker"}],
                     tools=ToolManager().get_tool_schemas())
    assert out["reply"] == "Made a 30 Hz Ricker."
    assert out["tools_used"] == ["make_ricker"]
    # auto-chained plot_ricker produced a png
    assert len(out["images"]) == 1 and out["images"][0].endswith(".png")
    # context stored for follow-ups
    assert runner.context_manager.get_context("last_ricker_wavelet") is not None


def test_run_tool_error_becomes_tool_message_and_loop_continues(fake_llm_factory):
    runner, llm = make_runner(fake_llm_factory, [
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": -5}')]},
        {"content": "<reply>That frequency is invalid.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    out = runner.run("SYS", [{"role": "user", "content": "bad"}],
                     tools=ToolManager().get_tool_schemas())
    assert out["tools_used"] == []
    # the error was surfaced to the model as a tool message
    sent = llm.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "failed" in m["content"] for m in sent)


def test_run_round_budget_forces_tool_free_completion(fake_llm_factory):
    tool_turn = {"content": "", "stop_reason": "tool_calls", "usage": None,
                 "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]}
    responses = [dict(tool_turn) for _ in range(5)] + [
        {"content": "<reply>Stopping here.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ]
    runner, llm = make_runner(fake_llm_factory, responses)
    out = runner.run("SYS", [{"role": "user", "content": "loop"}],
                     tools=ToolManager().get_tool_schemas())
    assert out["reply"] == "Stopping here."
    assert llm.calls[-1]["tools"] is None  # forced tool-free final call


def test_inject_context_inputs_last_image_always_wins(fake_llm_factory):
    runner, _ = make_runner(fake_llm_factory, [])
    runner.context_manager.set_context("last_image", "/sandbox/session/photo.png")
    filled = runner.inject_context_inputs(
        "interpret_outcrop", {"image_path": "/evil/elsewhere.png"})
    assert filled["image_path"] == "/sandbox/session/photo.png"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tool_loop.py -q`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'core.tool_loop'`.

- [ ] **Step 3: Create `core/tool_loop.py`**

Move the code listed above verbatim into this skeleton (do not retype logic — cut it from `chatbot_tool_use.py` and adjust only the names shown):

```python
"""Shared bounded tool-use loop.

One implementation of parse → inject-context → execute → compact → auto-plot →
harvest, used by both the classic SeismicChatBotToolUse and the agentic-mode
ExecutorAgent. Extracted verbatim from chatbot_tool_use.py — behavior changes
here change BOTH bots; keep it that way.
"""
import logging
import re
import json
import numpy as np
from typing import Dict, Any, List, Optional
from core.tool_registry import AUTO_PLOT
from workflows.engine import WORKFLOW_NAMES

logger = logging.getLogger(__name__)

_MAX_ARRAY_PREVIEW = 12  # moved from chatbot_tool_use


def extract_reply(text: str) -> Optional[str]:
    match = re.search(r'<reply>(.*?)</reply>', text, re.DOTALL)
    return match.group(1).strip() if match else None


class ToolLoopRunner:
    # Tools whose heavy inputs live in per-session context (tool, param, key).
    _CONTEXT_INPUTS = (
        ("interpret_outcrop", "image_path", "last_image"),
        ("outcrop_to_seismic", "image_path", "last_image"),
        ("outcrop_to_model", "interpretation", "last_outcrop"),
        ("synthetic_section", "model", "last_earth_model"),
    )

    def __init__(self, llm_client, tool_manager, context_manager, max_tool_rounds: int = 5):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.context_manager = context_manager
        self.max_tool_rounds = max_tool_rounds

    # ---- moved verbatim (drop the leading underscore) ----
    def parse_tool_input(self, tool_input): ...        # was _parse_tool_input
    def compact_tool_result(self, tool_result): ...    # was _compact_tool_result
    def compact_value(self, value): ...                # was _compact_value
    def inject_context_inputs(self, tool_name, tool_input): ...  # was _inject_context_inputs
    def harvest_images(self, tool_result, collected): ...        # was _harvest_images
    def handle_automatic_chaining(self, tool_name, tool_input, tool_result): ...
    def update_context(self, tool_name, tool_input, tool_result): ...

    def run(self, system_prompt: str, messages: List[dict], tools: list) -> Dict[str, Any]:
        """The bounded loop from _handle_tool_request, generalized.

        Differences from the original method: system prompt and tools are
        parameters; successfully executed tool names are recorded in
        'tools_used'; reply extraction uses module-level extract_reply.
        """
        collected_images: List[str] = []
        tools_used: List[str] = []
        for _ in range(self.max_tool_rounds):
            response = self.llm_client.get_completion(
                system_prompt=system_prompt, user_prompt="", tools=tools, messages=messages)
            if response.get("usage"):
                self.context_manager.update_token_usage(response["usage"])
            if not response.get("tool_calls"):
                messages.append({"role": "assistant", "content": response["content"]})
                reply = extract_reply(response["content"]) or response["content"]
                if isinstance(reply, bool):
                    reply = str(reply)
                return {"reply": reply, "images": collected_images, "tools_used": tools_used}
            tool_call = response["tool_calls"][0]
            tool_name = tool_call.function.name
            messages.append({"role": "assistant", "content": response["content"],
                             "tool_calls": [tool_call]})
            try:
                tool_input = self.inject_context_inputs(
                    tool_name, self.parse_tool_input(tool_call.function.arguments))
                tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
                tools_used.append(tool_name)
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": self.compact_tool_result(tool_result)})
                self.update_context(tool_name, tool_input, tool_result)
                self.harvest_images(tool_result, collected_images)
                chained = self.handle_automatic_chaining(tool_name, tool_input, tool_result)
                if chained:
                    self.harvest_images(chained, collected_images)
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": (f"Tool execution failed: {e}. Do not retry with "
                                             f"the same arguments; summarize what you have or "
                                             f"ask the user for clarification.")})
                continue
        final_response = self.llm_client.get_completion(
            system_prompt=system_prompt, user_prompt="", tools=None, messages=messages)
        if final_response.get("usage"):
            self.context_manager.update_token_usage(final_response["usage"])
        reply = extract_reply(final_response["content"]) or final_response["content"]
        if isinstance(reply, bool):
            reply = str(reply)
        return {"reply": reply, "images": collected_images, "tools_used": tools_used}
```

- [ ] **Step 4: Delegate from `SeismicChatBotToolUse`**

In `core/chatbot_tool_use.py`: delete the moved bodies; add `from core.tool_loop import ToolLoopRunner, extract_reply`; in `__init__` (after `self.context_manager = ...`) add `self._tool_loop = ToolLoopRunner(self.llm_client, self.tool_manager, self.context_manager)`. Replace the moved methods with one-line delegating shims so every existing test and the `chat()` REPL keep working:

```python
    def _parse_tool_input(self, tool_input):
        return self._tool_loop.parse_tool_input(tool_input)

    def _compact_tool_result(self, tool_result):
        return self._tool_loop.compact_tool_result(tool_result)

    def _compact_value(self, value):
        return self._tool_loop.compact_value(value)

    def _inject_context_inputs(self, tool_name, tool_input):
        return self._tool_loop.inject_context_inputs(tool_name, tool_input)

    def _harvest_images(self, tool_result, collected):
        return self._tool_loop.harvest_images(tool_result, collected)

    def _handle_automatic_chaining(self, tool_name, tool_input, tool_result):
        return self._tool_loop.handle_automatic_chaining(tool_name, tool_input, tool_result)

    def _update_context(self, tool_name, tool_input, tool_result):
        return self._tool_loop.update_context(tool_name, tool_input, tool_result)

    def _extract_reply(self, text):
        return extract_reply(text)

    def _handle_tool_request(self, user_input: str) -> Dict[str, Any]:
        result = self._tool_loop.run(
            self.system_prompt, [{"role": "user", "content": user_input}], self.tools)
        return {"reply": result["reply"], "images": result["images"]}
```

Keep `_CONTEXT_INPUTS` on the class as `_CONTEXT_INPUTS = ToolLoopRunner._CONTEXT_INPUTS` (one test-visible alias, no duplication). Remove now-unused imports (`re`, `np`, `AUTO_PLOT`, `WORKFLOW_NAMES`) only if nothing else in the file uses them.

- [ ] **Step 5: Run the new tests, then the full suite**

Run: `pytest tests/test_tool_loop.py -q` → PASS.
Run: `pytest -q` → all green, zero modifications to existing test files.

- [ ] **Step 6: Commit**

```bash
git add core/tool_loop.py core/chatbot_tool_use.py tests/test_tool_loop.py
git commit -m "refactor(core): extract shared bounded tool loop into core/tool_loop.py"
```

---

### Task 2: Extract `core/knowledge_router.py` (pure refactor)

**Files:**
- Create: `core/knowledge_router.py`
- Create: `tests/test_knowledge_router.py`
- Modify: `core/chatbot_tool_use.py`

**Interfaces:**
- Consumes: `llm_client.get_simple_completion(system_prompt, user_prompt) -> str`, `KnowledgeBase.query_knowledge / get_topic_response`.
- Produces: `class KnowledgeRouter` with `__init__(self, llm_client, knowledge_base)` and:
  - `is_knowledge_question(self, user_input: str) -> bool` (includes the `[image attached` → `False` guard)
  - `handle_knowledge_question(self, user_input: str) -> str`
  - `classify_intent_detailed(self, user_input: str) -> dict`

Move verbatim from `SeismicChatBotToolUse`: `_is_knowledge_question`, `_classify_intent_with_llm`, `classify_intent_detailed`, `_is_knowledge_question_keywords`, `_handle_knowledge_question`, `_handle_no_rag_results`, `_fallback_knowledge_response` (`core/chatbot_tool_use.py:374-558, 560-724`), including every system-prompt string byte-for-byte (the "⚠️ Not from the curated knowledge base" disclaimer is pinned by `tests/test_rag_no_results.py`). Public names on the router: strip the leading underscore from `_is_knowledge_question` and `_handle_knowledge_question`; the rest keep their underscore as router-private helpers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_knowledge_router.py
from core.knowledge_router import KnowledgeRouter


class FakeSimpleLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def get_simple_completion(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.reply


class RaisingLLM:
    def get_simple_completion(self, *a, **k):
        raise RuntimeError("no network")


class FakeKB:
    def __init__(self, response):
        self._response = response

    def query_knowledge(self, q):
        return self._response

    def get_topic_response(self, topic, section):
        return f"canned:{topic}"


def test_image_attached_is_never_knowledge():
    router = KnowledgeRouter(FakeSimpleLLM("KNOWLEDGE"), FakeKB({}))
    assert router.is_knowledge_question("[image attached: x.png] interpret this") is False


def test_llm_classification_yes_and_no():
    assert KnowledgeRouter(FakeSimpleLLM("KNOWLEDGE"), FakeKB({})).is_knowledge_question("what is tuning") is True
    assert KnowledgeRouter(FakeSimpleLLM("TOOL"), FakeKB({})).is_knowledge_question("make a wavelet") is False


def test_keyword_fallback_when_llm_fails():
    router = KnowledgeRouter(RaisingLLM(), FakeKB({}))
    assert router.is_knowledge_question("What is a Ricker wavelet?") is True
    assert router.is_knowledge_question("make a 30 Hz ricker wavelet please") is False


def test_handle_knowledge_question_rag_hit():
    router = KnowledgeRouter(FakeSimpleLLM("unused"), FakeKB({
        "rag_type": "retrieve_and_generate",
        "generated_response": "Tuning is ...",
        "total_retrieved": 3,
    }))
    out = router.handle_knowledge_question("what is tuning")
    assert out.startswith("Tuning is ...")
    assert "3 relevant documents" in out


def test_handle_no_rag_results_appends_disclaimer():
    router = KnowledgeRouter(FakeSimpleLLM("General answer."), FakeKB({
        "rag_type": "no_results",
    }))
    out = router.handle_knowledge_question("what is obscure thing")
    assert "Not from the curated knowledge base" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_knowledge_router.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.knowledge_router'`.

- [ ] **Step 3: Create `core/knowledge_router.py` and delegate**

Module skeleton (bodies moved verbatim; only `self.knowledge_base`/`self.llm_client` references already match):

```python
"""Intent split + knowledge (RAG) path, shared by both chat modes."""
import json
import logging

logger = logging.getLogger(__name__)


class KnowledgeRouter:
    def __init__(self, llm_client, knowledge_base):
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base

    def is_knowledge_question(self, user_input: str) -> bool: ...   # was _is_knowledge_question
    def _classify_intent_with_llm(self, user_input: str) -> bool: ...
    def classify_intent_detailed(self, user_input: str) -> dict: ...
    def _is_knowledge_question_keywords(self, user_input: str) -> bool: ...
    def handle_knowledge_question(self, user_input: str) -> str: ...  # was _handle_knowledge_question
    def _handle_no_rag_results(self, user_input: str) -> str: ...
    def _fallback_knowledge_response(self, user_input: str) -> str: ...
```

In `SeismicChatBotToolUse.__init__` add `self._knowledge_router = KnowledgeRouter(self.llm_client, self.knowledge_base)` and replace the moved methods with shims:

```python
    def _is_knowledge_question(self, user_input):
        return self._knowledge_router.is_knowledge_question(user_input)

    def _classify_intent_with_llm(self, user_input):
        return self._knowledge_router._classify_intent_with_llm(user_input)

    def classify_intent_detailed(self, user_input):
        return self._knowledge_router.classify_intent_detailed(user_input)

    def _is_knowledge_question_keywords(self, user_input):
        return self._knowledge_router._is_knowledge_question_keywords(user_input)

    def _handle_knowledge_question(self, user_input):
        return self._knowledge_router.handle_knowledge_question(user_input)

    def _handle_no_rag_results(self, user_input):
        return self._knowledge_router._handle_no_rag_results(user_input)

    def _fallback_knowledge_response(self, user_input):
        return self._knowledge_router._fallback_knowledge_response(user_input)
```

- [ ] **Step 4: Run new tests, then full suite**

Run: `pytest tests/test_knowledge_router.py -q` → PASS. Then `pytest -q` → all green, existing tests unmodified.

- [ ] **Step 5: Commit**

```bash
git add core/knowledge_router.py core/chatbot_tool_use.py tests/test_knowledge_router.py
git commit -m "refactor(core): extract intent split + knowledge path into core/knowledge_router.py"
```

---

### Task 3: `core/tool_index.py` — semantic tool discovery

**Files:**
- Create: `core/tool_index.py`
- Create: `tests/test_tool_index.py`

**Interfaces:**
- Consumes: `core.tool_registry.REGISTRY` (list of `ToolSpec(name, fn, description, params, required, defaults, validator, auto_plot)`), `knowledge.vector_db.content_id(text, metadata)`, `config.settings.RAG_EMBEDDING_MODEL`, `config.settings.RAG_VECTOR_DB_PATH`.
- Produces:
  - `render_card(spec) -> str`
  - `@dataclass(frozen=True) ToolCard(name: str, card: str, required: tuple, score: float)`
  - `class ToolIndex` with `__init__(self, persist_directory: str | None = None, specs: list | None = None)` (specs default `REGISTRY`; population runs in `__init__`) and `search(self, task_description: str, top_k: int = 5, threshold: float = 0.2) -> list[ToolCard]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_index.py
import pytest
from core.tool_index import ToolIndex, ToolCard, render_card
from core.tool_registry import REGISTRY, REGISTRY_BY_NAME, AUTO_PLOT


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    # module-scoped: building embeds ~30 cards once, not per test
    return ToolIndex(persist_directory=str(tmp_path_factory.mktemp("tool_index")))


def test_render_card_contains_name_description_and_params():
    spec = REGISTRY_BY_NAME["make_ricker"]
    card = render_card(spec)
    assert card.startswith("make_ricker: ")
    assert spec.description in card
    assert "frequency (number, required)" in card


def test_plot_tools_are_excluded(index):
    plot_targets = set(AUTO_PLOT.values())
    names = {c.name for c in index.search("plot a wavelet figure", top_k=10, threshold=-1.0)}
    assert names.isdisjoint(plot_targets)


def test_search_returns_relevant_ranked_cards(index):
    cards = index.search("create a ricker wavelet with a given frequency")
    assert cards, "on-topic query must never return empty"
    assert cards[0].score >= cards[-1].score
    assert "make_ricker" in {c.name for c in cards[:3]}


def test_search_always_returns_top3_even_below_threshold(index):
    cards = index.search("completely unrelated cooking recipe", threshold=0.99)
    assert len(cards) == 3  # top-3 floor; nothing beyond 3 clears 0.99


def test_population_is_idempotent(tmp_path):
    d = str(tmp_path / "idx")
    a = ToolIndex(persist_directory=d)
    count_a = a.collection.count()
    b = ToolIndex(persist_directory=d)  # second startup, same dir
    assert b.collection.count() == count_a


def test_stale_tools_are_deleted_on_repopulation(tmp_path):
    d = str(tmp_path / "idx")
    ToolIndex(persist_directory=d)  # full registry
    subset = [s for s in REGISTRY if s.name != "make_ricker"]
    rebuilt = ToolIndex(persist_directory=d, specs=subset)
    names = {c.name for c in rebuilt.search("ricker wavelet", top_k=10, threshold=-1.0)}
    assert "make_ricker" not in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tool_index.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.tool_index'`.

- [ ] **Step 3: Implement `core/tool_index.py`**

```python
"""Semantic index over the tool registry — the discovery half of agentic mode.

Derived entirely from core.tool_registry.REGISTRY: each compute tool renders to
one 'card' embedded into a dedicated ChromaDB collection. Auto-plot TARGETS are
excluded (the LLM must never call plot_* directly; chaining handles plots).
Population is idempotent (content-derived IDs) and self-cleaning (stored IDs no
longer derivable from the registry are deleted, so renamed/removed tools cannot
linger).
"""
import logging
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import RAG_EMBEDDING_MODEL, RAG_VECTOR_DB_PATH
from core.tool_registry import REGISTRY
from knowledge.vector_db import content_id

logger = logging.getLogger(__name__)

COLLECTION_NAME = "tool_index"


def render_card(spec) -> str:
    parts = []
    for pname, meta in spec.params.items():
        required = ", required" if pname in spec.required else ""
        parts.append(f"{pname} ({meta.get('type', 'any')}{required})")
    card = f"{spec.name}: {spec.description}"
    if parts:
        card += " Params: " + ", ".join(parts)
    return card


@dataclass(frozen=True)
class ToolCard:
    name: str
    card: str
    required: tuple
    score: float


class ToolIndex:
    def __init__(self, persist_directory: str | None = None, specs: list | None = None):
        specs = REGISTRY if specs is None else specs
        plot_targets = {s.auto_plot for s in specs if s.auto_plot}
        self._specs = [s for s in specs if s.name not in plot_targets]
        self.client = chromadb.PersistentClient(path=persist_directory or RAG_VECTOR_DB_PATH)
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(
                COLLECTION_NAME,
                metadata={"description": "LLM-facing tool cards", "hnsw:space": "cosine"})
        self.embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL)
        self._populate()

    def _populate(self) -> None:
        cards = {content_id(render_card(s), {"tool": s.name}): s for s in self._specs}
        stored = set(self.collection.get().get("ids", []))
        stale = sorted(stored - set(cards))
        if stale:
            self.collection.delete(ids=stale)
            logger.info(f"tool_index: deleted {len(stale)} stale cards")
        new_ids = [i for i in cards if i not in stored]
        if not new_ids:
            return
        texts = [render_card(cards[i]) for i in new_ids]
        metas = [{"tool": cards[i].name, "required": ",".join(cards[i].required)}
                 for i in new_ids]
        embeddings = self.embedding_model.encode(texts).tolist()
        self.collection.upsert(ids=new_ids, documents=texts,
                               metadatas=metas, embeddings=embeddings)
        logger.info(f"tool_index: embedded {len(new_ids)} cards")

    def search(self, task_description: str, top_k: int = 5,
               threshold: float = 0.2) -> list[ToolCard]:
        """Top-k cards by cosine similarity. The top 3 are ALWAYS returned
        (an on-topic request must never get an empty discovery); results
        beyond the top 3 must clear `threshold`."""
        n = min(top_k, self.collection.count())
        if n == 0:
            return []
        query_embedding = self.embedding_model.encode(task_description).tolist()
        res = self.collection.query(query_embeddings=[query_embedding], n_results=n,
                                    include=["documents", "metadatas", "distances"])
        cards = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                   res["distances"][0]):
            score = 1.0 - dist  # cosine space
            if len(cards) >= 3 and score < threshold:
                continue
            required = tuple(p for p in meta.get("required", "").split(",") if p)
            cards.append(ToolCard(name=meta["tool"], card=doc,
                                  required=required, score=score))
        return cards
```

- [ ] **Step 4: Run new tests, then full suite**

Run: `pytest tests/test_tool_index.py -q` → PASS. Then `pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add core/tool_index.py tests/test_tool_index.py
git commit -m "feat(agentic): semantic ToolIndex over the registry (idempotent, self-cleaning, plot tools excluded)"
```

---

### Task 4: `core/executor_agent.py` — scoped executor subagent

**Files:**
- Create: `core/executor_agent.py`
- Create: `tests/test_executor_agent.py`

**Interfaces:**
- Consumes: `ToolLoopRunner(llm_client, tool_manager, context_manager, max_tool_rounds).run(system_prompt, messages, tools) -> {"reply","images","tools_used"}` (Task 1), `core.tool_registry.REGISTRY_BY_NAME`, `core.tool_registry.to_openai_schema(spec) -> dict`, `core.tool_index.render_card`.
- Produces:
  - `@dataclass TaskResult(summary: str, images: list, tools_used: list, error: str | None = None)`
  - `class ExecutorAgent` with `__init__(self, llm_client, tool_manager, context_manager, max_tool_rounds: int = 5)` and `run(self, brief: str, tool_names: list[str]) -> TaskResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_executor_agent.py
import pytest
from core.executor_agent import ExecutorAgent, TaskResult
from core.tool_manager import ToolManager
from core.context_manager import ContextManager


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


def make_executor(fake_llm_factory, responses):
    llm = fake_llm_factory(responses)
    return ExecutorAgent(llm, ToolManager(), ContextManager()), llm


def test_scoped_schemas_only(fake_llm_factory):
    ex, llm = make_executor(fake_llm_factory, [
        {"content": "<reply>done</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    ex.run("make a 30 Hz ricker", ["make_ricker", "analyze_wedge"])
    sent = llm.calls[0]["tools"]
    assert {t["function"]["name"] for t in sent} == {"make_ricker", "analyze_wedge"}


def test_system_prompt_carries_assigned_cards_only(fake_llm_factory):
    ex, llm = make_executor(fake_llm_factory, [
        {"content": "<reply>done</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    ex.run("brief", ["make_ricker"])
    sys = llm.calls[0]["system_prompt"]
    assert "make_ricker:" in sys
    assert "wedge_model:" not in sys


def test_run_executes_and_returns_task_result(fake_llm_factory):
    ex, llm = make_executor(fake_llm_factory, [
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>30 Hz Ricker built.</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    result = ex.run("make a 30 Hz ricker", ["make_ricker"])
    assert isinstance(result, TaskResult)
    assert result.summary == "30 Hz Ricker built."
    assert result.tools_used == ["make_ricker"]
    assert result.error is None
    assert len(result.images) == 1 and result.images[0].endswith(".png")


def test_unknown_tool_name_is_error_not_exception(fake_llm_factory):
    ex, _ = make_executor(fake_llm_factory, [])
    result = ex.run("brief", ["make_ricker", "nonsense_tool"])
    assert result.error is not None and "nonsense_tool" in result.error
    assert result.summary == "" and result.images == []


def test_shared_context_manager_receives_results(fake_llm_factory):
    cm = ContextManager()
    llm = fake_llm_factory([
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>done</reply>", "tool_calls": None,
         "stop_reason": "stop", "usage": None},
    ])
    ExecutorAgent(llm, ToolManager(), cm).run("brief", ["make_ricker"])
    assert cm.get_context("last_ricker_wavelet") is not None  # visible to the NEXT executor
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_executor_agent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.executor_agent'`.

- [ ] **Step 3: Implement `core/executor_agent.py`**

```python
"""Executor subagent: one task brief, a scoped toolset, the shared tool loop."""
import logging
from dataclasses import dataclass, field
from typing import Optional

from core.tool_loop import ToolLoopRunner
from core.tool_registry import REGISTRY_BY_NAME, to_openai_schema
from core.tool_index import render_card

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM_PROMPT = """You are a seismic modeling task executor. Complete ONE task using only your assigned tools.

Rules:
- Tool results are compacted before you see them: long numeric arrays appear as summaries like "<61 values, min=..., max=...>".
- Any plot a tool produces is displayed to the user automatically — never print or mention image file paths.
- Plot tools run automatically after their matching compute tool — never call a plot_* tool yourself, and never pass raw array data as tool arguments.
- Never pass image_path, interpretation or model arguments yourself — they are supplied automatically from session context.
- When done, state the key quantitative results (e.g. tuning thickness, AVO class, intercept/gradient) inside <reply></reply> XML tags.

Your assigned tools:
{cards}
"""


@dataclass
class TaskResult:
    summary: str
    images: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    error: Optional[str] = None


class ExecutorAgent:
    def __init__(self, llm_client, tool_manager, context_manager, max_tool_rounds: int = 5):
        self._loop = ToolLoopRunner(llm_client, tool_manager, context_manager,
                                    max_tool_rounds=max_tool_rounds)

    def run(self, brief: str, tool_names: list) -> TaskResult:
        unknown = [n for n in tool_names if n not in REGISTRY_BY_NAME]
        if unknown:
            return TaskResult(summary="", error=f"Unknown tool(s): {', '.join(unknown)}")
        specs = [REGISTRY_BY_NAME[n] for n in tool_names]
        schemas = [{"type": "function", "function": to_openai_schema(s)} for s in specs]
        system_prompt = EXECUTOR_SYSTEM_PROMPT.format(
            cards="\n".join(f"- {render_card(s)}" for s in specs))
        try:
            out = self._loop.run(system_prompt,
                                 [{"role": "user", "content": brief}], schemas)
        except Exception as e:  # a failure here must not kill the orchestrator turn
            logger.error(f"Executor failed on brief {brief!r}: {e}")
            return TaskResult(summary="", error=str(e))
        return TaskResult(summary=out["reply"], images=out["images"],
                          tools_used=out["tools_used"])
```

- [ ] **Step 4: Run new tests, then full suite**

Run: `pytest tests/test_executor_agent.py -q` → PASS. Then `pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add core/executor_agent.py tests/test_executor_agent.py
git commit -m "feat(agentic): ExecutorAgent — scoped tool loop returning TaskResult"
```

---

### Task 5: `core/orchestrator.py` — the meta-tool loop

**Files:**
- Create: `core/orchestrator.py`
- Create: `tests/test_orchestrator.py`
- Modify: `tests/test_session_isolation.py` (add orchestrator cases)

**Interfaces:**
- Consumes: `KnowledgeRouter(llm_client, knowledge_base)` (Task 2), `ExecutorAgent(llm_client, tool_manager, context_manager).run(brief, tool_names) -> TaskResult` (Task 4), `ToolIndex.search(task_description, top_k=5, threshold=0.2) -> list[ToolCard]` (Task 3), `core.tool_loop.extract_reply`, `ContextManager`, `ToolManager`, `KnowledgeBase`, `LLMClient`.
- Produces: `class SeismicOrchestrator` with `__init__(self, llm_client=None, tool_manager=None, knowledge_base=None, tool_index=None)`, `new_session()`, `attach_image(path)`, `session_id`, `context_manager`, `process_single_input(user_input) -> {"reply": str, "images": list[str]}` — the same public contract as `SeismicChatBotToolUse`, consumed by Task 6.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_orchestrator.py
import json
import pytest
from core.orchestrator import SeismicOrchestrator, META_TOOL_NAMES, MAX_ORCH_ROUNDS
from core.tool_manager import ToolManager
from core.context_manager import ContextManager
from core.tool_index import ToolCard


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


class FakeToolIndex:
    def __init__(self, cards):
        self._cards = cards
        self.queries = []

    def search(self, task_description, top_k=5, threshold=0.2):
        self.queries.append(task_description)
        return self._cards


class FakeKB:
    def query_knowledge(self, q):
        return {"rag_type": "retrieve_and_generate", "generated_response": "kb", "total_retrieved": 1}

    def get_topic_response(self, topic, section):
        return "canned"


RICKER_CARD = ToolCard(name="make_ricker",
                       card="make_ricker: Creates a Ricker wavelet. Params: frequency (number, required)",
                       required=("frequency",), score=0.8)


def make_orchestrator(fake_llm_factory, responses, cards=(RICKER_CARD,)):
    llm = fake_llm_factory(responses)
    orch = SeismicOrchestrator(llm_client=llm, tool_manager=ToolManager(),
                               knowledge_base=FakeKB(), tool_index=FakeToolIndex(list(cards)))
    return orch, llm


def final(text):
    return {"content": text, "tool_calls": None, "stop_reason": "stop", "usage": None}


def meta(name, args, call_id="c1"):
    return {"content": "", "stop_reason": "tool_calls", "usage": None,
            "tool_calls": [FakeToolCall(name, json.dumps(args), call_id)]}


def test_meta_schemas_are_the_only_tools_sent(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [final("<reply>hi</reply>")])
    orch.process_single_input("make me a wavelet")
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == set(META_TOOL_NAMES) == {"discover_tools", "run_task"}


def test_discover_then_delegate_then_compose(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("discover_tools", {"task_description": "create ricker wavelet"}),
        meta("run_task", {"brief": "Create a 30 Hz Ricker wavelet.",
                          "tool_names": ["make_ricker"]}, "c2"),
        # executor's two calls:
        {"content": "", "stop_reason": "tool_calls", "usage": None,
         "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        final("<reply>Executor: built it.</reply>"),
        # orchestrator composes:
        final("<reply>Your 30 Hz Ricker wavelet is ready.</reply>"),
    ])
    out = orch.process_single_input("create a 30 Hz ricker wavelet")
    assert out["reply"] == "Your 30 Hz Ricker wavelet is ready."
    assert len(out["images"]) == 1 and out["images"][0].endswith(".png")
    # FakeLLMClient records a REFERENCE to the orchestrator's messages list,
    # so calls[0]["messages"] shows the final state; filter by role, don't index.
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "make_ricker:" in tool_msgs[0]["content"]        # discovery result card
    assert "Executor: built it." in tool_msgs[1]["content"]  # run_task summary
    assert ".png" not in tool_msgs[1]["content"]             # image paths stay out-of-band


def test_unknown_tool_name_reported_no_executor(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("run_task", {"brief": "x", "tool_names": ["bogus_tool"]}),
        final("<reply>Cannot do that.</reply>"),
    ])
    out = orch.process_single_input("do something odd")
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "bogus_tool" in tool_msgs[0]["content"]
    assert out["reply"] == "Cannot do that."


def test_empty_discovery_gets_informative_message(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [
        meta("discover_tools", {"task_description": "cook pasta"}),
        final("<reply>I have no tools for that.</reply>"),
    ], cards=())
    orch.process_single_input("cook pasta")
    tool_msgs = [m for m in llm.calls[0]["messages"] if m.get("role") == "tool"]
    assert "No tools matched" in tool_msgs[0]["content"]


def test_round_budget_forces_tool_free_completion(fake_llm_factory):
    responses = [meta("discover_tools", {"task_description": "x"}, f"c{i}")
                 for i in range(MAX_ORCH_ROUNDS)]
    responses.append(final("<reply>Ran out of rounds.</reply>"))
    orch, llm = make_orchestrator(fake_llm_factory, responses)
    out = orch.process_single_input("loop forever")
    assert out["reply"] == "Ran out of rounds."
    assert llm.calls[-1]["tools"] is None


def test_knowledge_question_routes_to_rag_not_meta_loop(fake_llm_factory):
    # get_simple_completion is absent on FakeLLMClient -> keyword fallback; '?' => knowledge
    orch, llm = make_orchestrator(fake_llm_factory, [])
    out = orch.process_single_input("What is seismic tuning?")
    assert out["reply"].startswith("kb")
    assert llm.calls == []  # no meta-loop LLM calls


def test_context_keys_line_in_system_prompt(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [final("<reply>ok</reply>")])
    orch.context_manager.set_context("last_wedge_model", {"x": 1})
    orch.process_single_input("tweak the wedge")
    assert "last_wedge_model" in llm.calls[0]["system_prompt"]


def test_attach_image_and_image_message_route_to_tools(fake_llm_factory):
    orch, llm = make_orchestrator(fake_llm_factory, [final("<reply>photo noted</reply>")])
    orch.attach_image("/sandbox/s1/photo.png")
    out = orch.process_single_input("[image attached: photo.png] interpret this")
    assert out["reply"] == "photo noted"
    assert orch.context_manager.get_context("last_image") == "/sandbox/s1/photo.png"
```

And in `tests/test_session_isolation.py`, add (mirroring the existing classic-bot cases):

```python
def test_orchestrator_new_session_isolates_context(fake_llm_factory):
    from core.orchestrator import SeismicOrchestrator
    base = SeismicOrchestrator(llm_client=fake_llm_factory([]), tool_manager=ToolManager(),
                               knowledge_base=object(), tool_index=object())
    a, b = base.new_session(), base.new_session()
    a.context_manager.set_context("last_wedge_model", {"x": 1})
    assert b.context_manager.get_context("last_wedge_model") is None
    assert a.session_id != b.session_id
    # shared heavy components are reused
    assert a.llm_client is base.llm_client
    assert a.tool_manager is base.tool_manager
    assert a.tool_index is base.tool_index
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.orchestrator'`.

- [ ] **Step 3: Implement `core/orchestrator.py`**

```python
"""Agentic-mode chatbot: a meta-tool loop that discovers tools semantically and
delegates domain tasks to scoped ExecutorAgents.

The LLM in this loop NEVER sees real tool schemas — only discover_tools and
run_task. The system prompt must not grow with the registry."""
import json
import logging
import uuid
from typing import Any, Dict, List

from core.llm_client import LLMClient
from core.tool_manager import ToolManager
from core.context_manager import ContextManager
from core.tool_registry import REGISTRY_BY_NAME
from core.tool_loop import extract_reply
from core.tool_index import ToolIndex
from core.executor_agent import ExecutorAgent
from core.knowledge_router import KnowledgeRouter
from knowledge.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

MAX_ORCH_ROUNDS = 8
META_TOOL_NAMES = ("discover_tools", "run_task")

META_TOOLS = [
    {"type": "function", "function": {
        "name": "discover_tools",
        "description": ("Semantic search over the seismic tool catalog. Describe ONE task "
                        "in plain words; returns the most relevant tools with their "
                        "required parameters. Call this before run_task."),
        "parameters": {"type": "object", "properties": {
            "task_description": {"type": "string",
                                 "description": "Plain-language description of one task."}},
            "required": ["task_description"]}}},
    {"type": "function", "function": {
        "name": "run_task",
        "description": ("Delegate one task to an executor subagent. Give a self-contained "
                        "brief (include every numeric parameter the user supplied) and the "
                        "tool names chosen from discover_tools results. Returns the "
                        "executor's summary; any plots are shown to the user automatically."),
        "parameters": {"type": "object", "properties": {
            "brief": {"type": "string",
                      "description": "Self-contained task instruction for the executor."},
            "tool_names": {"type": "array", "items": {"type": "string"},
                           "description": "Tool names the executor may use."}},
            "required": ["brief", "tool_names"]}}},
]

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a seismic modeling assistant.
You cannot run seismic tools yourself. Instead you decompose the user's request into task
briefs, discover the right tools for each (discover_tools), delegate each task to an
executor subagent (run_task), then compose ONE final answer.

Rules:
- Call discover_tools before run_task; pick tool names only from its results.
- Make each brief self-contained: repeat every parameter value the user gave.
- Executors share session context: a later task can rely on an earlier task's stored
  result (e.g. the earth model built by the previous run_task), so do NOT repeat work.
- A user message beginning "[image attached: ...]" means a photo was uploaded this turn:
  delegate to the outcrop tools. Never pass image_path, interpretation or model values in
  any brief — they are supplied automatically from session context.
- After an interpret_outcrop task, report the regions and the scale estimate WITH its
  confidence, and ask the user to confirm the height before building the model if
  confidence is low or no scale was found.
- Any plot an executor produces is displayed to the user automatically — never mention
  image file paths.
- In your final answer, state the key quantitative results (tuning thickness, AVO class,
  intercept/gradient, etc.).
- Place your final user-facing answer in <reply></reply> XML tags.

{context_line}"""


class SeismicOrchestrator:
    def __init__(self, llm_client=None, tool_manager=None,
                 knowledge_base=None, tool_index=None):
        self.llm_client = llm_client or LLMClient()
        self.tool_manager = tool_manager or ToolManager()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.tool_index = tool_index or ToolIndex()
        self.context_manager = ContextManager()  # per-session, never shared
        self.session_id = uuid.uuid4().hex
        self._knowledge_router = KnowledgeRouter(self.llm_client, self.knowledge_base)

    def new_session(self) -> "SeismicOrchestrator":
        return SeismicOrchestrator(llm_client=self.llm_client,
                                   tool_manager=self.tool_manager,
                                   knowledge_base=self.knowledge_base,
                                   tool_index=self.tool_index)

    def attach_image(self, path: str) -> None:
        self.context_manager.set_context("last_image", path)

    def _system_prompt(self) -> str:
        keys = sorted(self.context_manager.conversation_context.keys())
        line = f"Session context currently holds: {', '.join(keys)}." if keys \
            else "Session context is empty (fresh conversation)."
        return ORCHESTRATOR_SYSTEM_PROMPT.format(context_line=line)

    def process_single_input(self, user_input: str) -> Dict[str, Any]:
        try:
            if self._knowledge_router.is_knowledge_question(user_input):
                reply = self._knowledge_router.handle_knowledge_question(user_input)
                images: List[str] = []
            else:
                result = self._run_meta_loop(user_input)
                reply, images = result["reply"], result["images"]
            if isinstance(reply, bool):
                reply = str(reply)
            elif reply is None:
                reply = "I didn't get a response. Please try again."
            if not reply and images:
                reply = "Here are the results."
            return {"reply": reply, "images": images}
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {"reply": f"I encountered an error: {str(e)}", "images": []}

    def _run_meta_loop(self, user_input: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": user_input}]
        images: List[str] = []
        for _ in range(MAX_ORCH_ROUNDS):
            response = self.llm_client.get_completion(
                system_prompt=self._system_prompt(), user_prompt="",
                tools=META_TOOLS, messages=messages)
            if response.get("usage"):
                self.context_manager.update_token_usage(response["usage"])
            if not response.get("tool_calls"):
                messages.append({"role": "assistant", "content": response["content"]})
                reply = extract_reply(response["content"]) or response["content"]
                return {"reply": reply, "images": images}
            tool_call = response["tool_calls"][0]
            messages.append({"role": "assistant", "content": response["content"],
                             "tool_calls": [tool_call]})
            content = self._dispatch_meta(tool_call, images)
            messages.append({"role": "tool", "tool_call_id": tool_call.id,
                             "content": content})
        final_response = self.llm_client.get_completion(
            system_prompt=self._system_prompt(), user_prompt="",
            tools=None, messages=messages)
        if final_response.get("usage"):
            self.context_manager.update_token_usage(final_response["usage"])
        reply = extract_reply(final_response["content"]) or final_response["content"]
        return {"reply": reply, "images": images}

    def _dispatch_meta(self, tool_call, images: List[str]) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments) \
                if isinstance(tool_call.function.arguments, str) \
                else dict(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return f"Invalid arguments: {e}"
        if name == "discover_tools":
            return self._discover(args.get("task_description", ""))
        if name == "run_task":
            return self._run_task(args.get("brief", ""),
                                  args.get("tool_names") or [], images)
        return f"Unknown meta-tool: {name}. Use discover_tools or run_task."

    def _discover(self, task_description: str) -> str:
        cards = self.tool_index.search(task_description)
        if not cards:
            return "No tools matched; rephrase the task or answer directly."
        return "Matching tools:\n" + "\n".join(f"- {c.card}" for c in cards)

    def _run_task(self, brief: str, tool_names: List[str], images: List[str]) -> str:
        unknown = [n for n in tool_names if n not in REGISTRY_BY_NAME]
        if unknown:
            return (f"Unknown tool name(s): {', '.join(unknown)}. "
                    f"Use names exactly as returned by discover_tools.")
        executor = ExecutorAgent(self.llm_client, self.tool_manager, self.context_manager)
        result = executor.run(brief, tool_names)
        for p in result.images:
            if p not in images:
                images.append(p)
        payload = {"summary": result.summary, "tools_used": result.tools_used}
        if result.error:
            payload["error"] = result.error
        if result.images:
            payload["plots"] = f"{len(result.images)} plot(s) shown to the user"
        return json.dumps(payload)
```

- [ ] **Step 4: Run new tests, then full suite**

Run: `pytest tests/test_orchestrator.py tests/test_session_isolation.py -q` → PASS. Then `pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add core/orchestrator.py tests/test_orchestrator.py tests/test_session_isolation.py
git commit -m "feat(agentic): SeismicOrchestrator meta-tool loop (discover_tools + run_task)"
```

---

### Task 6: Mode wiring + docs

**Files:**
- Modify: `main.py` (add `agentic` to `--mode`, add `build_interface`)
- Modify: `interfaces/gradio_interface.py:52-57` (`create_chat_interface` accepts an injected base bot)
- Modify: `CLAUDE.md` (document agentic mode)
- Create: `tests/test_main_modes.py`

**Interfaces:**
- Consumes: `SeismicOrchestrator()` (Task 5), `create_chat_interface()` (existing).
- Produces: `create_chat_interface(base_bot=None)`; `main.build_interface(mode: str)` returning a Gradio Blocks demo.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_main_modes.py
"""Mode wiring: --mode agentic builds the orchestrator; injection bypasses the default bot."""
import pytest
import interfaces.gradio_interface as gi


class DummyBot:
    session_id = "dummy"

    def new_session(self):
        return self

    def process_single_input(self, text):
        return {"reply": "ok", "images": []}


def test_create_chat_interface_uses_injected_bot(monkeypatch):
    def boom():
        raise AssertionError("default bot must not be constructed when base_bot is given")
    monkeypatch.setattr(gi, "SeismicChatBotToolUse", boom)
    demo = gi.create_chat_interface(base_bot=DummyBot())
    assert demo is not None


def test_build_interface_agentic_uses_orchestrator(monkeypatch):
    import main
    built = {}

    def fake_orchestrator():
        built["orchestrator"] = True
        return DummyBot()

    def fake_create(base_bot=None):
        built["base_bot"] = base_bot
        return "demo"

    monkeypatch.setattr(main, "create_chat_interface", fake_create)
    monkeypatch.setattr("core.orchestrator.SeismicOrchestrator", fake_orchestrator)
    demo = main.build_interface("agentic")
    assert demo == "demo"
    assert built["orchestrator"] is True
    assert isinstance(built["base_bot"], DummyBot)


def test_build_interface_default_mode_passes_no_bot(monkeypatch):
    import main
    seen = {}

    def fake_create(base_bot=None):
        seen["base_bot"] = base_bot
        return "demo"

    monkeypatch.setattr(main, "create_chat_interface", fake_create)
    assert main.build_interface("tool-use") == "demo"
    assert seen["base_bot"] is None
```

Note: importing `main` executes `demo = create_chat_interface()` at module level, which builds a real bot needing credentials. Follow the existing suite's env-credential fixture pattern (see `tests/test_llm_credentials.py` for the `importlib.reload`/monkeypatched-env approach) — set fake `DEEPSEEK_API_KEY`/`DEEPSEEK_BASE_URL` env vars in a module-level fixture before importing `main`, exactly as other tests that import credential-touching modules do. If import stays too heavy (it builds `KnowledgeBase`), move the module-level `demo = create_chat_interface()` in `main.py` inside `main()` as part of this task — nothing but hot-reloading used it, and the tests then import `main` cheaply.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main_modes.py -q`
Expected: FAIL — `create_chat_interface() got an unexpected keyword argument 'base_bot'` / `main has no attribute 'build_interface'`.

- [ ] **Step 3: Implement the wiring**

`interfaces/gradio_interface.py` — only the factory head changes:

```python
def create_chat_interface(base_bot=None):
    """Create the Gradio chat interface.

    base_bot: any object with new_session()/process_single_input()/attach_image()
    (SeismicChatBotToolUse or SeismicOrchestrator). Default: the classic bot.
    """
    if base_bot is None:
        base_bot = SeismicChatBotToolUse()
    ...  # rest of the function unchanged
```

`main.py` — add `agentic` to choices and a testable builder; `main()` calls it:

```python
def build_interface(mode: str):
    """Build the Gradio demo for the given --mode."""
    if mode == "legacy":
        from interfaces.gradio_interface_legacy import create_chat_interface as create_legacy
        return create_legacy()
    if mode == "agentic":
        from core.orchestrator import SeismicOrchestrator
        return create_chat_interface(base_bot=SeismicOrchestrator())
    return create_chat_interface()
```

with `choices=["tool-use", "agentic", "legacy"]` on the `--mode` argument and the `main()` launch branch replaced by `demo = build_interface(args.mode)`.

`CLAUDE.md` — add a short `## Agentic mode (orchestrator + subagents)` section after "Request flow": one paragraph naming the four modules (`core/tool_index.py`, `core/executor_agent.py`, `core/orchestrator.py`, shared `core/tool_loop.py`/`core/knowledge_router.py`), the meta-tools, `python main.py --mode agentic`, the `tool_index` ChromaDB collection (regenerable, self-cleaning), and that the classic loop remains the default. Point to the spec file.

- [ ] **Step 4: Run new tests, then full suite**

Run: `pytest tests/test_main_modes.py -q` → PASS. Then `pytest -q` → green.

- [ ] **Step 5: Smoke-run the UI (manual, credential-gated)**

Run: `python main.py --mode agentic` and send "create a 30 Hz Ricker wavelet" plus one two-step request ("build a wedge at 25 Hz, then give me the AVO attributes for vp 3000→3200, vs 1500→1600, rho 2.3→2.4"). Verify plots render and the reply narrates numbers. This is a smoke check, not a test — skip if no credentials are configured, and say so in the commit/PR notes.

- [ ] **Step 6: Commit**

```bash
git add main.py interfaces/gradio_interface.py tests/test_main_modes.py CLAUDE.md
git commit -m "feat(agentic): --mode agentic wires SeismicOrchestrator into the Gradio UI"
```
