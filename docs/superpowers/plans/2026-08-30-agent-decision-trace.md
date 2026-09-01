# Agent Decision Trace (Tier 0 + Tier 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every agent decision (intent split, tool selection, arg provenance, auto-plot outcomes, failures, budgets, discovery scores, per-call tokens) observable via structured per-turn trace events, persisted as per-session JSONL and returned additively as a `"trace"` key.

**Architecture:** A new stdlib-only `core/turn_trace.py::TraceRecorder` hangs off the per-session `ContextManager` (already threaded into both request flows). Events are emitted at the three chokepoints (`ToolLoopRunner.run`, `KnowledgeRouter`, `SeismicOrchestrator` meta-loop) plus `LLMClient`. `process_single_input` in both bots wraps a turn with `begin_turn`/`end_turn` and returns the record. Tier-0 logging hardening rides alongside.

**Tech Stack:** Python 3.9.7, stdlib only (json/os/time/uuid/logging), pytest with the existing `fake_llm_factory` conftest fixture.

**Spec:** `docs/superpowers/specs/2026-08-30-agent-decision-trace-spec.md` (read it first — it defines the event vocabulary and constraints).

## Global Constraints

- Python 3.9.7 — every new/edited module keeps or adds `from __future__ import annotations`; use `typing.Optional[X]`, never `X | None` in annotations at runtime positions.
- No new runtime dependencies; `core/turn_trace.py` imports stdlib only.
- `ToolLoopRunner.run` must keep returning exactly the keys `{"reply", "images", "tools_used"}` (tests pin this).
- All reads of LLM response dicts use `.get()` — scripted `FakeLLMClient` responses are plain dicts without `model`/`latency_ms`.
- `tests/conftest.py::FakeLLMClient` has **no** `get_simple_completion` — existing tests rely on intent classification falling back to keywords via `AttributeError`. Do not add that method to the conftest fake and do not make the router call `get_completion` for classification.
- Trace events record parameter **names** (provenance), never parameter values; user input and briefs truncated to 200 chars.
- Working dir for all commands: `/Users/kumono/Desktop/devs/dash-apps/wedge-model/geo-mcp/seismic_chatbot` (its own git repo — commit from inside it, never from the outer repos).
- Branch: all work on `agent-decision-trace`, created from `stabilize-tool-layer` (Task 1 Step 0).
- Full suite takes ~200s — always run `pytest` with a generous timeout (≥ 360s); prefer single-file runs during tasks and one full run in the final task.

---

### Task 1: TraceRecorder module (`core/turn_trace.py`)

**Files:**
- Create: `core/turn_trace.py`
- Modify: `config/settings.py` (add `SEISMIC_TRACE_DIR` after the `SEISMIC_UPLOAD_DIR`/`MAX_IMAGE_MB` block, ~line 21)
- Test: `tests/test_turn_trace.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces (later tasks rely on these exact names):
  - `class TraceRecorder`: `__init__(self, session_id: Optional[str] = None, persist_dir: Optional[str] = None)`, attrs `session_id: str`, `persist_dir: Optional[str]`, `turn: int`, `events: List[Dict[str, Any]]`; methods `begin_turn(self, user_input: str) -> None`, `emit(self, t: str, **fields) -> None`, `end_turn(self) -> Dict[str, Any]`.
  - `emit_event(context_manager, t: str, **fields) -> None` (module function; safe no-op when `context_manager` is None or has no `trace` attr).
  - `usage_dict(usage) -> Dict[str, int]` (module function; `{}` for None/garbage).
  - `config.settings.SEISMIC_TRACE_DIR: str`.

- [ ] **Step 0: Create the working branch**

```bash
cd /Users/kumono/Desktop/devs/dash-apps/wedge-model/geo-mcp/seismic_chatbot
git checkout stabilize-tool-layer && git checkout -b agent-decision-trace
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_turn_trace.py`:

```python
from types import SimpleNamespace

from core.turn_trace import TraceRecorder, emit_event, usage_dict


def test_recorder_accumulates_and_flushes(tmp_path):
    rec = TraceRecorder(session_id="abc123", persist_dir=str(tmp_path))
    rec.begin_turn("make a 30 Hz ricker wavelet " + "x" * 300)
    rec.emit("tool_call", tool="make_ricker", ok=True, ms=1.2)
    rec.emit("tool_call", tool="bad_tool", ok=False, error="boom")
    record = rec.end_turn()
    assert record["session"] == "abc123"
    assert record["turn"] == 1
    assert record["tools_used"] == ["make_ricker"]  # ok=False excluded
    assert record["events"][0]["t"] == "turn_start"
    assert len(record["events"][0]["input"]) <= 200  # truncated
    # one JSONL line per turn, named by session
    lines = (tmp_path / "abc123.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    import json
    assert json.loads(lines[0])["turn"] == 1


def test_second_turn_resets_events(tmp_path):
    rec = TraceRecorder(session_id="s", persist_dir=str(tmp_path))
    rec.begin_turn("one")
    rec.emit("intent", verdict="TOOL", via="llm")
    rec.end_turn()
    rec.begin_turn("two")
    record = rec.end_turn()
    assert record["turn"] == 2
    assert all(e["t"] != "intent" for e in record["events"])
    assert len((tmp_path / "s.jsonl").read_text().strip().splitlines()) == 2


def test_persist_failure_is_swallowed():
    rec = TraceRecorder(session_id="s", persist_dir="/dev/null/not-a-dir")
    rec.begin_turn("hello")
    record = rec.end_turn()  # must not raise
    assert record["turn"] == 1


def test_no_persist_dir_disables_writes():
    rec = TraceRecorder(session_id="s", persist_dir="")
    rec.begin_turn("hello")
    assert rec.end_turn()["turn"] == 1


def test_emit_event_is_safe_without_trace():
    emit_event(None, "intent", verdict="TOOL")           # no-op, no raise
    emit_event(object(), "intent", verdict="TOOL")       # no trace attr: no-op
    cm = SimpleNamespace(trace=TraceRecorder(session_id="s", persist_dir=""))
    emit_event(cm, "intent", verdict="TOOL", via="llm")
    assert cm.trace.events[-1] == {**cm.trace.events[-1]}  # exists
    assert cm.trace.events[-1]["verdict"] == "TOOL"


def test_usage_dict_tolerates_shapes():
    assert usage_dict(None) == {}
    assert usage_dict({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}) == {
        "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    obj = SimpleNamespace(prompt_tokens=5, completion_tokens=6, total_tokens=11)
    assert usage_dict(obj)["total_tokens"] == 11


def test_settings_expose_trace_dir():
    from config.settings import SEISMIC_TRACE_DIR
    assert isinstance(SEISMIC_TRACE_DIR, str) and SEISMIC_TRACE_DIR
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_turn_trace.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'core.turn_trace'`

- [ ] **Step 3: Implement**

Add to `config/settings.py` (after the `MAX_IMAGE_MB` line, keeping the existing `tempfile` import at the top of the file — it is already imported):

```python
# Per-session decision-trace JSONL sink (core/turn_trace.py). One file per
# session at <SEISMIC_TRACE_DIR>/<session_id>.jsonl; set SEISMIC_TRACE_DIR=""
# is not supported via env (empty env vars fall through to the default) —
# pass persist_dir="" to TraceRecorder to disable writes in code/tests.
SEISMIC_TRACE_DIR = os.environ.get("SEISMIC_TRACE_DIR") or os.path.join(
    tempfile.gettempdir(), "seismic_traces"
)
```

Create `core/turn_trace.py`:

```python
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
        return record

    def _persist(self, record: Dict[str, Any]) -> None:
        if not self.persist_dir:
            return
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            path = os.path.join(self.persist_dir, f"{self.session_id}.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"trace persist failed: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_turn_trace.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add core/turn_trace.py config/settings.py tests/test_turn_trace.py
git commit -m "feat(trace): TraceRecorder — per-session JSONL decision trace (Tier 1 core)"
```

---

### Task 2: Tier-0 logging hardening

**Files:**
- Modify: `config/settings.py:40` (`LOG_LEVEL`)
- Modify: `interfaces/gradio_interface.py` (module top), `interfaces/api_interface.py` (module top)
- Modify: `core/tool_manager.py:54`, `knowledge/vector_db.py:207`
- Modify: `core/chatbot_tool_use.py:303`, `core/orchestrator.py:117`, `core/tool_loop.py:270,475`, `core/executor_agent.py:51` (`exc_info=True`)
- Test: `tests/test_turn_trace.py` (append one test)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `config.settings.LOG_LEVEL` env-overridable. No API changes.

- [ ] **Step 1: Write the failing test** — append to `tests/test_turn_trace.py`:

```python
def test_log_level_env_override(monkeypatch):
    import importlib
    import config.settings as settings
    try:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        importlib.reload(settings)
        assert settings.LOG_LEVEL == "DEBUG"
    finally:
        monkeypatch.undo()
        importlib.reload(settings)
    assert settings.LOG_LEVEL == "INFO"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_turn_trace.py::test_log_level_env_override -q`
Expected: FAIL — `assert settings.LOG_LEVEL == "DEBUG"` (currently hardcoded `"INFO"`)

- [ ] **Step 3: Implement**

In `config/settings.py` replace `LOG_LEVEL = "INFO"` with:

```python
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
```

In `interfaces/gradio_interface.py` add right after the imports (before `def append_bot_response`):

```python
import logging
from config.settings import LOG_LEVEL, LOG_FORMAT

# No-op when main.py already configured the root logger; makes direct imports
# / uvicorn-style launches emit logs instead of silence.
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
```

In `interfaces/api_interface.py` add the same 4 lines after its imports (note: it imports `os` already; add `import logging` and extend the existing `config` imports).

In `core/tool_manager.py` change line 54 `logger.debug(f"Calling {tool_name} with {full_params}")` → `logger.info(...)` (same f-string), and delete the now-redundant `logger.info(f"Processing tool call: {tool_name} with input: {tool_input}")` at line 58 **replacing** it with `logger.debug(f"process_tool_call: {tool_name} raw input keys: {sorted(tool_input)}")` (the INFO line now shows *resolved* params instead of pre-default ones; raw-arg values stop being duplicated at INFO).

In `knowledge/vector_db.py` line 207 change `logger.debug(f"Search query '{query}' returned {len(processed_results)} results")` → `logger.info(...)`.

Add `exc_info=True` to these existing `logger.error` calls (message text unchanged):
- `core/tool_loop.py:270` (`Error in automatic chaining`), `core/tool_loop.py:475` (`Tool execution failed`)
- `core/chatbot_tool_use.py:303` (`Error processing input`)
- `core/orchestrator.py:117` (`Error processing input`)
- `core/executor_agent.py:51` (`Executor failed on brief`)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_turn_trace.py tests/test_tool_manager.py -q`
Expected: all pass (tool_manager tests pin registry derivation, not log levels; if any asserts on the exact INFO string of `process_tool_call`, update that assertion to the new resolved-params line).

- [ ] **Step 5: Commit**

```bash
git add config/settings.py interfaces/gradio_interface.py interfaces/api_interface.py core/tool_manager.py knowledge/vector_db.py core/tool_loop.py core/chatbot_tool_use.py core/orchestrator.py core/executor_agent.py tests/test_turn_trace.py
git commit -m "feat(obs): Tier-0 logging — env LOG_LEVEL, interface basicConfig, resolved-params at INFO, exc_info on error paths"
```

---

### Task 3: Wire the recorder into ContextManager + LLMClient

**Files:**
- Modify: `core/context_manager.py` (`__init__`)
- Modify: `core/llm_client.py` (`get_completion`, `get_simple_completion`)
- Test: `tests/test_turn_trace.py` (append)

**Interfaces:**
- Consumes: Task 1 (`TraceRecorder`, `emit_event`, `usage_dict`).
- Produces:
  - `ContextManager.trace: TraceRecorder` (fresh per instance).
  - `LLMClient.get_completion(...)` result dict gains `"model": str` and `"latency_ms": float`.
  - `LLMClient.get_simple_completion(self, system_prompt: str, user_prompt: str, context_manager=None) -> str` — when `context_manager` given: updates its token usage and emits an `llm` event.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_turn_trace.py`:

```python
def test_context_manager_owns_a_recorder():
    from core.context_manager import ContextManager
    cm = ContextManager()
    assert isinstance(cm.trace, TraceRecorder)
    cm2 = ContextManager()
    assert cm.trace is not cm2.trace


def _bare_llm_client(content="hi", usage=None):
    """Real LLMClient minus credential resolution, with a stubbed transport."""
    from core.llm_client import LLMClient
    client = object.__new__(LLMClient)
    client.model, client.temperature, client.max_tokens = "test-model", 0.1, 100

    class _Msg:
        pass
    msg = _Msg()
    msg.content, msg.tool_calls = content, None

    class _Choice:
        pass
    choice = _Choice()
    choice.message = msg

    class _Resp:
        pass
    resp = _Resp()
    resp.choices, resp.usage = [choice], usage

    class _Completions:
        def create(self, **kw):
            return resp

    class _Chat:
        pass
    chat = _Chat()
    chat.completions = _Completions()

    class _Client:
        pass
    inner = _Client()
    inner.chat = chat
    client.client = inner
    return client


def test_get_completion_reports_model_and_latency():
    client = _bare_llm_client()
    res = client.get_completion("s", "u")
    assert res["model"] == "test-model"
    assert isinstance(res["latency_ms"], float)


def test_get_simple_completion_accounts_tokens_and_traces():
    from core.context_manager import ContextManager
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    client = _bare_llm_client(content="  KNOWLEDGE  ", usage=usage)
    cm = ContextManager()
    cm.trace.persist_dir = ""  # keep the test filesystem-clean
    out = client.get_simple_completion("s", "u", context_manager=cm)
    assert out == "KNOWLEDGE"
    assert cm.get_token_usage()["total_tokens"] == 10
    llm_events = [e for e in cm.trace.events if e["t"] == "llm"]
    assert llm_events and llm_events[0]["total_tokens"] == 10
    assert llm_events[0]["model"] == "test-model"


def test_get_simple_completion_without_context_manager_unchanged():
    client = _bare_llm_client(content="plain")
    assert client.get_simple_completion("s", "u") == "plain"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_turn_trace.py -q`
Expected: the four new tests FAIL (`AttributeError: ... 'trace'`, missing `model` key, unexpected kwarg `context_manager`).

- [ ] **Step 3: Implement**

`core/context_manager.py` — add at top `from core.turn_trace import TraceRecorder` and in `__init__` (after `self.token_usage = {...}`):

```python
        # Per-session decision-trace recorder (core/turn_trace.py). The owning
        # bot stamps its session_id onto it right after construction.
        self.trace = TraceRecorder()
```

`core/llm_client.py`:
- Add imports: `import time` and `from core.turn_trace import emit_event, usage_dict`.
- In `get_completion`, wrap the API call with timing and extend the result dict:

```python
            start = time.perf_counter()
            response = self.client.chat.completions.create(**api_params)
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
```

and in the returned `result` dict add two keys:

```python
                "model": self.model,
                "latency_ms": latency_ms,
```

- Replace `get_simple_completion` with:

```python
    def get_simple_completion(self, system_prompt: str, user_prompt: str,
                              context_manager=None) -> str:
        """Text-only completion. When a context_manager is supplied, its token
        counter and decision trace are updated — this is how KnowledgeRouter
        calls (intent classification, no-RAG fallback) become accountable."""
        response = self.get_completion(system_prompt, user_prompt)
        if context_manager is not None:
            if response.get("usage"):
                context_manager.update_token_usage(response["usage"])
            emit_event(context_manager, "llm",
                       model=response.get("model"),
                       latency_ms=response.get("latency_ms"),
                       **usage_dict(response.get("usage")))
        content = response.get("content", "")
        return content.strip() if content else ""
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_turn_trace.py tests/test_session_isolation.py tests/test_llm_credentials.py -q`
Expected: all pass (session-isolation tests confirm the new per-instance recorder doesn't leak across sessions).

- [ ] **Step 5: Commit**

```bash
git add core/context_manager.py core/llm_client.py tests/test_turn_trace.py
git commit -m "feat(trace): recorder on ContextManager; LLMClient reports model/latency and accounts simple completions"
```

---

### Task 4: KnowledgeRouter — intent provenance, RAG scores, token accounting

**Files:**
- Modify: `core/knowledge_router.py` (`__init__`, new `classify`, `is_knowledge_question`, `_classify_intent_with_llm`, `handle_knowledge_question`, `_handle_no_rag_results`)
- Modify: `core/orchestrator.py:84` and `core/chatbot_tool_use.py:70-79` (pass `context_manager` into the router)
- Test: `tests/test_knowledge_router.py` (append)

**Interfaces:**
- Consumes: Task 1 (`emit_event`), Task 3 (`get_simple_completion(..., context_manager=)`).
- Produces: `KnowledgeRouter.__init__(self, llm_client, knowledge_base, context_manager=None)`; `KnowledgeRouter.classify(self, user_input: str) -> dict` returning exactly `{"is_knowledge": bool, "via": str}` with `via` ∈ {"llm", "keyword_fallback", "image_shortcut"}; `is_knowledge_question` unchanged signature, delegates to `classify`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_knowledge_router.py`:

```python
from core.context_manager import ContextManager
from core.knowledge_router import KnowledgeRouter


def _cm():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    return cm


def test_classify_image_shortcut_emits_intent():
    cm = _cm()
    router = KnowledgeRouter(None, None, context_manager=cm)
    verdict = router.classify("[image attached: x.png] interpret this")
    assert verdict == {"is_knowledge": False, "via": "image_shortcut"}
    intents = [e for e in cm.trace.events if e["t"] == "intent"]
    assert intents[0]["verdict"] == "TOOL" and intents[0]["via"] == "image_shortcut"


def test_classify_via_llm():
    class SimpleFake:
        def get_simple_completion(self, s, u, context_manager=None):
            return "KNOWLEDGE"
    cm = _cm()
    router = KnowledgeRouter(SimpleFake(), None, context_manager=cm)
    verdict = router.classify("How does frequency affect resolution")
    assert verdict == {"is_knowledge": True, "via": "llm"}


def test_classify_falls_back_to_keywords_and_records_it():
    class BrokenFake:
        def get_simple_completion(self, s, u, context_manager=None):
            raise RuntimeError("down")
    cm = _cm()
    router = KnowledgeRouter(BrokenFake(), None, context_manager=cm)
    verdict = router.classify("what is a ricker wavelet?")
    assert verdict == {"is_knowledge": True, "via": "keyword_fallback"}
    intents = [e for e in cm.trace.events if e["t"] == "intent"]
    assert intents[0]["via"] == "keyword_fallback"


def test_classify_tolerates_legacy_two_arg_fake():
    class LegacyFake:  # old signature without context_manager kwarg
        def get_simple_completion(self, s, u):
            return "TOOL"
    router = KnowledgeRouter(LegacyFake(), None, context_manager=_cm())
    assert router.classify("make a wedge model") == {"is_knowledge": False, "via": "llm"}


def test_handle_knowledge_question_emits_rag_scores():
    class FakeKB:
        def query_knowledge(self, q):
            return {"rag_type": "retrieve_and_generate", "generated_response": "answer",
                    "total_retrieved": 2,
                    "retrieved_documents": [{"score": 0.8123}, {"score": 0.5}]}
    cm = _cm()
    router = KnowledgeRouter(None, FakeKB(), context_manager=cm)
    out = router.handle_knowledge_question("what is tuning?")
    assert "answer" in out
    rag = [e for e in cm.trace.events if e["t"] == "rag"][0]
    assert rag["rag_type"] == "retrieve_and_generate"
    assert rag["retrieved"] == 2
    assert rag["scores"] == [0.8123, 0.5]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_knowledge_router.py -q`
Expected: new tests FAIL (`__init__` rejects `context_manager`, no `classify`).

- [ ] **Step 3: Implement** in `core/knowledge_router.py`

Add import: `from core.turn_trace import emit_event`.

Replace `__init__` and `is_knowledge_question`, add `classify` and `_simple`:

```python
    def __init__(self, llm_client, knowledge_base, context_manager=None):
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base
        self.context_manager = context_manager

    def _simple(self, system_prompt: str, user_prompt: str) -> str:
        """get_simple_completion with token/trace accounting; tolerates legacy
        fakes whose signature lacks the context_manager kwarg."""
        try:
            return self.llm_client.get_simple_completion(
                system_prompt, user_prompt, context_manager=self.context_manager)
        except TypeError:
            return self.llm_client.get_simple_completion(system_prompt, user_prompt)

    def classify(self, user_input: str) -> dict:
        """Three-way intent decision with provenance: which branch decided."""
        if user_input.lstrip().startswith("[image attached"):
            verdict = {"is_knowledge": False, "via": "image_shortcut"}
        else:
            try:
                verdict = {"is_knowledge": self._classify_intent_with_llm(user_input),
                           "via": "llm"}
            except Exception as e:
                logger.error(f"LLM intent classification failed: {e}")
                verdict = {"is_knowledge": self._is_knowledge_question_keywords(user_input),
                           "via": "keyword_fallback"}
        label = "KNOWLEDGE" if verdict["is_knowledge"] else "TOOL"
        logger.info(f"intent: {label} (via {verdict['via']})")
        emit_event(self.context_manager, "intent", verdict=label, via=verdict["via"])
        return verdict

    def is_knowledge_question(self, user_input: str) -> bool:
        return self.classify(user_input)["is_knowledge"]
```

In `_classify_intent_with_llm` change the call `self.llm_client.get_simple_completion(system_prompt, user_input)` → `self._simple(system_prompt, user_input)`. Same change in `_handle_no_rag_results` (the `get_simple_completion` call at ~line 305).

In `handle_knowledge_question`, right after `rag_response = self.knowledge_base.query_knowledge(user_input)` insert:

```python
            docs = rag_response.get('retrieved_documents') or []
            emit_event(self.context_manager, "rag",
                       rag_type=rag_response.get('rag_type'),
                       retrieved=rag_response.get('total_retrieved', 0),
                       scores=[round(d.get('score', 0.0), 4) for d in docs
                               if isinstance(d, dict)])
```

Wire the call sites:
- `core/orchestrator.py:84`: `self._knowledge_router = KnowledgeRouter(self.llm_client, self.knowledge_base, self.context_manager)`
- `core/chatbot_tool_use.py` `_knowledge_router` property: add third arg `getattr(self, "context_manager", None)`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_knowledge_router.py tests/test_rag_no_results.py tests/test_chatbot.py -q`
Expected: all pass. (`test_rag_no_results` exercises `_handle_no_rag_results` end-to-end; the `TypeError`-fallback in `_simple` keeps any scripted fake working.)

- [ ] **Step 5: Commit**

```bash
git add core/knowledge_router.py core/orchestrator.py core/chatbot_tool_use.py tests/test_knowledge_router.py
git commit -m "feat(trace): intent provenance (llm/keyword/image), RAG scores, router token accounting"
```

---

### Task 5: ToolLoopRunner instrumentation

**Files:**
- Modify: `core/tool_loop.py` (`run`, `handle_automatic_chaining` error log only)
- Test: Create `tests/test_tool_loop_trace.py`

**Interfaces:**
- Consumes: Tasks 1+3 (`emit_event`, `usage_dict`, `ContextManager.trace`).
- Produces: no signature changes. `run` still returns `{"reply", "images", "tools_used"}`. New events on the session recorder: `llm`, `parallel_calls_dropped`, `tool_call` (with `injected`/`overridden`/`defaults_filled` name lists), `auto_plot`, `budget_exhausted`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_tool_loop_trace.py`:

```python
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


def _runner(responses, max_rounds=5):
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("test")
    return ToolLoopRunner(FakeLLM(responses), ToolManager(), cm,
                          max_tool_rounds=max_rounds), cm


def test_successful_tool_call_and_auto_plot_are_traced():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses)
    out = runner.run("sys", [{"role": "user", "content": "30 Hz ricker"}], tools=[])
    assert set(out) == {"reply", "images", "tools_used"}  # contract unchanged
    events = cm.trace.events
    tool_evts = [e for e in events if e["t"] == "tool_call"]
    assert tool_evts[0]["tool"] == "make_ricker" and tool_evts[0]["ok"] is True
    assert isinstance(tool_evts[0]["defaults_filled"], list)
    assert tool_evts[0]["injected"] == [] and tool_evts[0]["overridden"] == []
    auto = [e for e in events if e["t"] == "auto_plot"][0]
    assert auto == {**auto, "compute": "make_ricker", "plot": "plot_ricker", "fired": True}
    llm_evts = [e for e in events if e["t"] == "llm"]
    assert len(llm_evts) == 2 and llm_evts[0]["tool_call"] is True
    assert llm_evts[0]["total_tokens"] == 7
    assert cm.get_token_usage()["total_tokens"] == 7


def test_failed_tool_call_is_traced_with_error():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("no_such_tool", '{}')], "usage": None},
        {"content": "<reply>sorry</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["tools_used"] == []
    evt = [e for e in cm.trace.events if e["t"] == "tool_call"][0]
    assert evt["ok"] is False and "Unknown tool" in evt["error"]


def test_parallel_tool_calls_dropped_event():
    responses = [
        {"content": "", "tool_calls": [
            FakeToolCall("make_ricker", '{"frequency": 30}', call_id="c1"),
            FakeToolCall("wedge_model", '{}', call_id="c2")], "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses)
    runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    dropped = [e for e in cm.trace.events if e["t"] == "parallel_calls_dropped"][0]
    assert dropped["dropped"] == ["wedge_model"]


def test_budget_exhaustion_is_traced():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>forced</reply>", "tool_calls": None, "usage": None},
    ]
    runner, cm = _runner(responses, max_rounds=1)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert out["reply"] == "forced"
    budget = [e for e in cm.trace.events if e["t"] == "budget_exhausted"]
    assert budget and budget[0]["rounds"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_tool_loop_trace.py -q`
Expected: 4 FAIL (no events emitted yet).

- [ ] **Step 3: Implement** in `core/tool_loop.py`

Add imports: `import time` and `from core.turn_trace import emit_event, usage_dict`.

Add a private helper to `ToolLoopRunner`:

```python
    def _emit_llm(self, response: Dict[str, Any]) -> None:
        emit_event(self.context_manager, "llm",
                   model=response.get("model"),
                   latency_ms=response.get("latency_ms"),
                   tool_call=bool(response.get("tool_calls")),
                   **usage_dict(response.get("usage")))
```

In `run()`:

1. After each of the two `update_token_usage` blocks (the in-loop one at ~line 432-433 and the forced-completion one at ~line 495-496), add `self._emit_llm(response)` / `self._emit_llm(final_response)` respectively.

2. Right after `tool_call = response["tool_calls"][0]` (~line 445) insert:

```python
            if len(response["tool_calls"]) > 1:
                dropped = [tc.function.name for tc in response["tool_calls"][1:]]
                logger.warning(
                    f"Executing only the first of {len(response['tool_calls'])} "
                    f"requested tool calls; dropped: {dropped}")
                emit_event(self.context_manager, "parallel_calls_dropped", dropped=dropped)
```

3. Replace the body of the `try:` block (currently lines ~455-471) with the version below — the behavior-preserving diff is: parse and injection are split so arg provenance can be diffed, the tool call is timed, a `tool_call` event is emitted, and the auto-plot outcome is emitted:

```python
                raw_input = self.parse_tool_input(tool_input_str)
                tool_input = self.inject_context_inputs(tool_name, raw_input)
                injected = sorted(k for k in tool_input if k not in raw_input)
                overridden = sorted(
                    k for k in raw_input
                    if k in tool_input
                    and isinstance(raw_input.get(k), str)
                    and isinstance(tool_input.get(k), str)
                    and raw_input[k] != tool_input[k])
                spec = getattr(self.tool_manager, "specs", {}).get(tool_name)
                defaults_filled = sorted(
                    k for k in spec.defaults if k not in tool_input) if spec else []
                started = time.perf_counter()
                tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
                emit_event(self.context_manager, "tool_call", tool=tool_name, ok=True,
                           ms=round((time.perf_counter() - started) * 1000, 1),
                           injected=injected, overridden=overridden,
                           defaults_filled=defaults_filled)
                tools_used.append(tool_name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self.compact_tool_result(tool_result)
                })
                self.update_context(tool_name, tool_input, tool_result)
                self.harvest_images(tool_result, collected_images)

                # Auto-chaining still runs the partner plot tool; its plot now
                # joins the harvest instead of ending the turn.
                chained_result = self.handle_automatic_chaining(tool_name, tool_input, tool_result)
                if chained_result:
                    self.harvest_images(chained_result, collected_images)
                    emit_event(self.context_manager, "auto_plot", compute=tool_name,
                               plot=AUTO_PLOT.get(tool_name), fired=True)
                elif AUTO_PLOT.get(tool_name):
                    logger.warning(
                        f"auto-plot {AUTO_PLOT[tool_name]} did not run after "
                        f"{tool_name} (missing context or plot error)")
                    emit_event(self.context_manager, "auto_plot", compute=tool_name,
                               plot=AUTO_PLOT[tool_name], fired=False)
```

4. In the `except Exception as e:` handler, after the `logger.error(...)` line add:

```python
                emit_event(self.context_manager, "tool_call", tool=tool_name,
                           ok=False, error=str(e))
```

5. Before the forced final completion (just after the loop body ends, ~line 487), add:

```python
        logger.warning(
            f"Tool-round budget ({self.max_tool_rounds}) exhausted; forcing "
            f"tool-free completion")
        emit_event(self.context_manager, "budget_exhausted", rounds=self.max_tool_rounds)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_tool_loop_trace.py -q` then the neighbors: `pytest tests/test_chatbot_synthetic.py tests/test_wedge_gather.py tests/test_session_isolation.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/tool_loop.py tests/test_tool_loop_trace.py
git commit -m "feat(trace): tool-loop events — per-call llm/tool_call with arg provenance, parallel-drop, auto-plot outcome, budget"
```

---

### Task 6: Orchestrator instrumentation + `"trace"` in agentic replies

**Files:**
- Modify: `core/orchestrator.py` (`__init__`, `process_single_input`, `_run_meta_loop`, `_discover`, `_run_task`)
- Test: Create `tests/test_orchestrator_trace.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 5.
- Produces: `SeismicOrchestrator.process_single_input` returns `{"reply": str, "images": list, "trace": dict}` where `trace` is `TraceRecorder.end_turn()`'s record. New events: `discover` (`query`, `hits=[[name, score], ...]`), `run_task` (`brief` truncated to 200, `tool_names`, `tools_used`, `error`, `n_images`), plus `llm`/`budget_exhausted` from the meta loop.

- [ ] **Step 1: Write the failing tests** — create `tests/test_orchestrator_trace.py`:

```python
from core.context_manager import ContextManager  # noqa: F401  (parity import)
from core.orchestrator import SeismicOrchestrator
from core.tool_index import ToolCard
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
    # no get_simple_completion → intent classification keyword-falls-back (existing contract)


class FakeIndex:
    def search(self, q, top_k=5, threshold=0.2):
        return [ToolCard(name="make_ricker", card="make_ricker: Ricker wavelet",
                         required=("frequency",), score=0.9)]


def _orchestrator(responses):
    orch = SeismicOrchestrator(llm_client=FakeLLM(responses),
                               tool_manager=ToolManager(),
                               knowledge_base=object(),
                               tool_index=FakeIndex())
    orch.context_manager.trace.persist_dir = ""
    return orch


def test_agentic_turn_returns_full_trace():
    responses = [
        {"content": "", "tool_calls": [
            FakeToolCall("discover_tools", '{"task_description": "make a ricker"}')]},
        {"content": "", "tool_calls": [
            FakeToolCall("run_task",
                         '{"brief": "make a 30 Hz ricker", "tool_names": ["make_ricker"]}')]},
        # executor's inner loop:
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')]},
        {"content": "<reply>made it</reply>", "tool_calls": None},
        # orchestrator's final answer:
        {"content": "<reply>All done</reply>", "tool_calls": None},
    ]
    orch = _orchestrator(responses)
    out = orch.process_single_input("make a 30 Hz ricker wavelet")
    assert out["reply"] == "All done"
    trace = out["trace"]
    assert trace["session"] == orch.session_id
    kinds = [e["t"] for e in trace["events"]]
    for expected in ("turn_start", "intent", "discover", "tool_call", "run_task"):
        assert expected in kinds, f"missing {expected} in {kinds}"
    discover = [e for e in trace["events"] if e["t"] == "discover"][0]
    assert discover["hits"] == [["make_ricker", 0.9]]
    run_task = [e for e in trace["events"] if e["t"] == "run_task"][0]
    assert run_task["tools_used"] == ["make_ricker"]
    assert run_task["error"] is None
    assert trace["tools_used"] == ["make_ricker"]


def test_agentic_error_turn_still_returns_trace():
    orch = _orchestrator([])  # first meta-loop completion raises IndexError
    out = orch.process_single_input("make a 30 Hz ricker wavelet")
    assert "error" in out["reply"].lower()
    assert out["trace"]["turn"] == 1
    assert any(e["t"] == "turn_error" for e in out["trace"]["events"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_orchestrator_trace.py -q`
Expected: FAIL — `KeyError: 'trace'`.

- [ ] **Step 3: Implement** in `core/orchestrator.py`

Add import: `from core.turn_trace import emit_event, usage_dict`.

In `__init__`, after `self.session_id = uuid.uuid4().hex` add:

```python
        self.context_manager.trace.session_id = self.session_id
```

Replace `process_single_input` with:

```python
    def process_single_input(self, user_input: str) -> Dict[str, Any]:
        trace = self.context_manager.trace
        trace.begin_turn(user_input)
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
            return {"reply": reply, "images": images, "trace": trace.end_turn()}
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            trace.emit("turn_error", error=str(e))
            return {"reply": f"I encountered an error: {str(e)}", "images": [],
                    "trace": trace.end_turn()}
```

(The `exc_info=True` was added in Task 2; keep it.)

In `_run_meta_loop`, after each of the two `update_token_usage` blocks add an `llm` event (same shape as the tool loop's):

```python
            emit_event(self.context_manager, "llm",
                       model=response.get("model"),
                       latency_ms=response.get("latency_ms"),
                       tool_call=bool(response.get("tool_calls")),
                       **usage_dict(response.get("usage")))
```

(for the second block use `final_response` in place of `response`), and immediately before the `final_response = ...` forced completion add:

```python
        logger.warning(f"Orchestrator round budget ({MAX_ORCH_ROUNDS}) exhausted; "
                       f"forcing tool-free completion")
        emit_event(self.context_manager, "budget_exhausted", rounds=MAX_ORCH_ROUNDS)
```

Replace `_discover` with:

```python
    def _discover(self, task_description: str) -> str:
        cards = self.tool_index.search(task_description)
        hits = [[c.name, round(c.score, 4)] for c in cards]
        logger.info(f"discover_tools({task_description!r}) -> {hits}")
        emit_event(self.context_manager, "discover",
                   query=task_description[:200], hits=hits)
        if not cards:
            return "No tools matched; rephrase the task or answer directly."
        return "Matching tools:\n" + "\n".join(f"- {c.card}" for c in cards)
```

In `_run_task`, right after `result = executor.run(brief, tool_names)` add:

```python
        emit_event(self.context_manager, "run_task", brief=brief[:200],
                   tool_names=list(tool_names), tools_used=result.tools_used,
                   error=result.error, n_images=len(result.images))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_orchestrator_trace.py -q` then any existing orchestrator/executor suites: `pytest tests/ -k "orchestrator or executor or tool_index or meta" -q`
Expected: all pass. Existing orchestrator tests that assert `set(out) == {"reply", "images"}` (if any) must be updated to `{"reply", "images", "trace"}` — that is the one intended contract change; update the assertion, don't weaken anything else.

- [ ] **Step 5: Commit**

```bash
git add core/orchestrator.py tests/test_orchestrator_trace.py
git commit -m "feat(trace): orchestrator — discovery scores, undegraded TaskResult, meta-loop llm/budget events, trace in reply"
```

---

### Task 7: Classic bot — turn wrapping, `"trace"` key, tools_used threading

**Files:**
- Modify: `core/chatbot_tool_use.py` (`__init__`, `process_single_input`, `_handle_tool_request`)
- Test: Create `tests/test_chatbot_trace.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 5.
- Produces: `SeismicChatBotToolUse.process_single_input` returns `{"reply", "images", "trace"}`; `_handle_tool_request` returns `{"reply", "images", "tools_used"}` (previously dropped `tools_used`).

- [ ] **Step 1: Write the failing tests** — create `tests/test_chatbot_trace.py`:

```python
from core.chatbot_tool_use import SeismicChatBotToolUse
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
    # no get_simple_completion → keyword fallback routes "make a ..." to tools


def _bot(responses):
    bot = SeismicChatBotToolUse(llm_client=FakeLLM(responses),
                                tool_manager=ToolManager(),
                                knowledge_base=object())
    bot.context_manager.trace.persist_dir = ""
    return bot


def test_classic_turn_returns_trace_with_tools_used():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    bot = _bot(responses)
    out = bot.process_single_input("make a 30 Hz ricker wavelet")
    assert set(out) == {"reply", "images", "trace"}
    assert out["reply"] == "done"
    assert out["trace"]["session"] == bot.session_id
    assert out["trace"]["tools_used"] == ["make_ricker"]
    kinds = [e["t"] for e in out["trace"]["events"]]
    assert "intent" in kinds and "tool_call" in kinds and "auto_plot" in kinds


def test_classic_handle_tool_request_threads_tools_used():
    responses = [
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]
    bot = _bot(responses)
    result = bot._handle_tool_request("make a 30 Hz ricker wavelet")
    assert result["tools_used"] == ["make_ricker"]


def test_classic_error_turn_still_returns_trace():
    bot = _bot([])  # empty script → IndexError inside the loop
    out = bot.process_single_input("make a 30 Hz ricker wavelet")
    assert "error" in out["reply"].lower()
    assert any(e["t"] == "turn_error" for e in out["trace"]["events"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_chatbot_trace.py -q`
Expected: FAIL — no `"trace"` key, `_handle_tool_request` lacks `tools_used`.

- [ ] **Step 3: Implement** in `core/chatbot_tool_use.py`

In `__init__`, after `self.session_id = uuid.uuid4().hex` add:

```python
        self.context_manager.trace.session_id = self.session_id
```

Replace `_handle_tool_request` with:

```python
    def _handle_tool_request(self, user_input: str) -> Dict[str, Any]:
        result = self._tool_loop.run(
            self.system_prompt, [{"role": "user", "content": user_input}], self.tools)
        return {"reply": result["reply"], "images": result["images"],
                "tools_used": result["tools_used"]}
```

In `process_single_input`, wrap the turn: add as the first two lines of the method body (before the `try:`):

```python
        trace = self.context_manager.trace
        trace.begin_turn(user_input)
```

change the success return to:

```python
            return {"reply": reply, "images": images, "trace": trace.end_turn()}
```

and the except block to:

```python
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            trace.emit("turn_error", error=str(e))
            return {"reply": f"I encountered an error: {str(e)}", "images": [],
                    "trace": trace.end_turn()}
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_chatbot_trace.py tests/test_chatbot.py tests/test_session_isolation.py tests/test_chatbot_synthetic.py tests/test_chatbot_outcrop.py tests/test_gradio_upload.py -q`
Expected: all pass. Any existing test asserting `set(result) == {"reply", "images"}` gets updated to include `"trace"` — the one intended contract change.

- [ ] **Step 5: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_trace.py
git commit -m "feat(trace): classic bot — turn wrapping, trace in reply, tools_used no longer dropped"
```

---

### Task 8: Surfaces — API trace field, Gradio status line

**Files:**
- Modify: `interfaces/api_interface.py` (`ChatResponse`, `/chat` route)
- Modify: `interfaces/gradio_interface.py` (new `format_status` helper; `respond` success + error paths)
- Test: Create `tests/test_trace_surfaces.py`

**Interfaces:**
- Consumes: Tasks 6+7 (`"trace"` key).
- Produces: `ChatResponse.trace: Optional[dict] = None`; `interfaces.gradio_interface.format_status(token_usage: dict, trace: Optional[dict] = None) -> str`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_trace_surfaces.py`:

```python
def test_chat_response_carries_trace():
    from interfaces.api_interface import ChatResponse
    r = ChatResponse(response="x", success=True,
                     trace={"turn": 1, "tools_used": ["make_ricker"], "events": []})
    assert r.trace["turn"] == 1
    assert ChatResponse(response="x", success=True).trace is None


def test_format_status_appends_tools():
    from interfaces.gradio_interface import format_status
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert format_status(usage) == "Prompt: 10 | Completion: 5 | Total: 15"
    out = format_status(usage, {"tools_used": ["make_ricker", "analyze_wedge"]})
    assert out.endswith("| Tools: make_ricker → analyze_wedge")
    assert format_status(usage, {"tools_used": []}) == "Prompt: 10 | Completion: 5 | Total: 15"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_trace_surfaces.py -q`
Expected: FAIL — `ChatResponse` has no `trace` field; no `format_status`.

- [ ] **Step 3: Implement**

`interfaces/api_interface.py` — extend the model and the success return:

```python
class ChatResponse(BaseModel):
    response: str
    images: List[str] = []
    success: bool
    error: Optional[str] = None
    trace: Optional[dict] = None
```

and in the `/chat` route's dict branch add `trace=result.get("trace"),` to the `ChatResponse(...)` call.

`interfaces/gradio_interface.py` — add below `append_bot_response`:

```python
def format_status(token_usage, trace=None):
    """One-line session status: token totals plus the turn's tool chain."""
    status = (f"Prompt: {token_usage['prompt_tokens']} | "
              f"Completion: {token_usage['completion_tokens']} | "
              f"Total: {token_usage['total_tokens']}")
    tools = (trace or {}).get("tools_used") or []
    if tools:
        status += " | Tools: " + " → ".join(tools)
    return status
```

In `respond`, replace the token_str construction with:

```python
            token_usage = session_bot.context_manager.get_token_usage()
            trace = response.get("trace") if isinstance(response, dict) else None
            token_str = format_status(token_usage, trace)
```

and replace the error path so it stops blanking the counter:

```python
        except Exception as e:
            chat_history[-1][1] = f"Error processing request: {str(e)}"
            token_str = format_status(session_bot.context_manager.get_token_usage())
            return "", None, chat_history, token_str, session_bot
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_trace_surfaces.py tests/test_gradio_upload.py tests/test_security.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add interfaces/api_interface.py interfaces/gradio_interface.py tests/test_trace_surfaces.py
git commit -m "feat(trace): surface trace via API ChatResponse and Gradio status line; keep tokens on error"
```

---

### Task 9: Full suite, docs, wrap-up

**Files:**
- Modify: `CLAUDE.md` (new section)
- Test: full suite

- [ ] **Step 1: Full suite**

Run: `pytest -q` with a ≥ 360s timeout.
Expected: everything green except the one pre-existing known failure (a stdin-related test noted in the Aug 29 baseline: 1 failed / 542 passed then). Investigate and fix any NEW failure before proceeding; do not touch the pre-existing one.

- [ ] **Step 2: Document in CLAUDE.md**

Add this section to `CLAUDE.md` after the "Agentic mode" section:

```markdown
## Decision trace (agent observability, Tier 0+1)

Every turn in both modes is traced by `core/turn_trace.py::TraceRecorder`, hanging off the
per-session `ContextManager` (`.trace`). Events record decisions, never content: `intent`
(verdict + via llm/keyword_fallback/image_shortcut), `rag` (per-doc scores), `discover`
(tool-index hits with cosine scores), `run_task` (undegraded TaskResult), `tool_call`
(ok/ms + arg provenance as *names*: `injected`/`overridden`/`defaults_filled`),
`parallel_calls_dropped`, `auto_plot` (fired or not), `llm` (model/latency/tokens per call),
`budget_exhausted`, `turn_error`. `process_single_input` returns the turn record as an
additive `"trace"` key (`{"reply", "images", "trace"}`); the API exposes it as
`ChatResponse.trace` and the Gradio status line shows the tool chain. Records are appended
per session to `SEISMIC_TRACE_DIR/<session_id>.jsonl` (default `<tmpdir>/seismic_traces`;
write failures are swallowed). `LOG_LEVEL` is env-overridable now; interfaces call
`basicConfig` themselves. `LLMClient.get_simple_completion(..., context_manager=)` accounts
router-side tokens — but `knowledge/rag_system.py` still builds its own `LLMClient`, so RAG
*generation* tokens remain untracked (known gap, Tier 2). Tests: `tests/test_turn_trace.py`,
`test_tool_loop_trace.py`, `test_orchestrator_trace.py`, `test_chatbot_trace.py`,
`test_trace_surfaces.py`. When adding a decision point, emit an event via
`core.turn_trace.emit_event(context_manager, ...)` — don't invent a parallel channel.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: decision-trace section (TurnTrace events, trace key, SEISMIC_TRACE_DIR, LOG_LEVEL)"
```

- [ ] **Step 4: Report** — summarize suite results and remaining Tier 2+ follow-ups; hand off per superpowers:finishing-a-development-branch.
