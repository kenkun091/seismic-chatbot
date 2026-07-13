# Workflow Result UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every chatbot tool request returns LLM-narrated prose plus ALL plots produced during the turn — `process_single_input` always returns `{"reply": str, "images": list[str]}` — replacing the current behavior where an image short-circuits the agentic loop and the rich result dict is silently dropped.

**Architecture:** The bounded agentic tool loop in `core/chatbot_tool_use.py::_handle_tool_request` stops short-circuiting on images. Each round harvests image paths into an accumulator and feeds a *compacted* tool result (big arrays summarized, image paths masked) back to the LLM, which writes the final prose. The two derived surfaces (main Gradio UI, FastAPI `/chat`) render the new dict contract. Spec: `docs/superpowers/specs/2026-07-02-workflow-result-ux-design.md`.

**Tech Stack:** Python, pytest (scripted `FakeLLMClient` from `tests/conftest.py`, no network), Gradio 3.x pair-format chat history, FastAPI + `fastapi.testclient`.

## Global Constraints

- Run everything from this package dir (`geo-mcp/seismic_chatbot/`); it is its own git repo — commit with plain `git` from inside this dir, never from the parent repos.
- Branch: `stabilize-tool-layer`.
- No network in tests: anything touching the chatbot/LLM uses `fake_llm_factory` (`tests/conftest.py`) or stub objects.
- Do NOT modify: `workflows/` (recipes, sweep, engine), `core/tool_registry.py`, `core/tool_manager.py`, `core/context_manager.py`, `core/chatbot.py` (legacy bot), `interfaces/gradio_interface_legacy.py` (consumes the legacy bot — out of scope), the `chat()` REPL method in `core/chatbot_tool_use.py` (console-only, never surfaced images).
- `_handle_automatic_chaining` keeps its existing return contract (`{"image_path": path}` or `None`) — only the *loop's use* of it changes. Existing tests pinning that contract (`tests/test_chatbot.py`, `tests/test_rock_properties_plot.py::test_automatic_chaining_produces_image`) must stay green untouched.
- Full suite check: `pytest -q` (expect the one pre-existing known failure noted in the Jun 21 baseline: 290 passed, 1 known failure — do not fix or worsen it).
- End git commit messages with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: Result compaction helper (`_compact_tool_result` / `_compact_value`)

**Files:**
- Modify: `core/chatbot_tool_use.py` (add module constant `_MAX_ARRAY_PREVIEW` and two methods; place them right after `_parse_tool_input`, which ends at ~line 123)
- Create: `tests/test_result_compaction.py`

**Interfaces:**
- Consumes: nothing new (`json`, `numpy as np` are already imported in `core/chatbot_tool_use.py` — verify; add if missing).
- Produces: `SeismicChatBotToolUse._compact_tool_result(tool_result: Any) -> str` (JSON-ish string for the `role:"tool"` message) and `SeismicChatBotToolUse._compact_value(value: Any) -> Any` (recursive compactor). Task 3 calls `_compact_tool_result`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_result_compaction.py`:

```python
"""Unit tests for tool-result compaction: what the LLM sees in role:"tool"
messages. Large numeric arrays become summary strings; image paths are masked
(plots are shown to the user directly, the model must not echo them)."""
import numpy as np

from core.chatbot_tool_use import SeismicChatBotToolUse


def _bare_bot():
    # Build an instance WITHOUT running __init__ (avoids loading RAG/LLM deps).
    return object.__new__(SeismicChatBotToolUse)


def test_long_numeric_list_is_summarized():
    bot = _bare_bot()
    out = bot._compact_value(list(range(61)))
    assert out == "<61 values, min=0, max=60, first=0, last=60>"


def test_short_list_kept_verbatim():
    bot = _bare_bot()
    assert bot._compact_value([1, 2, 3]) == [1, 2, 3]


def test_large_ndarray_summarized_with_shape():
    bot = _bare_bot()
    out = bot._compact_value(np.zeros((100, 61)))
    assert out == "<array shape (100, 61), min=0, max=0>"


def test_small_ndarray_becomes_list():
    bot = _bare_bot()
    assert bot._compact_value(np.array([1.0, 2.0])) == [1.0, 2.0]


def test_image_path_masked():
    bot = _bare_bot()
    out = bot._compact_value({"tuning_thickness": 12.5, "image_path": "/tmp/x.png"})
    assert out == {"tuning_thickness": 12.5,
                   "image_path": "<plot generated and shown to the user>"}


def test_nested_dict_recursed():
    bot = _bare_bot()
    out = bot._compact_value({"cases": {"gas": {"rc": [0.1] * 20}}})
    assert out["cases"]["gas"]["rc"].startswith("<20 values")


def test_scalars_strings_and_bools_passthrough():
    bot = _bare_bot()
    assert bot._compact_value(3.5) == 3.5
    assert bot._compact_value("AVO class III") == "AVO class III"
    assert bot._compact_value(True) is True
    assert bot._compact_value(None) is None


def test_bool_list_not_treated_as_numeric():
    bot = _bare_bot()
    assert bot._compact_value([True] * 20) == [True] * 20


def test_tuple_of_arrays_compacted():
    bot = _bare_bot()
    time_array = np.linspace(-0.1, 0.1, 201)
    wavelet = np.zeros(201)
    out = bot._compact_value((time_array, wavelet))
    assert isinstance(out, list) and len(out) == 2
    assert out[0].startswith("<array shape (201,)")


def test_compact_tool_result_returns_string():
    bot = _bare_bot()
    s = bot._compact_tool_result({"a": 1, "rc": list(range(20))})
    assert isinstance(s, str)
    assert '"a": 1' in s
    assert "<20 values" in s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_result_compaction.py -q`
Expected: FAIL / ERROR with `AttributeError: ... has no attribute '_compact_value'`

- [ ] **Step 3: Implement the helpers**

In `core/chatbot_tool_use.py`, confirm `import json` and `import numpy as np` exist at the top (they do — the loop already uses both). Add a module-level constant near the other module-level names (after the imports/logger):

```python
# Numeric sequences longer than this are summarized before being sent back to
# the LLM as tool-message content (narration needs the stats, not 61 floats).
_MAX_ARRAY_PREVIEW = 12
```

Add these two methods to `SeismicChatBotToolUse`, directly after `_parse_tool_input`:

```python
    def _compact_tool_result(self, tool_result: Any) -> str:
        """Compact a tool result for the LLM's role:"tool" message.

        Large numeric arrays become summary strings and image paths are masked
        — plots are displayed to the user directly, so the model should narrate
        the numbers, not echo file paths.
        """
        compacted = self._compact_value(tool_result)
        try:
            return json.dumps(compacted, default=str)
        except (TypeError, ValueError):
            return str(compacted)

    def _compact_value(self, value: Any) -> Any:
        """Recursively compact one value (see _compact_tool_result)."""
        if isinstance(value, np.ndarray):
            if value.size > _MAX_ARRAY_PREVIEW:
                return (f"<array shape {value.shape}, "
                        f"min={value.min():.6g}, max={value.max():.6g}>")
            value = value.tolist()
        if isinstance(value, dict):
            return {
                k: ("<plot generated and shown to the user>"
                    if k == "image_path" and isinstance(v, str)
                    else self._compact_value(v))
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            seq = list(value)
            if (len(seq) > _MAX_ARRAY_PREVIEW
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            for x in seq)):
                arr = [float(x) for x in seq]
                return (f"<{len(arr)} values, min={min(arr):.6g}, max={max(arr):.6g}, "
                        f"first={arr[0]:.6g}, last={arr[-1]:.6g}>")
            return [self._compact_value(v) for v in seq]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_result_compaction.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_result_compaction.py core/chatbot_tool_use.py
git commit -m "feat(chatbot): _compact_tool_result — summarize arrays, mask image paths for LLM narration

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Image harvesting helper (`_harvest_images`)

**Files:**
- Modify: `core/chatbot_tool_use.py` (add one method, placed next to the existing `_workflow_image_output` at ~line 699 — the old helpers stay until Task 3)
- Modify: `tests/test_chatbot_workflow.py` (append new tests; do NOT touch the existing ones yet)

**Interfaces:**
- Consumes: nothing new.
- Produces: `SeismicChatBotToolUse._harvest_images(tool_result: Any, collected: List[str]) -> None` — mutates `collected` in place, deduped, order-preserving. Task 3 calls it with the loop's accumulator.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chatbot_workflow.py`:

```python
def test_harvest_images_from_plain_path(bot):
    collected = []
    bot._harvest_images("/tmp/a.png", collected)
    assert collected == ["/tmp/a.png"]


def test_harvest_images_from_dict(bot):
    collected = []
    bot._harvest_images({"avo_class": "III", "image_path": "/tmp/x.png"}, collected)
    assert collected == ["/tmp/x.png"]


def test_harvest_images_dedupes_preserving_order(bot):
    collected = ["/tmp/a.png"]
    bot._harvest_images({"image_path": "/tmp/b.png"}, collected)
    bot._harvest_images("/tmp/a.png", collected)
    bot._harvest_images({"image_path": "/tmp/b.png"}, collected)
    assert collected == ["/tmp/a.png", "/tmp/b.png"]


def test_harvest_images_ignores_non_images(bot):
    collected = []
    bot._harvest_images({"avo_class": "III"}, collected)
    bot._harvest_images({"image_path": 123}, collected)
    bot._harvest_images("not-a-path", collected)
    bot._harvest_images(("tuple", "result"), collected)
    bot._harvest_images(None, collected)
    assert collected == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chatbot_workflow.py -q -k harvest`
Expected: 4 FAIL/ERROR with `AttributeError: ... '_harvest_images'`

- [ ] **Step 3: Implement**

Add to `SeismicChatBotToolUse` in `core/chatbot_tool_use.py`, directly above `_workflow_image_output`:

```python
    def _harvest_images(self, tool_result: Any, collected: List[str]) -> None:
        """Collect .png paths from a tool result into `collected`.

        Handles the two shapes tools produce: a plain path string (plot tools)
        or a dict carrying an "image_path" key (workflow recipes, auto-chain
        results). Deduped, order-preserving.
        """
        path = None
        if isinstance(tool_result, str) and tool_result.endswith(".png"):
            path = tool_result
        elif isinstance(tool_result, dict):
            p = tool_result.get("image_path")
            if isinstance(p, str) and p.endswith(".png"):
                path = p
        if path is not None and path not in collected:
            collected.append(path)
```

Check the file's `typing` import includes `List` (top of file imports `Dict, Any, Optional` — add `List` if absent).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chatbot_workflow.py -q`
Expected: all PASS (old `_workflow_image_output` tests still green)

- [ ] **Step 5: Commit**

```bash
git add tests/test_chatbot_workflow.py core/chatbot_tool_use.py
git commit -m "feat(chatbot): _harvest_images — collect plot paths deduped, order-preserving

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: The contract flip — narrated loop + `{"reply", "images"}` from `process_single_input`

This is the atomic core: the loop stops short-circuiting, the old helpers die, and the return type changes. It cannot be split without leaving the suite red between commits.

**Files:**
- Modify: `core/chatbot_tool_use.py` — rewrite `_handle_tool_request` (~line 610) and `process_single_input` (~line 228); DELETE methods `_workflow_image_output` and `_is_image_output`.
- Modify: `tests/conftest.py` — `FakeLLMClient` records call kwargs.
- Modify: `tests/test_chatbot_workflow.py` — delete the two `_workflow_image_output` tests (lines 12–20: `test_workflow_image_output_from_dict`, `test_workflow_image_output_none_when_no_png`).
- Modify: `tests/test_rock_properties_plot.py` — update the two loop tests to the dict contract.
- Create: `tests/test_chatbot_narration.py`.

**Interfaces:**
- Consumes: `_compact_tool_result` (Task 1), `_harvest_images` (Task 2).
- Produces: `_handle_tool_request(user_input: str) -> Dict[str, Any]` and `process_single_input(user_input: str) -> Dict[str, Any]`, both returning `{"reply": str, "images": list[str]}`. Tasks 5–7 consume this contract.

- [ ] **Step 1: Make `FakeLLMClient` record calls**

In `tests/conftest.py` replace the `FakeLLMClient` class with:

```python
class FakeLLMClient:
    """Returns scripted completions; no network. Records each call's kwargs
    so tests can assert on the messages actually sent to the model."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get_completion(self, *a, **k):
        self.calls.append(k)
        return self._responses.pop(0)
```

Run: `pytest tests/test_chatbot_workflow.py tests/test_rock_properties_plot.py -q` — Expected: PASS (pure addition).

- [ ] **Step 2: Write the failing narration tests**

Create `tests/test_chatbot_narration.py`:

```python
"""End-to-end tests for the narrated response contract of the agentic tool
loop: _handle_tool_request / process_single_input return
{"reply": str, "images": list[str]}. No network — scripted FakeLLMClient
(tests/conftest.py) plus a scripted fake tool manager."""
import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = _FakeFunc(name, arguments)


class _ScriptedToolManager:
    """Returns a scripted result per tool name; a scripted Exception raises."""
    def __init__(self, results):
        self._results = dict(results)
        self.calls = []

    def process_tool_call(self, name, params):
        self.calls.append((name, params))
        result = self._results[name]
        if isinstance(result, Exception):
            raise result
        return result


def _completion(tool_calls=None, content=""):
    return {
        "content": content,
        "tool_calls": tool_calls,
        "stop_reason": "tool_calls" if tool_calls else "stop",
        "usage": None,
    }


@pytest.fixture
def bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


def test_workflow_result_is_narrated_with_image(bot, fake_llm_factory):
    tc = _FakeToolCall("tuning", '{"phit_sand": 0.25}')
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc]),
        _completion(content="<reply>Tuning thickness is 12.5 m at 30 Hz.</reply>"),
    ])
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/t.png"},
    })
    out = bot._handle_tool_request("tuning analysis for a 25% porosity sand")
    assert out == {"reply": "Tuning thickness is 12.5 m at 30 Hz.",
                   "images": ["/tmp/t.png"]}


def test_multiple_rounds_collect_all_images_in_order(bot, fake_llm_factory):
    tc1 = _FakeToolCall("tuning", '{"phit_sand": 0.25}', "c1")
    tc2 = _FakeToolCall("fluid_scenario", '{"phit_sand": 0.25}', "c2")
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1]),
        _completion(tool_calls=[tc2]),
        _completion(content="<reply>Both analyses are done.</reply>"),
    ])
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/a.png"},
        "fluid_scenario": {"cases": {}, "image_path": "/tmp/b.png"},
    })
    out = bot._handle_tool_request("tuning then fluid scenarios")
    assert out["reply"] == "Both analyses are done."
    assert out["images"] == ["/tmp/a.png", "/tmp/b.png"]


def test_repeated_image_path_deduped(bot, fake_llm_factory):
    tc1 = _FakeToolCall("tuning", '{"phit_sand": 0.25}', "c1")
    tc2 = _FakeToolCall("tuning", '{"phit_sand": 0.30}', "c2")
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1]),
        _completion(tool_calls=[tc2]),
        _completion(content="<reply>Done.</reply>"),
    ])
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/same.png"},
    })
    out = bot._handle_tool_request("two tuning runs")
    assert out["images"] == ["/tmp/same.png"]


def test_tool_message_content_is_compacted(bot, fake_llm_factory):
    tc = _FakeToolCall("tuning", '{"phit_sand": 0.25}')
    llm = fake_llm_factory([
        _completion(tool_calls=[tc]),
        _completion(content="<reply>Done.</reply>"),
    ])
    bot.llm_client = llm
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"curve": list(range(61)), "image_path": "/tmp/t.png"},
    })
    bot._handle_tool_request("tuning")
    final_messages = llm.calls[-1]["messages"]
    tool_msgs = [m for m in final_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "<61 values" in tool_msgs[0]["content"]
    assert "/tmp/t.png" not in tool_msgs[0]["content"]
    assert "plot generated and shown to the user" in tool_msgs[0]["content"]


def test_round_exhaustion_returns_reply_and_images(bot, fake_llm_factory):
    calls = [_FakeToolCall("tuning", '{"phit_sand": 0.25}', f"c{i}") for i in range(5)]
    bot.llm_client = fake_llm_factory(
        [_completion(tool_calls=[c]) for c in calls]
        + [_completion(content="<reply>Stopping here; tuning is 12.5 m.</reply>")]
    )
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/t.png"},
    })
    out = bot._handle_tool_request("keep going")
    assert out == {"reply": "Stopping here; tuning is 12.5 m.",
                   "images": ["/tmp/t.png"]}


def test_tool_error_returns_collected_images(bot, fake_llm_factory):
    tc1 = _FakeToolCall("tuning", '{"phit_sand": 0.25}', "c1")
    tc2 = _FakeToolCall("fluid_scenario", '{"phit_sand": 0.25}', "c2")
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1]),
        _completion(tool_calls=[tc2]),
    ])
    bot.tool_manager = _ScriptedToolManager({
        "tuning": {"tuning_thickness": 12.5, "image_path": "/tmp/t.png"},
        "fluid_scenario": ValueError("bad fluids"),
    })
    out = bot._handle_tool_request("tuning then fluids")
    assert out["reply"].startswith("Error executing tool:")
    assert "bad fluids" in out["reply"]
    assert out["images"] == ["/tmp/t.png"]


def test_process_single_input_passes_through_tool_dict(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: False)
    monkeypatch.setattr(bot, "_handle_tool_request",
                        lambda text: {"reply": "hi", "images": ["/tmp/a.png"]})
    assert bot.process_single_input("x") == {"reply": "hi", "images": ["/tmp/a.png"]}


def test_process_single_input_wraps_knowledge_string(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: True)
    monkeypatch.setattr(bot, "_handle_knowledge_question",
                        lambda text: "A Ricker wavelet is a zero-phase pulse.")
    assert bot.process_single_input("what is a ricker?") == {
        "reply": "A Ricker wavelet is a zero-phase pulse.", "images": []}


def test_process_single_input_error_returns_dict(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: False)

    def _boom(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(bot, "_handle_tool_request", _boom)
    out = bot.process_single_input("x")
    assert out["reply"].startswith("I encountered an error:")
    assert out["images"] == []


def test_process_single_input_none_reply_gets_fallback_text(bot, monkeypatch):
    monkeypatch.setattr(bot, "_is_knowledge_question", lambda text: True)
    monkeypatch.setattr(bot, "_handle_knowledge_question", lambda text: None)
    out = bot.process_single_input("x")
    assert out == {"reply": "I didn't get a response. Please try again.",
                   "images": []}
```

- [ ] **Step 3: Run new tests to verify they fail**

Run: `pytest tests/test_chatbot_narration.py -q`
Expected: FAIL — the current `_handle_tool_request` returns `{"image_path": ...}` / plain strings, not the dict contract.

- [ ] **Step 4: Rewrite `_handle_tool_request`**

Replace the entire method (currently ~line 610 through the round-exhaustion return at ~line 697) with:

```python
    def _handle_tool_request(self, user_input: str) -> Dict[str, Any]:
        """
        Handle a tool-use request through the bounded agentic tool loop.

        Returns:
            dict: {"reply": str, "images": list[str]} — the final prose answer
            plus every plot produced along the way (deduped, in order).
        """
        messages = [{"role": "user", "content": user_input}]
        collected_images: List[str] = []

        # Agentic tool loop: the model may chain several tool calls before
        # giving a final answer. Plots are harvested into collected_images and
        # a compacted tool result goes back to the model so it can narrate the
        # numbers (bounded to avoid runaways).
        MAX_TOOL_ROUNDS = 5
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.llm_client.get_completion(
                system_prompt=self.system_prompt,
                user_prompt="",
                tools=self.tools,
                messages=messages
            )
            if response.get("usage"):
                self.context_manager.update_token_usage(response["usage"])

            if not response.get("tool_calls"):
                # No tool requested: this is the final answer.
                messages.append({"role": "assistant", "content": response["content"]})
                reply = self._extract_reply(response["content"]) or response["content"]
                if isinstance(reply, bool):
                    reply = str(reply)
                return {"reply": reply, "images": collected_images}

            # Execute the (first) requested tool. Append only the tool_call we
            # respond to so every assistant tool_call has a matching tool result.
            tool_call = response["tool_calls"][0]
            tool_name = tool_call.function.name
            tool_input_str = tool_call.function.arguments
            messages.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": [tool_call]
            })

            try:
                tool_input = self._parse_tool_input(tool_input_str)
                tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self._compact_tool_result(tool_result)
                })
                self._update_context(tool_name, tool_input, tool_result)
                self._harvest_images(tool_result, collected_images)

                # Auto-chaining still runs the partner plot tool; its plot now
                # joins the harvest instead of ending the turn.
                chained_result = self._handle_automatic_chaining(tool_name, tool_input, tool_result)
                if chained_result:
                    self._harvest_images(chained_result, collected_images)

                # Loop so the model can narrate the result or chain another tool.
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return {"reply": f"Error executing tool: {str(e)}",
                        "images": collected_images}

        # Round budget exhausted while still calling tools: force a tool-free
        # completion so the user gets a textual answer instead of nothing.
        final_response = self.llm_client.get_completion(
            system_prompt=self.system_prompt,
            user_prompt="",
            tools=None,
            messages=messages
        )
        if final_response.get("usage"):
            self.context_manager.update_token_usage(final_response["usage"])
        reply = self._extract_reply(final_response["content"]) or final_response["content"]
        if isinstance(reply, bool):
            reply = str(reply)
        return {"reply": reply, "images": collected_images}
```

Then DELETE the `_workflow_image_output` and `_is_image_output` methods entirely.

- [ ] **Step 5: Rewrite `process_single_input`**

Replace the method (~line 228) with:

```python
    def process_single_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process a single user input and return a response.

        Args:
            user_input: The user's input text

        Returns:
            dict: {"reply": str, "images": list[str]} — images may be empty.
        """
        try:
            # Check if this is a knowledge question that should use RAG
            if self._is_knowledge_question(user_input):
                logger.info("Using RAG for knowledge question")
                response = self._handle_knowledge_question(user_input)
            else:
                # Otherwise, use the regular tool-based approach
                logger.info("Using tool-based approach")
                response = self._handle_tool_request(user_input)

            if isinstance(response, dict) and "reply" in response:
                reply = response["reply"]
                images = list(response.get("images") or [])
            else:
                reply, images = response, []

            # Final safety check: never surface booleans/None as the reply.
            if isinstance(reply, bool):
                reply = str(reply)
            elif reply is None:
                reply = "I didn't get a response. Please try again."

            return {"reply": reply, "images": images}

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {"reply": f"I encountered an error: {str(e)}", "images": []}
```

- [ ] **Step 6: Update the pinned tests**

In `tests/test_chatbot_workflow.py`, DELETE `test_workflow_image_output_from_dict` and `test_workflow_image_output_none_when_no_png` (the methods they test no longer exist; `_harvest_images` tests from Task 2 are their replacement).

In `tests/test_rock_properties_plot.py`, update the two loop tests:

```python
def test_tool_loop_executes_follow_up_tool_call(bot, fake_llm_factory):
    """A tool with no auto-plot followed by a second tool call must NOT drop the
    follow-up: the model's preamble is not the final answer."""
    # predict_elastic_layer has auto_plot=None and needs no network.
    tc1 = _FakeToolCall("predict_elastic_layer", '{"phit": [0.2], "vclay": [0.1]}', "c1")
    tc2 = _FakeToolCall("predict_elastic_layer", '{"phit": [0.25], "vclay": [0.15]}', "c2")

    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1], content=""),
        _completion(tool_calls=[tc2], content="Let me also look up some rock physics context."),
        _completion(tool_calls=None, content="<reply>Vp is about 4000 m/s.</reply>"),
    ])

    result = bot._handle_tool_request("predict the sand layer and explain it")
    assert "look up some rock physics context" not in result["reply"]
    assert result == {"reply": "Vp is about 4000 m/s.", "images": []}


def test_tool_loop_single_round_still_returns_text(bot, fake_llm_factory):
    tc1 = _FakeToolCall("predict_elastic_layer", '{"phit": [0.2], "vclay": [0.1]}', "c1")
    bot.llm_client = fake_llm_factory([
        _completion(tool_calls=[tc1], content=""),
        _completion(tool_calls=None, content="<reply>Done.</reply>"),
    ])
    assert bot._handle_tool_request("predict the layer") == {"reply": "Done.", "images": []}
```

- [ ] **Step 7: Run the affected files, then the full suite**

Run: `pytest tests/test_chatbot_narration.py tests/test_chatbot_workflow.py tests/test_rock_properties_plot.py tests/test_result_compaction.py tests/test_chatbot.py -q`
Expected: all PASS.

Run: `pytest -q`
Expected: same pass count as baseline (only the one pre-existing known failure).

- [ ] **Step 8: Commit**

```bash
git add core/chatbot_tool_use.py tests/conftest.py tests/test_chatbot_narration.py tests/test_chatbot_workflow.py tests/test_rock_properties_plot.py
git commit -m "feat(chatbot)!: narrated responses — {reply, images} contract, no image short-circuit

The agentic loop now harvests every plot into an accumulator and feeds
compacted tool results back to the model, which writes the final prose.
_workflow_image_output/_is_image_output are gone; process_single_input
always returns {\"reply\": str, \"images\": list[str]}.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: System prompt — narration instructions

**Files:**
- Modify: `core/chatbot_tool_use.py::_create_system_prompt` (~line 54; insert between the "Guidelines:" block and "In each conversational turn")
- Modify: `tests/test_chatbot_workflow.py` (append one test)

**Interfaces:**
- Consumes/Produces: text-only change to the returned prompt string.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatbot_workflow.py`:

```python
def test_system_prompt_explains_narration_contract(bot):
    prompt = bot._create_system_prompt()
    assert "displayed to the user automatically" in prompt
    assert "compacted" in prompt
```

Run: `pytest tests/test_chatbot_workflow.py -q -k narration_contract` — Expected: FAIL.

- [ ] **Step 2: Add the prompt text**

In `_create_system_prompt`, insert this block after the numbered "Guidelines:" list (after guideline 5, before the "In each conversational turn" paragraph):

```
Tool results and plots:
- Tool results are compacted before you see them: long numeric arrays appear as summaries like "<61 values, min=..., max=...>".
- Any plot a tool produces is displayed to the user automatically — never print or mention image file paths.
- After your tools finish, state the key quantitative results (e.g. tuning thickness, AVO class, intercept/gradient, sweep statistics) in your <reply>.
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_chatbot_workflow.py -q`
Expected: all PASS (the existing `- tuning:`-style bullet assertions must still pass).

- [ ] **Step 4: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(chatbot): system prompt — narrate results, plots auto-shown, no file paths

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Gradio interface renders reply + N images

**Files:**
- Modify: `interfaces/gradio_interface.py` (extract a testable module-level helper; use it in `respond`)
- Create: `tests/test_gradio_response_format.py`
- Modify: `docs/superpowers/specs/2026-07-02-workflow-result-ux-design.md` (two scope amendments discovered during planning)

**Interfaces:**
- Consumes: the `{"reply", "images"}` contract from Task 3.
- Produces: `append_bot_response(chat_history: list, response: Any) -> list` (module-level function in `interfaces/gradio_interface.py`; Gradio 3.x pair-format history).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gradio_response_format.py`:

```python
"""The Gradio UI renders the {"reply", "images"} contract: reply text fills the
pending slot, then one history row per image (Gradio 3.x shows one file per
message)."""
from interfaces.gradio_interface import append_bot_response


def _pending(msg="q"):
    return [[msg, None]]


def test_reply_with_images_appends_one_row_per_image():
    out = append_bot_response(_pending(), {"reply": "Tuning is 12.5 m.",
                                           "images": ["/tmp/a.png", "/tmp/b.png"]})
    assert out[0] == ["q", "Tuning is 12.5 m."]
    assert out[1] == [None, ("/tmp/a.png",)]
    assert out[2] == [None, ("/tmp/b.png",)]


def test_reply_without_images():
    assert append_bot_response(_pending(), {"reply": "Hello", "images": []}) == [["q", "Hello"]]


def test_missing_images_key_tolerated():
    assert append_bot_response(_pending(), {"reply": "Hello"}) == [["q", "Hello"]]


def test_plain_string_response():
    assert append_bot_response(_pending(), "plain") == [["q", "plain"]]


def test_non_string_response_stringified():
    assert append_bot_response(_pending(), 42) == [["q", "42"]]
```

Run: `pytest tests/test_gradio_response_format.py -q` — Expected: ImportError (no `append_bot_response`).

- [ ] **Step 2: Implement the helper and rewire `respond`**

In `interfaces/gradio_interface.py`, add at module level (after the imports, before `create_chat_interface`):

```python
def append_bot_response(chat_history, response):
    """Append a bot response to Gradio 3.x pair-format chat history.

    The tool-use bot returns {"reply": str, "images": list[str]}: the reply
    fills the pending assistant slot, then each image gets its own history row
    (Gradio renders one file per message). Plain strings render as-is.
    """
    if isinstance(response, dict) and "reply" in response:
        chat_history[-1][1] = response.get("reply") or ""
        for path in response.get("images") or []:
            chat_history.append([None, (path,)])
    elif isinstance(response, str):
        chat_history[-1][1] = response
    else:
        chat_history[-1][1] = str(response)
    return chat_history
```

In `respond` (lines 20–28), replace:

```python
            response = session_bot.process_single_input(message)

            # Handle different response types (Gradio 3.x format)
            if isinstance(response, dict) and 'image_path' in response:
                chat_history[-1][1] = (response['image_path'],)
            elif isinstance(response, str):
                chat_history[-1][1] = response
            else:
                chat_history[-1][1] = str(response)
```

with:

```python
            response = session_bot.process_single_input(message)
            chat_history = append_bot_response(chat_history, response)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_gradio_response_format.py -q`
Expected: all PASS.

- [ ] **Step 4: Amend the spec with the two planning discoveries**

In `docs/superpowers/specs/2026-07-02-workflow-result-ux-design.md`, section "4. Interfaces", replace the `gradio_interface_legacy.py` bullet with:

```markdown
- `interfaces/gradio_interface_legacy.py`: **out of scope** — it consumes the
  legacy `core/chatbot.py::SeismicChatBot.process_input`, whose contract this
  design does not change (discovered during planning; the spec originally
  assumed it consumed the tool-use bot).
```

And in section "1. Response contract", replace the `chat()` REPL bullet with:

```markdown
- The `chat()` REPL method is unchanged: it is a console-only loop that never
  surfaced images and does not use `_handle_tool_request`.
```

And in section "7. Testing", replace the `tests/test_chatbot.py` bullet with:

```markdown
- `tests/test_chatbot.py` — NO changes needed (discovered during planning):
  its `{"image_path": ...}` asserts pin `_handle_automatic_chaining`, whose
  return contract is intentionally preserved.
```

- [ ] **Step 5: Commit**

```bash
git add interfaces/gradio_interface.py tests/test_gradio_response_format.py docs/superpowers/specs/2026-07-02-workflow-result-ux-design.md
git commit -m "feat(ui): Gradio renders narrated reply + one row per harvested plot

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: FastAPI `/chat` carries images

**Files:**
- Modify: `interfaces/api_interface.py` (`ChatResponse` model at line 43; `chat` route at line 61)
- Create: `tests/test_api_chat_contract.py`

**Interfaces:**
- Consumes: the `{"reply", "images"}` contract from Task 3.
- Produces: `ChatResponse` with `response: str`, `images: List[str] = []`, `success: bool`, `error: Optional[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_chat_contract.py`:

```python
"""Contract test for POST /chat: the JSON body carries the narrated reply plus
plot paths. Importing interfaces.api_interface builds the real chatbot (heavy,
needs LLM credentials), so the fixture stubs SeismicChatBotToolUse before a
module reload — hermetic, no network, no credentials."""
import importlib

import pytest

pytest.importorskip("fastapi")


class _StubSession:
    def process_single_input(self, message):
        return {"reply": "Tuning is 12.5 m.", "images": ["/tmp/a.png", "/tmp/b.png"]}


class _StubBot:
    def __init__(self, *a, **k):
        pass

    def new_session(self):
        return _StubSession()


@pytest.fixture
def api(monkeypatch):
    import core.chatbot_tool_use as bot_module
    monkeypatch.setattr(bot_module, "SeismicChatBotToolUse", _StubBot)
    import interfaces.api_interface as api_module
    api_module = importlib.reload(api_module)
    monkeypatch.setattr(api_module, "API_AUTH_KEY", "sekret")
    return api_module


def test_chat_response_includes_reply_and_images(api):
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    r = client.post("/chat", json={"message": "run tuning"},
                    headers={"X-API-Key": "sekret"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["response"] == "Tuning is 12.5 m."
    assert body["images"] == ["/tmp/a.png", "/tmp/b.png"]


def test_chat_plain_string_response_has_empty_images(api):
    from fastapi.testclient import TestClient

    class _LegacySession:
        def process_single_input(self, message):
            return "plain text"

    class _LegacyBot:
        def new_session(self):
            return _LegacySession()

    api.base_chatbot = _LegacyBot()
    client = TestClient(api.app)
    r = client.post("/chat", json={"message": "hi"},
                    headers={"X-API-Key": "sekret"})
    body = r.json()
    assert body["response"] == "plain text"
    assert body["images"] == []
```

Run: `pytest tests/test_api_chat_contract.py -q` — Expected: FAIL (no `images` field / `response` is a stringified dict).

- [ ] **Step 2: Implement**

In `interfaces/api_interface.py`, change `ChatResponse`:

```python
class ChatResponse(BaseModel):
    response: str
    images: List[str] = []
    success: bool
    error: Optional[str] = None
```

and the route body:

```python
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_chat_policy)])
async def chat(request: ChatRequest):
    """Process a chat message; return the narrated reply plus any plot paths."""
    try:
        session = base_chatbot.new_session()
        result = session.process_single_input(request.message)
        if isinstance(result, dict) and "reply" in result:
            return ChatResponse(
                response=str(result["reply"]),
                images=[str(p) for p in result.get("images") or []],
                success=True,
            )
        return ChatResponse(response=str(result), success=True)
    except Exception as e:
        return ChatResponse(response="", success=False, error=str(e))
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_api_chat_contract.py tests/test_security.py -q`
Expected: all PASS (security tests confirm the auth/rate-limit gate is undisturbed).

- [ ] **Step 4: Commit**

```bash
git add interfaces/api_interface.py tests/test_api_chat_contract.py
git commit -m "feat(api): /chat returns narrated reply + images list

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Example flows print the new contract; final verification

**Files:**
- Modify: `example_tool_use.py` (six `print(f"Assistant: {response}")` sites at lines 27, 35, 43, 51, 59, 67)

**Interfaces:**
- Consumes: the `{"reply", "images"}` contract from Task 3.
- Produces: nothing downstream; console output only.

- [ ] **Step 1: Add a print helper and use it**

In `example_tool_use.py`, add below the imports:

```python
def _print_response(response):
    """Print a chatbot response: narrated reply first, then any plot paths."""
    if isinstance(response, dict) and "reply" in response:
        print(f"Assistant: {response['reply']}")
        for path in response.get("images") or []:
            print(f"  [plot saved: {path}]")
    else:
        print(f"Assistant: {response}")
```

Replace each of the six `print(f"Assistant: {response}")` lines with:

```python
    _print_response(response)
```

- [ ] **Step 2: Syntax check**

Run: `python -c "import ast; ast.parse(open('example_tool_use.py').read()); print('OK')"`
Expected: `OK` (the script needs live credentials to *run*; parse check only).

- [ ] **Step 3: Full suite**

Run: `pytest -q`
Expected: baseline pass count + the new tests from Tasks 1–6; only the one pre-existing known failure.

- [ ] **Step 4: Commit**

```bash
git add example_tool_use.py
git commit -m "chore(examples): print narrated reply + plot paths from the new contract

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Manual smoke test (needs `.env` credentials)**

Run: `python main.py` and in the browser UI try: "Run a tuning analysis for a 25% porosity clean sand under shale, 30 Hz, up to 60 m thickness" — expect a prose answer quoting the tuning thickness AND the tuning plot below it. Then: "Compare brine vs gas AVO for that sand" — expect narrated comparison + plot. Report results; do not mark the feature done without this check or an explicit user waiver.
