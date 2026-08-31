# Reusable Skills (Tier 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** YAML-defined, parameterized skills that run either as a deterministic replay of a recorded tool chain or as an LLM-guided procedure with a scoped toolset; captured from the last turn via `save_skill` or hand-authored; discoverable in both chat modes.

**Architecture:** `ToolLoopRunner.execute_call` (extracted from the loop body) becomes the single per-call code path used by live turns AND replay, and records each call's resolved args in session memory. `core/skills.py` holds the pure model (validation, slot substitution, parameterizer, two-layer registry, file IO) plus `execute_skill`. Session-scoped registry tools (`run_skill`/`save_skill`/`list_skills`) receive a hidden `_session` handle injected by the loop. `ToolIndex.refresh` makes skills discoverable.

**Tech Stack:** Python 3.9.7, PyYAML (already installed, 6.0.2 — added to requirements.txt), pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-reusable-skills-design.md` (binding — read first).

## Global Constraints

- Python 3.9.7 — `from __future__ import annotations`; `typing.Optional[X]`, never `X | None`.
- Skills are data: no `eval`, no imports, no code generation; slot substitution is value templating only; every tool name must exist in `REGISTRY_BY_NAME`.
- `ToolLoopRunner.run` keeps returning `{"reply", "images", "tools_used"}`; `process_single_input` keeps `{"reply", "images", "trace"}`.
- `_session` never appears in LLM-facing schemas, trace events, provenance sidecars, or recorded args.
- `last_turn_calls`/`current_turn_calls` (argument VALUES) live only in `ContextManager` memory — never persisted, never in the JSONL/OTel trace.
- Replay goes through `execute_call` (same validators, guards, sandboxes, events, sidecars, auto-plots as a live call).
- Circular-import rule: `core/tool_registry.py` imports `tools/skill_tools.py`; therefore `tools/skill_tools.py` and `core/skills.py` import `core.tool_registry`/`core.tool_loop`/`core.executor_agent` ONLY inside function bodies.
- Working dir: `/Users/kumono/Desktop/devs/dash-apps/wedge-model/geo-mcp/seismic_chatbot` (own git repo). Branch: `reusable-skills` (already created from `turn-transparency`, holds the spec commit b676812).
- Do NOT run the full suite until the final task; per-task runs use only the named files. The spec's illustrative `tuning_from_petro` example is replaced by a built-in `ricker_wavelet` skill (verified tool params) — ruling recorded in the plan.

---

### Task 1: `execute_call` extraction + per-turn call recording

**Files:**
- Modify: `core/tool_loop.py` (`ToolLoopRunner.__init__`, new `execute_call`, `run()` try-block)
- Modify: `core/context_manager.py` (new `begin_turn_recording`)
- Modify: `core/chatbot_tool_use.py`, `core/orchestrator.py` (one line each, after `trace.begin_turn(user_input)`)
- Test: Create `tests/test_execute_call.py`

**Interfaces:**
- Produces: `ToolLoopRunner.execute_call(self, tool_name: str, raw_input: Dict[str, Any], collected_images: List[str]) -> Any` — runs one tool with the full live-turn surroundings, appends `{"tool", "args", "ok": True}` to context key `current_turn_calls`, raises on failure, returns the raw result. `ToolLoopRunner.current_skill: Optional[str]` attribute (None). `ContextManager.begin_turn_recording(self, user_input: str) -> None` rotates `current_turn_calls`→`last_turn_calls`, `current_turn_input`→`last_turn_input`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_execute_call.py`:

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


def _cm():
    cm = ContextManager()
    cm.trace.persist_dir = ""
    return cm


def test_execute_call_runs_tool_with_full_surroundings():
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("make a 30 Hz ricker")
    runner = ToolLoopRunner(None, ToolManager(), cm)
    images = []
    result = runner.execute_call("make_ricker", {"frequency": 30}, images)
    assert isinstance(result, tuple) and len(result) == 2
    assert images and images[0].endswith(".png")  # auto-plot harvested
    kinds = [e["t"] for e in cm.trace.events]
    assert "tool_call" in kinds and "auto_plot" in kinds
    calls = cm.get_context("current_turn_calls")
    assert calls == [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]


def test_execute_call_raises_on_failure_and_records_nothing():
    import pytest
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    runner = ToolLoopRunner(None, ToolManager(), cm)
    with pytest.raises(ValueError):
        runner.execute_call("no_such_tool", {}, [])
    assert cm.get_context("current_turn_calls") == []


def test_begin_turn_recording_rotates():
    cm = _cm()
    cm.begin_turn_recording("first")
    cm.get_context("current_turn_calls").append({"tool": "a", "args": {}, "ok": True})
    cm.begin_turn_recording("second")
    assert cm.get_context("last_turn_calls") == [{"tool": "a", "args": {}, "ok": True}]
    assert cm.get_context("last_turn_input") == "first"
    assert cm.get_context("current_turn_calls") == []
    assert cm.get_context("current_turn_input") == "second"


def test_run_loop_still_returns_contract_and_records_calls():
    cm = _cm()
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    runner = ToolLoopRunner(FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 25}')],
         "usage": None},
        {"content": "<reply>done</reply>", "tool_calls": None, "usage": None},
    ]), ToolManager(), cm)
    out = runner.run("sys", [{"role": "user", "content": "x"}], tools=[])
    assert set(out) == {"reply", "images", "tools_used"}
    assert out["tools_used"] == ["make_ricker"]
    assert cm.get_context("current_turn_calls")[0]["args"] == {"frequency": 25}


def test_process_single_input_rotates_recording():
    from core.chatbot_tool_use import SeismicChatBotToolUse
    bot = SeismicChatBotToolUse(llm_client=FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>ok</reply>", "tool_calls": None, "usage": None},
        {"content": "<reply>second</reply>", "tool_calls": None, "usage": None},
    ]), tool_manager=ToolManager(), knowledge_base=object())
    bot.context_manager.trace.persist_dir = ""
    bot.process_single_input("make a 30 Hz ricker wavelet")
    assert bot.context_manager.get_context("current_turn_calls")[0]["tool"] == "make_ricker"
    bot.process_single_input("make another wavelet please")
    assert bot.context_manager.get_context("last_turn_calls")[0]["tool"] == "make_ricker"
    assert bot.context_manager.get_context("last_turn_input") == "make a 30 Hz ricker wavelet"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_execute_call.py -q`; expected `AttributeError` (no `execute_call` / `begin_turn_recording`).

- [ ] **Step 3: Implement**

`core/context_manager.py` — add method:

```python
    def begin_turn_recording(self, user_input: str) -> None:
        """Rotate the in-memory record of tool calls (name + resolved args) so
        save_skill can capture the PREVIOUS turn while the current one runs.
        Never persisted: argument values stay in process memory only."""
        self.set_context("last_turn_calls", self.get_context("current_turn_calls") or [])
        self.set_context("last_turn_input", self.get_context("current_turn_input") or "")
        self.set_context("current_turn_calls", [])
        self.set_context("current_turn_input", user_input)
```

`core/tool_loop.py`:
- In `__init__`, add `self.current_skill: Optional[str] = None`.
- Add the method (after `_write_provenance`):

```python
    def execute_call(self, tool_name: str, raw_input: Dict[str, Any],
                     collected_images: List[str]) -> Any:
        """Run ONE tool with everything a live turn does around it: context
        injection, warning capture, tool_call event, context update, image
        harvest + provenance sidecar, auto-plot chaining, and the in-memory
        current_turn_calls recording used by save_skill. Shared by run() and
        by skill replay. Raises on tool failure; returns the raw result."""
        tool_input = self.inject_context_inputs(tool_name, raw_input)
        public_input = {k: v for k, v in tool_input.items() if k != "_session"}
        injected = sorted(k for k in public_input if k not in raw_input)
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tool_result = self.tool_manager.process_tool_call(tool_name, tool_input)
        for w in caught:
            message = str(w.message)[:300]
            logger.warning(f"{tool_name}: {w.category.__name__}: {message}")
            emit_event(self.context_manager, "physics_warning",
                       tool=tool_name, category=w.category.__name__,
                       message=message)
        emit_event(self.context_manager, "tool_call", tool=tool_name, ok=True,
                   ms=round((time.perf_counter() - started) * 1000, 1),
                   injected=injected, overridden=overridden,
                   defaults_filled=defaults_filled)
        calls = self.context_manager.get_context("current_turn_calls")
        if isinstance(calls, list):
            calls.append({"tool": tool_name, "args": dict(public_input), "ok": True})
        self.update_context(tool_name, tool_input, tool_result)
        before_direct = len(collected_images)
        self.harvest_images(tool_result, collected_images)
        self._write_provenance(collected_images[before_direct:], tool_name, public_input)
        chained_result = self.handle_automatic_chaining(tool_name, tool_input, tool_result)
        if chained_result:
            before_chained = len(collected_images)
            self.harvest_images(chained_result, collected_images)
            self._write_provenance(collected_images[before_chained:],
                                   AUTO_PLOT.get(tool_name) or "auto_plot",
                                   {}, compute_tool=tool_name,
                                   compute_input=public_input)
            emit_event(self.context_manager, "auto_plot", compute=tool_name,
                       plot=AUTO_PLOT.get(tool_name), fired=True)
        elif AUTO_PLOT.get(tool_name):
            logger.warning(
                f"auto-plot {AUTO_PLOT[tool_name]} did not run after "
                f"{tool_name} (missing context or plot error)")
            emit_event(self.context_manager, "auto_plot", compute=tool_name,
                       plot=AUTO_PLOT[tool_name], fired=False)
        return tool_result
```

- In `run()`, replace the whole `try:` body (from `raw_input = self.parse_tool_input(tool_input_str)` through the `elif AUTO_PLOT.get(tool_name): ... fired=False)` block, keeping the `# Loop so the model...` comment) with:

```python
                raw_input = self.parse_tool_input(tool_input_str)
                tool_result = self.execute_call(tool_name, raw_input, collected_images)
                tools_used.append(tool_name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self.compact_tool_result(tool_result)
                })
```

(The `except Exception as e:` handler stays exactly as is. Ordering note: `tools_used`/the tool message are now appended after post-processing rather than before; post-processing helpers already swallow their own errors, so behavior is unchanged, and a post-processing exception can no longer leave a duplicate tool message.)

- `_write_provenance`: after building `payload`, add `if self.current_skill: payload["skill"] = self.current_skill`.

`core/chatbot_tool_use.py` and `core/orchestrator.py`: immediately after `trace.begin_turn(user_input)` add `self.context_manager.begin_turn_recording(user_input)`.

- [ ] **Step 4: Run to verify green** — `pytest tests/test_execute_call.py tests/test_tool_loop_trace.py tests/test_physics_warning_capture.py tests/test_provenance.py tests/test_chatbot_trace.py tests/test_orchestrator_trace.py -q`; all pass (the extraction is behavior-preserving for every Tier 1-3 test).

- [ ] **Step 5: Commit**

```bash
git add core/tool_loop.py core/context_manager.py core/chatbot_tool_use.py core/orchestrator.py tests/test_execute_call.py
git commit -m "refactor(loop): extract ToolLoopRunner.execute_call; record per-turn tool calls in session memory"
```

---

### Task 2: `core/skills.py` — model, validation, substitution, registry, parameterizer, file IO

**Files:**
- Create: `core/skills.py` (pure half; `execute_skill` comes in Task 4)
- Modify: `config/settings.py` (add `SEISMIC_SKILLS_DIR`, `SKILLS_REPO_DIR`), `requirements.txt` (add `pyyaml`)
- Test: Create `tests/test_skills_model.py`

**Interfaces (Produces):**
- `Skill` dataclass: `name, description, parameters: Dict[str, dict], tools: List[str], procedure: str, chain: List[dict]` (each `{"tool": str, "args": dict}`), `source: str` ("repo"/"runtime"/"memory"), `path: Optional[str]`.
- `validate_skill(data: dict, source="memory", path=None) -> Skill` (raises `ValueError`).
- `substitute(value, params) -> Any`; `fill_procedure(procedure, params) -> str`; `resolve_params(skill, params) -> Dict` (defaults/unknown/required).
- `build_chain(calls, param_values, context_params, max_list=12) -> Tuple[List[str], List[dict]]`; `build_procedure(input_text, param_values, tools) -> str`; `capture_skill(name, description, parameters, calls, input_text, context_params) -> dict` (a validated skill dict).
- `SkillRegistry(repo_dir, runtime_dir)` with `reload()`, `get(name)`, `names()`, `list()`, `specs()`, `save(skill_dict, overwrite=False) -> str`.
- `get_registry()` / `set_registry(reg)` module singleton; `SkillCard` duck-typed card spec (`name` = `"skill:<name>"`, `description`, `params`, `required`, `auto_plot=None`).
- `CONTEXT_PARAMS = ("image_path", "interpretation", "model")` (mirror of `_CONTEXT_INPUTS` param names — hardcoded to avoid importing tool_loop at module level).

- [ ] **Step 1: Write the failing tests** — create `tests/test_skills_model.py`:

```python
import pytest
import yaml

from core.skills import (CONTEXT_PARAMS, SkillRegistry, build_chain, build_procedure,
                         capture_skill, fill_procedure, resolve_params, substitute,
                         validate_skill)

_GOOD = {
    "name": "ricker_wavelet",
    "description": "Build a Ricker wavelet.",
    "parameters": {"frequency": {"type": "number", "description": "Hz", "default": 30}},
    "tools": ["make_ricker"],
    "procedure": "Create a {{frequency}} Hz Ricker wavelet.",
    "chain": [{"tool": "make_ricker", "args": {"frequency": "{{frequency}}"}}],
}


def test_validate_good_skill():
    s = validate_skill(_GOOD)
    assert s.name == "ricker_wavelet" and s.chain[0]["tool"] == "make_ricker"


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.pop("procedure"), "procedure"),
    (lambda d: d.update(name="Bad Name"), "name"),
    (lambda d: d.update(tools=["no_such_tool"]), "no_such_tool"),
    (lambda d: d.update(chain=[{"tool": "wedge_model", "args": {}}]), "wedge_model"),
    (lambda d: d.update(procedure="use {{nope}}"), "nope"),
    (lambda d: d.update(chain=[{"tool": "make_ricker", "args": {"frequency": "{{nope}}"}}]), "nope"),
])
def test_validate_rejects(mutate, msg):
    data = yaml.safe_load(yaml.safe_dump(_GOOD))
    mutate(data)
    with pytest.raises(ValueError) as exc:
        validate_skill(data)
    assert msg in str(exc.value)


def test_substitute_typed_and_textual():
    params = {"freq": 30, "name": "sand"}
    assert substitute("{{freq}}", params) == 30            # exact slot -> typed value
    assert substitute("{{freq}} Hz {{name}}", params) == "30 Hz sand"
    assert substitute({"a": ["{{freq}}", "x"]}, params) == {"a": [30, "x"]}
    assert substitute(5, params) == 5


def test_resolve_params_defaults_unknown_required():
    s = validate_skill(_GOOD)
    assert resolve_params(s, {}) == {"frequency": 30}
    assert resolve_params(s, {"frequency": 45}) == {"frequency": 45}
    with pytest.raises(ValueError):
        resolve_params(s, {"bogus": 1})
    strict = dict(_GOOD, parameters={"frequency": {"type": "number"}})
    with pytest.raises(ValueError):
        resolve_params(validate_skill(strict), {})


def test_build_chain_parameterizes_by_value_and_drops_context_args():
    calls = [
        {"tool": "make_ricker", "args": {"frequency": 30, "time_length": 200}, "ok": True},
        {"tool": "interpret_outcrop", "args": {"image_path": "/tmp/x.png"}, "ok": True},
        {"tool": "wedge_model", "args": {"wavelet_freq": 30.0, "v1": 2500,
                                        "big": list(range(50))}, "ok": True},
    ]
    tools, chain = build_chain(calls, {"freq": 30}, set(CONTEXT_PARAMS))
    assert tools == ["make_ricker", "interpret_outcrop", "wedge_model"]
    assert chain[0]["args"] == {"frequency": "{{freq}}", "time_length": 200}
    assert chain[1]["args"] == {}                      # context arg dropped
    assert chain[2]["args"] == {"wavelet_freq": "{{freq}}", "v1": 2500}  # 30.0 == 30; big list dropped


def test_build_chain_rejects_unused_parameter():
    calls = [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]
    with pytest.raises(ValueError) as exc:
        build_chain(calls, {"thickness": 100}, set(CONTEXT_PARAMS))
    assert "thickness=100" in str(exc.value)


def test_build_procedure_substitutes_values_or_falls_back():
    assert build_procedure("make a 30 Hz ricker with 0.25 porosity",
                           {"freq": 30, "phit": 0.25}, ["make_ricker"]) == \
        "make a {{freq}} Hz ricker with {{phit}} porosity"
    assert build_procedure("", {"freq": 30}, ["make_ricker", "analyze_wedge"]) == \
        "Run the recorded chain: make_ricker → analyze_wedge."


def test_capture_skill_produces_valid_skill():
    calls = [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]
    data = capture_skill("my_ricker", "A ricker.", {"freq": 30}, calls,
                         "make a 30 Hz ricker", set(CONTEXT_PARAMS))
    s = validate_skill(data)
    assert s.tools == ["make_ricker"]
    assert s.parameters["freq"]["default"] == 30
    assert s.chain == [{"tool": "make_ricker", "args": {"frequency": "{{freq}}"}}]
    assert fill_procedure(s.procedure, {"freq": 45}) == "make a 45 Hz ricker"


def test_capture_accepts_rich_parameter_form():
    calls = [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}]
    data = capture_skill("r", "d", {"freq": {"value": 30, "description": "Hz"}}, calls,
                         "x", set(CONTEXT_PARAMS))
    assert data["parameters"]["freq"] == {"type": "number", "description": "Hz", "default": 30}


def test_registry_two_layers_override_and_save(tmp_path, caplog):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    (repo / "ricker_wavelet.yaml").write_text(yaml.safe_dump(_GOOD, sort_keys=False))
    reg = SkillRegistry(repo_dir=str(repo), runtime_dir=str(runtime))
    assert reg.names() == ["ricker_wavelet"]
    assert reg.get("ricker_wavelet").source == "repo"
    override = dict(_GOOD, description="runtime version")
    path = reg.save(override)
    assert path == str(runtime / "ricker_wavelet.yaml")
    assert reg.get("ricker_wavelet").description == "runtime version"
    assert reg.get("ricker_wavelet").source == "runtime"
    assert any("overrides" in r.message for r in caplog.records)
    with pytest.raises(ValueError):
        reg.save(override)  # exists, overwrite=False
    reg.save(override, overwrite=True)
    with pytest.raises(ValueError):
        reg.save(dict(_GOOD, name="make_ricker"))  # collides with a registry tool
    cards = reg.specs()
    assert cards[0].name == "skill:ricker_wavelet" and cards[0].required == ()
    assert reg.list()[0]["has_chain"] is True


def test_registry_skips_invalid_files_with_warning(tmp_path, caplog):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "broken.yaml").write_text("name: broken\n")
    reg = SkillRegistry(repo_dir=str(tmp_path / "none"), runtime_dir=str(runtime))
    assert reg.names() == []
    assert any("broken" in r.message for r in caplog.records)


def test_settings_expose_skill_dirs():
    from config.settings import SEISMIC_SKILLS_DIR, SKILLS_REPO_DIR
    assert SEISMIC_SKILLS_DIR and SKILLS_REPO_DIR.endswith("skills")
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_skills_model.py -q`; expected `ModuleNotFoundError: core.skills`.

- [ ] **Step 3: Implement**

`config/settings.py` — after the `SEISMIC_TRACE_DIR` block:

```python
# Reusable skills (core/skills.py): curated skills ship in <package>/skills/;
# captured skills are written to SEISMIC_SKILLS_DIR (0o700), runtime overriding
# repo on a name clash (WARNING).
SKILLS_REPO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")
SEISMIC_SKILLS_DIR = os.environ.get("SEISMIC_SKILLS_DIR") or os.path.join(
    tempfile.gettempdir(), "seismic_skills")
```

`requirements.txt` — append a line `pyyaml>=6.0`.

Create `core/skills.py`:

```python
"""Reusable skills (Tier 4): YAML-defined, parameterized flows that run either
as a deterministic replay of a recorded tool chain or as an LLM-guided
procedure with a scoped toolset.

Pure half (this file's first part): model, validation, slot substitution,
parameterizer, two-layer registry, file IO. Execution (execute_skill) sits at
the bottom and is the only part that touches the loop/executor — imported
lazily to keep core.tool_registry free of import cycles.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_SLOT_RE = re.compile(r"\{\{(\w+)\}\}")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SCALARS = (str, int, float, bool, type(None))
CONTEXT_PARAMS = ("image_path", "interpretation", "model")  # mirrors _CONTEXT_INPUTS


@dataclass
class Skill:
    name: str
    description: str
    parameters: Dict[str, dict]
    tools: List[str]
    procedure: str
    chain: List[dict] = field(default_factory=list)
    source: str = "memory"
    path: Optional[str] = None


@dataclass(frozen=True)
class SkillCard:
    """Duck-typed ToolSpec stand-in so ToolIndex.render_card works unchanged."""
    name: str
    description: str
    params: dict
    required: tuple
    auto_plot: Optional[str] = None


def _slots_in(value: Any) -> List[str]:
    if isinstance(value, str):
        return _SLOT_RE.findall(value)
    if isinstance(value, dict):
        return [s for v in value.values() for s in _slots_in(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _slots_in(v)]
    return []


def _registry_names() -> set:
    from core.tool_registry import REGISTRY_BY_NAME  # lazy: avoids import cycle
    return set(REGISTRY_BY_NAME)


def validate_skill(data: Any, source: str = "memory",
                   path: Optional[str] = None) -> Skill:
    """Turn a raw mapping into a Skill or raise ValueError naming the problem."""
    where = f" ({path})" if path else ""
    if not isinstance(data, dict):
        raise ValueError(f"skill{where}: expected a mapping")
    for key in ("name", "description", "parameters", "tools", "procedure"):
        if key not in data:
            raise ValueError(f"skill{where}: missing required key '{key}'")
    name = data["name"]
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"skill{where}: invalid name {name!r} (use [a-z][a-z0-9_]*)")
    parameters = data["parameters"] or {}
    if not isinstance(parameters, dict) or not all(isinstance(v, dict) for v in parameters.values()):
        raise ValueError(f"skill {name}: parameters must map names to schema dicts")
    tools = list(data["tools"] or [])
    known = _registry_names()
    for t in tools:
        if t not in known:
            raise ValueError(f"skill {name}: unknown tool '{t}'")
    chain = list(data.get("chain") or [])
    for i, step in enumerate(chain):
        if not isinstance(step, dict) or "tool" not in step:
            raise ValueError(f"skill {name}: chain step {i} must have a 'tool'")
        if step["tool"] not in known:
            raise ValueError(f"skill {name}: chain step {i} unknown tool '{step['tool']}'")
        if step["tool"] not in tools:
            raise ValueError(f"skill {name}: chain tool '{step['tool']}' not in tools")
        if not isinstance(step.get("args", {}), dict):
            raise ValueError(f"skill {name}: chain step {i} args must be a mapping")
    used = set(_slots_in(data["procedure"])) | set(_slots_in([s.get("args", {}) for s in chain]))
    undeclared = sorted(used - set(parameters))
    if undeclared:
        raise ValueError(f"skill {name}: undeclared slot(s) {', '.join(undeclared)}")
    return Skill(name=name, description=str(data["description"]), parameters=parameters,
                 tools=tools, procedure=str(data["procedure"]),
                 chain=[{"tool": s["tool"], "args": dict(s.get("args", {}))} for s in chain],
                 source=source, path=path)


def substitute(value: Any, params: Dict[str, Any]) -> Any:
    """Value-level templating: an exact '{{slot}}' string becomes the typed
    parameter value; slots inside longer strings are replaced textually."""
    if isinstance(value, str):
        m = _SLOT_RE.fullmatch(value)
        if m:
            return params[m.group(1)]
        return _SLOT_RE.sub(lambda mm: str(params[mm.group(1)]), value)
    if isinstance(value, dict):
        return {k: substitute(v, params) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [substitute(v, params) for v in value]
    return value


def fill_procedure(procedure: str, params: Dict[str, Any]) -> str:
    return _SLOT_RE.sub(lambda m: str(params[m.group(1)]), procedure)


def resolve_params(skill: Skill, params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params or {})
    unknown = sorted(set(params) - set(skill.parameters))
    if unknown:
        raise ValueError(f"skill {skill.name}: unknown parameter(s) {', '.join(unknown)}")
    bound: Dict[str, Any] = {}
    for pname, schema in skill.parameters.items():
        if pname in params:
            bound[pname] = params[pname]
        elif "default" in schema:
            bound[pname] = schema["default"]
        else:
            raise ValueError(f"skill {skill.name}: missing required parameter '{pname}'")
    return bound


# --- capture -----------------------------------------------------------------

def _is_portable(value: Any, max_list: int) -> bool:
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) <= max_list and all(isinstance(v, _SCALARS) for v in value)
    return False


def _match_slot(value: Any, param_values: Dict[str, Any]) -> Optional[str]:
    for slot, pv in param_values.items():
        if isinstance(value, bool) or isinstance(pv, bool):
            if value is pv:
                return slot
            continue
        if isinstance(value, (int, float)) and isinstance(pv, (int, float)):
            if float(value) == float(pv):
                return slot
        elif isinstance(value, str) and isinstance(pv, str) and value == pv:
            return slot
    return None


def build_chain(calls: Iterable[dict], param_values: Dict[str, Any],
                context_params: set, max_list: int = 12) -> Tuple[List[str], List[dict]]:
    tools: List[str] = []
    chain: List[dict] = []
    used: set = set()
    for call in calls:
        args: Dict[str, Any] = {}
        for k, v in (call.get("args") or {}).items():
            if k in context_params or k == "_session" or not _is_portable(v, max_list):
                continue
            slot = _match_slot(v, param_values)
            if slot is not None:
                args[k] = "{{" + slot + "}}"
                used.add(slot)
            else:
                args[k] = v
        chain.append({"tool": call["tool"], "args": args})
        if call["tool"] not in tools:
            tools.append(call["tool"])
    for slot, pv in param_values.items():
        if slot not in used:
            raise ValueError(f"parameter {slot}={pv!r} was not used by any tool call")
    return tools, chain


def _value_forms(value: Any) -> List[str]:
    forms = [str(value)]
    if isinstance(value, float) and value.is_integer():
        forms.append(str(int(value)))
    if isinstance(value, int) and not isinstance(value, bool):
        forms.append(f"{value}.0")
    return forms


def build_procedure(input_text: str, param_values: Dict[str, Any],
                    tools: List[str]) -> str:
    text = (input_text or "").strip()
    if not text:
        return "Run the recorded chain: " + " → ".join(tools) + "."
    for slot, pv in sorted(param_values.items(), key=lambda kv: -len(str(kv[1]))):
        for form in _value_forms(pv):
            text = re.sub(rf"(?<![\w.]){re.escape(form)}(?![\w.])", "{{" + slot + "}}", text)
    return text


def _normalize_parameters(parameters: Dict[str, Any]) -> Tuple[Dict[str, dict], Dict[str, Any]]:
    """Accept {slot: value} or {slot: {value, description?, type?}}."""
    schemas: Dict[str, dict] = {}
    values: Dict[str, Any] = {}
    for slot, spec in (parameters or {}).items():
        if not _NAME_RE.match(str(slot)):
            raise ValueError(f"invalid parameter name {slot!r}")
        if isinstance(spec, dict) and "value" in spec:
            value = spec["value"]
            ptype = spec.get("type") or ("number" if isinstance(value, (int, float)) and not isinstance(value, bool) else "string")
            schemas[slot] = {"type": ptype, "description": spec.get("description", slot), "default": value}
        else:
            value = spec
            ptype = "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else "string"
            schemas[slot] = {"type": ptype, "description": slot, "default": value}
        values[slot] = value
    if not values:
        raise ValueError("at least one parameter is required to make a skill reusable")
    return schemas, values


def capture_skill(name: str, description: str, parameters: Dict[str, Any],
                  calls: List[dict], input_text: str, context_params: set) -> dict:
    if not calls:
        raise ValueError("the last turn ran no tools — nothing to capture")
    schemas, values = _normalize_parameters(parameters)
    tools, chain = build_chain(calls, values, context_params)
    data = {"name": name, "description": description, "parameters": schemas,
            "tools": tools, "procedure": build_procedure(input_text, values, tools),
            "chain": chain}
    validate_skill(data)
    return data


# --- registry ----------------------------------------------------------------

class SkillRegistry:
    def __init__(self, repo_dir: Optional[str] = None, runtime_dir: Optional[str] = None):
        if repo_dir is None or runtime_dir is None:
            from config.settings import SEISMIC_SKILLS_DIR, SKILLS_REPO_DIR
            repo_dir = repo_dir or SKILLS_REPO_DIR
            runtime_dir = runtime_dir or SEISMIC_SKILLS_DIR
        self.repo_dir = repo_dir
        self.runtime_dir = runtime_dir
        self._skills: Dict[str, Skill] = {}
        self.reload()

    def _load_dir(self, directory: str, source: str) -> None:
        if not directory or not os.path.isdir(directory):
            return
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, fname)
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                skill = validate_skill(data, source=source, path=path)
            except Exception as e:
                logger.warning(f"skipping skill file {path}: {e}")
                continue
            if skill.name in self._skills and source == "runtime":
                logger.warning(f"runtime skill '{skill.name}' overrides the repo skill")
            self._skills[skill.name] = skill

    def reload(self) -> None:
        self._skills = {}
        self._load_dir(self.repo_dir, "repo")
        self._load_dir(self.runtime_dir, "runtime")

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise ValueError(f"unknown skill '{name}'; use list_skills to see available skills")
        return self._skills[name]

    def names(self) -> List[str]:
        return sorted(self._skills)

    def list(self) -> List[dict]:
        return [{"name": s.name, "description": s.description, "parameters": s.parameters,
                 "has_chain": bool(s.chain), "source": s.source}
                for s in (self._skills[n] for n in self.names())]

    def specs(self) -> List[SkillCard]:
        return [SkillCard(name=f"skill:{s.name}", description=s.description,
                          params=s.parameters,
                          required=tuple(p for p, sch in s.parameters.items() if "default" not in sch))
                for s in (self._skills[n] for n in self.names())]

    def save(self, data: dict, overwrite: bool = False) -> str:
        skill = validate_skill(data, source="runtime")
        if skill.name in _registry_names():
            raise ValueError(f"skill name '{skill.name}' collides with a registry tool")
        os.makedirs(self.runtime_dir, mode=0o700, exist_ok=True)
        path = os.path.join(self.runtime_dir, f"{skill.name}.yaml")
        if os.path.exists(path) and not overwrite:
            raise ValueError(f"skill '{skill.name}' already exists at {path}; pass overwrite=true to replace")
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        self.reload()
        return path


_REGISTRY: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SkillRegistry()
    return _REGISTRY


def set_registry(registry: Optional[SkillRegistry]) -> None:
    global _REGISTRY
    _REGISTRY = registry
```

- [ ] **Step 4: Run to verify green** — `pytest tests/test_skills_model.py -q`; expected 13 passed (parametrized counts as 6).

- [ ] **Step 5: Commit**

```bash
git add core/skills.py config/settings.py requirements.txt tests/test_skills_model.py
git commit -m "feat(skills): skill model, validation, slot substitution, parameterizer, two-layer registry"
```

---

### Task 3: Session-scoped tools, `SessionHandle`, `ToolIndex.refresh`, sidecar no-overwrite

**Files:**
- Modify: `core/tool_registry.py` (`ToolSpec.session_scoped`), `core/tool_manager.py` (`execute_tool`), `core/tool_loop.py` (`inject_context_inputs`), `core/tool_index.py` (`refresh`), `core/provenance.py` (no-overwrite)
- Create: `core/session_handle.py`
- Test: Create `tests/test_session_scoped.py`; append to `tests/test_provenance.py`

**Interfaces (Produces):**
- `ToolSpec.session_scoped: bool = False`.
- `core.session_handle.SessionHandle(llm_client, tool_manager, context_manager, runner)` dataclass.
- `ToolManager.execute_tool` strips `_session` before validation/logging and passes it to `spec.fn` only when the spec is session-scoped.
- `ToolLoopRunner.inject_context_inputs` adds `_session` for session-scoped specs.
- `ToolIndex.refresh(self, extra_specs=None) -> None`.
- `write_plot_provenance` returns the existing sidecar path without rewriting when one already exists.

- [ ] **Step 1: Write the failing tests** — create `tests/test_session_scoped.py`:

```python
from core.context_manager import ContextManager
from core.session_handle import SessionHandle
from core.tool_loop import ToolLoopRunner
from core.tool_manager import ToolManager
from core.tool_registry import ToolSpec


def _scoped_spec(fn):
    return ToolSpec(name="probe_session", fn=fn, description="probe", params={},
                    required=[], session_scoped=True)


def test_execute_tool_passes_session_only_to_scoped_tools():
    seen = {}

    def probe(_session=None):
        seen["session"] = _session
        return "ok"

    tm = ToolManager()
    tm.specs = dict(tm.specs, probe_session=_scoped_spec(probe))
    tm.tools["probe_session"] = probe
    handle = object()
    assert tm.execute_tool("probe_session", {"_session": handle}) == "ok"
    assert seen["session"] is handle
    # a non-scoped tool never receives _session even if present
    tm.execute_tool("make_ricker", {"frequency": 30, "_session": handle})


def test_loop_injects_session_handle_and_keeps_it_out_of_events_and_recording():
    seen = {}

    def probe(_session=None):
        seen["session"] = _session
        return {"value": 1}

    tm = ToolManager()
    tm.specs = dict(tm.specs, probe_session=_scoped_spec(probe))
    tm.tools["probe_session"] = probe
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    runner = ToolLoopRunner(None, tm, cm)
    runner.execute_call("probe_session", {}, [])
    assert isinstance(seen["session"], SessionHandle)
    assert seen["session"].runner is runner
    assert seen["session"].context_manager is cm
    evt = [e for e in cm.trace.events if e["t"] == "tool_call"][0]
    assert evt["injected"] == []
    assert cm.get_context("current_turn_calls")[0]["args"] == {}


def test_tool_index_refresh_adds_and_removes_extra_cards(tmp_path):
    from core.skills import SkillCard
    from core.tool_index import ToolIndex
    idx = ToolIndex(persist_directory=str(tmp_path))
    base = idx.collection.count()
    card = SkillCard(name="skill:demo", description="Demo skill for tests.",
                     params={"freq": {"type": "number"}}, required=("freq",))
    idx.refresh([card])
    assert idx.collection.count() == base + 1
    assert any(c.name == "skill:demo" for c in idx.search("demo skill"))
    idx.refresh([])
    assert idx.collection.count() == base
```

Append to `tests/test_provenance.py`:

```python
def test_existing_sidecar_is_not_overwritten(tmp_path):
    png = tmp_path / "plot.png"
    png.write_bytes(b"x")
    write_plot_provenance(str(png), {"tool": "first"})
    write_plot_provenance(str(png), {"tool": "second"})
    assert json.loads((tmp_path / "plot.png.prov.json").read_text())["tool"] == "first"
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_session_scoped.py tests/test_provenance.py -q`; expected: `ToolSpec` rejects `session_scoped`, no `core.session_handle`, no `refresh`, overwrite test fails.

- [ ] **Step 3: Implement**

Create `core/session_handle.py`:

```python
"""Handle injected into session-scoped registry tools (run_skill, save_skill,
list_skills) as the hidden `_session` kwarg — never in LLM-facing schemas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SessionHandle:
    llm_client: Any
    tool_manager: Any
    context_manager: Any
    runner: Any  # the ToolLoopRunner executing the current call
```

`core/tool_registry.py` — add `session_scoped: bool = False` as the last field of `ToolSpec`.

`core/tool_manager.py` — replace `execute_tool` with:

```python
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        spec = self.specs.get(tool_name)
        if spec is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        params = dict(params)
        session = params.pop("_session", None)  # hidden handle, never validated/logged
        # Fill defaults BEFORE validating so required-after-default works correctly.
        full_params = dict(spec.defaults)
        full_params.update(params)
        is_valid, msg = self.validate_parameters(tool_name, full_params)
        if not is_valid:
            raise ValueError(msg)
        logger.info(f"Calling {tool_name} with {full_params}")
        if getattr(spec, "session_scoped", False):
            return spec.fn(**full_params, _session=session)
        return spec.fn(**full_params)
```

`core/tool_loop.py` — add `from core.session_handle import SessionHandle` to imports; at the end of `inject_context_inputs` (before `return filled`):

```python
        spec = getattr(self.tool_manager, "specs", {}).get(tool_name)
        if spec is not None and getattr(spec, "session_scoped", False):
            filled["_session"] = SessionHandle(self.llm_client, self.tool_manager,
                                               self.context_manager, self)
```

`core/tool_index.py` — add method:

```python
    def refresh(self, extra_specs: Optional[list] = None) -> None:
        """Re-index REGISTRY plus extra card specs (e.g. skills); stale cards
        are deleted by _populate, so removed extras disappear too."""
        specs = list(REGISTRY) + list(extra_specs or [])
        plot_targets = {s.auto_plot for s in specs if getattr(s, "auto_plot", None)}
        self._specs = [s for s in specs if s.name not in plot_targets]
        self._populate()
```

(`Optional` needs importing from typing in tool_index.py: `from typing import Optional`.)

`core/provenance.py` — in `write_plot_provenance`, right after computing `sidecar`: 

```python
    if os.path.exists(sidecar):
        return sidecar  # first writer wins (plots are unique mkstemp paths)
```

- [ ] **Step 4: Run to verify green** — `pytest tests/test_session_scoped.py tests/test_provenance.py tests/test_tool_manager.py tests/test_tool_registry.py tests/test_tool_index.py -q` (if `test_tool_index.py` doesn't exist, run `pytest tests -k tool_index -q`); all pass.

- [ ] **Step 5: Commit**

```bash
git add core/tool_registry.py core/tool_manager.py core/tool_loop.py core/tool_index.py core/provenance.py core/session_handle.py tests/test_session_scoped.py tests/test_provenance.py
git commit -m "feat(skills): session-scoped tools (_session handle), ToolIndex.refresh, first-writer-wins sidecars"
```

---

### Task 4: `execute_skill`, the three registry tools, prompts, orchestrator index

**Files:**
- Modify: `core/skills.py` (append execution half)
- Create: `tools/skill_tools.py`
- Modify: `core/tool_registry.py` (three `ToolSpec` entries), `core/chatbot_tool_use.py` (system prompt lines), `core/orchestrator.py` (prompt rule + index refresh)
- Test: Create `tests/test_skill_execution.py`

**Interfaces (Produces):**
- `core.skills.execute_skill(skill, params, mode, session) -> Dict[str, Any]` (spec §4 shapes; `extra_image_paths` key carries plots; `skill_run` event; depth guard via context key `_skill_depth`).
- Registry tools `run_skill(name, params=None, mode="auto", _session=None)`, `save_skill(name, description, parameters, overwrite=False, _session=None)`, `list_skills(_session=None)` in `tools/skill_tools.py`; all `session_scoped=True`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_skill_execution.py`:

```python
import pytest
import yaml

from core.context_manager import ContextManager
from core.session_handle import SessionHandle
from core.skills import SkillRegistry, set_registry, validate_skill
from core.tool_loop import ToolLoopRunner
from core.tool_manager import ToolManager
from core.tool_registry import REGISTRY_BY_NAME


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


_RICKER = {
    "name": "ricker_wavelet", "description": "Build a Ricker wavelet.",
    "parameters": {"frequency": {"type": "number", "description": "Hz", "default": 30}},
    "tools": ["make_ricker"], "procedure": "Create a {{frequency}} Hz Ricker wavelet.",
    "chain": [{"tool": "make_ricker", "args": {"frequency": "{{frequency}}"}}],
}


@pytest.fixture
def registry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ricker_wavelet.yaml").write_text(yaml.safe_dump(_RICKER, sort_keys=False))
    reg = SkillRegistry(repo_dir=str(repo), runtime_dir=str(tmp_path / "rt"))
    set_registry(reg)
    yield reg
    set_registry(None)


def _session(llm=None):
    cm = ContextManager()
    cm.trace.persist_dir = ""
    cm.trace.begin_turn("t")
    cm.begin_turn_recording("x")
    tm = ToolManager()
    runner = ToolLoopRunner(llm, tm, cm)
    return SessionHandle(llm, tm, cm, runner), cm


def test_registry_declares_skill_tools_as_session_scoped():
    for name in ("run_skill", "save_skill", "list_skills"):
        assert REGISTRY_BY_NAME[name].session_scoped is True


def test_run_skill_replay_executes_chain_with_bound_params(registry):
    from tools.skill_tools import run_skill
    session, cm = _session()
    out = run_skill("ricker_wavelet", {"frequency": 45}, _session=session)
    assert out["mode"] == "replay"
    assert out["steps"] == [{"tool": "make_ricker", "ok": True}]
    assert out["extra_image_paths"] and out["extra_image_paths"][0].endswith(".png")
    assert cm.get_context("last_ricker_wavelet")["parameters"]["frequency"] == 45
    kinds = [e["t"] for e in cm.trace.events]
    assert "skill_run" in kinds and "auto_plot" in kinds
    import json
    sidecar = json.load(open(out["extra_image_paths"][0] + ".prov.json"))
    assert sidecar["skill"] == "ricker_wavelet"
    assert cm.get_context("_skill_depth") == 0


def test_run_skill_replay_stops_on_failed_step(registry):
    from tools.skill_tools import run_skill
    broken = dict(_RICKER, name="broken_ricker",
                  chain=[{"tool": "make_ricker", "args": {}},  # missing required frequency
                         {"tool": "make_ricker", "args": {"frequency": "{{frequency}}"}}])
    registry.save(broken)
    session, cm = _session()
    out = run_skill("broken_ricker", {}, _session=session)
    assert out["mode"] == "replay" and "error" in out
    assert out["steps"][0]["ok"] is False and len(out["steps"]) == 1
    assert any(e["t"] == "tool_call" and not e["ok"] for e in cm.trace.events)


def test_run_skill_guided_uses_executor(registry):
    from tools.skill_tools import run_skill
    llm = FakeLLM([
        {"content": "", "tool_calls": [FakeToolCall("make_ricker", '{"frequency": 30}')],
         "usage": None},
        {"content": "<reply>built it</reply>", "tool_calls": None, "usage": None},
    ])
    session, cm = _session(llm)
    out = run_skill("ricker_wavelet", {"frequency": 30}, mode="guided", _session=session)
    assert out["mode"] == "guided" and out["summary"] == "built it"
    assert out["tools_used"] == ["make_ricker"]
    assert out["extra_image_paths"]


def test_run_skill_rejections(registry):
    from tools.skill_tools import run_skill
    session, _ = _session()
    with pytest.raises(ValueError):
        run_skill("nope", {}, _session=session)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {"bogus": 1}, _session=session)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {}, mode="weird", _session=session)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {}, _session=None)
    session.context_manager.set_context("_skill_depth", 1)
    with pytest.raises(ValueError):
        run_skill("ricker_wavelet", {}, _session=session)


def test_save_skill_captures_previous_turn(registry):
    from tools.skill_tools import list_skills, save_skill
    session, cm = _session()
    cm.set_context("last_turn_calls", [{"tool": "make_ricker", "args": {"frequency": 35}, "ok": True}])
    cm.set_context("last_turn_input", "make a 35 Hz ricker")
    out = save_skill("my_ricker", "My ricker", {"freq": 35}, _session=session)
    assert out["name"] == "my_ricker" and out["n_steps"] == 1
    saved = registry.get("my_ricker")
    assert saved.chain == [{"tool": "make_ricker", "args": {"frequency": "{{freq}}"}}]
    assert saved.procedure == "make a {{freq}} Hz ricker"
    assert [s["name"] for s in list_skills(_session=session)] == ["my_ricker", "ricker_wavelet"]
    with pytest.raises(ValueError):
        save_skill("my_ricker", "again", {"freq": 35}, _session=session)  # no overwrite


def test_save_skill_requires_a_prior_tool_turn(registry):
    from tools.skill_tools import save_skill
    session, cm = _session()
    with pytest.raises(ValueError):
        save_skill("x", "d", {"freq": 1}, _session=session)


def test_classic_prompt_and_orchestrator_prompt_mention_skills():
    from core.chatbot_tool_use import SeismicChatBotToolUse
    from core.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
    prompt = SeismicChatBotToolUse._create_system_prompt(object.__new__(SeismicChatBotToolUse))
    assert "run_skill" in prompt and "save_skill" in prompt and "list_skills" in prompt
    assert "skill:" in ORCHESTRATOR_SYSTEM_PROMPT and "run_skill" in ORCHESTRATOR_SYSTEM_PROMPT
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_skill_execution.py -q`; expected: `KeyError: 'run_skill'`, `ModuleNotFoundError: tools.skill_tools`.

- [ ] **Step 3: Implement**

Append to `core/skills.py`:

```python
# --- execution -----------------------------------------------------------------

def execute_skill(skill: Skill, params: Optional[Dict[str, Any]], mode: str,
                  session: Any) -> Dict[str, Any]:
    """Run a skill: deterministic replay of its chain through the session's
    ToolLoopRunner.execute_call, or LLM-guided via a scoped ExecutorAgent."""
    from core.turn_trace import emit_event  # lazy: keep this module import-light
    if session is None:
        raise ValueError("run_skill requires a live session (call it from the chat loop)")
    cm = session.context_manager
    if (cm.get_context("_skill_depth") or 0) >= 1:
        raise ValueError("run_skill cannot be invoked from inside a running skill")
    if mode not in ("auto", "replay", "guided"):
        raise ValueError(f"mode must be auto, replay or guided (got {mode!r})")
    if mode == "replay" and not skill.chain:
        raise ValueError(f"skill '{skill.name}' has no recorded chain; use mode='guided'")
    bound = resolve_params(skill, params or {})
    use_replay = mode == "replay" or (mode == "auto" and bool(skill.chain))
    runner = session.runner
    cm.set_context("_skill_depth", 1)
    runner.current_skill = skill.name
    try:
        if use_replay:
            images: List[str] = []
            steps: List[dict] = []
            last: Any = None
            for step in skill.chain:
                args = substitute(step["args"], bound)
                try:
                    last = runner.execute_call(step["tool"], args, images)
                    steps.append({"tool": step["tool"], "ok": True})
                except Exception as e:
                    steps.append({"tool": step["tool"], "ok": False, "error": str(e)})
                    emit_event(cm, "tool_call", tool=step["tool"], ok=False, error=str(e))
                    emit_event(cm, "skill_run", name=skill.name, mode="replay",
                               n_steps=len(steps), error=str(e))
                    return {"mode": "replay", "steps": steps,
                            "error": f"step {step['tool']} failed: {e}",
                            "extra_image_paths": images}
            emit_event(cm, "skill_run", name=skill.name, mode="replay", n_steps=len(steps))
            return {"mode": "replay", "steps": steps,
                    "result": runner.compact_value(last), "extra_image_paths": images}
        from core.executor_agent import ExecutorAgent
        brief = fill_procedure(skill.procedure, bound)
        result = ExecutorAgent(session.llm_client, session.tool_manager, cm).run(
            brief, list(skill.tools))
        emit_event(cm, "skill_run", name=skill.name, mode="guided",
                   n_steps=len(result.tools_used))
        out: Dict[str, Any] = {"mode": "guided", "summary": result.summary,
                               "tools_used": list(result.tools_used),
                               "extra_image_paths": list(result.images)}
        if result.error:
            out["error"] = result.error
        return out
    finally:
        runner.current_skill = None
        cm.set_context("_skill_depth", 0)
```

Create `tools/skill_tools.py`:

```python
"""LLM-facing skill tools (session-scoped: the loop injects `_session`).
All core.skills imports are lazy so core.tool_registry can import this module
without a cycle."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def run_skill(name: str, params: Optional[Dict[str, Any]] = None, mode: str = "auto",
              _session: Any = None) -> Dict[str, Any]:
    from core.skills import execute_skill, get_registry
    return execute_skill(get_registry().get(name), params or {}, mode, _session)


def save_skill(name: str, description: str, parameters: Dict[str, Any],
               overwrite: bool = False, _session: Any = None) -> Dict[str, Any]:
    from core.skills import CONTEXT_PARAMS, capture_skill, get_registry
    if _session is None:
        raise ValueError("save_skill requires a live session")
    cm = _session.context_manager
    calls = cm.get_context("last_turn_calls") or []
    data = capture_skill(name, description, parameters, calls,
                         cm.get_context("last_turn_input") or "", set(CONTEXT_PARAMS))
    registry = get_registry()
    path = registry.save(data, overwrite=bool(overwrite))
    _refresh_index(_session, registry)
    return {"name": name, "path": path, "n_steps": len(data["chain"]),
            "parameters": sorted(data["parameters"])}


def list_skills(_session: Any = None) -> List[dict]:
    from core.skills import get_registry
    return get_registry().list()


def _refresh_index(session: Any, registry: Any) -> None:
    """Re-index discovery if the session's runner exposes a tool index."""
    index = getattr(session, "tool_index", None)
    if index is not None and hasattr(index, "refresh"):
        try:
            index.refresh(registry.specs())
        except Exception:  # discovery is best-effort; the skill is saved regardless
            pass
```

`core/tool_registry.py` — import `from tools.skill_tools import run_skill, save_skill, list_skills` alongside the other tool imports, and append three specs to the base `REGISTRY` list (before the workflow specs are concatenated):

```python
    ToolSpec(
        name="run_skill",
        fn=run_skill,
        description=("Run a saved reusable skill by name with parameter values. Skills "
                     "appear in discovery as 'skill:<name>' cards; list_skills shows them "
                     "with their parameters. mode 'auto' replays the recorded tool chain "
                     "when the skill has one (deterministic, no reasoning) and otherwise "
                     "follows the skill's procedure with its scoped tools; 'replay' and "
                     "'guided' force one or the other."),
        params={
            "name": {"type": "string", "description": "Skill name (without the 'skill:' prefix)."},
            "params": {"type": "object", "description": "Parameter values keyed by the skill's parameter names."},
            "mode": {"type": "string", "enum": ["auto", "replay", "guided"],
                     "description": "auto | replay | guided."},
        },
        required=["name"],
        defaults={"params": {}, "mode": "auto"},
        session_scoped=True,
    ),
    ToolSpec(
        name="save_skill",
        fn=save_skill,
        description=("Save the PREVIOUS turn's tool calls as a reusable, parameterized skill. "
                     "parameters maps each new parameter name to the value it had in that turn "
                     "(e.g. {\"freq\": 30}); every matching tool argument becomes a slot. "
                     "Use when the user asks to save/remember/reuse what was just done."),
        params={
            "name": {"type": "string", "description": "New skill name: lowercase letters, digits, underscores."},
            "description": {"type": "string", "description": "One-line description of what the skill does."},
            "parameters": {"type": "object", "description": "{parameter_name: value_used_last_turn, ...}"},
            "overwrite": {"type": "boolean", "description": "Replace an existing skill of the same name."},
        },
        required=["name", "description", "parameters"],
        defaults={"overwrite": False},
        session_scoped=True,
    ),
    ToolSpec(
        name="list_skills",
        fn=list_skills,
        description="List the saved reusable skills with their parameters.",
        params={},
        required=[],
        session_scoped=True,
    ),
```

`core/chatbot_tool_use.py` — in `_create_system_prompt`, after the `- outcrop_to_seismic: ...` line add:

```
- run_skill: Run a saved reusable skill by name with parameter values (replays its recorded tool chain, or follows its procedure); skills are listed by list_skills
- save_skill: Save the previous turn's tool calls as a reusable parameterized skill — use when the user asks to save/remember/reuse what was just done, mapping each parameter name to the value used
- list_skills: List saved reusable skills and their parameters
```

`core/orchestrator.py` — add a rule line to `ORCHESTRATOR_SYSTEM_PROMPT` (before `- Any plot an executor produces...`):

```
- discover_tools may return 'skill:<name>' cards: these are saved reusable flows. Run one by
  delegating a task whose tool_names include run_skill and whose brief says to call
  run_skill with that name and the parameter values. To save the previous turn's work as a
  skill, delegate to save_skill.
```

and in `__init__`, after `self.tool_index = tool_index or ToolIndex()` add:

```python
        if tool_index is None:  # injected fakes need not support refresh
            from core.skills import get_registry
            try:
                self.tool_index.refresh(get_registry().specs())
            except Exception as e:
                logger.warning(f"skill discovery refresh failed: {e}")
```

Also give `SessionHandle` access to the index for `save_skill`'s refresh: in `core/session_handle.py` add field `tool_index: Any = None`, and in `inject_context_inputs` pass `tool_index=getattr(self, "tool_index", None)`; `SeismicOrchestrator._run_task` sets `executor._loop.tool_index = self.tool_index` right after constructing the `ExecutorAgent` (classic mode has no index — refresh is a no-op there).

- [ ] **Step 4: Run to verify green** — `pytest tests/test_skill_execution.py tests/test_session_scoped.py tests/test_tool_registry.py tests/test_orchestrator.py tests/test_chatbot.py -q`; all pass.

- [ ] **Step 5: Commit**

```bash
git add core/skills.py tools/skill_tools.py core/tool_registry.py core/chatbot_tool_use.py core/orchestrator.py core/session_handle.py core/tool_loop.py tests/test_skill_execution.py
git commit -m "feat(skills): execute_skill (replay/guided), run_skill/save_skill/list_skills tools, prompts, discovery refresh"
```

---

### Task 5: Gradio — save-skill form and skills list

**Files:**
- Modify: `interfaces/gradio_interface.py`
- Test: Create `tests/test_gradio_skills.py`

**Interfaces (Produces):**
- `interfaces.gradio_interface.parse_parameter_lines(text: str) -> Dict[str, Any]` (lines `slot=value`; numbers parsed as int/float, else string; blank lines ignored; `ValueError` on malformed lines).
- `interfaces.gradio_interface.save_skill_from_ui(session_bot, name, description, params_text) -> str` (status markdown; uses the most recent completed turn — `current_turn_calls`/`current_turn_input`).
- `interfaces.gradio_interface.skills_markdown() -> str`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_gradio_skills.py`:

```python
import pytest

from core.skills import SkillRegistry, set_registry
from interfaces.gradio_interface import (parse_parameter_lines, save_skill_from_ui,
                                         skills_markdown)


@pytest.fixture
def registry(tmp_path):
    reg = SkillRegistry(repo_dir=str(tmp_path / "none"), runtime_dir=str(tmp_path / "rt"))
    set_registry(reg)
    yield reg
    set_registry(None)


def test_parse_parameter_lines():
    assert parse_parameter_lines("freq=30\nphit = 0.25\nname=sand\n\n") == \
        {"freq": 30, "phit": 0.25, "name": "sand"}
    with pytest.raises(ValueError):
        parse_parameter_lines("no equals sign")


def test_save_skill_from_ui_uses_last_completed_turn(registry):
    from core.chatbot_tool_use import SeismicChatBotToolUse
    from core.tool_manager import ToolManager
    bot = SeismicChatBotToolUse(llm_client=object(), tool_manager=ToolManager(),
                                knowledge_base=object())
    bot.context_manager.trace.persist_dir = ""
    bot.context_manager.set_context("current_turn_calls",
                                    [{"tool": "make_ricker", "args": {"frequency": 30}, "ok": True}])
    bot.context_manager.set_context("current_turn_input", "make a 30 Hz ricker")
    status = save_skill_from_ui(bot, "ui_ricker", "From the UI", "freq=30")
    assert "Saved skill" in status and "ui_ricker" in status
    assert registry.get("ui_ricker").chain[0]["args"] == {"frequency": "{{freq}}"}
    assert "ui_ricker" in skills_markdown()


def test_save_skill_from_ui_reports_errors(registry):
    from core.chatbot_tool_use import SeismicChatBotToolUse
    from core.tool_manager import ToolManager
    bot = SeismicChatBotToolUse(llm_client=object(), tool_manager=ToolManager(),
                                knowledge_base=object())
    assert "no tools" in save_skill_from_ui(bot, "x", "d", "freq=1").lower()
    assert save_skill_from_ui(None, "x", "d", "freq=1").startswith("⚠️")
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_gradio_skills.py -q`; expected `ImportError`.

- [ ] **Step 3: Implement** in `interfaces/gradio_interface.py`

Add import: `from core.skills import CONTEXT_PARAMS, capture_skill, get_registry`.

Add helpers after `format_status`:

```python
def parse_parameter_lines(text: str) -> dict:
    """'slot=value' per line → {slot: typed value}; numbers become int/float."""
    params = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"expected 'name=value', got {line!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    return params


def save_skill_from_ui(session_bot, name, description, params_text) -> str:
    """Save the most recent completed turn as a skill; returns status markdown."""
    if session_bot is None:
        return "⚠️ Send a message first — there is no turn to save yet."
    try:
        cm = session_bot.context_manager
        calls = cm.get_context("current_turn_calls") or []
        data = capture_skill((name or "").strip(), (description or "").strip(),
                             parse_parameter_lines(params_text), calls,
                             cm.get_context("current_turn_input") or "",
                             set(CONTEXT_PARAMS))
        registry = get_registry()
        path = registry.save(data)
        index = getattr(session_bot, "tool_index", None)
        if index is not None and hasattr(index, "refresh"):
            index.refresh(registry.specs())
        return f"✅ Saved skill **{name}** ({len(data['chain'])} step(s)) → `{path}`"
    except Exception as e:
        return f"⚠️ Could not save skill: {e}"


def skills_markdown() -> str:
    skills = get_registry().list()
    if not skills:
        return "_No skills saved yet._"
    lines = []
    for s in skills:
        params = ", ".join(f"{p}={sch.get('default', '?')}" for p, sch in s["parameters"].items())
        kind = "replay" if s["has_chain"] else "guided"
        lines.append(f"- **{s['name']}** ({kind}, {s['source']}) — {s['description']}  \n  parameters: {params or 'none'}")
    return "\n".join(lines)
```

In the layout, after the "Decision trace" accordion add:

```python
                with gr.Accordion("🧩 Skills", open=False):
                    skills_display = gr.Markdown(skills_markdown())
                    with gr.Row():
                        skill_name = gr.Textbox(label="Name", placeholder="tuning_from_petro", scale=2)
                        skill_desc = gr.Textbox(label="Description", placeholder="What it does", scale=3)
                    skill_params = gr.Textbox(label="Parameters (one per line: name=value used last turn)",
                                              placeholder="freq=30\nphit=0.25", lines=3)
                    save_btn = gr.Button("💾 Save last turn as skill")
                    save_status = gr.Markdown("")
```

and wire it after the existing submit bindings:

```python
        def on_save_skill(name, description, params_text, session_bot):
            status = save_skill_from_ui(session_bot, name, description, params_text)
            return status, skills_markdown()

        save_btn.click(on_save_skill, [skill_name, skill_desc, skill_params, session_state],
                       [save_status, skills_display])
```

- [ ] **Step 4: Run to verify green** — `pytest tests/test_gradio_skills.py tests/test_gradio_trace_panel.py tests/test_gradio_upload.py tests/test_main_modes.py -q`; all pass.

- [ ] **Step 5: Commit**

```bash
git add interfaces/gradio_interface.py tests/test_gradio_skills.py
git commit -m "feat(skills): Gradio save-last-turn-as-skill form and skills list"
```

---

### Task 6: Built-in skill + end-to-end discovery test

**Files:**
- Create: `skills/ricker_wavelet.yaml`
- Test: Create `tests/test_skills_e2e.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_skills_e2e.py`:

```python
import pytest

from core.skills import SkillRegistry, get_registry, set_registry


def test_builtin_ricker_skill_loads_from_repo():
    set_registry(None)
    reg = get_registry()
    try:
        skill = reg.get("ricker_wavelet")
        assert skill.source == "repo" and skill.chain[0]["tool"] == "make_ricker"
    finally:
        set_registry(None)


def test_builtin_skill_is_discoverable(tmp_path):
    from core.tool_index import ToolIndex
    reg = SkillRegistry(runtime_dir=str(tmp_path / "rt"))
    idx = ToolIndex(persist_directory=str(tmp_path))
    idx.refresh(reg.specs())
    names = [c.name for c in idx.search("build a ricker wavelet skill")]
    assert "skill:ricker_wavelet" in names


def test_orchestrator_can_run_skill_via_run_task(tmp_path):
    from core.orchestrator import SeismicOrchestrator
    from core.tool_manager import ToolManager

    class FakeFunc:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, name, arguments, call_id="c1"):
            self.id = call_id
            self.function = FakeFunc(name, arguments)

    class FakeLLM:
        def __init__(self, responses):
            self._responses = list(responses)

        def get_completion(self, *a, **k):
            return self._responses.pop(0)

    set_registry(SkillRegistry(runtime_dir=str(tmp_path / "rt")))
    try:
        llm = FakeLLM([
            {"content": "", "tool_calls": [FakeToolCall(
                "run_task", '{"brief": "call run_skill ricker_wavelet with frequency 40", '
                            '"tool_names": ["run_skill"]}')]},
            {"content": "", "tool_calls": [FakeToolCall(
                "run_skill", '{"name": "ricker_wavelet", "params": {"frequency": 40}}')]},
            {"content": "<reply>ran the skill</reply>", "tool_calls": None},
            {"content": "<reply>Done: 40 Hz wavelet</reply>", "tool_calls": None},
        ])
        orch = SeismicOrchestrator(llm_client=llm, tool_manager=ToolManager(),
                                   knowledge_base=object(), tool_index=object())
        orch.context_manager.trace.persist_dir = ""
        out = orch.process_single_input("run the ricker skill at 40 Hz")
        assert out["reply"] == "Done: 40 Hz wavelet"
        assert out["images"] and out["images"][0].endswith(".png")
        kinds = [e["t"] for e in out["trace"]["events"]]
        assert "skill_run" in kinds
        assert orch.context_manager.get_context("last_ricker_wavelet")["parameters"]["frequency"] == 40
    finally:
        set_registry(None)
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_skills_e2e.py -q`; expected: `ValueError: unknown skill 'ricker_wavelet'`.

- [ ] **Step 3: Implement** — create `skills/ricker_wavelet.yaml`:

```yaml
name: ricker_wavelet
description: Build and plot a Ricker wavelet of a given dominant frequency.
parameters:
  frequency:
    type: number
    description: Dominant frequency in Hz.
    default: 30
tools: [make_ricker]
procedure: |
  Create a {{frequency}} Hz Ricker wavelet with make_ricker and report its
  dominant frequency and time-length; the plot is generated automatically.
chain:
  - tool: make_ricker
    args: {frequency: "{{frequency}}"}
```

Also add `skills*` to the `include` list in `pyproject.toml`'s `[tool.setuptools.packages.find]`? No — YAML isn't a package. Instead add to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
# Built-in skills ship with the wheel (loaded from <package>/skills/ at runtime).
"*" = ["../skills/*.yaml"]
```

(If setuptools rejects the parent-relative glob in this layout, drop this hunk and note in the report that built-in skills are repo-only until packaging is revisited — do not spend more than one attempt on it.)

- [ ] **Step 4: Run to verify green** — `pytest tests/test_skills_e2e.py tests/test_skill_execution.py -q`; all pass.

- [ ] **Step 5: Commit**

```bash
git add skills/ricker_wavelet.yaml tests/test_skills_e2e.py pyproject.toml
git commit -m "feat(skills): built-in ricker_wavelet skill + discovery/orchestrator end-to-end tests"
```

---

### Task 7: Full suite + docs

**Files:**
- Modify: `CLAUDE.md`
- Test: full suite

- [ ] **Step 1: Full suite** — `pytest -q` (≥ 420s timeout). Expected: green except the pre-existing `test_tool_use.py::test_tool_use_pattern` stdin failure (644 passed at the Tier-3 head; this branch adds ~30 tests). Fix any NEW failure traced to this branch — likely candidates: tests pinning the exact registry tool count or the classic system prompt's tool list, and `test_no_dead_code`-style guards; fix additively, never by weakening.

- [ ] **Step 2: CLAUDE.md** — add a new section after "Decision trace (agent observability, Tier 0-3)":

```markdown
## Reusable skills (Tier 4)

A **skill** is a YAML file (`name`, `description`, `parameters` JSON-schema-ish with optional
`default`s, `tools` scope, `procedure` brief template with `{{slot}}`s, optional `chain` of
recorded `{tool, args}` steps) loaded by `core/skills.py::SkillRegistry` from two layers —
curated `skills/*.yaml` in the repo and captured skills in `SEISMIC_SKILLS_DIR` (default
`<tmpdir>/seismic_skills`, 0o700; runtime overrides repo on a name clash with a WARNING).
Skills are data: validated against `REGISTRY_BY_NAME`, no code, value-only slot substitution.
Three session-scoped registry tools (they receive a hidden `_session` `SessionHandle` injected
by `ToolLoopRunner.inject_context_inputs`; `ToolSpec.session_scoped=True`): `run_skill(name,
params, mode)` — `auto` replays the chain when present (each step through
`ToolLoopRunner.execute_call`, the same per-call path as a live turn: validators, guards,
sandboxes, events, sidecars with a `skill` key, auto-plots; no LLM) else runs the filled
procedure through a scoped `ExecutorAgent`; `save_skill(name, description, parameters,
overwrite)` captures the PREVIOUS turn's calls (`ContextManager.begin_turn_recording`
rotates `current_turn_calls`/`last_turn_calls` — argument VALUES kept in memory only, never
persisted) and parameterizes by explicit value matching (a parameter matching no argument is
an error; context-fed and non-scalar args are dropped); `list_skills()`. Discovery: skills
render as `skill:<name>` cards via `ToolIndex.refresh(registry.specs())` so agentic-mode
`discover_tools` returns them; the orchestrator prompt explains invoking them via `run_skill`.
Gradio has a "Skills" accordion with a save-last-turn form (uses the most recent completed
turn directly, no LLM). Recursion is blocked (`_skill_depth`); a failed replay step stops the
chain and returns `error`. Built-in example: `skills/ricker_wavelet.yaml`. Tests:
`tests/test_execute_call.py`, `test_skills_model.py`, `test_session_scoped.py`,
`test_skill_execution.py`, `test_gradio_skills.py`, `test_skills_e2e.py`. Out of scope for
now: evals over recorded traces, LLM-assisted slot inference, `run_sweep` over skills,
versioning.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: reusable skills (Tier 4) section"
```

- [ ] **Step 4: Report** — suite tally; hand off per superpowers:finishing-a-development-branch.
