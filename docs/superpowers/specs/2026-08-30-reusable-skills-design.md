# Reusable Skills (Tier 4) — Design

Status: approved in brainstorming (2026-08-30), pending user review of this document.
Builds on Tiers 0-3 (PRs #3 → #4 → #5). Branch: `reusable-skills`, stacked on `turn-transparency`.

## Goal

Let a user turn a successful conversation flow into a named, parameterized, reusable
**skill** without writing Python — and run it again either deterministically (replaying
the recorded tool chain with new parameter values) or adaptively (an LLM executor following
the skill's procedure with a scoped toolset). Deterministic replay of recorded flows is
delivered by construction, which is the substrate the roadmap's Tier 4 asked for.

Decisions taken in brainstorming:
- A skill is **one format with two execution modes** (guided + optional replayable chain).
- Skills come from **both** capture ("save the last turn as a skill") and hand-authored YAML.
- Storage is **two-layer**: curated `skills/*.yaml` in the repo; captured skills in
  `SEISMIC_SKILLS_DIR` (default `<tmpdir>/seismic_skills`, 0o700), runtime overriding repo
  on name clash with a WARNING.
- Approach A: saved `run_task` + replayable chain, executed by the existing machinery
  (registry, `ToolLoopRunner`, `ExecutorAgent`, `ToolIndex`). No code generation, no
  prompt-only skills.

## 1. Skill format

One YAML file per skill:

```yaml
name: tuning_from_petro                  # [a-z][a-z0-9_]*, unique across layers
description: Tuning analysis for a sand encased in shale from porosity/clay.
parameters:                              # same shape as ToolSpec.params (JSON-schema-ish)
  phit:  {type: number, description: Sand porosity (0-1)}
  vclay: {type: number, description: Sand clay volume (0-1)}
  freq:  {type: number, description: Wavelet frequency (Hz), default: 30}
tools: [predict_layer, wedge_model, analyze_wedge]   # allowed toolset (guided mode scope)
procedure: |                                          # guided-mode brief template
  Predict sand and shale elastic properties for porosity {{phit}} and clay
  {{vclay}}, build a wedge with a {{freq}} Hz Ricker wavelet, and report the
  tuning thickness.
chain:                                                # optional: replay-mode steps
  - tool: predict_layer
    args: {phit: "{{phit}}", vclay: "{{vclay}}", fluid: brine}
  - tool: wedge_model
    args: {frequency: "{{freq}}", thickness: 100, v1: 2500, v2: 2800, v3: 2500}
  - tool: analyze_wedge
    args: {}                                          # context-fed at replay, like live
```

Validation (`core/skills.py::validate_skill`, applied on load and on save; violations
are `ValueError`s with the file/skill name):
- required keys `name`, `description`, `parameters`, `tools`, `procedure`; `chain` optional.
- every tool in `tools` and every `chain[*].tool` exists in `REGISTRY_BY_NAME`; chain tools
  must be a subset of `tools`.
- every `{{slot}}` in `procedure` and in chain arg values is a declared parameter; every
  declared parameter without a `default` is required at invocation.
- parameter `default`s are literals; slot substitution is value-level string templating —
  an arg value that is exactly `"{{slot}}"` is replaced by the parameter's typed value,
  and slots embedded in longer strings are substituted textually. No expressions.

## 2. Registry and discovery (`core/skills.py`)

- `SkillRegistry.load(repo_dir="skills", runtime_dir=SEISMIC_SKILLS_DIR)` reads both layers
  (repo first, runtime second; a runtime skill with a repo name overrides it and logs a
  WARNING). Exposes `get(name)`, `names()`, `specs()` (skill cards, see below), `reload()`.
- Discovery reuses `core/tool_index.py::ToolIndex` unchanged in spirit: `SkillRegistry`
  produces duck-typed card specs (`name`, `description`, `params`, `required`,
  `auto_plot=None`) whose rendered card is prefixed `skill:` — so `discover_tools` returns
  skills next to tools. `ToolIndex` gains `refresh(extra_specs)` that re-runs its
  idempotent, self-cleaning `_populate` with `REGISTRY + extra_specs`; `save_skill` calls
  it. The orchestrator's system prompt tells the LLM that `skill:` cards are invoked via
  `run_skill(name, params)`.
- Classic mode needs nothing special: `run_skill`, `save_skill`, `list_skills` are ordinary
  registry tools (schemas derived from `ToolSpec` as usual).

## 3. Session-scoped tools (small registry extension)

`run_skill` and `save_skill` need the session (LLM client, tool manager, context manager).
Registry tool functions are plain callables, so:
- `ToolSpec` gains `session_scoped: bool = False`. For such tools the loop injects a hidden
  kwarg `_session=SessionHandle(llm_client, tool_manager, context_manager, runner)` in
  `inject_context_inputs` (alongside the existing `_CONTEXT_INPUTS` mechanism); `_session`
  is never in the LLM-facing schema and is stripped before validation (`execute_tool`
  passes it through untouched).
- `SessionHandle` is a tiny dataclass in `core/session_handle.py`.

## 4. Execution: `run_skill(name, params, mode="auto")`

`core/skills.py::execute_skill(skill, params, mode, session)`:
1. Validate `params` against the skill's parameter schema — fill declared defaults, reject
   unknown keys, require the undefaulted ones (`ValueError` → the calling loop records a
   failed tool call as it does for any tool).
2. **Mode**: `replay` if `mode == "replay"`, or `auto` with a chain present; `guided`
   otherwise. `mode="replay"` on a chain-less skill is a `ValueError`.
3. **Replay**: for each step, substitute slots into `args`, then run
   `session.runner.execute_call(step.tool, args)` — a method **extracted from
   `ToolLoopRunner.run`'s try-block body** (context injection, warning capture, `tool_call`
   event, `update_context`, harvest + provenance sidecar, auto-plot chaining). Because it is
   the same code path, context-fed steps, physics warnings, sidecars, and auto-plots behave
   exactly as in a live turn, with no LLM call. First failing step stops the chain.
   Returns `{"mode": "replay", "steps": [{"tool", "ok", "error"?}], "result": <compacted
   last result>, "extra_image_paths": [...]}` — the `extra_image_paths` key is what the
   outer loop's `harvest_images` already collects, so skill plots surface in the chat.
4. **Guided**: fill the procedure template, then `ExecutorAgent(session.llm_client,
   session.tool_manager, session.context_manager).run(brief, skill.tools)`; returns
   `{"mode": "guided", "summary", "tools_used", "error"?, "extra_image_paths"}`.
5. Emits one `skill_run` event `{name, mode, n_steps}`; provenance sidecars written during a
   skill carry `"skill": name` (the runner exposes a `current_skill` attribute the replay
   sets/clears; `_write_provenance` includes it when set).

`list_skills()` returns `[{name, description, parameters, has_chain, source}]`.

## 5. Capture: `save_skill(name, description, parameters, overwrite=False)`

- **Recording**: `ToolLoopRunner.execute_call` appends `{"tool", "args": resolved_args,
  "ok": True}` to `context_manager` key `last_turn_calls` (in-memory session state only —
  never persisted, never in the JSONL/OTel trace, which stay names-not-values).
  `begin_turn` clears it (`process_single_input` calls `set_context("last_turn_calls", [])`).
- `save_skill` (session-scoped): errors if `last_turn_calls` is empty ("the last turn ran no
  tools"). `parameters` is a dict `{slot_name: {value, description?, type?}}` naming the
  values to parameterize.
- **Parameterization (explicit value matching, no LLM)**: for every recorded arg whose
  value equals a parameter's value (numbers compared as floats with exact equality; strings
  exact), the arg becomes `"{{slot}}"`. A parameter whose value matches no arg is a
  `ValueError` ("parameter freq=30 was not used by any tool call"). Context-fed args
  (`_CONTEXT_INPUTS` names) and non-scalar values (arrays, dicts, lists longer than 12) are
  dropped from the chain; those steps re-read session context at replay.
- **Procedure**: the turn's `turn_start.input` text with the same value→slot textual
  substitution; if the input is unavailable, a generated line "Run the recorded chain:
  tool1 → tool2 …".
- `tools` = the distinct recorded tools in order. The file is validated (Section 1), written
  to `SEISMIC_SKILLS_DIR/<name>.yaml` (refuse to overwrite unless `overwrite=True`;
  refuse names that collide with registry tools), the registry reloaded, and the index
  refreshed. Returns `{"path", "name", "n_steps", "parameters"}`.
- **Gradio**: a "💾 Save last turn as skill" button opens a small form (name, description,
  parameters as `slot=value` lines) and sends the equivalent tool request through
  `process_single_input` so both modes handle it identically; a "Skills" accordion lists
  `list_skills()`.

## 6. Safety

- Skills are data. No `eval`, no imports, no code generation; slot substitution is value
  templating only. Tool names must exist in the registry.
- Replay goes through the same validators, physics guards, path/image sandboxes and the
  same tool loop code path as a live turn — a skill cannot do what a chat turn cannot.
- Captured files live in `SEISMIC_SKILLS_DIR` (0o700), never in the source tree; promotion
  to `skills/` is a manual copy + review.
- `last_turn_calls` (argument values) never leaves process memory.
- Guided mode is bounded by the executor's round budget; a skill invoking `run_skill`
  recursively is rejected (`ValueError`) via a `_session.depth` guard (max 1).

## 7. Testing

- Unit: registry load/validate/layering/override warning; parameterizer (value→slot,
  multi-use values, unused parameter error, context/non-scalar arg dropping); template
  filling; `validate_skill` failure cases; `list_skills`.
- Loop: `last_turn_calls` recording + reset per turn; `execute_call` extraction is
  behavior-preserving (existing `test_tool_loop_trace`, provenance and physics tests keep
  passing unchanged).
- End-to-end replay: run a real `make_ricker` turn with `FakeLLMClient`, `save_skill` with
  `frequency` parameterized, `run_skill` in replay with a different frequency → wavelet
  differs, auto-plot fired, sidecar carries `skill`, `skill_run` event present.
- Guided: `run_skill(mode="guided")` through `ExecutorAgent` with scripted completions.
- Discovery: a saved skill appears in `ToolIndex.search` results with the `skill:` prefix.
- Gradio: the save form path builds the right tool request; skills accordion renders.
- One built-in example skill ships in `skills/tuning_from_petro.yaml` and is exercised by
  the replay test.

## 8. Out of scope (next tier if wanted)

Automated evals / regression suites over recorded traces; LLM-assisted slot inference;
`run_sweep` over skills (adapter); skill versioning/migration; sharing beyond file copy;
capture from agentic-mode multi-task turns beyond the last executor's calls (the recording
covers every `execute_call` in the turn, so multi-task turns are captured as one chain —
acceptable for v1).
