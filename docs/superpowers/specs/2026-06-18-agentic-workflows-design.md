# Agentic Workflows for End-to-End Geophysical Analysis — Design

**Date:** 2026-06-18
**Branch:** `stabilize-tool-layer`
**Status:** Approved design; per-phase implementation plans to follow.

## Problem

The chatbot has 20 registry-driven leaf tools (wavelets, wedge, AVO, rock physics,
EEI, RAG). The only "chaining" that exists is a single hop: one LLM turn → one
compute tool → one auto-plot (`AUTO_PLOT`). Multi-step geophysical analysis —
*petrophysical inputs (φ, Vclay, Sw) → predict elastic properties → model AVO /
EEI / tuning responses* — is currently reachable only by the LLM re-typing
numbers between turns. That is lossy, cannot hold parallel scenarios (brine vs
gas), and makes parameter sweeps impossible.

This design adds an orchestration layer that chains existing tools into named,
deterministic, auditable workflows, built in phased slices.

## What the tool layer can already form (the map)

Canonical end-to-end workflows the existing tools can compose:

1. **AVO feasibility / fluid modeling** — `calculate_rock_properties` → interface →
   `shuey`/`zoeppritz` → `avo_attributes`.
2. **Fluid-substitution scenario** — add `gassmann_substitution` (brine vs gas).
3. **Tuning & resolution** — wavelet → `wedge_model` → `analyze_wedge`.
4. **AVO + tuning** — `wedge_avo_gather` → `analyze_wedge_gather`.
5. **EEI optimal-angle** — `extended_elastic_impedance` swept χ → correlate vs a target log.
6. **Wavelet/frequency sensitivity** — wavelet sweep → wedge → analyze per frequency.
7. **Stochastic / Monte-Carlo** — parameter ranges → workflow 1 → response distribution.

### Gap ledger (verified against actual tool return contracts)

**Mechanical glue gaps** (plumbing that blocks chaining today):

| ID | Gap | Exact mismatch |
|----|-----|----------------|
| G1 | Interface assembly | Rock physics describes **1 layer**; AVO needs **2** (`vp1/vp2…`), wedge needs **3** (`v1/v2/v3`). No assembler. |
| G2 | Unpack + rename | `calculate_rock_properties` returns a **positional tuple** `(vp, vs, rhob, vp_vs, ai, si)` — no keys; `gassmann_substitution` returns a **dict** `{vp, vs, rho, vp_vs, k_dry, k_sat, mu}`. Targets want `vp1/v1/rho1`. No adapter. |
| G3 | Array→scalar reduce | Rock physics is shape-preserving (a *log* of samples → arrays); AVO/wedge want **scalars per layer**. No "pick representative rock type / sample" reducer. |

**Science gaps** (missing physics, not plumbing):

| ID | Gap | Why it matters |
|----|-----|----------------|
| S1 | Saturation | `calculate_rock_properties` takes a fluid *string*; `gassmann_substitution` does end-member swaps. No continuous Sw, no Reuss/Voigt-Hill/Brie fluid mixing, no patchy-vs-uniform. |
| S2 | EEI optimal-χ | EEI(χ) is computed per layer, but nothing correlates EEI(χ) against a target log to find the best χ — which *is* the workflow. |
| S3 | Sweep/scenario runner | No engine to run a base chain across φ/Vclay/Sw ranges → distributions. |

**Infrastructure gap:**

- **I1 — No data spine.** Multi-step chains rely on the LLM re-typing numbers
  between turns: lossy, no scenario branches, no sweeps.

**Headline finding:** the blocker is not a smarter agent loop — it is the missing
*primitives* (G1–G3) and the missing *data spine* (I1). Orchestration is cheap
once data flows in code instead of through the chat transcript.

## Architecture

Three layers; the 20 leaf tools stay pure, JSON-friendly, and individually callable.

```
LLM / chatbot  ──picks & parameterizes──►  Workflow (a "fat tool")
                                                 │
                              WorkflowEngine runs a recipe (DAG, in code)
                                                 │
   ┌─────────────────────────────────────────────┼───────────────┐
   │ typed spine (Layer/Scenario)   adapters (G1–G3)   leaf tools │
   │  flows between steps IN CODE    glue the seams    UNCHANGED  │
   └─────────────────────────────────────────────────────────────┘
```

The engine calls leaf **functions** directly, so:

- data passes in code (closes I1; enables sweeps),
- the chatbot's turn-by-turn `AUTO_PLOT` chaining never fires mid-recipe — each
  workflow owns its own plotting and surfaces its own composite figure(s).

### Components

New top-level package `workflows/` (matches the flat layout: `config/`, `core/`,
`tools/`, `knowledge/`, `parsing/`, `interfaces/`, `workflows/`).

```
workflows/
  types.py        # Layer, Scenario
  adapters.py     # predict_layer (G3), build_interface (G1), unpack/rename (G2)
  engine.py       # WorkflowSpec, WORKFLOW_REGISTRY, run(), to_meta_tool_specs()
  recipes/        # petro_to_avo.py, fluid_scenario.py, eei_optimal_chi.py, tuning.py
  sweep.py        # generic grid runner (Phase 4)
```

**`workflows/types.py` — the data spine.** Two frozen dataclasses, the vocabulary
every recipe speaks:

- `Layer` = `(vp, vs, rho, label)` — one rock. Adapters produce these (closes G1/G2 typing).
- `Scenario` = a named bundle of `Layer`s / interfaces, e.g. `{"brine": …, "gas": …}`
  — makes "brine vs gas" first-class instead of two manual chat turns.

These live *inside* the engine. Leaf tools never see them; adapters translate
`Layer ⇄ {vp1, vs1, …}` at the boundary.

**`workflows/adapters.py` — the glue (closes G1–G3).**

- `predict_layer(phit, vclay, fluid, *, reduce="mean"|"median"|index)` — wraps
  `calculate_rock_properties`, unpacks the positional tuple, reduces an array log
  to a representative scalar `Layer` (G2 + G3).
- `build_interface(upper: Layer, lower: Layer)` → the `{vp1, vs1, rho1, vp2, vs2,
  rho2}` dict AVO tools expect (G1 for AVO).
- `build_earth_model(layers: list[Layer])` → the `{v1, v2, v3, rho1, …, vs1, …}`
  dict wedge tools expect (G1 for wedge).
- `layer_from_gassmann(result: dict, label)` — adapts the `gassmann_substitution`
  dict into a `Layer` (G2 for fluid substitution).

**`workflows/engine.py` — the orchestration core.** A workflow is declared exactly
like a tool, mirroring `ToolSpec`/`REGISTRY`:

```python
@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    fn: Callable          # the recipe: takes params, returns structured result + image_path(s)
    description: str
    params: dict[str, dict]
    required: list[str]
    defaults: dict = field(default_factory=dict)
    auto_plot: Optional[str] = None   # usually None: recipes plot themselves
```

`WORKFLOW_REGISTRY = [...]`, `WorkflowEngine.run(name, params)` validates/fills
defaults/executes, and `to_meta_tool_specs()` emits `ToolSpec`s.

**The idiomatic integration move:** `core/tool_registry.py` appends the workflow
meta-tool specs into `REGISTRY`. `TOOL_SCHEMAS`, `TOOL_FUNCTIONS`, `AUTO_PLOT`,
`REGISTRY_BY_NAME`, chatbot dispatch — **all derive automatically**. No changes to
`tool_manager.py` or chatbot routing logic. Workflows ride the existing
"registry is the single source of truth" machinery.

**Recipe representation:** a recipe is a plain Python function (no DSL). It calls
leaf functions and adapters, returns `{...structured result..., "image_path": ...}`
(or a list of image paths). A generic sweep runner (Phase 4) wraps *any* recipe fn
over a parameter grid — so the Python-function choice does not block S3.

### Data flow (flagship recipe `petro_to_avo`)

```
params {phit_sand, vclay_sand, fluid_sand, phit_shale, vclay_shale, angles, method}
  → predict_layer(sand)  → Layer(vp,vs,rho,"sand")
  → predict_layer(shale) → Layer(vp,vs,rho,"shale")
  → build_interface(shale, sand) → {vp1,vs1,rho1,vp2,vs2,rho2}
  → shuey_reflectivity / zoeppritz_reflectivity (per `method`) → R(θ)
  → avo_attributes → {intercept, gradient, avo_class, ...}
  → composite plot (rock-physics summary + R(θ) curve + A-B crossplot)
  → return {layers, rc, attributes, image_path}
```

## Phasing (each phase ships independently and gets its own plan)

| Phase | Delivers | Closes | Workflow unlocked |
|------|----------|--------|-------------------|
| **0 — Spine + adapters** | `Layer`/`Scenario`; `predict_layer`, `build_interface`, `build_earth_model`, `layer_from_gassmann`. Expose 1–2 adapters as leaf tools for instant chat payoff. | G1, G2, G3, I1 | — (enables all) |
| **1 — Engine + flagship** | `WorkflowEngine` + `WorkflowSpec`/`WORKFLOW_REGISTRY` + meta-tool wiring into `tool_registry.py` + `petro_to_avo` recipe + composite plot. | engine infra | #1 AVO feasibility |
| **2 — Scenario/EEI/tuning recipes** | `fluid_scenario` (Gassmann brine vs gas via `Scenario`); `eei_optimal_chi` (adds S2 correlation tool + target log); `tuning` (wraps wedge/analyze). | S2 | #2, #4, #5 |
| **3 — Saturation science** | Continuous Sw via cited fluid mixing (Reuss/Brie/Voigt-Hill) feeding `predict_layer`. | S1 | full (φ, Vclay, **Sw**) spine |
| **4 — Sweep engine** | Generic `run_sweep(recipe, grid)` → distributions + plots over any recipe. | S3 | #7 Monte-Carlo |

**Ordering note:** Phase 1's flagship initially uses the discrete `fluid_type`
string; continuous **Sw** becomes real only in Phase 3. If saturation is the
priority, S1 can be pulled earlier (before or alongside Phase 2).

## Chatbot integration (Phase 1)

Minimal, because workflows are registry meta-tools:

- Workflow specs appended to `REGISTRY` → schemas/functions/dispatch auto-derived.
- `ContextManager` gains `last_workflow_result` so follow-ups can reuse it.
- Image surfacing: if a workflow returns multiple image paths, the chatbot's
  image detection handles a list (currently single `{"image_path": ...}`); extend
  to accept a list. This is the only touch to chatbot image-handling code.
- System prompt tool listing: workflow names included (the listing is currently
  hard-coded; add the workflow entries alongside the leaf tools).

## Error handling

- The engine validates params against the `WorkflowSpec` (reuse the leaf-tool
  validation pattern: fill defaults, then validate) before running the recipe.
- Adapters enforce physical validity via existing `tools/physics_guards.py`
  (vp>0, 0<vs<vp, ρ>0, φ/Vclay∈[0,1]) — REJECT non-physical, WARN out-of-range.
- A recipe step that raises surfaces a clear error naming the failed step; partial
  results (e.g. computed layers) are included in the error payload where possible.
- Sweep (Phase 4): a failed grid cell is recorded as `null`/NaN and reported in a
  coverage summary rather than aborting the whole sweep (no silent truncation).

## Testing & correctness posture

- **Adapters & engine:** unit tests — tuple-unpack correctness, key renaming,
  array→scalar reduction, interface assembly, registry derivation
  (meta-tools appear in `TOOL_SCHEMAS`/`TOOL_FUNCTIONS`), default-fill + validation.
- **Recipes:** end-to-end test per recipe with a known-answer fixture (e.g. a
  brine sand vs shale interface → expected AVO class).
- **Science tools (S1, S2):** same cited-model rigor and regression tests this
  package already demands of Gassmann/Nur. S1 fluid mixing verified against the
  Rock Physics Handbook (Reuss bound / Brie); S2 correlation verified against a
  synthetic log with a known optimal χ.
- **No-dead-code guard:** consistent with `tests/test_no_dead_code.py` philosophy.

## Decisions (made during brainstorming; revisit if needed)

1. **Orchestration foundation:** Workflow engine + named recipes with an in-code
   typed context (option B), built in phased slices ("B but phased"). Adapter-only
   (A) and LLM-planner-loop (C) were rejected: A cannot do sweeps and keeps I1's
   precision loss; C is non-deterministic and unauditable — a poor fit for
   reproducible geophysics.
2. **Exposure:** workflows are meta-tools in the one `REGISTRY` (not a separate
   invocation path), so all derivation machinery is reused.
3. **Recipe representation:** Python functions, not a declarative DAG DSL — no
   premature abstraction; a generic sweep runner still wraps any recipe.
4. **Location:** new top-level `workflows/` package, matching the flat-layout
   absolute-import convention.

## Out of scope (YAGNI for now)

- Declarative DAG DSL / visual workflow builder.
- Persisting workflow runs to disk beyond existing CSV export.
- True offset/angle-gather migration, anisotropy, attenuation (Q), multiples, NMO
  (pre-existing wedge limitations, unchanged here).
- Real well-log file ingestion (recipes take parameters / arrays, not LAS files).

## Next step

Per-phase implementation plans, starting with **Phase 0 (spine + adapters)**, via
the writing-plans skill.
