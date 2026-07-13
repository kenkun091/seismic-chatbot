# Workflows Phase 2c — `tuning` Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `tuning` workflow — a sand wedge encased in shale, predicted from petrophysics, built into a wedge model and analyzed for tuning thickness, resolution limit, and the amplitude-vs-thickness curve — exposed as a chatbot meta-tool.

**Architecture:** A new recipe `workflows/recipes/tuning.py` that reuses the Phase 0 adapters (`predict_layer`, `build_earth_model` — exercised here for the first time) and the existing wedge tools (`create_wedge_model`, `analyze_wedge`), returning a JSON-friendly dict with the tuning analysis plus a focused `plot_tuning` amplitude-vs-thickness figure. Declared once as a `WorkflowSpec` in `workflows/engine.py`; the Phase 1 meta-tool wiring + `WORKFLOW_NAMES`-keyed caching + dict-`image_path` surfacing pick it up automatically — only a system-prompt bullet is added.

**Tech Stack:** Python 3.9+, NumPy, Matplotlib (existing headless plot convention), pytest. Run from inside `geo-mcp/seismic_chatbot`.

**Spec:** `docs/superpowers/specs/2026-06-18-agentic-workflows-design.md` (Phase 2 row — the `tuning` slice).
**Builds on:**
- Phase 0 — `workflows/adapters.py::{predict_layer, build_earth_model}`.
- Phase 1 — the engine + meta-tool wiring + chatbot generalizations (same as `petro_to_avo`/`fluid_scenario` rode). `REGISTRY` currently has 23 tools; `fluid_scenario` is the most recent precedent recipe to mirror.

**Working dir for all commands:** `geo-mcp/seismic_chatbot` (its own git repo, branch `stabilize-tool-layer`).

## Verified contracts (read from source — do not re-derive)

- `build_earth_model([l1, l2, l3]) -> {"v1","v2","v3","rho1","rho2","rho3","vs1","vs2","vs3"}` mapping layer1→1, layer2→2, layer3→3 (`workflows/adapters.py`). Requires exactly 3 `Layer`s; guards elastic validity.
- `create_wedge_model(max_thickness, v1, v2, v3, rho1, rho2, rho3, wavelet_freq=30.0, num_traces=61, vs1=None, vs2=None, vs3=None, incident_angle=0, export_path=None) -> (time_array, model, synthetic, parameters)` (`tools/wedge_tools.py:813`). The `parameters` dict it returns contains `"v2"`, `"wavelet_freq"`, `"max_thickness"`. **Proven** to accept `**build_earth_model([...])` by the passing Phase 0 test `test_build_earth_model_feeds_create_wedge_model`.
- `analyze_wedge(synthetic_data, parameters) -> {"tuning_thickness", "tuning_amplitude", "resolution_limit", "max_amplitudes" (list), "thicknesses" (list)}` (`tools/wedge_tools.py:1009`). Internals: `tuning_thickness = parameters["v2"] / (4*wavelet_freq)`; `resolution_limit = tuning_thickness / 2`; `max_amplitudes = max|synthetic| over time per trace`; `thicknesses = linspace(0, max_thickness, num_traces)`. So with `build_earth_model([shale, sand, shale])`, `v2` is the **sand Vp** → `tuning_thickness = sand.vp / (4*freq)` exactly (a deterministic check).
- `predict_layer(phit, vclay, fluid="water", *, label="") -> Layer(vp, vs, rho, label)` (Phase 0).
- Do NOT reuse `plot_wedge_analysis` (`tools/wedge_tools.py:1033`) — it calls `plt.show()` and returns `None`. Use a file-based `plot_tuning` following the `tempfile.mkstemp(suffix=".png")` + `savefig(dpi=300, bbox_inches="tight")` + `plt.close(fig)` + `return output_path` convention (as in `plot_petro_to_avo`/`plot_fluid_scenario`).
- Engine/meta-tool/chatbot pattern is identical to `fluid_scenario` (Phase 2a): add a `WorkflowSpec`; `core/tool_registry.py` converts the whole `WORKFLOW_REGISTRY`; chatbot caching (`_update_context` keyed on `WORKFLOW_NAMES`) and image surfacing (`_workflow_image_output`) generalize; only `_create_system_prompt`'s hardcoded list needs a bullet.

---

## File Structure

- `workflows/recipes/tuning.py` — new. The `tuning` recipe + its `plot_tuning` figure.
- `workflows/engine.py` — modify: import `tuning`, add a third `WorkflowSpec`.
- `core/tool_registry.py` — **no change** (converts the whole `WORKFLOW_REGISTRY`).
- `core/chatbot_tool_use.py` — modify: one `tuning` system-prompt bullet.
- Tests: `tests/test_tuning.py`, `tests/test_workflow_meta_tool.py` (append a case), `tests/test_tool_registry.py` (count 23→24), `tests/test_chatbot_workflow.py` (append a prompt test).

---

### Task 1: `tuning` compute recipe (no plot yet)

**Files:**
- Create: `workflows/recipes/tuning.py`
- Create: `tests/test_tuning.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tuning.py`:

```python
import numpy as np
import pytest

from workflows.adapters import predict_layer
from workflows.recipes.tuning import tuning


def test_tuning_keys_and_shapes():
    res = tuning(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        max_thickness=40.0, wavelet_freq=30.0, num_traces=41,
    )
    assert {"sand", "shale", "tuning_thickness", "tuning_amplitude",
            "resolution_limit", "thicknesses", "max_amplitudes",
            "wavelet_freq", "max_thickness"} <= set(res)
    assert len(res["thicknesses"]) == 41
    assert len(res["max_amplitudes"]) == 41
    assert all(np.isfinite(res["max_amplitudes"]))
    assert res["tuning_thickness"] > 0
    assert res["resolution_limit"] > 0


def test_tuning_thickness_known_answer():
    # analyze_wedge defines tuning_thickness = v2/(4f), resolution_limit = v2/(8f),
    # where v2 is the sand Vp (layer 2 from build_earth_model). This pins the recipe
    # to BOTH the rock-physics prediction AND the correct shale/sand/shale mapping.
    res = tuning(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        max_thickness=40.0, wavelet_freq=30.0,
    )
    sand = predict_layer(0.28, 0.10, fluid="brine", label="sand")
    expected = sand.vp / (4.0 * 30.0)
    assert np.isclose(res["tuning_thickness"], expected, rtol=1e-6)
    assert np.isclose(res["resolution_limit"], expected / 2.0, rtol=1e-6)
    assert res["sand"]["vp"] == pytest.approx(sand.vp)


def test_tuning_higher_freq_resolves_thinner():
    # tuning_thickness = v2/(4f): higher frequency -> thinner tuning / better resolution.
    lo = tuning(0.28, 0.10, 0.10, 0.50, max_thickness=40.0, wavelet_freq=20.0)
    hi = tuning(0.28, 0.10, 0.10, 0.50, max_thickness=40.0, wavelet_freq=50.0)
    assert hi["tuning_thickness"] < lo["tuning_thickness"]
    assert hi["resolution_limit"] < lo["resolution_limit"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tuning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.recipes.tuning'`

- [ ] **Step 3: Write the compute recipe**

Create `workflows/recipes/tuning.py`:

```python
"""tuning: wedge tuning-thickness / vertical-resolution analysis from petrophysics.

Predict a sand and an encasing shale from porosity and clay volume, build a sand
wedge between two shale layers, convolve to a synthetic gather, and analyze the
amplitude-vs-thickness curve for tuning thickness and resolution limit. The
composite plot is added in Task 2.
"""
import numpy as np

from workflows.adapters import predict_layer, build_earth_model
from tools.wedge_tools import create_wedge_model, analyze_wedge


def tuning(phit_sand, vclay_sand, phit_shale, vclay_shale, max_thickness,
           wavelet_freq=30.0, num_traces=61, fluid_sand="brine"):
    """Tuning-wedge analysis for a sand encased in shale, predicted from petrophysics.

    Returns a JSON-friendly dict with the sand/shale layers, the tuning thickness,
    tuning amplitude, resolution limit, and the amplitude-vs-thickness curve.
    """
    sand = predict_layer(phit_sand, vclay_sand, fluid=fluid_sand, label="sand")
    shale = predict_layer(phit_shale, vclay_shale, fluid="water", label="shale")
    earth = build_earth_model([shale, sand, shale])

    time_array, model, synthetic, parameters = create_wedge_model(
        max_thickness=max_thickness, wavelet_freq=wavelet_freq,
        num_traces=num_traces, **earth,
    )
    analysis = analyze_wedge(synthetic, parameters)

    return {
        "sand": {"vp": sand.vp, "vs": sand.vs, "rho": sand.rho, "label": sand.label},
        "shale": {"vp": shale.vp, "vs": shale.vs, "rho": shale.rho, "label": shale.label},
        "tuning_thickness": float(analysis["tuning_thickness"]),
        "tuning_amplitude": float(analysis["tuning_amplitude"]),
        "resolution_limit": float(analysis["resolution_limit"]),
        "thicknesses": [float(t) for t in analysis["thicknesses"]],
        "max_amplitudes": [float(a) for a in analysis["max_amplitudes"]],
        "wavelet_freq": float(wavelet_freq),
        "max_thickness": float(max_thickness),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tuning.py -v`
Expected: PASS (3 passed)

(If `create_wedge_model` does not honor `num_traces` — i.e. `len(res["thicknesses"]) != 41` — STOP and report BLOCKED with the actual length; do not change the test.)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/tuning.py tests/test_tuning.py
git commit -m "feat(workflows): tuning compute recipe (wedge tuning from petrophysics)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `plot_tuning` amplitude-vs-thickness figure + wire `image_path`

**Files:**
- Modify: `workflows/recipes/tuning.py` (add plot fn + imports; call it)
- Modify: `tests/test_tuning.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tuning.py`:

```python
import os


def test_tuning_returns_image_path():
    res = tuning(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        max_thickness=40.0, wavelet_freq=30.0, num_traces=41,
    )
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tuning.py::test_tuning_returns_image_path -v`
Expected: FAIL with `KeyError: 'image_path'`

- [ ] **Step 3: Add the plot and call it**

In `workflows/recipes/tuning.py`, change the top import block to add `os`, `tempfile`, and matplotlib:

```python
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from workflows.adapters import predict_layer, build_earth_model
from tools.wedge_tools import create_wedge_model, analyze_wedge
```

Add this function at the END of the file (it plots the amplitude-vs-thickness curve with the tuning thickness and resolution limit marked — the standard tuning figure):

```python
def plot_tuning(analysis, wavelet_freq, output_path=None):
    """Amplitude-vs-thickness curve with tuning-thickness and resolution-limit markers."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    thicknesses = np.asarray(analysis["thicknesses"], dtype=float)
    max_amplitudes = np.asarray(analysis["max_amplitudes"], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thicknesses, max_amplitudes, "b-", label="Max amplitude")
    ax.axvline(analysis["tuning_thickness"], color="r", ls="--",
               label=f"Tuning thickness: {analysis['tuning_thickness']:.2f} m")
    ax.axvline(analysis["resolution_limit"], color="g", ls="--",
               label=f"Resolution limit: {analysis['resolution_limit']:.2f} m")
    ax.set_xlabel("Thickness (m)")
    ax.set_ylabel("Maximum amplitude")
    ax.set_title(f"Wedge tuning curve ({wavelet_freq:g} Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

Then, inside `tuning`, AFTER the `analysis = analyze_wedge(synthetic, parameters)` line and BEFORE the `return {`, insert:

```python
    image_path = plot_tuning(analysis, wavelet_freq)
```

And add this key to the returned dict (last entry before the closing `}`):

```python
        "image_path": image_path,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tuning.py -v`
Expected: PASS (4 passed — the 3 from Task 1 still pass; they check a key subset)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/tuning.py tests/test_tuning.py
git commit -m "feat(workflows): tuning amplitude-vs-thickness plot + image_path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Register `tuning` in the workflow engine

**Files:**
- Modify: `workflows/engine.py` (import + third `WorkflowSpec`)
- Modify: `tests/test_tool_registry.py` (count 23 → 24)
- Modify: `tests/test_workflow_meta_tool.py` (append a case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_meta_tool.py` (`reg` and `ToolManager` are already imported at the top):

```python
def test_tuning_is_registered_meta_tool():
    assert "tuning" in reg.REGISTRY_BY_NAME
    assert "tuning" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"tuning"}


def test_tuning_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("tuning", {
        "phit_sand": 0.28, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "max_thickness": 40.0,
    })
    assert res["tuning_thickness"] > 0
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_meta_tool.py -k tuning -v`
Expected: FAIL — `tuning` not in `reg.REGISTRY_BY_NAME`.

- [ ] **Step 3: Add the `WorkflowSpec`**

In `workflows/engine.py`, add the recipe import after the existing `from workflows.recipes.fluid_scenario import fluid_scenario` line:

```python
from workflows.recipes.tuning import tuning
```

Then add this `WorkflowSpec` to the `WORKFLOW_REGISTRY` list (after the `fluid_scenario` spec, before the closing `]`):

```python
    WorkflowSpec(
        name="tuning",
        fn=tuning,
        description=(
            "Wedge tuning / vertical-resolution analysis: predict a sand and encasing "
            "shale from porosity and clay volume, build a sand wedge between two shale "
            "layers, and analyze the amplitude-vs-thickness response for the tuning "
            "thickness and resolution limit at a given wavelet frequency. Returns the "
            "tuning thickness, resolution limit, the amplitude-vs-thickness curve, and "
            "a tuning-curve plot."
        ),
        params={
            "phit_sand": {"type": "number", "description": "Sand porosity (fraction, 0-1)."},
            "vclay_sand": {"type": "number", "description": "Sand clay volume (fraction, 0-1)."},
            "phit_shale": {"type": "number", "description": "Shale porosity (fraction, 0-1)."},
            "vclay_shale": {"type": "number", "description": "Shale clay volume (fraction, 0-1)."},
            "max_thickness": {"type": "number", "description": "Maximum wedge thickness in meters."},
            "wavelet_freq": {"type": "number", "description": "Ricker wavelet dominant frequency in Hz (default 30)."},
            "num_traces": {"type": "integer", "description": "Number of thickness traces across the wedge (default 61)."},
            "fluid_sand": {"type": "string", "description": "Sand pore fluid: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
        },
        required=["phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "max_thickness"],
        defaults={"wavelet_freq": 30.0, "num_traces": 61, "fluid_sand": "brine"},
        auto_plot=None,
    ),
```

- [ ] **Step 4: Bump the registry count assertion**

In `tests/test_tool_registry.py`, change `assert len(reg.REGISTRY) == 23` to:

```python
    assert len(reg.REGISTRY) == 24
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_workflow_meta_tool.py tests/test_tool_registry.py tests/test_workflow_engine.py -v`
Expected: PASS. Also confirm `python -c "from core.tool_registry import REGISTRY; print(len(REGISTRY))"` prints 24.

- [ ] **Step 6: Commit**

```bash
git add workflows/engine.py tests/test_workflow_meta_tool.py tests/test_tool_registry.py
git commit -m "feat(workflows): register tuning as a workflow meta-tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: List `tuning` in the system prompt; full-suite check

**Files:**
- Modify: `core/chatbot_tool_use.py` (system-prompt bullet only)
- Modify: `tests/test_chatbot_workflow.py` (append a prompt test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatbot_workflow.py` (reuses the `bot` fixture):

```python
def test_system_prompt_lists_tuning(bot):
    # Match the bullet prefix specifically: the word "tuning" may already appear
    # inside other wedge-tool descriptions, but "- tuning:" is the new bullet.
    prompt = bot._create_system_prompt()
    assert "- tuning:" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_workflow.py -k "lists_tuning" -v`
Expected: FAIL — `"- tuning:"` not yet in the system prompt.

- [ ] **Step 3: Add the system-prompt bullet**

In `core/chatbot_tool_use.py`, in `_create_system_prompt`'s hardcoded "Available tools:" bullet list, add this bullet directly after the `fluid_scenario` bullet (match the neighboring bullet formatting exactly):

```
- tuning: Wedge tuning / vertical-resolution analysis — predicts a sand & encasing shale from porosity/clay, builds a sand wedge, and returns the tuning thickness, resolution limit, and amplitude-vs-thickness curve with a plot.
```

(No other chatbot changes are needed: `_update_context` already caches any tool in `WORKFLOW_NAMES`, and `_workflow_image_output` already surfaces the dict `image_path`.)

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: PASS (the prior chatbot-workflow tests plus the new one)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: the new Phase 2c tests pass and nothing else regressed. The standalone `test_tool_use.py::test_tool_use_pattern` (stdin) is a KNOWN pre-existing failure — if ONLY that fails, it is expected. If any OTHER test fails, STOP and report BLOCKED with the failure.

- [ ] **Step 6: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(workflows): list tuning in chatbot system prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `WorkflowEngine().run("tuning", {...})` and `ToolManager().execute_tool("tuning", {...})` return a dict with the sand/shale layers, `tuning_thickness`, `resolution_limit`, the amplitude-vs-thickness curve (`thicknesses`/`max_amplitudes`), and a `tuning`-curve `image_path`.
- `tuning_thickness == sand.vp/(4*wavelet_freq)` and `resolution_limit == that/2` (deterministic), confirming `build_earth_model` mapped sand→v2; higher frequency yields a thinner tuning thickness.
- `tuning` is in `REGISTRY_BY_NAME`/`TOOL_SCHEMAS`/`TOOL_FUNCTIONS` (registry count 24); the chatbot caches `last_workflow_result`, surfaces the plot, and lists `tuning` in the prompt — via the Phase 1 generalizations plus one prompt bullet.
- Full suite green (modulo the pre-existing stdin-script failure).

## Spec coverage (Phase 2 — `tuning` slice)

- `tuning` recipe (wraps `wedge_model`/`analyze_wedge`) → Tasks 1–2. This also closes the Phase 0 "`build_earth_model` is built but not yet exercised by a recipe" gap.
- Meta-tool registration + chatbot exposure → Tasks 3–4 (mostly free via Phase 1's generalized wiring).

## Not in this plan (later sub-plans / beyond)

- **2b `eei_optimal_chi`** — sweep χ and correlate `extended_elastic_impedance` against a target log; needs a NEW correlation tool (gap S2) and a target-log input. Its own plan.
- Continuous saturation Sw (Phase 3, gap S1) and the generic sweep runner (Phase 4, gap S3) — note Phase 4's sweep is the natural home for a tuning-vs-frequency sweep (workflow #6).
- The angle-gather tuning variant (`wedge_avo_gather`/`analyze_wedge_gather`) is intentionally out of scope here — this recipe is the zero-offset tuning analysis (workflow #3).
- Carry-over deferred items (zoeppritz-path coverage for `fluid_scenario`, recipe-level input guards, multi-image output, LLM narration of result dicts).
