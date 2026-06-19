# Workflows Phase 2a — `fluid_scenario` Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `fluid_scenario` workflow — predict an in-situ sand and overlying shale from petrophysics, then use Gassmann fluid substitution to model and compare the AVO response across scenario fluids (brine vs gas vs …), exposed as a chatbot meta-tool.

**Architecture:** A new recipe `workflows/recipes/fluid_scenario.py` that reuses Phase 0 adapters (`predict_layer`, `layer_from_gassmann`, `build_interface`) and Phase 1 leaf tools (`shuey`/`zoeppritz_reflectivity`, `avo_attributes`), organizing the per-fluid sand layers in a `Scenario` (the Phase 0 type, exercised here for the first time) and returning a JSON-friendly dict with per-case AVO results plus an overlaid composite plot. It is declared once as a `WorkflowSpec` in `workflows/engine.py`; the registry meta-tool wiring (Phase 1) picks it up automatically, and the chatbot's `WORKFLOW_NAMES`-keyed caching + dict-`image_path` surfacing already generalize to it — only the hardcoded system-prompt list needs a new bullet.

**Tech Stack:** Python 3.9+, NumPy, Matplotlib (existing headless plot convention), pytest. Run from inside `geo-mcp/seismic_chatbot`.

**Spec:** `docs/superpowers/specs/2026-06-18-agentic-workflows-design.md` (Phase 2 row — the `fluid_scenario` slice).
**Builds on:**
- Phase 0 — `workflows/types.py::Scenario`, `workflows/adapters.py::{predict_layer, build_interface, layer_from_gassmann}`.
- Phase 1 — `workflows/engine.py` (`WorkflowSpec`, `WORKFLOW_REGISTRY`, `WORKFLOW_NAMES`), the meta-tool wiring in `core/tool_registry.py`, and the chatbot's `_workflow_image_output` + `WORKFLOW_NAMES`-keyed `_update_context`. `REGISTRY` currently has 22 tools; `petro_to_avo` is the precedent recipe to mirror.

**Working dir for all commands:** `geo-mcp/seismic_chatbot` (its own git repo, branch `stabilize-tool-layer`).

## Verified contracts (from Phase 0/1 — do not re-derive)

- `predict_layer(phit, vclay, fluid="water", *, reduce="mean", label="") -> Layer(vp, vs, rho, label)`.
- `layer_from_gassmann(vp, vs, rho, phi, fluid_in, fluid_out, *, reduce="mean", label="", **kwargs) -> Layer`. Brine→gas LOWERS Vp, RAISES Vs (shear-modulus-independent), LOWERS rho.
- `build_interface(upper: Layer, lower: Layer) -> {"vp1","vs1","rho1","vp2","vs2","rho2"}`.
- `shuey_reflectivity(**iface, angles=...) -> np.ndarray`; `avo_attributes(**iface) -> {"intercept","gradient","avo_class","avo_class_description"}`. For Shuey, `R(theta=0) == intercept`.
- `Scenario(name: str, cases: dict)` — frozen dataclass (`workflows/types.py`).
- Plot convention: `output_path=None` → `tempfile.mkstemp(suffix=".png")` + `os.close(fd)`; `fig.savefig(output_path, dpi=300, bbox_inches="tight")`; `plt.close(fig)`; `return output_path`.
- Engine pattern (from `petro_to_avo`): add a `WorkflowSpec` to `WORKFLOW_REGISTRY`; `core/tool_registry.py` already converts the whole list to `ToolSpec`s. Chatbot `_update_context` caches any `tool_name in WORKFLOW_NAMES`; `_workflow_image_output` surfaces any dict with a `.png` `image_path`. Only `_create_system_prompt`'s hardcoded list needs a bullet.

---

## File Structure

- `workflows/recipes/fluid_scenario.py` — new. The `fluid_scenario` recipe + its `plot_fluid_scenario` overlaid composite plot.
- `workflows/engine.py` — modify: import `fluid_scenario`, add a second `WorkflowSpec`.
- `core/tool_registry.py` — **no change** (it converts the whole `WORKFLOW_REGISTRY`).
- `core/chatbot_tool_use.py` — modify: add a `fluid_scenario` bullet to the system prompt (caching + image surfacing already generalize).
- Tests: `tests/test_fluid_scenario.py`, `tests/test_workflow_meta_tool.py` (append a case), `tests/test_tool_registry.py` (count 22→23), `tests/test_chatbot_workflow.py` (append a prompt test).

---

### Task 1: `fluid_scenario` compute recipe (no plot yet)

**Files:**
- Create: `workflows/recipes/fluid_scenario.py`
- Create: `tests/test_fluid_scenario.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fluid_scenario.py`:

```python
import numpy as np
import pytest

from workflows.recipes.fluid_scenario import fluid_scenario


def test_fluid_scenario_keys_and_cases():
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluids=["brine", "gas"], fluid_in="brine",
    )
    assert {"shale", "fluids", "fluid_in", "cases", "angles", "method"} <= set(res)
    assert set(res["cases"]) == {"brine", "gas"}
    for f in ("brine", "gas"):
        c = res["cases"][f]
        assert {"layer", "rc", "intercept", "gradient", "avo_class"} <= set(c)
        assert len(c["rc"]) == 4


def test_fluid_scenario_gas_softens_sand():
    # Gassmann brine->gas: Vp down, Vs up (shear-independent, lower density), rho down.
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluids=["brine", "gas"], fluid_in="brine",
    )
    brine = res["cases"]["brine"]["layer"]
    gas = res["cases"]["gas"]["layer"]
    assert gas["vp"] < brine["vp"]
    assert gas["vs"] > brine["vs"]
    assert gas["rho"] < brine["rho"]


def test_fluid_scenario_shuey_intercept_consistency():
    # Per case, Shuey R(0) == that case's intercept; classes are valid labels.
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], method="shuey",  # default fluids ["brine","gas"]
    )
    for c in res["cases"].values():
        assert np.isclose(c["rc"][0], c["intercept"], rtol=1e-6, atol=1e-9)
        assert c["avo_class"] in {"I", "I*", "II", "IIp", "III", "IV"}


def test_fluid_scenario_rejects_bad_method():
    with pytest.raises(ValueError):
        fluid_scenario(0.28, 0.10, 0.10, 0.50, [0, 10], method="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fluid_scenario.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.recipes.fluid_scenario'`

- [ ] **Step 3: Write the compute recipe**

Create `workflows/recipes/fluid_scenario.py`:

```python
"""fluid_scenario: AVO fluid-substitution scenarios (brine vs gas vs ...).

Predict the in-situ sand and overlying shale from petrophysics, then use Gassmann
fluid substitution to model the AVO response for each scenario fluid. This mirrors
the interpretation workflow: log-derived in-situ properties, substituted to
alternate fluids to test the AVO / DHI response. The per-fluid sand layers are
organized in a Scenario. The composite plot is added in Task 2.
"""
import numpy as np

from workflows.types import Scenario
from workflows.adapters import predict_layer, build_interface, layer_from_gassmann
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity, avo_attributes


def fluid_scenario(phit_sand, vclay_sand, phit_shale, vclay_shale, angles,
                   fluids=None, fluid_in="brine", method="shuey"):
    """Model AVO for a sand under shale across several pore fluids (Gassmann).

    Returns a JSON-friendly dict with the shale layer and, per fluid case, the
    substituted sand layer, the reflectivity-vs-angle curve, and AVO attributes.
    """
    if method not in ("shuey", "zoeppritz"):
        raise ValueError(f"method must be 'shuey' or 'zoeppritz' (got {method!r})")
    if fluids is None:
        fluids = ["brine", "gas"]

    rc_fn = shuey_reflectivity if method == "shuey" else zoeppritz_reflectivity
    shale = predict_layer(phit_shale, vclay_shale, fluid="water", label="shale")
    in_situ = predict_layer(phit_sand, vclay_sand, fluid=fluid_in, label="sand")

    # Build the per-fluid sand layers into a Scenario (in-situ used as-is; others
    # via Gassmann substitution from the in-situ state).
    sand_layers = {}
    for f in fluids:
        if f == fluid_in:
            sand_layers[f] = in_situ
        else:
            sand_layers[f] = layer_from_gassmann(
                in_situ.vp, in_situ.vs, in_situ.rho, phi=phit_sand,
                fluid_in=fluid_in, fluid_out=f, label=f"sand-{f}",
            )
    scenario = Scenario(name="fluid", cases=sand_layers)

    cases = {}
    for f, sand_f in scenario.cases.items():
        iface = build_interface(shale, sand_f)
        rc = np.asarray(rc_fn(**iface, angles=angles), dtype=float)
        attrs = avo_attributes(**iface)
        cases[f] = {
            "layer": {"vp": sand_f.vp, "vs": sand_f.vs, "rho": sand_f.rho, "label": sand_f.label},
            "rc": [float(x) for x in rc],
            "intercept": float(attrs["intercept"]),
            "gradient": float(attrs["gradient"]),
            "avo_class": attrs["avo_class"],
            "avo_class_description": attrs["avo_class_description"],
        }

    return {
        "shale": {"vp": shale.vp, "vs": shale.vs, "rho": shale.rho, "label": shale.label},
        "fluids": list(fluids),
        "fluid_in": fluid_in,
        "cases": cases,
        "angles": [float(a) for a in angles],
        "method": method,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fluid_scenario.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/fluid_scenario.py tests/test_fluid_scenario.py
git commit -m "feat(workflows): fluid_scenario compute recipe (Gassmann AVO scenarios)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `plot_fluid_scenario` overlaid composite + wire `image_path`

**Files:**
- Modify: `workflows/recipes/fluid_scenario.py` (add plot fn + imports; call it)
- Modify: `tests/test_fluid_scenario.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fluid_scenario.py`:

```python
import os


def test_fluid_scenario_returns_image_path():
    res = fluid_scenario(
        phit_sand=0.28, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluids=["brine", "gas"],
    )
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fluid_scenario.py::test_fluid_scenario_returns_image_path -v`
Expected: FAIL with `KeyError: 'image_path'`

- [ ] **Step 3: Add the overlaid plot and call it**

In `workflows/recipes/fluid_scenario.py`, change the top import block to add `os`, `tempfile`, and matplotlib:

```python
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from workflows.types import Scenario
from workflows.adapters import predict_layer, build_interface, layer_from_gassmann
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity, avo_attributes
```

Add this function at the end of the file (it overlays the per-fluid R(theta) curves and A-B points):

```python
def plot_fluid_scenario(shale, angles, cases, method, output_path=None):
    """Overlaid composite: R(theta) per fluid (left) and A-B points per fluid (right)."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    angles = np.asarray(angles, dtype=float)

    fig, (ax_rc, ax_ab) = plt.subplots(1, 2, figsize=(12, 5))

    for f, res in cases.items():
        ax_rc.plot(angles, np.asarray(res["rc"], dtype=float), "o-", label=f)
        ax_ab.plot([res["intercept"]], [res["gradient"]], "s", markersize=10,
                   label=f"{f} ({res['avo_class']})")

    ax_rc.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax_rc.set_xlabel("Incidence angle (deg)")
    ax_rc.set_ylabel("Reflection coefficient")
    ax_rc.set_title(f"AVO by fluid ({method})")
    ax_rc.legend()
    ax_rc.grid(True, alpha=0.3)

    vals = [abs(res["intercept"]) for res in cases.values()]
    vals += [abs(res["gradient"]) for res in cases.values()]
    lim = max(0.1, max(vals) * 1.5)
    ax_ab.axhline(0.0, color="grey", lw=0.8)
    ax_ab.axvline(0.0, color="grey", lw=0.8)
    ax_ab.set_xlim(-lim, lim)
    ax_ab.set_ylim(-lim, lim)
    ax_ab.set_xlabel("Intercept A")
    ax_ab.set_ylabel("Gradient B")
    ax_ab.set_title("Intercept-Gradient by fluid")
    ax_ab.legend()

    fig.suptitle(f"Fluid scenarios: sand below {shale.label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

Then, inside `fluid_scenario`, AFTER the `for f, sand_f in scenario.cases.items():` loop completes (i.e. after `cases` is fully built) and BEFORE the `return {`, insert:

```python
    image_path = plot_fluid_scenario(shale, angles, cases, method)
```

And add this key to the returned dict (last entry before the closing `}`):

```python
        "image_path": image_path,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fluid_scenario.py -v`
Expected: PASS (5 passed — the 4 from Task 1 still pass; they check a key subset)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/fluid_scenario.py tests/test_fluid_scenario.py
git commit -m "feat(workflows): fluid_scenario overlaid composite plot + image_path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Register `fluid_scenario` in the workflow engine

**Files:**
- Modify: `workflows/engine.py` (import + second `WorkflowSpec`)
- Modify: `tests/test_tool_registry.py` (count 22 → 23)
- Modify: `tests/test_workflow_meta_tool.py` (append a case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_meta_tool.py`:

```python
def test_fluid_scenario_is_registered_meta_tool():
    assert "fluid_scenario" in reg.REGISTRY_BY_NAME
    assert "fluid_scenario" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"fluid_scenario"}


def test_fluid_scenario_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("fluid_scenario", {
        "phit_sand": 0.28, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "angles": [0, 10, 20, 30],
    })
    assert set(res["cases"]) == {"brine", "gas"}  # default fluids
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
```

(`reg` and `ToolManager` are already imported at the top of this test file from Phase 1.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_meta_tool.py -k fluid_scenario -v`
Expected: FAIL — `fluid_scenario` not in `reg.REGISTRY_BY_NAME`.

- [ ] **Step 3: Add the `WorkflowSpec`**

In `workflows/engine.py`, add the recipe import after the existing `from workflows.recipes.petro_to_avo import petro_to_avo` line:

```python
from workflows.recipes.fluid_scenario import fluid_scenario
```

Then add this second `WorkflowSpec` to the `WORKFLOW_REGISTRY` list (after the `petro_to_avo` spec, before the closing `]`):

```python
    WorkflowSpec(
        name="fluid_scenario",
        fn=fluid_scenario,
        description=(
            "AVO fluid-substitution scenarios: predict an in-situ sand and overlying "
            "shale from porosity and clay volume, then use Gassmann fluid substitution "
            "to model and compare the AVO response (reflectivity curve, intercept, "
            "gradient, AVO class) for each pore fluid (e.g. brine vs gas). Returns the "
            "per-fluid results and an overlaid comparison plot. Useful for DHI / "
            "fluid-feasibility assessment."
        ),
        params={
            "phit_sand": {"type": "number", "description": "Sand porosity (fraction, 0-1)."},
            "vclay_sand": {"type": "number", "description": "Sand clay volume (fraction, 0-1)."},
            "phit_shale": {"type": "number", "description": "Shale porosity (fraction, 0-1)."},
            "vclay_shale": {"type": "number", "description": "Shale clay volume (fraction, 0-1)."},
            "angles": {"type": "array", "items": {"type": "number"}, "description": "Incidence angles in degrees for the AVO curves."},
            "fluids": {"type": "array", "items": {"type": "string"}, "description": "Pore fluids to compare, e.g. ['brine','gas'] (default). Each is 'brine'/'water', 'oil', or 'gas'."},
            "fluid_in": {"type": "string", "description": "In-situ pore fluid the sand is predicted at before substitution (default 'brine')."},
            "method": {"type": "string", "description": "Reflectivity method: 'shuey' (default) or 'zoeppritz'."},
        },
        required=["phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "angles"],
        defaults={"fluids": None, "fluid_in": "brine", "method": "shuey"},
        auto_plot=None,
    ),
```

NOTE: `defaults={"fluids": None, ...}` is correct — the recipe treats `fluids=None` as `["brine","gas"]`. The meta-tool path fills this default before calling, so `fluids=None` reaches the recipe and is normalized there.

- [ ] **Step 4: Bump the registry count assertion**

In `tests/test_tool_registry.py`, change `assert len(reg.REGISTRY) == 22` to:

```python
    assert len(reg.REGISTRY) == 23
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_workflow_meta_tool.py tests/test_tool_registry.py tests/test_workflow_engine.py -v`
Expected: PASS. Also confirm `python -c "from core.tool_registry import REGISTRY; print(len(REGISTRY))"` prints 23.

- [ ] **Step 6: Commit**

```bash
git add workflows/engine.py tests/test_workflow_meta_tool.py tests/test_tool_registry.py
git commit -m "feat(workflows): register fluid_scenario as a workflow meta-tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: List `fluid_scenario` in the system prompt; full-suite check

**Files:**
- Modify: `core/chatbot_tool_use.py` (system-prompt bullet only)
- Modify: `tests/test_chatbot_workflow.py` (append a prompt test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatbot_workflow.py` (reuses the `bot` fixture):

```python
def test_system_prompt_lists_fluid_scenario(bot):
    prompt = bot._create_system_prompt()
    assert "fluid_scenario" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_workflow.py -k fluid_scenario -v`
Expected: FAIL — `"fluid_scenario"` not in the system prompt.

- [ ] **Step 3: Add the system-prompt bullet**

In `core/chatbot_tool_use.py`, in `_create_system_prompt`'s hardcoded "Available tools:" bullet list, add this bullet directly after the `petro_to_avo` bullet (match the neighboring bullet formatting exactly):

```
- fluid_scenario: AVO fluid-substitution scenarios — predicts sand & shale from porosity/clay, Gassmann-substitutes the sand across fluids (e.g. brine vs gas), and returns per-fluid AVO class/intercept/gradient with an overlaid comparison plot (DHI feasibility).
```

(No other chatbot changes are needed: `_update_context` already caches any tool in `WORKFLOW_NAMES`, and `_workflow_image_output` already surfaces the dict `image_path` — both generalize to `fluid_scenario` automatically.)

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: PASS (5 passed — the 4 from Phase 1 plus the new one)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: the new Phase 2a tests pass and nothing else regressed. The standalone `test_tool_use.py::test_tool_use_pattern` (stdin) is a KNOWN pre-existing failure — if ONLY that fails, it is expected. If any OTHER test fails, STOP and report BLOCKED with the failure.

- [ ] **Step 6: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(workflows): list fluid_scenario in chatbot system prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `WorkflowEngine().run("fluid_scenario", {...})` and `ToolManager().execute_tool("fluid_scenario", {...})` return a dict with `shale`, per-fluid `cases` (each with substituted `layer`, `rc`, `intercept`/`gradient`/`avo_class`), and a composite `image_path`.
- Gassmann direction holds (brine→gas: sand Vp↓, Vs↑, rho↓) and per-case Shuey `R(0)==intercept`.
- `fluid_scenario` is in `REGISTRY_BY_NAME`/`TOOL_SCHEMAS`/`TOOL_FUNCTIONS` (registry count 23); the chatbot caches `last_workflow_result`, surfaces the plot, and lists `fluid_scenario` in the prompt — all via the Phase 1 generalizations plus one prompt bullet.
- Full suite green (modulo the pre-existing stdin-script failure).

## Spec coverage (Phase 2 — `fluid_scenario` slice)

- `fluid_scenario` recipe (Gassmann brine-vs-gas via a `Scenario`) → Tasks 1–2. This also closes the Phase 0/1 "`Scenario` is introduced but not yet exercised" gap.
- Meta-tool registration + chatbot exposure → Tasks 3–4 (mostly free via Phase 1's generalized wiring).

## Not in this plan (later Phase 2 sub-plans / beyond)

- **2b `eei_optimal_chi`** — sweep χ and correlate `extended_elastic_impedance` against a target log to find the best fluid/lithology projection; needs a NEW correlation tool (gap S2) and a target-log input. Its own plan.
- **2c `tuning`** — wrap `wedge_model`/`analyze_wedge` (and the gather) as a recipe. Its own plan.
- Continuous saturation Sw (Phase 3, gap S1) and the generic sweep runner (Phase 4, gap S3).
- Multi-image workflow output, LLM narration of the result dict, and recipe-level input guards remain deferred (carry-over from the Phase 1 review).
