# Workflows Phase 1 — Engine + Flagship `petro_to_avo` Recipe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the workflow engine (declarative `WorkflowSpec` registry + a `WorkflowEngine.run` entry point) and the flagship `petro_to_avo` recipe, exposed to the chatbot as a single meta-tool that chains Phase 0 adapters end-to-end (porosity/clay → elastic layers → interface → AVO reflectivity + attributes + composite plot).

**Architecture:** Recipes are plain Python functions that call the Phase 0 adapters (`predict_layer`, `build_interface`) and existing leaf functions (`shuey_reflectivity`/`zoeppritz_reflectivity`, `avo_attributes`), returning a JSON-friendly dict that includes a composite `image_path`. Each recipe is declared once as a frozen `WorkflowSpec` in `workflows/engine.py`; `core/tool_registry.py` converts those specs into `ToolSpec`s and appends them to `REGISTRY`, so schemas/function-map/dispatch all derive automatically and the chatbot runs a workflow exactly like any other tool. The chatbot is taught to surface a dict result's `image_path` and to cache the workflow result.

**Tech Stack:** Python 3.9+, NumPy, Matplotlib (Agg/headless via the existing plot convention), pytest. Run from inside `geo-mcp/seismic_chatbot`.

**Spec:** `docs/superpowers/specs/2026-06-18-agentic-workflows-design.md` (Phase 1 row + "Chatbot integration" section).
**Builds on:** Phase 0 (`docs/superpowers/plans/2026-06-19-workflows-phase-0-spine-adapters.md`) — `workflows/types.py` (`Layer`), `workflows/adapters.py` (`predict_layer`, `build_interface`, `build_earth_model`, `layer_from_gassmann`, `predict_elastic_layer`) are all in place; `REGISTRY` currently has 21 tools.

**Working dir for all commands:** `geo-mcp/seismic_chatbot` (its own git repo, branch `stabilize-tool-layer`).

## Verified integration contracts (from a grounding scan — do not re-derive)

- Plot tools take `output_path=None`, and when None do `fd, output_path = tempfile.mkstemp(suffix=".png"); os.close(fd)`, then `fig.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(fig); return output_path` (a **raw string path**). Pattern: `tools/avo_tools.py` `plot_avo_crossplot`.
- `core/tool_registry.py`: `ToolSpec(name, fn, description, params, required, defaults={}, validator=None, auto_plot=None)`. After the `REGISTRY = [...]` literal, the module derives `REGISTRY_BY_NAME`, `TOOL_SCHEMAS = [to_openai_schema(s) ...]`, `TOOL_FUNCTIONS`, `AUTO_PLOT`. Appending a `ToolSpec` to `REGISTRY` is all that's needed.
- `core/tool_manager.py` `execute_tool(self, tool_name, params)`: fills `spec.defaults`, then `params`, validates required + optional `spec.validator`, then `spec.fn(**full_params)`. Imports from `core.tool_registry`.
- `core/chatbot_tool_use.py` (line numbers approximate):
  - ~656: `if self._is_image_output(tool_name, tool_result): return {"image_path": tool_result}` (`_is_image_output` ~694 checks `str` ending `.png` AND `tool_name in [hardcoded plot list]`).
  - ~709 `_handle_automatic_chaining` uses `AUTO_PLOT` (workflow `auto_plot=None` → no chaining; the recipe plots itself).
  - ~784 `_update_context` stores `last_*` via `self.context_manager.set_context(key, dict)` in per-tool `elif` branches (last block is `calculate_rock_properties`).
  - ~53–93 `_create_system_prompt` returns a string with a **hardcoded** "Available tools:" bullet list.
- `core/context_manager.py`: generic `set_context(key, value)` / `get_context(key, default=None)` over a `conversation_context` dict — no new methods needed for `last_workflow_result`.

---

## File Structure

- `workflows/recipes/__init__.py` — new (marks the recipes subpackage).
- `workflows/recipes/petro_to_avo.py` — new. The `petro_to_avo` recipe function **and** its `plot_petro_to_avo` composite plot. (One recipe per module; compute + its plot live together.)
- `workflows/engine.py` — new. `WorkflowSpec` dataclass, `WORKFLOW_REGISTRY`, `WORKFLOW_REGISTRY_BY_NAME`, `WORKFLOW_NAMES`, `WorkflowEngine`.
- `core/tool_registry.py` — modify: import `WORKFLOW_REGISTRY`, convert to `ToolSpec`s, append to `REGISTRY`.
- `core/chatbot_tool_use.py` — modify: surface dict `image_path`; cache `last_workflow_result`; add the workflow to the system-prompt tool list.
- Tests: `tests/test_petro_to_avo.py`, `tests/test_workflow_engine.py`, `tests/test_workflow_meta_tool.py`, `tests/test_chatbot_workflow.py`; bump count in `tests/test_tool_registry.py`.

No existing leaf tool, `workflows/types.py`, or `workflows/adapters.py` is modified.

---

### Task 1: `petro_to_avo` compute recipe (no plot yet)

**Files:**
- Create: `workflows/recipes/__init__.py`
- Create: `workflows/recipes/petro_to_avo.py`
- Create: `tests/test_petro_to_avo.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_petro_to_avo.py`:

```python
import numpy as np

from workflows.adapters import predict_layer
from workflows.recipes.petro_to_avo import petro_to_avo


def test_petro_to_avo_keys_and_layers():
    res = petro_to_avo(
        phit_sand=0.25, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluid_sand="brine", method="shuey",
    )
    # The compute recipe returns at least these keys (image_path added in Task 2).
    assert {"upper", "lower", "angles", "rc", "intercept", "gradient",
            "avo_class", "method"} <= set(res)
    # Layers match predict_layer exactly (upper = shale, lower = sand).
    up = predict_layer(0.10, 0.50, fluid="water", label="shale")
    lo = predict_layer(0.25, 0.10, fluid="brine", label="sand")
    assert np.isclose(res["upper"]["vp"], up.vp)
    assert np.isclose(res["lower"]["vp"], lo.vp)
    assert res["upper"]["label"] == "shale"
    assert res["lower"]["label"] == "sand"


def test_petro_to_avo_shuey_intercept_consistency():
    # For Shuey, R(theta=0) is exactly the intercept A (same Aki-Richards R0
    # that avo_attributes reports). This pins the reflectivity curve and the
    # attributes to the same physics.
    res = petro_to_avo(
        phit_sand=0.25, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluid_sand="brine", method="shuey",
    )
    assert len(res["rc"]) == 4
    assert np.isclose(res["rc"][0], res["intercept"], rtol=1e-6, atol=1e-9)
    assert res["avo_class"] in {"I", "I*", "II", "IIp", "III", "IV"}


def test_petro_to_avo_rejects_bad_method():
    import pytest
    with pytest.raises(ValueError):
        petro_to_avo(0.25, 0.10, 0.10, 0.50, [0, 10], method="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_petro_to_avo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.recipes'`

- [ ] **Step 3: Create the recipes subpackage and the compute recipe**

Create `workflows/recipes/__init__.py`:

```python
"""Workflow recipes: each module chains adapters + leaf tools end-to-end."""
```

Create `workflows/recipes/petro_to_avo.py`:

```python
"""petro_to_avo: AVO feasibility from petrophysics.

Predict elastic properties of a sand and an overlying shale from porosity and
clay volume (Han 1986), assemble the shale-over-sand interface, and model the
AVO reflectivity curve plus interpretation attributes (intercept, gradient,
AVO class). The composite plot is added in Task 2.
"""
import numpy as np

from workflows.adapters import predict_layer, build_interface
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity, avo_attributes


def petro_to_avo(phit_sand, vclay_sand, phit_shale, vclay_shale, angles,
                 fluid_sand="brine", fluid_shale="water", method="shuey"):
    """Run the petrophysics -> elastic -> interface -> AVO chain.

    Returns a JSON-friendly dict with the two layers, the reflectivity-vs-angle
    curve, and the AVO attributes.
    """
    if method not in ("shuey", "zoeppritz"):
        raise ValueError(f"method must be 'shuey' or 'zoeppritz' (got {method!r})")
    upper = predict_layer(phit_shale, vclay_shale, fluid=fluid_shale, label="shale")
    lower = predict_layer(phit_sand, vclay_sand, fluid=fluid_sand, label="sand")
    iface = build_interface(upper, lower)
    rc_fn = shuey_reflectivity if method == "shuey" else zoeppritz_reflectivity
    rc = np.asarray(rc_fn(**iface, angles=angles), dtype=float)
    attrs = avo_attributes(**iface)
    return {
        "upper": {"vp": upper.vp, "vs": upper.vs, "rho": upper.rho, "label": upper.label},
        "lower": {"vp": lower.vp, "vs": lower.vs, "rho": lower.rho, "label": lower.label},
        "angles": [float(a) for a in angles],
        "rc": [float(x) for x in rc],
        "intercept": float(attrs["intercept"]),
        "gradient": float(attrs["gradient"]),
        "avo_class": attrs["avo_class"],
        "avo_class_description": attrs["avo_class_description"],
        "method": method,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_petro_to_avo.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/__init__.py workflows/recipes/petro_to_avo.py tests/test_petro_to_avo.py
git commit -m "feat(workflows): petro_to_avo compute recipe (petrophysics -> AVO)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `plot_petro_to_avo` composite plot + wire `image_path` into the result

**Files:**
- Modify: `workflows/recipes/petro_to_avo.py` (add plot fn + imports; call it in `petro_to_avo`)
- Modify: `tests/test_petro_to_avo.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_petro_to_avo.py`:

```python
import os


def test_petro_to_avo_returns_image_path():
    res = petro_to_avo(
        phit_sand=0.25, vclay_sand=0.10,
        phit_shale=0.10, vclay_shale=0.50,
        angles=[0, 10, 20, 30], fluid_sand="gas", method="shuey",
    )
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_petro_to_avo.py::test_petro_to_avo_returns_image_path -v`
Expected: FAIL with `KeyError: 'image_path'`

- [ ] **Step 3: Add the composite plot and call it**

In `workflows/recipes/petro_to_avo.py`, change the import block at the top to add `os`, `tempfile`, and matplotlib (place these with the existing imports):

```python
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from workflows.adapters import predict_layer, build_interface
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity, avo_attributes
```

Add this function below `petro_to_avo` (it builds a 3-panel composite figure following the package's `tempfile.mkstemp` + `savefig(dpi=300, bbox_inches="tight")` plot convention):

```python
def plot_petro_to_avo(upper, lower, angles, rc, attrs, output_path=None):
    """Composite plot: model/attribute summary, R(theta) curve, A-B point."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    angles = np.asarray(angles, dtype=float)
    rc = np.asarray(rc, dtype=float)

    fig, (ax_tbl, ax_rc, ax_ab) = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: layer + attribute summary (monospace text)
    ax_tbl.axis("off")
    lines = [
        f"{'':10s}{'Vp':>8s}{'Vs':>8s}{'rho':>7s}",
        f"{upper.label:10s}{upper.vp:8.0f}{upper.vs:8.0f}{upper.rho:7.2f}",
        f"{lower.label:10s}{lower.vp:8.0f}{lower.vs:8.0f}{lower.rho:7.2f}",
        "",
        f"Intercept A = {attrs['intercept']:.4f}",
        f"Gradient  B = {attrs['gradient']:.4f}",
        f"AVO class   = {attrs['avo_class']}",
    ]
    ax_tbl.text(0.0, 0.95, "\n".join(lines), family="monospace", va="top", fontsize=11)
    ax_tbl.set_title("Model & AVO attributes")

    # Panel 2: R(theta)
    ax_rc.plot(angles, rc, "o-", color="C0")
    ax_rc.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax_rc.set_xlabel("Incidence angle (deg)")
    ax_rc.set_ylabel("Reflection coefficient")
    ax_rc.set_title("AVO reflectivity")
    ax_rc.grid(True, alpha=0.3)

    # Panel 3: intercept-gradient point
    A = float(attrs["intercept"])
    B = float(attrs["gradient"])
    lim = max(0.1, abs(A) * 1.5, abs(B) * 1.5)
    ax_ab.axhline(0.0, color="grey", lw=0.8)
    ax_ab.axvline(0.0, color="grey", lw=0.8)
    ax_ab.plot([A], [B], "s", color="C3", markersize=10)
    ax_ab.annotate(attrs["avo_class"], (A, B),
                   textcoords="offset points", xytext=(8, 8))
    ax_ab.set_xlim(-lim, lim)
    ax_ab.set_ylim(-lim, lim)
    ax_ab.set_xlabel("Intercept A")
    ax_ab.set_ylabel("Gradient B")
    ax_ab.set_title("Intercept-Gradient")

    fig.suptitle(f"Petro -> AVO: {lower.label} below {upper.label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

Then, inside `petro_to_avo`, AFTER the `attrs = avo_attributes(**iface)` line and BEFORE the `return {`, insert:

```python
    image_path = plot_petro_to_avo(upper, lower, angles, rc, attrs)
```

And add this key to the returned dict (e.g. as the last entry before the closing `}`):

```python
        "image_path": image_path,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_petro_to_avo.py -v`
Expected: PASS (4 passed — the 3 from Task 1 still pass because they only check a key subset)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/petro_to_avo.py tests/test_petro_to_avo.py
git commit -m "feat(workflows): petro_to_avo composite plot + image_path in result

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Workflow engine — `WorkflowSpec`, `WORKFLOW_REGISTRY`, `WorkflowEngine.run`

**Files:**
- Create: `workflows/engine.py`
- Create: `tests/test_workflow_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_engine.py`:

```python
import pytest

from workflows.engine import (
    WorkflowSpec, WORKFLOW_REGISTRY, WORKFLOW_REGISTRY_BY_NAME,
    WORKFLOW_NAMES, WorkflowEngine,
)


def test_registry_contains_petro_to_avo():
    assert "petro_to_avo" in WORKFLOW_REGISTRY_BY_NAME
    assert "petro_to_avo" in WORKFLOW_NAMES
    spec = WORKFLOW_REGISTRY_BY_NAME["petro_to_avo"]
    assert isinstance(spec, WorkflowSpec)
    assert callable(spec.fn)
    assert set(spec.required) == {"phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "angles"}
    assert spec.defaults == {"fluid_sand": "brine", "fluid_shale": "water", "method": "shuey"}


def test_run_fills_defaults_and_executes():
    eng = WorkflowEngine()
    res = eng.run("petro_to_avo", {
        "phit_sand": 0.25, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "angles": [0, 10, 20, 30],
    })
    assert res["method"] == "shuey"            # default filled
    assert res["lower"]["label"] == "sand"
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")


def test_run_unknown_workflow_raises():
    with pytest.raises(ValueError):
        WorkflowEngine().run("does_not_exist", {})


def test_run_missing_required_raises():
    with pytest.raises(ValueError):
        WorkflowEngine().run("petro_to_avo", {"phit_sand": 0.25})  # missing others
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.engine'`

- [ ] **Step 3: Write the engine**

Create `workflows/engine.py`:

```python
"""Workflow engine: declarative recipe registry + a run() entry point.

Each recipe is declared once as a frozen `WorkflowSpec`. `core/tool_registry.py`
converts these into `ToolSpec`s and appends them to the tool REGISTRY, so the
chatbot runs a workflow exactly like any other tool. `WorkflowEngine.run` is the
programmatic / future-sweep entry point that fills defaults, checks required
params, and calls the recipe.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from workflows.recipes.petro_to_avo import petro_to_avo


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    fn: Callable
    description: str
    params: dict
    required: list
    defaults: dict = field(default_factory=dict)
    auto_plot: Optional[str] = None


WORKFLOW_REGISTRY = [
    WorkflowSpec(
        name="petro_to_avo",
        fn=petro_to_avo,
        description=(
            "End-to-end AVO feasibility: predict elastic properties (Vp, Vs, density) "
            "of a sand and an overlying shale from porosity and clay volume (Han 1986), "
            "build the shale-over-sand interface, and model the AVO reflectivity curve "
            "and interpretation attributes (intercept A, gradient B, AVO class). Returns "
            "the two layers, the reflectivity-vs-angle curve, the AVO attributes, and a "
            "composite plot."
        ),
        params={
            "phit_sand": {"type": "number", "description": "Sand porosity (fraction, 0-1)."},
            "vclay_sand": {"type": "number", "description": "Sand clay volume (fraction, 0-1)."},
            "phit_shale": {"type": "number", "description": "Shale porosity (fraction, 0-1)."},
            "vclay_shale": {"type": "number", "description": "Shale clay volume (fraction, 0-1)."},
            "angles": {"type": "array", "items": {"type": "number"}, "description": "Incidence angles in degrees for the AVO curve."},
            "fluid_sand": {"type": "string", "description": "Sand pore fluid: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
            "fluid_shale": {"type": "string", "description": "Shale pore fluid (default 'water')."},
            "method": {"type": "string", "description": "Reflectivity method: 'shuey' (default) or 'zoeppritz'."},
        },
        required=["phit_sand", "vclay_sand", "phit_shale", "vclay_shale", "angles"],
        defaults={"fluid_sand": "brine", "fluid_shale": "water", "method": "shuey"},
        auto_plot=None,
    ),
]

WORKFLOW_REGISTRY_BY_NAME = {w.name: w for w in WORKFLOW_REGISTRY}
WORKFLOW_NAMES = frozenset(WORKFLOW_REGISTRY_BY_NAME)


class WorkflowEngine:
    """Runs a registered workflow recipe by name (programmatic / sweep entry)."""

    def run(self, name, params):
        spec = WORKFLOW_REGISTRY_BY_NAME.get(name)
        if spec is None:
            raise ValueError(f"Unknown workflow: {name}")
        full = dict(spec.defaults)
        full.update(params)
        missing = [p for p in spec.required if p not in full]
        if missing:
            raise ValueError(f"{name}: missing required parameters: {missing}")
        return spec.fn(**full)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_engine.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/engine.py tests/test_workflow_engine.py
git commit -m "feat(workflows): WorkflowSpec registry + WorkflowEngine.run

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire workflows into the tool registry as meta-tools

**Files:**
- Modify: `core/tool_registry.py` (import `WORKFLOW_REGISTRY`, convert + append before the derivations)
- Modify: `tests/test_tool_registry.py:6` (count 21 → 22)
- Create: `tests/test_workflow_meta_tool.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_meta_tool.py`:

```python
from core import tool_registry as reg
from core.tool_manager import ToolManager


def test_petro_to_avo_is_registered_meta_tool():
    assert "petro_to_avo" in reg.REGISTRY_BY_NAME
    assert "petro_to_avo" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"petro_to_avo"}
    spec = reg.REGISTRY_BY_NAME["petro_to_avo"]
    assert spec.auto_plot is None  # the recipe plots itself


def test_petro_to_avo_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("petro_to_avo", {
        "phit_sand": 0.25, "vclay_sand": 0.10,
        "phit_shale": 0.10, "vclay_shale": 0.50, "angles": [0, 10, 20, 30],
    })
    assert isinstance(res, dict)
    assert res["avo_class"] in {"I", "I*", "II", "IIp", "III", "IV"}
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_meta_tool.py -v`
Expected: FAIL — `petro_to_avo` not in `reg.REGISTRY_BY_NAME` (KeyError / assertion error).

- [ ] **Step 3: Append workflow meta-tools to the registry**

In `core/tool_registry.py`, add this import after the existing tool imports (right after `from workflows.adapters import predict_elastic_layer`):

```python
from workflows.engine import WORKFLOW_REGISTRY
```

Then, immediately AFTER the `REGISTRY = [ ... ]` list literal's closing `]` and BEFORE the line `def to_openai_schema(spec: ToolSpec) -> dict:`, insert:

```python
# Workflows are declared in workflows/engine.py and exposed here as meta-tools,
# so all schema/function/dispatch derivation below applies to them unchanged.
_WORKFLOW_TOOL_SPECS = [
    ToolSpec(
        name=w.name,
        fn=w.fn,
        description=w.description,
        params=w.params,
        required=w.required,
        defaults=w.defaults,
        validator=None,
        auto_plot=w.auto_plot,
    )
    for w in WORKFLOW_REGISTRY
]
REGISTRY = REGISTRY + _WORKFLOW_TOOL_SPECS
```

(The `REGISTRY_BY_NAME` / `TOOL_SCHEMAS` / `TOOL_FUNCTIONS` / `AUTO_PLOT` derivations at the bottom of the file already run after this, so they pick up the appended specs automatically — do not touch them.)

- [ ] **Step 4: Bump the registry count assertion**

In `tests/test_tool_registry.py`, line 6, change:

```python
    assert len(reg.REGISTRY) == 21
```

to:

```python
    assert len(reg.REGISTRY) == 22
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_workflow_meta_tool.py tests/test_tool_registry.py -v`
Expected: PASS (registry contract tests + the 2 new tests all green).

- [ ] **Step 6: Commit**

```bash
git add core/tool_registry.py tests/test_workflow_meta_tool.py tests/test_tool_registry.py
git commit -m "feat(workflows): expose workflows as registry meta-tools

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Chatbot surfaces a workflow result's `image_path`

**Files:**
- Modify: `core/chatbot_tool_use.py` (add `_workflow_image_output` + call it in the result path)
- Create: `tests/test_chatbot_workflow.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chatbot_workflow.py`. The bot fails fast without LLM
credentials, so inject a fake client via the `fake_llm_factory` fixture from
`tests/conftest.py` (the documented injection point — see the package CLAUDE.md
"Per-session state": `__init__(llm_client=..., tool_manager=..., knowledge_base=...)`).
None of these unit checks call the LLM:

```python
import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse


@pytest.fixture
def bot(fake_llm_factory):
    # fake_llm_factory(responses) -> a no-network FakeLLMClient (tests/conftest.py).
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


def test_workflow_image_output_from_dict(bot):
    out = bot._workflow_image_output({"avo_class": "III", "image_path": "/tmp/x.png"})
    assert out == {"image_path": "/tmp/x.png"}


def test_workflow_image_output_none_when_no_png(bot):
    assert bot._workflow_image_output({"avo_class": "III"}) is None
    assert bot._workflow_image_output({"image_path": 123}) is None
    assert bot._workflow_image_output("not-a-dict") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: FAIL with `AttributeError: 'SeismicChatBotToolUse' object has no attribute '_workflow_image_output'` (the fixture constructs the bot fine; the method just doesn't exist yet).

(If the `llm_client=` keyword is rejected by the constructor, STOP and report BLOCKED with the actual `__init__` signature — the controller will supply the correct injection. Do not work around it by editing unrelated code or removing credential checks.)

- [ ] **Step 3: Add the method and call it in the result path**

In `core/chatbot_tool_use.py`, add this method to the `SeismicChatBotToolUse` class (place it directly above the existing `_is_image_output` method):

```python
    def _workflow_image_output(self, tool_result):
        """Surface a composite plot path from a workflow's dict result, if present."""
        if isinstance(tool_result, dict):
            path = tool_result.get("image_path")
            if isinstance(path, str) and path.endswith(".png"):
                return {"image_path": path}
        return None
```

Then find the existing result-handling line (around line 656):

```python
        if self._is_image_output(tool_name, tool_result):
            return {"image_path": tool_result}
```

Immediately AFTER that `if` block, insert:

```python
        workflow_image = self._workflow_image_output(tool_result)
        if workflow_image is not None:
            return workflow_image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(workflows): chatbot surfaces workflow composite plot path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Chatbot caches `last_workflow_result` + lists the workflow in the system prompt; full-suite check

**Files:**
- Modify: `core/chatbot_tool_use.py` (import `WORKFLOW_NAMES`; store result in `_update_context`; add a system-prompt bullet)
- Modify: `tests/test_chatbot_workflow.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatbot_workflow.py` (reuses the `bot` fixture defined above):

```python
def test_update_context_caches_workflow_result(bot):
    result = {"avo_class": "III", "image_path": "/tmp/x.png"}
    bot._update_context("petro_to_avo", {"phit_sand": 0.25}, result)
    assert bot.context_manager.get_context("last_workflow_result") == result


def test_system_prompt_lists_petro_to_avo(bot):
    prompt = bot._create_system_prompt()
    assert "petro_to_avo" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_workflow.py -k "caches_workflow_result or lists_petro" -v`
Expected: FAIL — `last_workflow_result` is `None` (not stored) and/or `"petro_to_avo"` not in the system prompt.

- [ ] **Step 3: Implement the two changes**

In `core/chatbot_tool_use.py`:

(a) Add this import near the other registry import(s) (e.g. wherever `AUTO_PLOT` is imported from `core.tool_registry`):

```python
from workflows.engine import WORKFLOW_NAMES
```

(b) In `_update_context`, after the final `elif tool_name == "calculate_rock_properties":` block (the last `elif` in the method), add a new branch:

```python
        elif tool_name in WORKFLOW_NAMES:
            if isinstance(tool_result, dict):
                self.context_manager.set_context("last_workflow_result", tool_result)
```

(c) In `_create_system_prompt`, in the hardcoded "Available tools:" bullet list, add a bullet (place it after the rock-physics/EEI tool lines):

```
- petro_to_avo: End-to-end AVO feasibility from petrophysics — predicts sand & shale elastic properties from porosity/clay, models the AVO response, and returns the intercept/gradient/AVO class with a composite plot.
```

(Match the existing bullet formatting in that string exactly — same leading `- ` and indentation as the neighboring lines.)

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: the new Phase 1 tests pass and nothing else broke. The standalone `test_tool_use.py::test_tool_use_pattern` (stdin read) is a KNOWN pre-existing failure unrelated to this work — if ONLY that fails, it is expected. If any OTHER test fails, STOP and report BLOCKED with the failure.

- [ ] **Step 6: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(workflows): cache last_workflow_result + list petro_to_avo in prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `WorkflowEngine().run("petro_to_avo", {...})` and `ToolManager().execute_tool("petro_to_avo", {...})` both return a dict with `upper`/`lower` layers, `rc`, `intercept`/`gradient`/`avo_class`, and a composite `image_path` PNG.
- `petro_to_avo` appears in `REGISTRY_BY_NAME`/`TOOL_SCHEMAS`/`TOOL_FUNCTIONS` (registry count 22); schemas/dispatch derive automatically; no edits to `tool_manager.py`.
- The chatbot surfaces the workflow's composite plot (`{"image_path": ...}`), caches `last_workflow_result`, and lists `petro_to_avo` in the system prompt.
- Full suite green (modulo the pre-existing stdin-script failure).

## Spec coverage (Phase 1 row + integration section)

- WorkflowEngine + WorkflowSpec/WORKFLOW_REGISTRY → Task 3.
- Meta-tool wiring into `tool_registry.py` → Task 4.
- `petro_to_avo` recipe (predict_layer×2 → build_interface → Shuey/Zoeppritz → avo_attributes) + composite plot → Tasks 1–2.
- ContextManager `last_workflow_result` → Task 6.
- Image surfacing for the workflow result → Task 5.
- System-prompt listing → Task 6.

## Not in this phase (Phase 2+)

`fluid_scenario` (Gassmann brine-vs-gas `Scenario`), `eei_optimal_chi` (+ the S2 correlation tool), `tuning` recipe, continuous saturation (S1), and the generic sweep runner (S3). Multi-image workflow results (a list of `image_path`s) are deferred until a recipe needs them — Phase 1 surfaces a single composite plot.
