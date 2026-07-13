# Workflows Phase 0 — Data Spine + Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the typed data spine (`Layer`/`Scenario`) and the adapter functions that glue rock-physics tool outputs to AVO/wedge tool inputs, closing the mechanical chaining gaps (G1–G3) and the data-spine gap (I1).

**Architecture:** A new top-level `workflows/` package holds two frozen dataclasses (`Layer`, `Scenario`) and pure adapter functions. Adapters unpack the existing leaf tools' returns (`calculate_rock_properties` → positional tuple; `gassmann_substitution` → dict), reduce array logs to representative scalars, enforce physical validity via `tools/physics_guards.py`, and assemble layers into the exact `{vp1, vs1, …}` / `{v1, v2, v3, …}` dicts the AVO and wedge tools require. The leaf tools are **not modified**. One adapter is exposed as a new leaf tool for immediate chat payoff.

**Tech Stack:** Python 3.9+, NumPy, pytest. Absolute top-level imports (flat layout: `from workflows.adapters import …`). Run pytest from inside the package dir (`geo-mcp/seismic_chatbot`).

**Spec:** `docs/superpowers/specs/2026-06-18-agentic-workflows-design.md` (Phase 0 row).

**Working dir for all commands:** `geo-mcp/seismic_chatbot` (the package is its own git repo).

---

### Task 1: Package scaffold + `Layer`/`Scenario` types

**Files:**
- Create: `workflows/__init__.py`
- Create: `workflows/types.py`
- Create: `tests/test_workflow_types.py`
- Modify: `pyproject.toml` (packages-find include list)

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_types.py`:

```python
import pytest

from workflows.types import Layer, Scenario


def test_layer_fields_and_default_label():
    ly = Layer(vp=3000.0, vs=1500.0, rho=2.2)
    assert (ly.vp, ly.vs, ly.rho) == (3000.0, 1500.0, 2.2)
    assert ly.label == ""


def test_layer_is_frozen():
    ly = Layer(vp=3000.0, vs=1500.0, rho=2.2)
    with pytest.raises(Exception):
        ly.vp = 9999.0  # frozen dataclass forbids reassignment


def test_scenario_holds_named_layers():
    brine = Layer(3000.0, 1500.0, 2.20, "brine")
    gas = Layer(2700.0, 1550.0, 2.05, "gas")
    sc = Scenario(name="fluid", cases={"brine": brine, "gas": gas})
    assert sc.name == "fluid"
    assert sc.cases["gas"].vp < sc.cases["brine"].vp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows'`

- [ ] **Step 3: Create the package and types**

Create `workflows/__init__.py`:

```python
"""Workflow orchestration layer: typed spine + adapters that chain leaf tools."""
```

Create `workflows/types.py`:

```python
"""Typed data spine for workflow recipes.

A `Layer` is one rock (Vp, Vs, density); a `Scenario` bundles named layers
(e.g. a brine case vs a gas case). These live inside the workflow engine —
leaf tools never see them; adapters translate Layer <-> the {vp1, vs1, ...}
dicts the leaf tools expect.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Layer:
    """One elastic rock layer. Units: vp, vs in m/s; rho in g/cm^3."""
    vp: float
    vs: float
    rho: float
    label: str = ""


@dataclass(frozen=True)
class Scenario:
    """A named bundle of layers, e.g. Scenario("fluid", {"brine": ..., "gas": ...})."""
    name: str
    cases: dict = field(default_factory=dict)
```

- [ ] **Step 4: Register the package for installs**

In `pyproject.toml`, find this line (under `[tool.setuptools.packages.find]`):

```python
include = ["config*", "core*", "tools*", "knowledge*", "parsing*", "interfaces*"]
```

Replace it with (adds `workflows*`):

```python
include = ["config*", "core*", "tools*", "knowledge*", "parsing*", "interfaces*", "workflows*"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_workflow_types.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add workflows/__init__.py workflows/types.py tests/test_workflow_types.py pyproject.toml
git commit -m "feat(workflows): Layer/Scenario data-spine types + package scaffold

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_reduce` + `predict_layer` adapter (G2 unpack, G3 reduce)

**Files:**
- Create: `workflows/adapters.py`
- Create: `tests/test_workflow_adapters.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow_adapters.py`:

```python
import numpy as np
import pytest

from workflows.types import Layer
from workflows.adapters import _reduce, predict_layer


def test_reduce_mean_median_index():
    assert _reduce([1.0, 3.0], "mean") == 2.0
    assert _reduce([1.0, 2.0, 9.0], "median") == 2.0
    assert _reduce([5.0, 6.0, 7.0], 0) == 5.0


def test_reduce_empty_raises():
    with pytest.raises(ValueError):
        _reduce([], "mean")


def test_predict_layer_scalar_known_answer():
    # Han et al. (1986), water, vclay=0, phit=0.2:
    #   vp = (5.59 - 6.93*0.2) * 1000 = 4204 m/s
    #   vs = (3.52 - 4.91*0.2) * 1000 = 2538 m/s
    ly = predict_layer(0.2, 0.0, fluid="water")
    assert isinstance(ly, Layer)
    assert np.isclose(ly.vp, 4204.0, rtol=1e-3)
    assert np.isclose(ly.vs, 2538.0, rtol=1e-3)
    assert 0 < ly.vs < ly.vp and ly.rho > 0


def test_predict_layer_reduces_array_log_to_scalar():
    # A two-sample log; mean of phit 0.1 & 0.3 reproduces the phit=0.2 scalar
    # (the Han model is linear in phit), and the result is a plain float (G3).
    ly = predict_layer([0.1, 0.3], [0.0, 0.0], fluid="water", reduce="mean")
    assert isinstance(ly.vp, float)
    assert np.isclose(ly.vp, 4204.0, rtol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_adapters.py -v`
Expected: FAIL with `ImportError: cannot import name '_reduce' from 'workflows.adapters'` (module does not exist)

- [ ] **Step 3: Write the adapter module**

Create `workflows/adapters.py`:

```python
"""Adapters that glue rock-physics outputs to AVO/wedge tool inputs.

Closes the mechanical gaps between the leaf tools:
- G2: unpack `calculate_rock_properties` (a positional tuple) / `gassmann_substitution`
  (a dict) into a typed `Layer`.
- G3: reduce a log of samples (arrays) to one representative scalar layer.
- G1: assemble layers into the {vp1, vs1, ...} / {v1, v2, v3, ...} dicts that the
  AVO and wedge tools require as inputs.

The leaf tools themselves are not modified.
"""
import numpy as np

from tools.rock_physics_tools import calculate_rock_properties, gassmann_substitution
from tools.physics_guards import require_elastic_medium
from workflows.types import Layer


def _reduce(values, reduce="mean"):
    """Reduce a scalar/array to one float.

    `reduce` is 'mean', 'median', or an integer sample index.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("cannot reduce an empty array to a scalar Layer")
    if reduce == "mean":
        return float(np.mean(arr))
    if reduce == "median":
        return float(np.median(arr))
    if isinstance(reduce, int) and not isinstance(reduce, bool):
        return float(arr[reduce])
    raise ValueError(f"unknown reduce mode: {reduce!r}")


def predict_layer(phit, vclay, fluid="water", *, reduce="mean", label=""):
    """Predict a representative elastic `Layer` from porosity + clay volume (G2 + G3)."""
    vp, vs, rho, *_ = calculate_rock_properties(
        phit, vclay, fluid_type=fluid, print_results=False
    )
    layer = Layer(
        vp=_reduce(vp, reduce),
        vs=_reduce(vs, reduce),
        rho=_reduce(rho, reduce),
        label=label,
    )
    require_elastic_medium(layer.vp, layer.vs, layer.rho, label=label or "layer")
    return layer
```

Note: `gassmann_substitution` is imported here but first used in Task 5; importing
it now avoids editing the import line later. (No linter/CI runs in this repo.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_adapters.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/adapters.py tests/test_workflow_adapters.py
git commit -m "feat(workflows): predict_layer adapter (tuple-unpack + array reduce)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `build_interface` adapter (G1 for AVO)

**Files:**
- Modify: `workflows/adapters.py` (append function)
- Modify: `tests/test_workflow_adapters.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_adapters.py`:

```python
from tools.avo_tools import shuey_reflectivity
from workflows.adapters import build_interface


def test_build_interface_keys_and_values():
    upper = Layer(2500.0, 1200.0, 2.30, "shale")
    lower = Layer(3200.0, 1800.0, 2.15, "sand")
    iface = build_interface(upper, lower)
    assert iface == {
        "vp1": 2500.0, "vs1": 1200.0, "rho1": 2.30,
        "vp2": 3200.0, "vs2": 1800.0, "rho2": 2.15,
    }


def test_build_interface_feeds_shuey():
    # The whole point of G1: the assembled dict must satisfy shuey_reflectivity's
    # exact kwargs, proving the contract connects end-to-end.
    upper = Layer(2500.0, 1200.0, 2.30, "shale")
    lower = Layer(3200.0, 1800.0, 2.15, "sand")
    rc = shuey_reflectivity(**build_interface(upper, lower), angles=[0, 10, 20, 30])
    assert np.asarray(rc).shape == (4,)


def test_build_interface_rejects_nonphysical_layer():
    bad = Layer(3000.0, 3200.0, 2.2, "bad")  # vs >= vp is non-physical
    good = Layer(2500.0, 1200.0, 2.3, "ok")
    with pytest.raises(ValueError):
        build_interface(good, bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_adapters.py -k build_interface -v`
Expected: FAIL with `ImportError: cannot import name 'build_interface'`

- [ ] **Step 3: Implement `build_interface`**

Append to `workflows/adapters.py`:

```python
def build_interface(upper: Layer, lower: Layer) -> dict:
    """Assemble two layers into the AVO interface dict the reflectivity tools expect.

    Returns {vp1, vs1, rho1, vp2, vs2, rho2} (G1). Rejects non-physical layers.
    """
    require_elastic_medium(upper.vp, upper.vs, upper.rho, label=upper.label or "upper")
    require_elastic_medium(lower.vp, lower.vs, lower.rho, label=lower.label or "lower")
    return {
        "vp1": upper.vp, "vs1": upper.vs, "rho1": upper.rho,
        "vp2": lower.vp, "vs2": lower.vs, "rho2": lower.rho,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_adapters.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/adapters.py tests/test_workflow_adapters.py
git commit -m "feat(workflows): build_interface adapter (layers -> AVO inputs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `build_earth_model` adapter (G1 for wedge)

**Files:**
- Modify: `workflows/adapters.py` (append function)
- Modify: `tests/test_workflow_adapters.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_adapters.py`:

```python
from tools.wedge_tools import create_wedge_model
from workflows.adapters import build_earth_model


def test_build_earth_model_keys():
    l1 = Layer(2500.0, 1200.0, 2.30, "shale")
    l2 = Layer(3200.0, 1800.0, 2.15, "sand")
    l3 = Layer(2600.0, 1250.0, 2.32, "shale")
    em = build_earth_model([l1, l2, l3])
    assert em == {
        "v1": 2500.0, "v2": 3200.0, "v3": 2600.0,
        "rho1": 2.30, "rho2": 2.15, "rho3": 2.32,
        "vs1": 1200.0, "vs2": 1800.0, "vs3": 1250.0,
    }


def test_build_earth_model_requires_three_layers():
    l1 = Layer(2500.0, 1200.0, 2.30)
    with pytest.raises(ValueError):
        build_earth_model([l1, l1])  # only 2 layers


def test_build_earth_model_feeds_create_wedge_model():
    l1 = Layer(2500.0, 1200.0, 2.30, "shale")
    l2 = Layer(3200.0, 1800.0, 2.15, "sand")
    l3 = Layer(2600.0, 1250.0, 2.32, "shale")
    result = create_wedge_model(max_thickness=50.0, **build_earth_model([l1, l2, l3]))
    # create_wedge_model returns (time_array, model, synthetic, parameters)
    assert len(result) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_adapters.py -k build_earth_model -v`
Expected: FAIL with `ImportError: cannot import name 'build_earth_model'`

- [ ] **Step 3: Implement `build_earth_model`**

Append to `workflows/adapters.py`:

```python
def build_earth_model(layers) -> dict:
    """Assemble exactly 3 layers into the wedge-model input dict (G1).

    Returns {v1, v2, v3, rho1, rho2, rho3, vs1, vs2, vs3} keyed for create_wedge_model.
    """
    layers = list(layers)
    if len(layers) != 3:
        raise ValueError(f"build_earth_model expects exactly 3 layers (got {len(layers)})")
    for i, ly in enumerate(layers, start=1):
        require_elastic_medium(ly.vp, ly.vs, ly.rho, label=ly.label or f"layer{i}")
    l1, l2, l3 = layers
    return {
        "v1": l1.vp, "v2": l2.vp, "v3": l3.vp,
        "rho1": l1.rho, "rho2": l2.rho, "rho3": l3.rho,
        "vs1": l1.vs, "vs2": l2.vs, "vs3": l3.vs,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_adapters.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/adapters.py tests/test_workflow_adapters.py
git commit -m "feat(workflows): build_earth_model adapter (layers -> wedge inputs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `layer_from_gassmann` adapter (G2 for fluid substitution)

**Files:**
- Modify: `workflows/adapters.py` (append function)
- Modify: `tests/test_workflow_adapters.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_adapters.py`:

```python
from workflows.adapters import layer_from_gassmann


def test_layer_from_gassmann_brine_to_gas():
    # Shear modulus is fluid-independent: gas LOWERS Vp and RAISES Vs (lower density),
    # and lowers bulk density. Result must be a scalar Layer (G2 + G3).
    ly = layer_from_gassmann(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas", label="gas-sand",
    )
    assert isinstance(ly.vp, float)
    assert ly.vp < 3000.0
    assert ly.vs > 1500.0
    assert ly.rho < 2.2
    assert ly.label == "gas-sand"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_adapters.py -k gassmann -v`
Expected: FAIL with `ImportError: cannot import name 'layer_from_gassmann'`

- [ ] **Step 3: Implement `layer_from_gassmann`**

Append to `workflows/adapters.py`:

```python
def layer_from_gassmann(vp, vs, rho, phi, fluid_in, fluid_out,
                        *, reduce="mean", label="", **kwargs) -> Layer:
    """Run Gassmann fluid substitution and adapt the result dict into a `Layer` (G2).

    Extra kwargs (k_mineral, k_fl_in, rho_fl_in, k_fl_out, rho_fl_out) pass through
    to gassmann_substitution.
    """
    res = gassmann_substitution(
        vp=vp, vs=vs, rho=rho, phi=phi,
        fluid_in=fluid_in, fluid_out=fluid_out,
        print_results=False, **kwargs,
    )
    layer = Layer(
        vp=_reduce(res["vp"], reduce),
        vs=_reduce(res["vs"], reduce),
        rho=_reduce(res["rho"], reduce),
        label=label,
    )
    require_elastic_medium(layer.vp, layer.vs, layer.rho, label=label or "substituted")
    return layer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_adapters.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/adapters.py tests/test_workflow_adapters.py
git commit -m "feat(workflows): layer_from_gassmann adapter (fluid-sub dict -> Layer)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Expose `predict_elastic_layer` as a leaf tool (chat payoff)

**Files:**
- Modify: `workflows/adapters.py` (append function)
- Modify: `core/tool_registry.py` (import + new `ToolSpec`)
- Create: `tests/test_predict_elastic_layer_tool.py`
- Modify: `tests/test_tool_registry.py:6` (count 20 → 21)

- [ ] **Step 1: Write the failing test**

Create `tests/test_predict_elastic_layer_tool.py`:

```python
import numpy as np

from core import tool_registry as reg
from workflows.adapters import predict_elastic_layer


def test_predict_elastic_layer_returns_plain_dict():
    out = predict_elastic_layer(0.2, 0.0, fluid="water")
    assert set(out) == {"vp", "vs", "rho", "vp_vs"}
    assert np.isclose(out["vp"], 4204.0, rtol=1e-3)
    assert np.isclose(out["vp_vs"], out["vp"] / out["vs"], rtol=1e-9)


def test_predict_elastic_layer_registered():
    assert "predict_elastic_layer" in reg.REGISTRY_BY_NAME
    assert "predict_elastic_layer" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"predict_elastic_layer"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict_elastic_layer_tool.py -v`
Expected: FAIL with `ImportError: cannot import name 'predict_elastic_layer'`

- [ ] **Step 3: Implement the dict-returning wrapper**

Append to `workflows/adapters.py`:

```python
def predict_elastic_layer(phit, vclay, fluid="water", reduce="mean") -> dict:
    """LLM-facing leaf tool: representative elastic properties of one layer as a dict.

    Thin wrapper over predict_layer that returns a JSON-friendly dict (no Layer type),
    so it can be registered as a standard leaf tool.
    """
    layer = predict_layer(phit, vclay, fluid=fluid, reduce=reduce)
    return {"vp": layer.vp, "vs": layer.vs, "rho": layer.rho, "vp_vs": layer.vp / layer.vs}
```

- [ ] **Step 4: Register it in the tool registry**

In `core/tool_registry.py`, add this import alongside the other tool imports (after the
`from tools.rag_tools import knowledge_rag` line):

```python
from workflows.adapters import predict_elastic_layer
```

Then add this `ToolSpec` to the `REGISTRY` list, immediately before the closing `]`
(after the `knowledge_rag` spec):

```python
    ToolSpec(
        name="predict_elastic_layer",
        fn=predict_elastic_layer,
        description="Predicts representative elastic properties (Vp, Vs, density, Vp/Vs ratio) of a single rock layer from porosity (phit) and clay volume (vclay) using the Han et al. (1986) rock-physics model. Reduces a log of samples to one representative scalar layer. Returns a dict; no plot.",
        params={
            "phit": {"type": "array", "items": {"type": "number"}, "description": "Porosity values (fraction, 0-1); a single value or a log array."},
            "vclay": {"type": "array", "items": {"type": "number"}, "description": "Clay volume values (fraction, 0-1)."},
            "fluid": {"type": "string", "description": "Pore fluid: 'water'/'brine', 'oil', or 'gas' (default 'water')."},
            "reduce": {"type": "string", "description": "How to reduce a log array to one scalar layer: 'mean' or 'median' (default 'mean')."},
        },
        required=["phit", "vclay"],
        defaults={"fluid": "water", "reduce": "mean"},
        validator=None,
        auto_plot=None,
    ),
```

- [ ] **Step 5: Update the registry count assertion**

In `tests/test_tool_registry.py`, line 6, change:

```python
    assert len(reg.REGISTRY) == 20
```

to:

```python
    assert len(reg.REGISTRY) == 21
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_predict_elastic_layer_tool.py tests/test_tool_registry.py -v`
Expected: PASS (registry contract tests + the two new tests all green)

- [ ] **Step 7: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: PASS — all pre-existing tests plus the new Phase 0 tests. (The standalone
`test_tool_use_pattern` script that reads stdin is a known pre-existing failure
unrelated to this work; everything under `tests/` passes.)

- [ ] **Step 8: Commit**

```bash
git add workflows/adapters.py core/tool_registry.py tests/test_predict_elastic_layer_tool.py tests/test_tool_registry.py
git commit -m "feat(workflows): register predict_elastic_layer leaf tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `workflows/` package exists with `types.py` (`Layer`, `Scenario`) and `adapters.py`
  (`_reduce`, `predict_layer`, `build_interface`, `build_earth_model`,
  `layer_from_gassmann`, `predict_elastic_layer`).
- `pyproject.toml` installs `workflows*`.
- Adapters enforce physical validity via `require_elastic_medium` and reject
  non-physical layers.
- Round-trip tests prove the assembled dicts satisfy `shuey_reflectivity` and
  `create_wedge_model` kwargs exactly (G1 closed), tuples/dicts are unpacked (G2),
  and array logs reduce to scalars (G3).
- `predict_elastic_layer` is a registered leaf tool (registry count 21); full suite green.

## Closes (from the spec gap ledger)

- **G1** interface assembly — `build_interface`, `build_earth_model`.
- **G2** unpack + rename — `predict_layer`, `layer_from_gassmann`.
- **G3** array→scalar reduce — `_reduce`.
- **I1** data spine — `Layer`/`Scenario` + in-code adapter passing.

## Not in this phase

The `WorkflowEngine`, `WorkflowSpec`/`WORKFLOW_REGISTRY`, recipes, meta-tool wiring,
saturation science, and the sweep runner are Phases 1–4 (separate plans). `Scenario`
is introduced here as a type but is not exercised by a recipe until Phase 2.
