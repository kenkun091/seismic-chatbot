# Workflows Phase 4 — Sweep Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic parameter-sweep engine `run_sweep(recipe, grid, metric, fixed)` that runs any registered workflow recipe over the cartesian product of a parameter grid, collects a chosen scalar metric per cell, and returns a results table, summary statistics, coverage, and an aggregate plot — closing gap **S3** and unlocking the Monte-Carlo / sensitivity workflow (#7).

**Architecture:** A new `workflows/sweep.py` holds a pure grid-expander `_expand_grid`, a pure `_summarize` (numeric stats / categorical counts), the orchestrator `run_sweep`, and `plot_sweep`. `run_sweep` runs each grid cell through the existing `WorkflowEngine` (lazy-imported to avoid an import cycle), deletes each cell's self-plotted PNG, extracts the metric, and tolerates per-cell failures (recorded, never aborting). It is registered as one `WorkflowSpec` in `workflows/engine.py`, so it rides the registry-derivation machinery and the chatbot's generic image surfacing exactly like every other recipe.

**Tech Stack:** Python 3.9+, NumPy, Matplotlib (existing headless plot convention), pytest, `itertools.product`. Run from inside `geo-mcp/seismic_chatbot`.

**Spec:** parent design `docs/superpowers/specs/2026-06-18-agentic-workflows-design.md` (Phase 4 row; gap **S3**; `workflows/sweep.py` "generic grid runner"; error handling "a failed grid cell is recorded as null/NaN and reported in a coverage summary rather than aborting the whole sweep").

**Builds on (verified contracts — do not re-derive):**
- `workflows/engine.py::WorkflowEngine.run(name, params)` — fills `spec.defaults`, checks `spec.required`, calls `spec.fn(**full)`; raises `ValueError(f"Unknown workflow: {name}")` for an unknown name and `ValueError(f"{name}: missing required parameters: {missing}")` for missing required params.
- `workflows/engine.py::WORKFLOW_REGISTRY_BY_NAME` (name → `WorkflowSpec`) and `WORKFLOW_NAMES` (frozenset of recipe names). After this phase, both include `run_sweep`.
- Every recipe returns a **flat JSON-friendly dict** whose scalar metrics are top-level keys, e.g. `petro_to_avo` → `intercept`, `gradient`, `avo_class` (+ `rc` list, `image_path`); `tuning` → `tuning_thickness`, `tuning_amplitude`, `resolution_limit`; `saturation_sweep` → curves; all recipes return a self-plotted `image_path` (a `.png` temp file).
- Registry: workflows are declared in `workflows/engine.py::WORKFLOW_REGISTRY` and auto-converted into `ToolSpec`s by `core/tool_registry.py`. **`REGISTRY` currently has 28 tools.**
- Plot convention: `output_path=None` → `tempfile.mkstemp(suffix=".png")` + `os.close(fd)`; `fig.savefig(output_path, dpi=300, bbox_inches="tight")`; `plt.close(fig)`; `return output_path`.
- Chatbot: `_workflow_image_output` surfaces `{"image_path": ...}` for ANY dict whose `image_path` is a `.png` string; `_update_context` caches `tool_name in WORKFLOW_NAMES`; `_create_system_prompt` has a hardcoded bullet list.

**Working dir for all commands:** `geo-mcp/seismic_chatbot` (its own git repo, branch `stabilize-tool-layer`).

## Global Constraints

- All new code lives in `workflows/sweep.py` and one `WorkflowSpec` in `workflows/engine.py`. No edits to `core/tool_registry.py` or `core/tool_manager.py` (the registry auto-conversion picks the spec up). The only `core/` edit is the chatbot system-prompt bullet (Task 6).
- **No import cycle:** `workflows/sweep.py` must NOT import `workflows.engine` at module top level (engine imports sweep to register it). Import `WorkflowEngine` / `WORKFLOW_NAMES` **lazily inside `run_sweep`**.
- Sweep type is the **cartesian product of explicit value lists** (deterministic; no Monte-Carlo / RNG this phase).
- **No silent truncation / no abort:** a cell whose recipe raises is recorded in `coverage.failures` with `value=None`; the sweep continues. If *every* cell fails, raise `ValueError` naming the first failure (surfaces a bad recipe name, missing required param, or metric typo).
- **Per-cell PNG hygiene:** each recipe self-plots a temp PNG; `run_sweep` deletes every cell's `image_path` and emits exactly ONE aggregate plot.
- `run_sweep` targets registered **workflow recipes only** (names in `WORKFLOW_NAMES`) and must reject sweeping itself (`recipe == "run_sweep"`).
- Registry count moves **28 → 29** (one new meta-tool, `run_sweep`).
- The standalone `test_tool_use.py::test_tool_use_pattern` (stdin) is a KNOWN pre-existing failure; ignore it in full-suite runs.

---

## File Structure

- `workflows/sweep.py` — new: `_expand_grid`, `_summarize`, `run_sweep`, `plot_sweep`.
- `workflows/engine.py` — modify: import `run_sweep` + one `WorkflowSpec`.
- `core/chatbot_tool_use.py` — modify: one system-prompt bullet.
- Tests: `tests/test_sweep.py` (expander + summarize + run_sweep + plot), appends to `tests/test_tool_registry.py` (28→29), `tests/test_workflow_meta_tool.py`, `tests/test_chatbot_workflow.py`.

The 6 tasks: (1) `_expand_grid`; (2) `_summarize`; (3) `run_sweep` core (no plot); (4) `plot_sweep` + wire `image_path`; (5) register `run_sweep` meta-tool; (6) system-prompt bullet + full suite.

---

### Task 1: `_expand_grid` cartesian product

**Files:**
- Create: `workflows/sweep.py`
- Create: `tests/test_sweep.py`

**Interfaces:**
- Produces: `_expand_grid(grid, fixed=None)` → `list[dict]`. Each dict is one full parameter set: the `fixed` params merged with one combination of the swept params (swept keys override on collision). Order follows `itertools.product` over the grid keys in insertion order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sweep.py`:

```python
import os
import numpy as np
import pytest

from workflows.sweep import _expand_grid


def test_expand_grid_cartesian_product():
    combos = _expand_grid({"a": [1, 2, 3], "b": ["x", "y"]})
    assert len(combos) == 6
    assert {"a": 1, "b": "x"} in combos
    assert {"a": 3, "b": "y"} in combos


def test_expand_grid_merges_fixed():
    combos = _expand_grid({"a": [1, 2]}, fixed={"c": 9})
    assert combos == [{"c": 9, "a": 1}, {"c": 9, "a": 2}]


def test_expand_grid_swept_overrides_fixed():
    combos = _expand_grid({"a": [5]}, fixed={"a": 0, "c": 9})
    assert combos == [{"c": 9, "a": 5}]


def test_expand_grid_rejects_empty_grid():
    with pytest.raises(ValueError):
        _expand_grid({})


def test_expand_grid_rejects_empty_value_list():
    with pytest.raises(ValueError):
        _expand_grid({"a": [1], "b": []})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep.py -v`
Expected: FAIL with `ImportError: cannot import name '_expand_grid' from 'workflows.sweep'` (module does not exist yet).

- [ ] **Step 3: Implement the expander**

Create `workflows/sweep.py`:

```python
"""Generic parameter-sweep engine for workflow recipes (gap S3).

run_sweep runs a registered recipe over the cartesian product of a parameter
grid, collects a chosen scalar metric per cell, and returns a results table,
summary statistics, coverage, and an aggregate plot. Per-cell recipe plots are
deleted; one aggregate plot is produced. A cell whose recipe raises is recorded
(value None) and the sweep continues.
"""
import itertools
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt


def _expand_grid(grid, fixed=None):
    """Cartesian product of a {param: [values]} grid, each merged with `fixed`.

    Returns a list of full parameter dicts. Swept keys override `fixed` on
    collision. Raises ValueError on an empty grid or an empty value list.
    """
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid must be a non-empty {param: [values]} mapping")
    fixed = dict(fixed or {})
    keys = list(grid)
    value_lists = []
    for k in keys:
        vals = grid[k]
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            raise ValueError(f"grid['{k}'] must be a non-empty list of values")
        value_lists.append(list(vals))

    combos = []
    for values in itertools.product(*value_lists):
        combo = dict(fixed)
        combo.update(dict(zip(keys, values)))
        combos.append(combo)
    return combos
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sweep.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/sweep.py tests/test_sweep.py
git commit -m "feat(workflows): _expand_grid cartesian product for sweeps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_summarize` statistics

**Files:**
- Modify: `workflows/sweep.py` (append `_summarize`)
- Modify: `tests/test_sweep.py` (append tests)

**Interfaces:**
- Produces: `_summarize(values)` → dict. `values` is a list that may contain `None` (failed cells). Numeric metric (all non-None entries are real numbers, excluding bool) → `{"kind": "numeric", "count", "min", "max", "mean", "std"}`. Otherwise → `{"kind": "categorical", "count", "counts": {value: n}}`. Empty / all-None → `{"kind": "empty", "count": 0}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep.py`:

```python
from workflows.sweep import _summarize


def test_summarize_numeric():
    s = _summarize([1.0, 2.0, 3.0, None])
    assert s["kind"] == "numeric"
    assert s["count"] == 3
    assert s["min"] == 1.0 and s["max"] == 3.0
    assert np.isclose(s["mean"], 2.0)
    assert np.isclose(s["std"], np.std([1.0, 2.0, 3.0]))


def test_summarize_categorical():
    s = _summarize(["III", "III", "IIp", None])
    assert s["kind"] == "categorical"
    assert s["count"] == 3
    assert s["counts"] == {"III": 2, "IIp": 1}


def test_summarize_empty():
    s = _summarize([None, None])
    assert s["kind"] == "empty"
    assert s["count"] == 0


def test_summarize_bool_is_categorical():
    # bool must NOT be treated as numeric (True/False are categorical here).
    s = _summarize([True, False, True])
    assert s["kind"] == "categorical"
    assert s["counts"] == {True: 2, False: 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep.py -k summarize -v`
Expected: FAIL with `ImportError: cannot import name '_summarize'`.

- [ ] **Step 3: Implement `_summarize`**

Append to `workflows/sweep.py`:

```python
def _is_number(x):
    """True for real numeric scalars, excluding bool (bool is treated categorically)."""
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _summarize(values):
    """Summarize swept metric values (numeric stats or categorical counts)."""
    present = [v for v in values if v is not None]
    if not present:
        return {"kind": "empty", "count": 0}
    if all(_is_number(v) for v in present):
        arr = np.asarray([float(v) for v in present], dtype=float)
        return {
            "kind": "numeric",
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    return {"kind": "categorical", "count": len(present), "counts": counts}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sweep.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/sweep.py tests/test_sweep.py
git commit -m "feat(workflows): _summarize numeric/categorical sweep stats

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `run_sweep` core (no plot yet)

**Files:**
- Modify: `workflows/sweep.py` (append `run_sweep`; the plot is added in Task 4)
- Modify: `tests/test_sweep.py` (append tests)

**Interfaces:**
- Consumes: `_expand_grid`, `_summarize`; `WorkflowEngine`, `WORKFLOW_NAMES` (lazy-imported from `workflows.engine`).
- Produces: `run_sweep(recipe, grid, metric, fixed=None)` → dict:
  `{"recipe", "metric", "swept_params": [grid keys], "rows": [{"params": {swept-only}, "value": metric-or-None}], "stats": {summary}, "coverage": {"total", "ran", "failed", "failures": [{"params", "error"}]}}`.
  (No `image_path` yet — added in Task 4.)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep.py`:

```python
from workflows.sweep import run_sweep

# Fixed params that make petro_to_avo run for any fluid_sand value.
_PETRO_FIXED = {
    "phit_sand": 0.25, "vclay_sand": 0.15,
    "phit_shale": 0.10, "vclay_shale": 0.55,
    "angles": [0, 10, 20, 30],
}


def test_run_sweep_1d_numeric_metric():
    res = run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert res["recipe"] == "petro_to_avo"
    assert res["metric"] == "gradient"
    assert res["swept_params"] == ["fluid_sand"]
    assert len(res["rows"]) == 2
    assert all(isinstance(r["value"], float) for r in res["rows"])
    assert res["coverage"] == {"total": 2, "ran": 2, "failed": 0, "failures": []}
    assert res["stats"]["kind"] == "numeric"


def test_run_sweep_2d_grid_cell_count():
    res = run_sweep("petro_to_avo",
                    {"phit_sand": [0.20, 0.30], "fluid_sand": ["brine", "gas"]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert len(res["rows"]) == 4
    assert res["coverage"]["ran"] == 4


def test_run_sweep_categorical_metric():
    res = run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                    metric="avo_class", fixed=_PETRO_FIXED)
    assert res["stats"]["kind"] == "categorical"
    assert sum(res["stats"]["counts"].values()) == 2


def test_run_sweep_records_failed_cell_without_aborting():
    # phit_sand=1.5 is non-physical -> that cell raises; the brine/gas valid
    # cells still run. (Grid value overrides the fixed phit_sand.)
    res = run_sweep("petro_to_avo",
                    {"phit_sand": [0.25, 1.5], "fluid_sand": ["gas"]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert res["coverage"]["total"] == 2
    assert res["coverage"]["ran"] == 1
    assert res["coverage"]["failed"] == 1
    assert len(res["coverage"]["failures"]) == 1


def test_run_sweep_rejects_self():
    with pytest.raises(ValueError):
        run_sweep("run_sweep", {"a": [1]}, metric="x")


def test_run_sweep_rejects_unknown_recipe():
    with pytest.raises(ValueError):
        run_sweep("not_a_recipe", {"a": [1]}, metric="x")


def test_run_sweep_all_cells_fail_raises():
    # A bad metric makes every cell fail -> ValueError naming the problem.
    with pytest.raises(ValueError):
        run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                  metric="nonexistent_metric", fixed=_PETRO_FIXED)


def test_run_sweep_cleans_up_cell_pngs(monkeypatch):
    removed = []
    real_remove = os.remove
    monkeypatch.setattr("workflows.sweep.os.remove",
                        lambda p: (removed.append(p), real_remove(p)))
    run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
              metric="gradient", fixed=_PETRO_FIXED)
    # Two cells -> two per-cell PNGs deleted.
    assert len([p for p in removed if p.endswith(".png")]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep.py -k run_sweep -v`
Expected: FAIL with `ImportError: cannot import name 'run_sweep'`.

- [ ] **Step 3: Implement `run_sweep`**

Append to `workflows/sweep.py`:

```python
def run_sweep(recipe, grid, metric, fixed=None):
    """Run `recipe` over the cartesian product of `grid`, collecting `metric` per cell.

    Returns a results table, summary stats, and coverage. Per-cell recipe PNGs are
    deleted. Cells whose recipe raises (or that lack `metric`) are recorded in
    coverage.failures with value None; the sweep does not abort. Raises ValueError
    for a self-sweep, an unknown recipe, a bad grid, or if every cell fails.
    """
    # Lazy import to avoid the engine<->sweep import cycle.
    from workflows.engine import WorkflowEngine, WORKFLOW_NAMES

    if recipe == "run_sweep":
        raise ValueError("cannot sweep run_sweep itself")
    if recipe not in WORKFLOW_NAMES:
        raise ValueError(f"unknown recipe {recipe!r}; choose one of {sorted(WORKFLOW_NAMES)}")

    swept_params = list(grid)
    combos = _expand_grid(grid, fixed)
    engine = WorkflowEngine()

    rows = []
    failures = []
    for combo in combos:
        swept_only = {k: combo[k] for k in swept_params}
        try:
            result = engine.run(recipe, combo)
        except Exception as exc:  # a recipe raised -> record, do not abort
            failures.append({"params": swept_only, "error": str(exc)})
            rows.append({"params": swept_only, "value": None})
            continue
        img = result.get("image_path")
        if isinstance(img, str) and os.path.exists(img):
            os.remove(img)
        if metric not in result:
            failures.append({"params": swept_only,
                             "error": f"metric {metric!r} not in result keys {sorted(result)}"})
            rows.append({"params": swept_only, "value": None})
            continue
        rows.append({"params": swept_only, "value": result[metric]})

    total = len(rows)
    ran = sum(1 for r in rows if r["value"] is not None)
    if ran == 0:
        first = failures[0]["error"] if failures else "no cells produced a value"
        raise ValueError(f"all {total} sweep cells failed; first error: {first}")

    return {
        "recipe": recipe,
        "metric": metric,
        "swept_params": swept_params,
        "rows": rows,
        "stats": _summarize([r["value"] for r in rows]),
        "coverage": {"total": total, "ran": ran, "failed": total - ran,
                     "failures": failures},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sweep.py -v`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/sweep.py tests/test_sweep.py
git commit -m "feat(workflows): run_sweep core (grid runner, coverage, fault-tolerant)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `plot_sweep` + wire `image_path` into `run_sweep`

**Files:**
- Modify: `workflows/sweep.py` (append `plot_sweep`; add `image_path` to the `run_sweep` return)
- Modify: `tests/test_sweep.py` (append tests)

**Interfaces:**
- Consumes: the `run_sweep` result dict (`swept_params`, `rows`, `metric`, `stats`).
- Produces: `plot_sweep(result, output_path=None)` → `.png` path. Dispatch: categorical metric → bar of value counts; numeric metric with exactly one numeric swept param → line (metric vs param); numeric metric with exactly two numeric swept params → heatmap; otherwise numeric → histogram of metric values. `run_sweep` now sets `result["image_path"] = plot_sweep(result)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sweep.py`:

```python
def _png_ok(path):
    return isinstance(path, str) and path.endswith(".png") and os.path.getsize(path) > 0


def test_run_sweep_adds_image_path_1d():
    res = run_sweep("petro_to_avo", {"phit_sand": [0.15, 0.25, 0.35]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert _png_ok(res["image_path"])
    os.remove(res["image_path"])


def test_run_sweep_adds_image_path_2d_heatmap():
    res = run_sweep("petro_to_avo",
                    {"phit_sand": [0.15, 0.25], "vclay_sand": [0.10, 0.20]},
                    metric="gradient", fixed=_PETRO_FIXED)
    assert _png_ok(res["image_path"])
    os.remove(res["image_path"])


def test_run_sweep_adds_image_path_categorical():
    res = run_sweep("petro_to_avo", {"fluid_sand": ["brine", "gas"]},
                    metric="avo_class", fixed=_PETRO_FIXED)
    assert _png_ok(res["image_path"])
    os.remove(res["image_path"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sweep.py -k image_path -v`
Expected: FAIL — `KeyError: 'image_path'` (run_sweep does not set it yet).

- [ ] **Step 3: Implement `plot_sweep` and wire it in**

Append `plot_sweep` to `workflows/sweep.py`:

```python
def _numeric_param_values(rows, key):
    """Return the per-row float values of a swept param, or None if any is non-numeric."""
    vals = [r["params"][key] for r in rows]
    if all(_is_number(v) for v in vals):
        return [float(v) for v in vals]
    return None


def plot_sweep(result, output_path=None):
    """Aggregate plot for a sweep: line (1-D numeric), heatmap (2-D numeric),
    histogram (other numeric), or bar of counts (categorical metric)."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    rows = result["rows"]
    metric = result["metric"]
    swept = result["swept_params"]
    numeric_metric = result["stats"]["kind"] == "numeric"

    fig, ax = plt.subplots(figsize=(9, 6))

    if not numeric_metric:
        # Categorical metric: bar of value counts.
        counts = result["stats"].get("counts", {})
        labels = [str(k) for k in counts]
        ax.bar(labels, [counts[k] for k in counts], color="C0")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.set_title(f"{result['recipe']}: {metric} distribution")
    elif len(swept) == 1 and _numeric_param_values(rows, swept[0]) is not None:
        x = _numeric_param_values(rows, swept[0])
        y = [r["value"] if r["value"] is not None else np.nan for r in rows]
        order = np.argsort(x)
        ax.plot(np.asarray(x)[order], np.asarray(y, dtype=float)[order], "o-", color="C0")
        ax.set_xlabel(swept[0])
        ax.set_ylabel(metric)
        ax.set_title(f"{result['recipe']}: {metric} vs {swept[0]}")
        ax.grid(True, alpha=0.3)
    elif (len(swept) == 2 and _numeric_param_values(rows, swept[0]) is not None
          and _numeric_param_values(rows, swept[1]) is not None):
        xs = sorted({r["params"][swept[0]] for r in rows})
        ys = sorted({r["params"][swept[1]] for r in rows})
        grid_z = np.full((len(ys), len(xs)), np.nan)
        for r in rows:
            i = ys.index(r["params"][swept[1]])
            j = xs.index(r["params"][swept[0]])
            grid_z[i, j] = r["value"] if r["value"] is not None else np.nan
        im = ax.imshow(grid_z, origin="lower", aspect="auto",
                       extent=[min(xs), max(xs), min(ys), max(ys)])
        fig.colorbar(im, ax=ax, label=metric)
        ax.set_xlabel(swept[0])
        ax.set_ylabel(swept[1])
        ax.set_title(f"{result['recipe']}: {metric}")
    else:
        # >2 swept dims or non-numeric params: distribution histogram.
        vals = [r["value"] for r in rows if r["value"] is not None]
        ax.hist(vals, bins=min(20, max(5, len(vals))), color="C0")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.set_title(f"{result['recipe']}: {metric} distribution ({len(swept)} params)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

Then, in `run_sweep`, replace the `return {...}` block so it builds the result, attaches the plot, and returns it:

```python
    out = {
        "recipe": recipe,
        "metric": metric,
        "swept_params": swept_params,
        "rows": rows,
        "stats": _summarize([r["value"] for r in rows]),
        "coverage": {"total": total, "ran": ran, "failed": total - ran,
                     "failures": failures},
    }
    out["image_path"] = plot_sweep(out)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sweep.py -v`
Expected: PASS (20 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/sweep.py tests/test_sweep.py
git commit -m "feat(workflows): plot_sweep (line/heatmap/hist/bar) + image_path

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Register `run_sweep` as a workflow meta-tool

**Files:**
- Modify: `workflows/engine.py` (import + `WorkflowSpec`)
- Modify: `tests/test_tool_registry.py` (count 28 → 29)
- Modify: `tests/test_workflow_meta_tool.py` (append a case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_meta_tool.py` (`reg` and `ToolManager` already imported):

```python
def test_run_sweep_is_registered_meta_tool():
    assert "run_sweep" in reg.REGISTRY_BY_NAME
    assert "run_sweep" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"run_sweep"}


def test_run_sweep_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("run_sweep", {
        "recipe": "petro_to_avo",
        "grid": {"fluid_sand": ["brine", "gas"]},
        "metric": "gradient",
        "fixed": {"phit_sand": 0.25, "vclay_sand": 0.15,
                  "phit_shale": 0.10, "vclay_shale": 0.55,
                  "angles": [0, 10, 20, 30]},
    })
    assert res["coverage"]["ran"] == 2
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_meta_tool.py -k run_sweep -v`
Expected: FAIL — not in `reg.REGISTRY_BY_NAME`.

- [ ] **Step 3: Register the meta-tool**

In `workflows/engine.py`, add the import after the existing recipe imports (after `from workflows.recipes.saturation_sweep import saturation_sweep`):

```python
from workflows.sweep import run_sweep
```

Add this `WorkflowSpec` to `WORKFLOW_REGISTRY` (after the `saturation_sweep` spec, before the closing `]`):

```python
    WorkflowSpec(
        name="run_sweep",
        fn=run_sweep,
        description=(
            "Parameter sweep / sensitivity analysis: run another workflow recipe over a "
            "grid of parameter values (the cartesian product) and collect one scalar "
            "result metric per run. Returns a results table, summary statistics "
            "(min/max/mean/std for numeric metrics, or value counts for categorical ones "
            "like AVO class), a coverage report (which cells ran or failed), and an "
            "aggregate plot (a line for a 1-parameter sweep, a heatmap for two, a "
            "histogram otherwise). Use it to test how an output responds across ranges of "
            "porosity, clay volume, fluid, saturation, frequency, etc."
        ),
        params={
            "recipe": {"type": "string", "description": "Name of the workflow recipe to sweep, e.g. 'petro_to_avo', 'fluid_scenario', 'tuning', or 'saturation_sweep'."},
            "grid": {"type": "object", "description": "Swept parameters mapped to lists of values, e.g. {\"phit_sand\": [0.1, 0.2, 0.3], \"fluid_sand\": [\"brine\", \"gas\"]}. The cartesian product of these is run."},
            "metric": {"type": "string", "description": "Name of the scalar field in the recipe's result to collect per run, e.g. 'gradient', 'intercept', 'avo_class', 'tuning_thickness', or 'resolution_limit'."},
            "fixed": {"type": "object", "description": "Parameters held constant across every run (the recipe's other required/optional params)."},
        },
        required=["recipe", "grid", "metric"],
        defaults={"fixed": None},
        auto_plot=None,
    ),
```

- [ ] **Step 4: Bump the registry count**

In `tests/test_tool_registry.py`, change `assert len(reg.REGISTRY) == 28` to:

```python
    assert len(reg.REGISTRY) == 29
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_workflow_meta_tool.py tests/test_tool_registry.py tests/test_workflow_engine.py -v`
Expected: PASS. Also confirm `python -c "from core.tool_registry import REGISTRY; print(len(REGISTRY))"` prints 29.

- [ ] **Step 6: Commit**

```bash
git add workflows/engine.py tests/test_workflow_meta_tool.py tests/test_tool_registry.py
git commit -m "feat(workflows): register run_sweep meta-tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: System-prompt bullet; full-suite check

**Files:**
- Modify: `core/chatbot_tool_use.py` (one system-prompt bullet)
- Modify: `tests/test_chatbot_workflow.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatbot_workflow.py` (reuses the `bot` fixture):

```python
def test_system_prompt_lists_run_sweep(bot):
    prompt = bot._create_system_prompt()
    assert "- run_sweep:" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_workflow.py -k run_sweep -v`
Expected: FAIL — the bullet is not present yet.

- [ ] **Step 3: Add the bullet**

In `core/chatbot_tool_use.py`, in `_create_system_prompt`'s hardcoded "Available tools:" bullet list, add this bullet after the `saturation_sweep` bullet (match the neighboring bullet formatting exactly):

```
- run_sweep: Sweep another workflow recipe over a grid of parameter values (cartesian product) and collect one scalar metric per run — returns a results table, summary statistics, a coverage report, and an aggregate plot (line for 1 parameter, heatmap for 2). Use for sensitivity / Monte-Carlo-style analysis across ranges of porosity, clay, fluid, saturation, or frequency.
```

(No other chatbot changes: `run_sweep` returns a dict with `image_path`, so `_workflow_image_output` surfaces the plot; it is in `WORKFLOW_NAMES`, so it auto-caches via the existing branch.)

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: PASS (the prior chatbot-workflow tests plus the new one).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: the new Phase 4 tests pass and nothing else regressed. The standalone
`test_tool_use.py::test_tool_use_pattern` (stdin) is a KNOWN pre-existing failure —
if ONLY that fails, it is expected. If any OTHER test fails, STOP and report BLOCKED.

- [ ] **Step 6: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(workflows): list run_sweep in chatbot system prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `_expand_grid` produces the cartesian product merged with `fixed` (swept keys win), and rejects an empty grid / empty value list.
- `_summarize` returns numeric stats (min/max/mean/std) for numeric metrics, value counts for categorical, `empty` for all-None, and treats `bool` as categorical.
- `run_sweep` runs a recipe over the grid, deletes each cell's PNG, records failed cells in `coverage` without aborting, raises `ValueError` for a self-sweep / unknown recipe / all-cells-failed, and returns the table + stats + coverage + one aggregate `image_path`.
- `plot_sweep` dispatches correctly: 1-D numeric → line, 2-D numeric → heatmap, categorical → bar, else → histogram.
- `run_sweep` is registered as a meta-tool (registry count **29**), runs end-to-end via `ToolManager`, surfaces its plot through the generic `_workflow_image_output`, caches via `WORKFLOW_NAMES`, and appears in the chatbot system prompt.
- Full suite green (modulo the pre-existing stdin-script failure).

## Spec coverage

- Gap **S3** (generic sweep/scenario runner → distributions) → Tasks 1–4.
- Fault-tolerant coverage (failed cell recorded, not aborted; no silent truncation) → Task 3.
- Registry exposure as a meta-tool → Task 5; chatbot exposure → Task 6 (image surfacing + caching come free).
- Aggregate plot (line / heatmap / histogram / bar) → Task 4.
- The Monte-Carlo / sensitivity workflow (#7) is unlocked by `run_sweep` over `petro_to_avo` (AVO response across φ/Vclay/fluid), exercised in the Task 3/5 tests.

## Not in this plan

- Random / Monte-Carlo sampling from continuous ranges (grid only this phase; the
  grid runner is the foundation a sampler would build on).
- Sweeping leaf tools (non-recipe registry tools); `run_sweep` targets `WORKFLOW_NAMES`.
- Multi-metric collection in one sweep (one `metric` per call); nested/parallel sweeps;
  persisting the results table to CSV.
- 3-D+ structured plots (the histogram fallback covers >2 swept dimensions).
- Plus the standing deferred cleanups (carry-over task #14).
```
