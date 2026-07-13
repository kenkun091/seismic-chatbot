# Workflows Phase 2b — EEI Optimal-χ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the EEI rotation angle χ that best correlates with a target property over a log — as a raw-logs leaf tool (`eei_optimal_chi`) and a petrophysics-driven recipe (`eei_optimal_chi_petro`), over one shared science core.

**Architecture:** A private `_eei_chi_scan` computes the Whitcombe EEI log at each χ (scalar background K) and its Pearson correlation with the target, returning `χ* = argmax|r|`, the r-vs-χ curve, and the EEI log at χ\*. A shared `plot_eei_chi_scan` renders the r-vs-χ curve. The leaf tool (raw logs) and the recipe (predicts logs from porosity/clay via `calculate_rock_properties`) both call the core, self-plot, and return a dict containing `image_path` — so the chatbot's generic `_workflow_image_output` surfaces the plot with no per-tool wiring.

**Tech Stack:** Python 3.9+, NumPy, Matplotlib (existing headless plot convention), pytest. Run from inside `geo-mcp/seismic_chatbot`.

**Spec:** `docs/superpowers/specs/2026-06-20-eei-optimal-chi-design.md`.
**Builds on:** existing `tools/avo_tools.py::extended_elastic_impedance` (Whitcombe formula, scalar layer) and `require_elastic_medium`; `tools/rock_physics_tools.py::calculate_rock_properties` (shape-preserving tuple `(vp, vs, rhob, vp_vs, ai, si)`); the Phase 1 engine + chatbot generalizations. `REGISTRY` currently has 24 tools.

**Working dir for all commands:** `geo-mcp/seismic_chatbot` (its own git repo, branch `stabilize-tool-layer`).

## Verified contracts (from source — do not re-derive)

- `extended_elastic_impedance(vp, vs, rho, chi, vp0=None, vs0=None, rho0=None, k=None)` uses scalar `vp/vs/rho`, array `chi`; formula `p=cosχ+sinχ`, `q=−8K sinχ`, `r=cosχ−4K sinχ`, `K=(vs/vp)²`. The new core reimplements this **vectorized over a log with a scalar background K** (a different shape), reusing the formula — it does NOT call `extended_elastic_impedance`.
- `tools/physics_guards.py::require_elastic_medium(vp, vs, rho, label="medium")` raises `ValueError` for non-physical scalars (vp>0, rho>0, 0<vs<vp). Pure floats — call per-sample or replicate the checks vectorized.
- `calculate_rock_properties(phit, vclay, fluid_type='water', print_results=True)` → tuple `(vp, vs, rhob, vp_vs_ratio, ai, si)`, **shape-preserving** (array in → array out). Use `print_results=False`.
- Plot convention: `output_path=None` → `tempfile.mkstemp(suffix=".png")` + `os.close(fd)`; `fig.savefig(output_path, dpi=300, bbox_inches="tight")`; `plt.close(fig)`; `return output_path`.
- Registry: append a `ToolSpec` to `REGISTRY` (`core/tool_registry.py`); all derivations (`REGISTRY_BY_NAME`/`TOOL_SCHEMAS`/`TOOL_FUNCTIONS`/`AUTO_PLOT`) follow. Workflows are declared in `workflows/engine.py::WORKFLOW_REGISTRY` and converted automatically.
- Chatbot: `_workflow_image_output(tool_result)` returns `{"image_path": ...}` for ANY dict whose `image_path` is a `.png` string (not gated on tool name) — so a self-plotting leaf tool or recipe is surfaced with no edits. `_update_context` caches `tool_name in WORKFLOW_NAMES`. `_create_system_prompt` has a hardcoded bullet list.

---

## File Structure

- `tools/avo_tools.py` — modify: add `_eei_chi_scan`, `plot_eei_chi_scan`, `eei_optimal_chi`.
- `core/tool_registry.py` — modify: import + one `ToolSpec` for `eei_optimal_chi`.
- `workflows/recipes/eei_optimal_chi_petro.py` — new: the petrophysics recipe.
- `workflows/engine.py` — modify: import + one `WorkflowSpec`.
- `core/chatbot_tool_use.py` — modify: two system-prompt bullets.
- Tests: `tests/test_eei_optimal_chi.py` (core + leaf tool), `tests/test_eei_optimal_chi_petro.py` (recipe), and appends to `tests/test_tool_registry.py` (count 24→26), `tests/test_workflow_meta_tool.py`, `tests/test_chatbot_workflow.py`.

The 6 tasks: (1) core `_eei_chi_scan`; (2) `plot_eei_chi_scan` + leaf tool `eei_optimal_chi`; (3) register leaf tool; (4) petro recipe; (5) register recipe; (6) system-prompt bullets + full suite.

---

### Task 1: `_eei_chi_scan` science core

**Files:**
- Modify: `tools/avo_tools.py` (append the private core; reuse the existing `np` and `require_elastic_medium` imports already in the file)
- Create: `tests/test_eei_optimal_chi.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eei_optimal_chi.py`:

```python
import numpy as np
import pytest

from tools.avo_tools import _eei_chi_scan


def _logs(n=40, seed=0):
    rng = np.random.RandomState(seed)
    vp = 3000.0 + 800.0 * rng.rand(n)
    vs = vp / 2.0 + 50.0 * rng.rand(n)   # keeps 0 < vs < vp
    rho = 2.2 + 0.3 * rng.rand(n)
    return vp, vs, rho


def test_scan_ai_target_recovers_chi_zero():
    # At chi=0, EEI = Vp*rho (acoustic impedance). With target = Vp*rho, the
    # correlation peaks at chi=0 with |r| = 1.
    vp, vs, rho = _logs()
    target = vp * rho
    res = _eei_chi_scan(vp, vs, rho, target, np.arange(-90.0, 90.0 + 1.0, 1.0))
    assert abs(res["optimal_chi"]) <= 1.0
    assert np.isclose(abs(res["max_correlation"]), 1.0, atol=1e-6)
    assert len(res["chi"]) == len(res["correlation"])
    assert len(res["eei_optimal"]) == len(vp)


def test_scan_is_shift_scale_invariant():
    # Pearson r is invariant to affine transforms of the target -> same optimal chi.
    vp, vs, rho = _logs(seed=3)
    target = vp * rho
    chi = np.arange(-90.0, 90.0 + 1.0, 1.0)
    a = _eei_chi_scan(vp, vs, rho, target, chi)["optimal_chi"]
    b = _eei_chi_scan(vp, vs, rho, 5.0 * target + 17.0, chi)["optimal_chi"]
    assert a == b


def test_scan_constant_target_raises():
    vp, vs, rho = _logs()
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, np.ones_like(vp), np.arange(-90.0, 91.0, 1.0))


def test_scan_length_mismatch_raises():
    vp, vs, rho = _logs()
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, (vp * rho)[:-1], np.arange(-90.0, 91.0, 1.0))


def test_scan_nonphysical_sample_raises():
    vp, vs, rho = _logs()
    vs = vs.copy()
    vs[0] = vp[0] + 1.0   # vs >= vp -> non-physical
    with pytest.raises(ValueError):
        _eei_chi_scan(vp, vs, rho, vp * rho, np.arange(-90.0, 91.0, 1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eei_optimal_chi.py -v`
Expected: FAIL with `ImportError: cannot import name '_eei_chi_scan' from 'tools.avo_tools'`

- [ ] **Step 3: Implement the core**

Append to `tools/avo_tools.py` (uses the existing module-level `numpy as np`; no new imports needed — physical validity is checked vectorized inline, not via `require_elastic_medium`):

```python
def _eei_chi_scan(vp, vs, rho, target, chi, k=None):
    """Scan rotation angle chi for the EEI projection best correlated with a target log.

    EEI(chi) over a log (Whitcombe 2002), with a single SCALAR background K so chi has
    a consistent meaning across the interval. Returns chi*, the Pearson r vs chi curve,
    the signed correlation at chi*, and the EEI log at chi*. Raw (un-normalized) EEI is
    used: Pearson r is scale-invariant, so normalization is unnecessary.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    target = np.asarray(target, dtype=float)
    chi = np.atleast_1d(np.asarray(chi, dtype=float))

    if not (vp.shape == vs.shape == rho.shape == target.shape) or vp.ndim != 1:
        raise ValueError("vp, vs, rho, target must be 1-D logs of equal length")
    if vp.size < 2:
        raise ValueError("logs must have at least 2 samples to correlate")
    if chi.size == 0:
        raise ValueError("chi sweep is empty")
    if np.any(np.abs(chi) > 90):
        raise ValueError("chi (rotation angle) must be within [-90, 90] degrees")
    # Per-sample physical validity (vp>0, rho>0, 0<vs<vp).
    if np.any(vp <= 0) or np.any(rho <= 0) or np.any(vs <= 0) or np.any(vs >= vp):
        raise ValueError("non-physical elastic sample: require vp>0, rho>0, 0<vs<vp")
    if np.std(target) == 0:
        raise ValueError("target log has zero variance; cannot correlate")

    K = float(np.mean((vs / vp) ** 2)) if k is None else float(k)
    x = np.radians(chi)
    # exponents per chi (1-D, length n_chi)
    p = np.cos(x) + np.sin(x)
    q = -8.0 * K * np.sin(x)
    r = np.cos(x) - 4.0 * K * np.sin(x)

    # EEI log per chi: shape (n_samples, n_chi) via outer broadcasting.
    log_eei = (np.log(vp)[:, None] * p[None, :]
               + np.log(vs)[:, None] * q[None, :]
               + np.log(rho)[:, None] * r[None, :])
    eei = np.exp(log_eei)  # (n_samples, n_chi)

    t = target - target.mean()
    t_norm = np.sqrt(np.sum(t ** 2))
    e = eei - eei.mean(axis=0, keepdims=True)
    e_norm = np.sqrt(np.sum(e ** 2, axis=0))
    e_norm = np.where(e_norm == 0, np.nan, e_norm)  # guard flat EEI columns
    corr = (t @ e) / (t_norm * e_norm)  # Pearson r per chi, shape (n_chi,)

    best = int(np.nanargmax(np.abs(corr)))
    return {
        "chi": [float(c) for c in chi],
        "correlation": [float(c) for c in corr],
        "optimal_chi": float(chi[best]),
        "max_correlation": float(corr[best]),
        "eei_optimal": [float(v) for v in eei[:, best]],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eei_optimal_chi.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/avo_tools.py tests/test_eei_optimal_chi.py
git commit -m "feat(avo): _eei_chi_scan core (EEI-vs-target chi correlation)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `plot_eei_chi_scan` + leaf tool `eei_optimal_chi`

**Files:**
- Modify: `tools/avo_tools.py` (append plot + leaf tool)
- Modify: `tests/test_eei_optimal_chi.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eei_optimal_chi.py`:

```python
import os

from tools.avo_tools import eei_optimal_chi


def test_eei_optimal_chi_tool_ai_target():
    vp, vs, rho = _logs(seed=1)
    res = eei_optimal_chi(
        vp.tolist(), vs.tolist(), rho.tolist(), (vp * rho).tolist(),
        chi_min=-90, chi_max=90, chi_step=1,
    )
    assert abs(res["optimal_chi"]) <= 1.0
    assert np.isclose(abs(res["max_correlation"]), 1.0, atol=1e-6)
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eei_optimal_chi.py::test_eei_optimal_chi_tool_ai_target -v`
Expected: FAIL with `ImportError: cannot import name 'eei_optimal_chi'`

- [ ] **Step 3: Implement the plot and the leaf tool**

Append to `tools/avo_tools.py`. (The module already imports `numpy as np` and
`matplotlib.pyplot as plt`; add `import os` / `import tempfile` at the top of the
file if not already present — `plot_extended_elastic_impedance` imports them inside
its body, so add module-level `import os`/`import tempfile` near the other imports.)

```python
def plot_eei_chi_scan(chi, correlation, optimal_chi, output_path=None):
    """Plot Pearson correlation vs rotation angle chi, marking the optimal chi."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    chi = np.asarray(chi, dtype=float)
    correlation = np.asarray(correlation, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chi, correlation, "b-", linewidth=2, label="Pearson r(χ)")
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.axvline(optimal_chi, color="r", lw=1.2, ls="--",
               label=f"χ* = {optimal_chi:.1f}°")
    ax.set_xlabel("Rotation angle χ (degrees)")
    ax.set_ylabel("Correlation with target")
    ax.set_title("EEI–target correlation vs χ")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def eei_optimal_chi(vp, vs, rho, target, chi_min=-90.0, chi_max=90.0,
                    chi_step=1.0, k=None):
    """Find the EEI rotation angle chi best correlated with a target log (raw-logs mode).

    Sweeps chi in [chi_min, chi_max] (step chi_step), correlates EEI(chi) against the
    target log, and returns chi*, the correlation curve, the EEI log at chi*, and a
    correlation-vs-chi plot path.
    """
    chi = np.arange(chi_min, chi_max + chi_step, chi_step)
    result = _eei_chi_scan(vp, vs, rho, target, chi, k=k)
    result["image_path"] = plot_eei_chi_scan(
        result["chi"], result["correlation"], result["optimal_chi"]
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eei_optimal_chi.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/avo_tools.py tests/test_eei_optimal_chi.py
git commit -m "feat(avo): plot_eei_chi_scan + eei_optimal_chi leaf tool (raw logs)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Register `eei_optimal_chi` as a leaf tool

**Files:**
- Modify: `core/tool_registry.py` (import + `ToolSpec`)
- Modify: `tests/test_tool_registry.py` (count 24 → 25)
- Create: `tests/test_eei_optimal_chi_tool_registration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eei_optimal_chi_tool_registration.py`:

```python
import numpy as np

from core import tool_registry as reg
from core.tool_manager import ToolManager


def test_eei_optimal_chi_registered():
    assert "eei_optimal_chi" in reg.REGISTRY_BY_NAME
    assert "eei_optimal_chi" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"eei_optimal_chi"}
    assert reg.REGISTRY_BY_NAME["eei_optimal_chi"].auto_plot is None  # self-plots


def test_eei_optimal_chi_runs_through_tool_manager():
    n = 30
    vp = list(3000.0 + 10.0 * np.arange(n))
    vs = [v / 2.0 for v in vp]
    rho = list(2.2 + 0.01 * np.arange(n))
    target = [a * b for a, b in zip(vp, rho)]  # acoustic impedance
    tm = ToolManager()
    res = tm.execute_tool("eei_optimal_chi", {
        "vp": vp, "vs": vs, "rho": rho, "target": target,
    })
    assert abs(res["optimal_chi"]) <= 1.0
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eei_optimal_chi_tool_registration.py -v`
Expected: FAIL — `eei_optimal_chi` not in `reg.REGISTRY_BY_NAME`.

- [ ] **Step 3: Register the leaf tool**

In `core/tool_registry.py`, add `eei_optimal_chi` to the existing avo_tools import line:

```python
from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity, plot_avo_reflectivity, avo_attributes, plot_avo_crossplot, extended_elastic_impedance, plot_extended_elastic_impedance, eei_optimal_chi
```

Then add this `ToolSpec` to the `REGISTRY` list (place it after the
`plot_extended_elastic_impedance` spec, before the `calculate_rock_properties` spec):

```python
    ToolSpec(
        name="eei_optimal_chi",
        fn=eei_optimal_chi,
        description="Finds the Extended Elastic Impedance rotation angle chi whose EEI log best correlates with a target property log (e.g. Vclay, Sw, porosity, or an acoustic-impedance log). Takes Vp, Vs, density, and target as equal-length logs; sweeps chi and returns the optimal chi, the correlation-vs-chi curve, the EEI log at the optimal chi, and a plot.",
        params={
            "vp": {"type": "array", "items": {"type": "number"}, "description": "P-wave velocity log in m/s."},
            "vs": {"type": "array", "items": {"type": "number"}, "description": "S-wave velocity log in m/s."},
            "rho": {"type": "array", "items": {"type": "number"}, "description": "Bulk density log in g/cm³."},
            "target": {"type": "array", "items": {"type": "number"}, "description": "Target property log to correlate against (same length as the elastic logs)."},
            "chi_min": {"type": "number", "description": "Minimum rotation angle in degrees (default -90)."},
            "chi_max": {"type": "number", "description": "Maximum rotation angle in degrees (default 90)."},
            "chi_step": {"type": "number", "description": "Rotation-angle step in degrees (default 1)."},
            "k": {"type": "number", "description": "Optional background (Vs/Vp)² constant; default is the mean over the log."},
        },
        required=["vp", "vs", "rho", "target"],
        defaults={"chi_min": -90.0, "chi_max": 90.0, "chi_step": 1.0},
        validator=None,
        auto_plot=None,
    ),
```

- [ ] **Step 4: Bump the registry count**

In `tests/test_tool_registry.py`, change `assert len(reg.REGISTRY) == 24` to:

```python
    assert len(reg.REGISTRY) == 25
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_eei_optimal_chi_tool_registration.py tests/test_tool_registry.py -v`
Expected: PASS. Also confirm `python -c "from core.tool_registry import REGISTRY; print(len(REGISTRY))"` prints 25.

- [ ] **Step 6: Commit**

```bash
git add core/tool_registry.py tests/test_eei_optimal_chi_tool_registration.py tests/test_tool_registry.py
git commit -m "feat(avo): register eei_optimal_chi leaf tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `eei_optimal_chi_petro` recipe

**Files:**
- Create: `workflows/recipes/eei_optimal_chi_petro.py`
- Create: `tests/test_eei_optimal_chi_petro.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eei_optimal_chi_petro.py`:

```python
import os
import numpy as np
import pytest

from workflows.recipes.eei_optimal_chi_petro import eei_optimal_chi_petro


def _petro_logs(n=40, seed=0):
    rng = np.random.RandomState(seed)
    phit = 0.05 + 0.25 * rng.rand(n)
    vclay = 0.50 * rng.rand(n)
    return phit.tolist(), vclay.tolist()


def test_petro_recipe_vclay_target():
    phit, vclay = _petro_logs()
    res = eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine")
    assert {"optimal_chi", "max_correlation", "chi", "correlation",
            "eei_optimal", "target", "image_path"} <= set(res)
    assert res["target"] == "vclay"
    assert -90.0 <= res["optimal_chi"] <= 90.0
    assert abs(res["max_correlation"]) <= 1.0 + 1e-9
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)


def test_petro_recipe_phit_target_runs():
    phit, vclay = _petro_logs(seed=2)
    res = eei_optimal_chi_petro(phit, vclay, target="phit")
    assert res["target"] == "phit"
    os.remove(res["image_path"])


def test_petro_recipe_rejects_bad_target():
    phit, vclay = _petro_logs()
    with pytest.raises(ValueError):
        eei_optimal_chi_petro(phit, vclay, target="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eei_optimal_chi_petro.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.recipes.eei_optimal_chi_petro'`

- [ ] **Step 3: Write the recipe**

Create `workflows/recipes/eei_optimal_chi_petro.py`:

```python
"""eei_optimal_chi_petro: EEI optimal-chi from petrophysical logs.

Predict Vp/Vs/density logs from porosity and clay-volume logs (rock physics), then
find the EEI rotation angle chi whose EEI log best correlates with a chosen
petrophysical target (Vclay for lithology, or porosity). Wraps the shared
_eei_chi_scan core; self-plots via plot_eei_chi_scan.
"""
import numpy as np

from tools.rock_physics_tools import calculate_rock_properties
from tools.avo_tools import _eei_chi_scan, plot_eei_chi_scan


_TARGETS = {"vclay", "phit"}


def eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine",
                          chi_min=-90.0, chi_max=90.0, chi_step=1.0):
    """Find optimal EEI chi against a petrophysical target, from porosity/clay logs."""
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {sorted(_TARGETS)} (got {target!r})")

    phit = np.asarray(phit, dtype=float)
    vclay = np.asarray(vclay, dtype=float)
    vp, vs, rhob, *_ = calculate_rock_properties(
        phit, vclay, fluid_type=fluid, print_results=False
    )
    target_log = vclay if target == "vclay" else phit

    chi = np.arange(chi_min, chi_max + chi_step, chi_step)
    result = _eei_chi_scan(vp, vs, rhob, target_log, chi)
    result["target"] = target
    result["fluid"] = fluid
    result["image_path"] = plot_eei_chi_scan(
        result["chi"], result["correlation"], result["optimal_chi"]
    )
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eei_optimal_chi_petro.py -v`
Expected: PASS (3 passed)

(If `calculate_rock_properties` warns about clay/porosity clipping, that is expected
and not a failure. If the AI-style assertions are unstable due to the random logs,
they are not used here — the petro tests only check structure/targets, so a BLOCKED
report is only warranted if an unexpected exception is raised.)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/eei_optimal_chi_petro.py tests/test_eei_optimal_chi_petro.py
git commit -m "feat(workflows): eei_optimal_chi_petro recipe (optimal chi from petrophysics)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Register `eei_optimal_chi_petro` in the workflow engine

**Files:**
- Modify: `workflows/engine.py` (import + `WorkflowSpec`)
- Modify: `tests/test_tool_registry.py` (count 25 → 26)
- Modify: `tests/test_workflow_meta_tool.py` (append a case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_meta_tool.py` (`reg` and `ToolManager` already imported):

```python
def test_eei_optimal_chi_petro_is_registered_meta_tool():
    assert "eei_optimal_chi_petro" in reg.REGISTRY_BY_NAME
    assert "eei_optimal_chi_petro" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"eei_optimal_chi_petro"}


def test_eei_optimal_chi_petro_runs_through_tool_manager():
    import numpy as np
    n = 30
    phit = list(0.10 + 0.002 * np.arange(n))
    vclay = list(0.10 + 0.01 * np.arange(n))
    tm = ToolManager()
    res = tm.execute_tool("eei_optimal_chi_petro", {"phit": phit, "vclay": vclay})
    assert res["target"] == "vclay"  # default
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_meta_tool.py -k eei_optimal_chi_petro -v`
Expected: FAIL — not in `reg.REGISTRY_BY_NAME`.

- [ ] **Step 3: Register the recipe**

In `workflows/engine.py`, add the import after the existing recipe imports (after
`from workflows.recipes.tuning import tuning`):

```python
from workflows.recipes.eei_optimal_chi_petro import eei_optimal_chi_petro
```

Add this `WorkflowSpec` to `WORKFLOW_REGISTRY` (after the `tuning` spec, before `]`):

```python
    WorkflowSpec(
        name="eei_optimal_chi_petro",
        fn=eei_optimal_chi_petro,
        description=(
            "EEI optimal-rotation-angle analysis from petrophysics: predict Vp/Vs/density "
            "logs from porosity and clay-volume logs, then find the Extended Elastic "
            "Impedance angle chi whose EEI log best correlates with a chosen target "
            "(Vclay for lithology, or porosity). Returns the optimal chi, the "
            "correlation-vs-chi curve, the EEI log at the optimal chi, and a plot."
        ),
        params={
            "phit": {"type": "array", "items": {"type": "number"}, "description": "Porosity log (fraction, 0-1)."},
            "vclay": {"type": "array", "items": {"type": "number"}, "description": "Clay-volume log (fraction, 0-1)."},
            "target": {"type": "string", "description": "Target property to correlate against: 'vclay' (default) or 'phit'."},
            "fluid": {"type": "string", "description": "Pore fluid for the rock-physics prediction: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
            "chi_min": {"type": "number", "description": "Minimum rotation angle in degrees (default -90)."},
            "chi_max": {"type": "number", "description": "Maximum rotation angle in degrees (default 90)."},
            "chi_step": {"type": "number", "description": "Rotation-angle step in degrees (default 1)."},
        },
        required=["phit", "vclay"],
        defaults={"target": "vclay", "fluid": "brine", "chi_min": -90.0, "chi_max": 90.0, "chi_step": 1.0},
        auto_plot=None,
    ),
```

- [ ] **Step 4: Bump the registry count**

In `tests/test_tool_registry.py`, change `assert len(reg.REGISTRY) == 25` to:

```python
    assert len(reg.REGISTRY) == 26
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_workflow_meta_tool.py tests/test_tool_registry.py tests/test_workflow_engine.py -v`
Expected: PASS. Also confirm `python -c "from core.tool_registry import REGISTRY; print(len(REGISTRY))"` prints 26.

- [ ] **Step 6: Commit**

```bash
git add workflows/engine.py tests/test_workflow_meta_tool.py tests/test_tool_registry.py
git commit -m "feat(workflows): register eei_optimal_chi_petro meta-tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: System-prompt bullets; full-suite check

**Files:**
- Modify: `core/chatbot_tool_use.py` (two system-prompt bullets)
- Modify: `tests/test_chatbot_workflow.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatbot_workflow.py` (reuses the `bot` fixture):

```python
def test_system_prompt_lists_eei_optimal_chi(bot):
    prompt = bot._create_system_prompt()
    assert "- eei_optimal_chi:" in prompt
    assert "- eei_optimal_chi_petro:" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_workflow.py -k eei_optimal_chi -v`
Expected: FAIL — neither bullet present yet.

- [ ] **Step 3: Add the two bullets**

In `core/chatbot_tool_use.py`, in `_create_system_prompt`'s hardcoded "Available
tools:" bullet list, add these two bullets after the `tuning` bullet (match the
neighboring bullet formatting exactly):

```
- eei_optimal_chi: Finds the Extended Elastic Impedance rotation angle χ whose EEI log best correlates with a target property log (Vp, Vs, density, and target supplied as logs); returns the optimal χ, the correlation-vs-χ curve, and a plot.
- eei_optimal_chi_petro: EEI optimal-χ from petrophysics — predicts Vp/Vs/density logs from porosity & clay-volume logs, then finds the χ whose EEI best correlates with Vclay (lithology) or porosity, with a plot.
```

(No other chatbot changes: both entry points return a dict with `image_path`, so
`_workflow_image_output` surfaces the plot; the recipe auto-caches via `WORKFLOW_NAMES`.)

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: PASS (the prior chatbot-workflow tests plus the new one)

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: the new Phase 2b tests pass and nothing else regressed. The standalone
`test_tool_use.py::test_tool_use_pattern` (stdin) is a KNOWN pre-existing failure —
if ONLY that fails, it is expected. If any OTHER test fails, STOP and report BLOCKED.

- [ ] **Step 6: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(workflows): list eei_optimal_chi(+petro) in chatbot system prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `_eei_chi_scan` correctly returns `χ*=argmax|r|`, the r-vs-χ curve, and the EEI log at χ\*, with `target=Vp·ρ` → `χ*≈0`, `|r|≈1`.
- `eei_optimal_chi` (leaf tool, raw logs) and `eei_optimal_chi_petro` (recipe, petrophysics) both run via `ToolManager`, return a dict with `image_path`, and are registered (registry count 26).
- The chatbot surfaces both plots via the generic `_workflow_image_output`, the recipe caches `last_workflow_result`, and both names appear in the system prompt — with no per-tool chatbot hardcoding.
- Full suite green (modulo the pre-existing stdin-script failure).

## Spec coverage

- Shared core `_eei_chi_scan` (argmax|r|, scalar background K) → Task 1.
- `plot_eei_chi_scan` + leaf tool `eei_optimal_chi` (raw logs) → Tasks 2–3.
- `eei_optimal_chi_petro` recipe (petrophysics, Vclay/φ targets) → Tasks 4–5.
- Chatbot exposure → Task 6 (image surfacing + caching come free).
- AI known-answer + invariance + guard tests → Tasks 1–5.

## Not in this plan

- Continuous Sw target (Phase 3, gap S1); cross-well sweep (Phase 4, gap S3);
  the EEI(χ*)-vs-target scatter panel; LAS ingestion. Plus the standing deferred
  cleanups (carry-over task list).
