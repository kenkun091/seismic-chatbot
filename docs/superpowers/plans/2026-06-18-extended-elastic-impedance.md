# Extended Elastic Impedance (EEI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `extended_elastic_impedance` tool (Whitcombe 2002 EEI(χ)) plus an EEI-vs-χ plot, registered and wired into the chatbot.

**Architecture:** A vectorized `extended_elastic_impedance` in `tools/avo_tools.py` computing `EEI(χ) = Vp0·ρ0·(Vp/Vp0)^p·(Vs/Vs0)^q·(ρ/ρ0)^r` (raw when no reference constants; Whitcombe-normalized when all three supplied), a companion plot, and registry + chatbot wiring mirroring the existing AVO-reflectivity ndarray/auto-plot pattern.

**Tech Stack:** Python, NumPy, Matplotlib (Agg), pytest. Work in `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`); run tests with `python -m pytest tests/ -q` from the package dir.

---

## File Structure

- `tools/avo_tools.py` — add `extended_elastic_impedance` and `plot_extended_elastic_impedance`. (Module already imports `numpy as np`, `matplotlib.pyplot as plt`, `warnings`, and `require_elastic_medium`/`angles_error` from `tools.physics_guards`.)
- `core/tool_registry.py` — two new `ToolSpec`s + extended import.
- `core/chatbot_tool_use.py` — `_is_image_output`, `_handle_automatic_chaining`, `_update_context`, system-prompt list.
- `tests/test_extended_elastic_impedance.py` — new test module.
- `tests/test_tool_registry.py` — bump count guard 18 → 20.
- `CLAUDE.md` — document under the AVO section.

---

## Task 1: `extended_elastic_impedance` core

**Files:**
- Modify: `tools/avo_tools.py` (add near the other AVO functions, e.g. after `shuey_reflectivity`)
- Test: `tests/test_extended_elastic_impedance.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extended_elastic_impedance.py`:

```python
import numpy as np
import pytest

from tools.avo_tools import extended_elastic_impedance


def test_chi_zero_is_acoustic_impedance():
    # At chi=0: p=1, q=0, r=1 -> EEI = Vp*rho (acoustic impedance), raw.
    vp, vs, rho = 3000.0, 1500.0, 2.3
    eei = extended_elastic_impedance(vp, vs, rho, chi=[0.0])
    assert np.isclose(eei[0], vp * rho)


def test_closed_form_at_chi_30():
    vp, vs, rho = 3000.0, 1500.0, 2.3
    chi_deg = 30.0
    x = np.radians(chi_deg)
    K = (vs / vp) ** 2
    p = np.cos(x) + np.sin(x)
    q = -8.0 * K * np.sin(x)
    r = np.cos(x) - 4.0 * K * np.sin(x)
    expected = vp ** p * vs ** q * rho ** r
    eei = extended_elastic_impedance(vp, vs, rho, chi=[chi_deg])
    assert np.isclose(eei[0], expected)


def test_eei_varies_with_chi():
    eei = extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[-45.0, 0.0, 45.0])
    assert eei.shape == (3,)
    assert not np.allclose(eei, eei[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extended_elastic_impedance.py -q`
Expected: FAIL with `ImportError: cannot import name 'extended_elastic_impedance'`.

- [ ] **Step 3: Implement `extended_elastic_impedance`**

Add to `tools/avo_tools.py` (after `shuey_reflectivity`):

```python
def extended_elastic_impedance(vp, vs, rho, chi,
                               vp0=None, vs0=None, rho0=None, k=None):
    """Extended Elastic Impedance, EEI(chi) (Whitcombe, 2002).

    EEI(chi) = Vp0*rho0 * (Vp/Vp0)^p * (Vs/Vs0)^q * (rho/rho0)^r
        p = cos(chi) + sin(chi);  q = -8K*sin(chi);  r = cos(chi) - 4K*sin(chi)
        K = (Vs/Vp)^2  (background; override with `k`)
    chi is the rotation angle in degrees. At chi=0, EEI = Vp*rho (acoustic impedance).
    Raw EEI (reference Vp0=Vs0=rho0=1) is returned unless ALL of vp0/vs0/rho0 are
    supplied, in which case Whitcombe normalization is applied.

    Args:
        vp, vs, rho: scalar layer P/S velocity (m/s) and density (g/cc).
        chi: array-like rotation angles in degrees (|chi| <= 90).
        vp0, vs0, rho0: optional reference constants (all-or-nothing).
        k: optional background (Vs/Vp)^2; default computed from vs/vp.

    Returns:
        np.ndarray of EEI values, one per chi.
    """
    require_elastic_medium(vp, vs, rho)

    refs = (vp0, vs0, rho0)
    n_set = sum(ref is not None for ref in refs)
    if n_set not in (0, 3):
        raise ValueError(
            "reference constants vp0/vs0/rho0 are all-or-nothing: supply all three "
            "(Whitcombe normalization) or none (raw EEI)."
        )
    if n_set == 3:
        for name, ref in (("vp0", vp0), ("vs0", vs0), ("rho0", rho0)):
            if ref <= 0:
                raise ValueError(f"{name} must be positive (got {ref})")
        rvp, rvs, rrho = float(vp0), float(vs0), float(rho0)
    else:
        rvp = rvs = rrho = 1.0

    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    if np.any(np.abs(chi) > 90):
        raise ValueError("chi (rotation angle) must be within [-90, 90] degrees")

    K = (vs / vp) ** 2 if k is None else float(k)
    x = np.radians(chi)
    p = np.cos(x) + np.sin(x)
    q = -8.0 * K * np.sin(x)
    r = np.cos(x) - 4.0 * K * np.sin(x)

    return rvp * rrho * (vp / rvp) ** p * (vs / rvs) ** q * (rho / rrho) ** r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extended_elastic_impedance.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/avo_tools.py tests/test_extended_elastic_impedance.py
git commit -m "feat(avo): extended_elastic_impedance EEI(chi) (Whitcombe 2002)"
```

---

## Task 2: Normalization, `k` override, and guards

**Files:**
- Test: `tests/test_extended_elastic_impedance.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extended_elastic_impedance.py`:

```python
def test_normalization_anchors_at_chi_zero_differs_elsewhere():
    vp, vs, rho = 3000.0, 1500.0, 2.3
    # A background reference different from the sample.
    ref = dict(vp0=2800.0, vs0=1400.0, rho0=2.2)
    chi = [0.0, 45.0]
    raw = extended_elastic_impedance(vp, vs, rho, chi=chi)
    norm = extended_elastic_impedance(vp, vs, rho, chi=chi, **ref)
    # chi=0: normalization scale is 1 -> same as raw (== Vp*rho).
    assert np.isclose(norm[0], raw[0])
    assert np.isclose(norm[0], vp * rho)
    # chi=45: reference rescales the value -> differs from raw.
    assert not np.isclose(norm[1], raw[1])


def test_k_override_changes_result():
    vp, vs, rho = 3000.0, 1500.0, 2.3
    default = extended_elastic_impedance(vp, vs, rho, chi=[45.0])
    overridden = extended_elastic_impedance(vp, vs, rho, chi=[45.0], k=0.1)
    assert not np.isclose(default[0], overridden[0])


def test_guards():
    # vs >= vp -> unphysical medium.
    with pytest.raises(ValueError):
        extended_elastic_impedance(2000.0, 2200.0, 2.3, chi=[0.0])
    # |chi| > 90.
    with pytest.raises(ValueError):
        extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[91.0])
    # Partial reference constants (only vp0).
    with pytest.raises(ValueError):
        extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[0.0], vp0=2800.0)
    # Non-positive reference constant.
    with pytest.raises(ValueError):
        extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=[0.0],
                                   vp0=2800.0, vs0=1400.0, rho0=-1.0)
```

- [ ] **Step 2: Run tests to verify they pass (Task 1 implementation already satisfies them)**

Run: `python -m pytest tests/test_extended_elastic_impedance.py -q`
Expected: 6 passed (3 from Task 1 + these 3).

> If `test_k_override_changes_result` is borderline: `k=0.1` vs the default `(1500/3000)²=0.25` changes `q`/`r`, so the χ=45° value must differ. If `test_normalization_anchors...` fails at χ=0, recheck that the reference factor cancels (it must, since p=1,q=0,r=1 at χ=0). Do NOT weaken assertions — a failure here means the implementation diverged from the spec; report it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_extended_elastic_impedance.py
git commit -m "test(avo): EEI normalization, k override, and guards"
```

---

## Task 3: `plot_extended_elastic_impedance`

**Files:**
- Modify: `tools/avo_tools.py` (add after `extended_elastic_impedance`)
- Test: `tests/test_extended_elastic_impedance.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extended_elastic_impedance.py`:

```python
import os

from tools.avo_tools import plot_extended_elastic_impedance


def test_eei_plot_returns_png_path():
    chi = np.linspace(-90, 90, 37)
    eei = extended_elastic_impedance(3000.0, 1500.0, 2.3, chi=chi)
    path = plot_extended_elastic_impedance(chi, eei)
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.exists(path)
    os.remove(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extended_elastic_impedance.py::test_eei_plot_returns_png_path -q`
Expected: FAIL with `ImportError: cannot import name 'plot_extended_elastic_impedance'`.

- [ ] **Step 3: Implement `plot_extended_elastic_impedance`**

Add to `tools/avo_tools.py` (after `extended_elastic_impedance`):

```python
def plot_extended_elastic_impedance(chi, eei, output_path=None):
    """Plot EEI vs rotation angle chi and return the PNG path.

    Marks chi=0 (where EEI equals the acoustic impedance) for reference.
    """
    import tempfile
    import os

    chi = np.asarray(chi, dtype=float)
    eei = np.asarray(eei, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chi, eei, "b-", linewidth=2, label="EEI(χ)")
    ax.axvline(0.0, color="k", linewidth=0.6, alpha=0.5, label="χ=0 (AI)")
    ax.set_xlabel("Rotation angle χ (degrees)")
    ax.set_ylabel("Extended Elastic Impedance")
    ax.set_title("Extended Elastic Impedance vs χ")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extended_elastic_impedance.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/avo_tools.py tests/test_extended_elastic_impedance.py
git commit -m "feat(avo): plot_extended_elastic_impedance (EEI vs chi)"
```

---

## Task 4: Register tools + chatbot wiring

**Files:**
- Modify: `core/tool_registry.py` (AVO import line; add two `ToolSpec`s after the `plot_avo_crossplot` spec)
- Modify: `core/chatbot_tool_use.py` (`_is_image_output`; `_handle_automatic_chaining`; `_update_context`; `_create_system_prompt`)
- Modify: `tests/test_tool_registry.py` (count guard 18 → 20)
- Test: `tests/test_extended_elastic_impedance.py` (append)

- [ ] **Step 1: Write the failing registry test**

Append to `tests/test_extended_elastic_impedance.py`:

```python
def test_eei_registered_and_chained():
    from core.tool_registry import REGISTRY_BY_NAME, AUTO_PLOT, TOOL_SCHEMAS

    assert "extended_elastic_impedance" in REGISTRY_BY_NAME
    assert "plot_extended_elastic_impedance" in REGISTRY_BY_NAME
    assert AUTO_PLOT.get("extended_elastic_impedance") == "plot_extended_elastic_impedance"
    spec = REGISTRY_BY_NAME["extended_elastic_impedance"]
    assert set(spec.required) == {"vp", "vs", "rho", "chi"}
    assert spec.validator is None
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "extended_elastic_impedance" in names and "plot_extended_elastic_impedance" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extended_elastic_impedance.py::test_eei_registered_and_chained -q`
Expected: FAIL (KeyError / assertion — not registered yet).

- [ ] **Step 3: Register the tools**

In `core/tool_registry.py`, extend the AVO import (currently `from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity, plot_avo_reflectivity, avo_attributes, plot_avo_crossplot`):

```python
from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity, plot_avo_reflectivity, avo_attributes, plot_avo_crossplot, extended_elastic_impedance, plot_extended_elastic_impedance
```

Add these two `ToolSpec`s to `REGISTRY`, immediately AFTER the existing `plot_avo_crossplot` spec:

```python
    ToolSpec(
        name="extended_elastic_impedance",
        fn=extended_elastic_impedance,
        description="Computes Extended Elastic Impedance EEI(χ) (Whitcombe 2002) for a layer across rotation angles χ in degrees. At χ=0 it equals the acoustic impedance Vp·ρ. Optionally Whitcombe-normalized when reference constants vp0/vs0/rho0 are all supplied. Auto-plots EEI vs χ.",
        params={
            "vp": {"type": "number", "description": "P-wave velocity of the layer in m/s."},
            "vs": {"type": "number", "description": "S-wave velocity of the layer in m/s."},
            "rho": {"type": "number", "description": "Density of the layer in g/cm³."},
            "chi": {"type": "array", "items": {"type": "number"}, "description": "Rotation angles χ in degrees (|χ| ≤ 90)."},
            "vp0": {"type": "number", "description": "Optional reference P-wave velocity (m/s) for Whitcombe normalization; supply with vs0 and rho0."},
            "vs0": {"type": "number", "description": "Optional reference S-wave velocity (m/s) for Whitcombe normalization; supply with vp0 and rho0."},
            "rho0": {"type": "number", "description": "Optional reference density (g/cm³) for Whitcombe normalization; supply with vp0 and vs0."},
            "k": {"type": "number", "description": "Optional background (Vs/Vp)² constant; defaults to (vs/vp)² of the layer."},
        },
        required=["vp", "vs", "rho", "chi"],
        defaults={},
        validator=None,
        auto_plot="plot_extended_elastic_impedance",
    ),
    ToolSpec(
        name="plot_extended_elastic_impedance",
        fn=plot_extended_elastic_impedance,
        description="Plots Extended Elastic Impedance EEI vs rotation angle χ.",
        params={
            "chi": {"type": "array", "items": {"type": "number"}, "description": "Rotation angles χ in degrees."},
            "eei": {"type": "array", "items": {"type": "number"}, "description": "EEI values, one per χ."},
        },
        required=["chi", "eei"],
        defaults={},
    ),
```

- [ ] **Step 4: Bump the registry count guard**

In `tests/test_tool_registry.py`, find `assert len(REGISTRY) == 18` and change `18` to `20`. (If the current number is not 18, STOP and report what it actually is — do not guess.)

- [ ] **Step 5: Run the registry test + count guard**

Run: `python -m pytest tests/test_extended_elastic_impedance.py::test_eei_registered_and_chained tests/test_tool_registry.py -q`
Expected: PASS.

- [ ] **Step 6: `_is_image_output` — add the EEI plot**

In `core/chatbot_tool_use.py`, in `_is_image_output`, add `"plot_extended_elastic_impedance"` to the plot-tool name list (it currently ends with `"plot_avo_crossplot"]`). Result:

```python
                tool_name in ["plot_ricker", "plot_wedge_model", "plot_avo_reflectivity", "plot_rock_properties", "plot_wedge_gather", "plot_avo_crossplot", "plot_extended_elastic_impedance"])
```

- [ ] **Step 7: `_handle_automatic_chaining` — add the EEI branch**

In `_handle_automatic_chaining`, add this branch BEFORE the final `else: return None` (it can go after the `avo_attributes` branch):

```python
            elif tool_name == "extended_elastic_impedance":
                if not (isinstance(tool_result, np.ndarray) and "chi" in tool_input):
                    return None
                plot_input = {"chi": tool_input["chi"], "eei": tool_result}
```

(The shared code after the if/elif chain calls `process_tool_call(plot_tool, plot_input)` and returns `{"image_path": ...}` — do not duplicate it.)

- [ ] **Step 8: `_update_context` — store last_eei**

In `_update_context`, add this branch (e.g. after the `avo_attributes` branch):

```python
            elif tool_name == "extended_elastic_impedance":
                if isinstance(tool_result, np.ndarray) and "chi" in tool_input:
                    self.context_manager.set_context("last_eei", {
                        "chi": tool_input["chi"],
                        "eei": tool_result,
                        "parameters": tool_input,
                    })
```

- [ ] **Step 9: System prompt — add the tool line**

In `_create_system_prompt`, after the `- avo_attributes: ...` line, add:

```
- extended_elastic_impedance: Extended Elastic Impedance EEI(χ) for a layer (AI at χ=0), with an EEI-vs-χ plot
```

Match the exact surrounding bullet format.

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (was 152; now 152 + new EEI tests; count guard green at 20). Only the pre-existing multi-angle wedge warning should appear.

- [ ] **Step 11: Commit**

```bash
git add core/tool_registry.py core/chatbot_tool_use.py tests/test_tool_registry.py tests/test_extended_elastic_impedance.py
git commit -m "feat(registry): register extended_elastic_impedance + plot, wire chaining"
```

---

## Task 5: Document the tool

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add documentation**

In `CLAUDE.md`, in the "## Gotchas specific to this package" section, immediately after the "**AVO interpretation attributes.**" bullet (added by the avo_attributes feature), add a new sibling bullet:

```markdown
- **Extended Elastic Impedance.** `tools/avo_tools.py::extended_elastic_impedance` computes Whitcombe (2002) EEI(χ) for a layer across rotation angles χ (deg): `EEI(χ)=Vp0·ρ0·(Vp/Vp0)^p·(Vs/Vs0)^q·(ρ/ρ0)^r` with `p=cosχ+sinχ, q=−8K·sinχ, r=cosχ−4K·sinχ, K=(Vs/Vp)²`. At χ=0 it equals AI (Vp·ρ). Raw by default; Whitcombe normalization is applied only when all three reference constants `vp0/vs0/rho0` are supplied (partial → `ValueError`). Returns an ndarray and auto-plots `plot_extended_elastic_impedance` (EEI vs χ). Subsumes Connolly (1999) EI(θ) via `tan χ = sin²θ`. Covered by `tests/test_extended_elastic_impedance.py`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document extended_elastic_impedance (EEI)"
```

---

## Self-Review Notes

- **Spec coverage:** core EEI formula + raw/normalized + K override (Task 1, Task 2); χ=0 AI anchor (Task 1 test); guards reject vs≥vp / |χ|>90 / partial refs / non-positive ref (Task 1 impl, Task 2 tests); plot EEI vs χ with χ=0 marker (Task 3); registry two specs + auto_plot + count bump (Task 4); chaining (chi key)/context/`_is_image_output`/system-prompt (Task 4 steps 6-9); docs incl. the Connolly subsumption note (Task 5). Single-layer scalar honored (no log/optimal-χ). All spec sections map to a task.
- **Placeholder scan:** none — every code step shows complete code.
- **Type consistency:** `extended_elastic_impedance` returns an `np.ndarray`; the chaining branch reads `tool_input["chi"]` + the returned ndarray and feeds `plot_extended_elastic_impedance(chi, eei)`; the plot signature is `(chi, eei, output_path=None)`; registry `required` for EEI is `{vp,vs,rho,chi}` matching the test. Reference-constant rule (all-or-nothing) implemented once in Task 1 and exercised in Task 2. Count guard 18 → 20 matches adding exactly two tools.
