# AVO Attributes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `avo_attributes` tool (intercept A, gradient B, AVO class I–IV) and a `plot_avo_crossplot` companion, derived from the existing Shuey coefficients.

**Architecture:** Extract a pure `_shuey_coefficients` helper from `shuey_reflectivity` (so intercept/gradient are identical to the AVO curve), build `avo_attributes` + a `_classify_avo` helper on top, add a quadrant-shaded crossplot, and wire both into the registry and chatbot auto-chaining.

**Tech Stack:** Python, NumPy, Matplotlib (Agg), pytest. Work in `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`); run tests with `python -m pytest tests/ -q` from the package dir.

---

## File Structure

- `tools/avo_tools.py` — add `_shuey_coefficients`, `_classify_avo`, `avo_attributes`, `plot_avo_crossplot`; refactor `shuey_reflectivity` to use the helper. (Module already imports `numpy as np`, `matplotlib.pyplot as plt`, `warnings`, and `require_elastic_medium`/`angles_error`.)
- `core/tool_registry.py` — two new `ToolSpec`s + extended import.
- `core/chatbot_tool_use.py` — auto-chain + context + `_is_image_output` + system-prompt list.
- `tests/test_avo_attributes.py` — new test module.
- `tests/test_tool_registry.py` — bump the registry count guard (16 → 18).
- `CLAUDE.md` — document under the AVO section.

---

## Task 1: Extract `_shuey_coefficients` and refactor `shuey_reflectivity`

**Files:**
- Modify: `tools/avo_tools.py` (add helper above `shuey_reflectivity` at line 55; refactor its body at lines 74-84)
- Test: `tests/test_avo_attributes.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_avo_attributes.py`:

```python
import numpy as np

from tools.avo_tools import _shuey_coefficients, shuey_reflectivity


def test_shuey_coefficients_intercept_matches_closed_form():
    args = dict(vp1=2400, vs1=1200, rho1=2.35, vp2=2000, vs2=1300, rho2=2.0)
    R0, G, F = _shuey_coefficients(**args)
    d_vp, d_rho = args["vp2"] - args["vp1"], args["rho2"] - args["rho1"]
    avg_vp, avg_rho = 0.5 * (args["vp1"] + args["vp2"]), 0.5 * (args["rho1"] + args["rho2"])
    assert np.isclose(R0, 0.5 * (d_vp / avg_vp + d_rho / avg_rho))
    # Intercept == zero-angle reflectivity.
    assert np.isclose(shuey_reflectivity(angles=[0.0], **args)[0], R0)


def test_shuey_reflectivity_unchanged_by_refactor():
    args = dict(vp1=2400, vs1=1200, rho1=2.35, vp2=3000, vs2=1500, rho2=2.4)
    angles = [0.0, 10.0, 20.0, 30.0]
    R0, G, F = _shuey_coefficients(**args)
    th = np.radians(angles)
    expected = R0 + G * np.sin(th) ** 2 + F * (np.tan(th) ** 2 - np.sin(th) ** 2)
    got = shuey_reflectivity(angles=angles, **args)
    assert np.allclose(got, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_avo_attributes.py -q`
Expected: FAIL with `ImportError: cannot import name '_shuey_coefficients'`.

- [ ] **Step 3: Add the helper and refactor `shuey_reflectivity`**

In `tools/avo_tools.py`, add this function immediately above `shuey_reflectivity` (before line 55):

```python
def _shuey_coefficients(vp1, vs1, rho1, vp2, vs2, rho2):
    """Shuey three-term coefficients: intercept R0, gradient G, curvature F.

    Pure (no guards, no angle). R0 is the normal-incidence reflectivity (AVO
    intercept) and G is the AVO gradient. Shared by shuey_reflectivity and
    avo_attributes so the two never diverge.
    """
    d_vp = vp2 - vp1
    d_vs = vs2 - vs1
    d_rho = rho2 - rho1
    avg_vp = 0.5 * (vp1 + vp2)
    avg_vs = 0.5 * (vs1 + vs2)
    avg_rho = 0.5 * (rho1 + rho2)
    R0 = 0.5 * (d_vp / avg_vp + d_rho / avg_rho)
    G = 0.5 * d_vp / avg_vp - 2 * (avg_vs ** 2 / avg_vp ** 2) * (d_rho / avg_rho + 2 * d_vs / avg_vs)
    F = 0.5 * d_vp / avg_vp
    return R0, G, F
```

Then in `shuey_reflectivity`, replace the inline coefficient block (currently lines 74-83, from the `# Shuey coefficients` comment through the `F = 0.5 * d_vp / avg_vp` line) with:

```python
    # Shuey coefficients (intercept R0, gradient G, curvature F).
    R0, G, F = _shuey_coefficients(vp1, vs1, rho1, vp2, vs2, rho2)
```

Leave the surrounding code unchanged: the guards/angle handling above it, and the
`rc = R0 + G * np.sin(angles) ** 2 + F * (np.tan(angles) ** 2 - np.sin(angles) ** 2)`
line plus `return rc` below it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_avo_attributes.py -q`
Expected: 2 passed.

- [ ] **Step 5: Run the AVO regression tests**

Run: `python -m pytest tests/test_tools.py -q`
Expected: all pass (the existing `test_zoeppritz_vs_shuey_small_angles` etc. still hold — the refactor is behavior-preserving).

- [ ] **Step 6: Commit**

```bash
git add tools/avo_tools.py tests/test_avo_attributes.py
git commit -m "refactor(avo): extract _shuey_coefficients helper (DRY intercept/gradient)"
```

---

## Task 2: `avo_attributes` + `_classify_avo`

**Files:**
- Modify: `tools/avo_tools.py` (add after `_shuey_coefficients` / near the other AVO functions)
- Test: `tests/test_avo_attributes.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_avo_attributes.py`:

```python
import pytest

from tools.avo_tools import avo_attributes


def test_intercept_gradient_match_helper():
    args = dict(vp1=2400, vs1=1200, rho1=2.35, vp2=2000, vs2=1300, rho2=2.0)
    R0, G, _ = _shuey_coefficients(**args)
    res = avo_attributes(**args)
    assert np.isclose(res["intercept"], R0)
    assert np.isclose(res["gradient"], G)


def test_class_iii_gas_sand():
    # Shale over gas sand: Vp and rho both drop -> A<0; gradient B<0 -> Class III.
    res = avo_attributes(vp1=2400, vs1=1100, rho1=2.35, vp2=2000, vs2=1250, rho2=2.0)
    assert res["intercept"] < 0 and res["gradient"] < 0
    assert res["avo_class"] == "III"


def test_class_i_hard_event():
    # Soft shale over hard limestone: A>0, B<0 -> Class I.
    res = avo_attributes(vp1=2500, vs1=1200, rho1=2.3, vp2=4000, vs2=2200, rho2=2.55)
    assert res["intercept"] > 0 and res["gradient"] < 0
    assert res["avo_class"] == "I"


def test_class_iv_soft_sand_low_shear():
    # Hard cap over soft gas sand with lower Vs: A<0, B>0 -> Class IV.
    res = avo_attributes(vp1=3000, vs1=1700, rho1=2.4, vp2=2600, vs2=1100, rho2=2.15)
    assert res["intercept"] < 0 and res["gradient"] > 0
    assert res["avo_class"] == "IV"


def test_class_ii_near_zero_intercept():
    # Tuned so |A| <= 0.02 -> Class II.
    res = avo_attributes(vp1=2500, vs1=1200, rho1=2.30, vp2=2560, vs2=1250, rho2=2.28)
    assert abs(res["intercept"]) <= 0.02
    assert res["avo_class"] in ("II", "IIp")


def test_avo_attributes_rejects_unphysical_medium():
    with pytest.raises(ValueError):
        avo_attributes(vp1=2000, vs1=2200, rho1=2.3, vp2=2500, vs2=1200, rho2=2.4)  # vs1>=vp1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_avo_attributes.py -q`
Expected: FAIL with `ImportError: cannot import name 'avo_attributes'`.

- [ ] **Step 3: Implement `_classify_avo` and `avo_attributes`**

In `tools/avo_tools.py`, add (after `_shuey_coefficients`):

```python
_CLASS_II_INTERCEPT = 0.02  # |intercept| band that defines a Class-II response

_AVO_CLASS_DESCRIPTIONS = {
    "I": "High-impedance contrast; amplitude dims (and may reverse polarity) with offset.",
    "I*": "Atypical: positive intercept with non-negative gradient (amplitude rises with offset).",
    "II": "Near-zero intercept; weak amplitude with a phase/polarity reversal with offset.",
    "IIp": "Near-zero negative intercept; polarity reversal with offset.",
    "III": "Classic bright spot (e.g. gas sand); amplitude brightens with offset.",
    "IV": "Bright spot whose amplitude magnitude decreases with offset.",
}


def _classify_avo(intercept, gradient):
    """Return (class_label, description) from the AVO intercept and gradient.

    Rutherford & Williams (1989) / Castagna & Swan (1997), on the signs of
    A (intercept) and B (gradient) with a near-zero-intercept band for Class II.
    """
    A, B = intercept, gradient
    if abs(A) <= _CLASS_II_INTERCEPT:
        cls = "IIp" if A < 0 else "II"
    elif A > 0 and B < 0:
        cls = "I"
    elif A < 0 and B < 0:
        cls = "III"
    elif A < 0 and B > 0:
        cls = "IV"
    else:  # A > 0 and B >= 0
        cls = "I*"
    return cls, _AVO_CLASS_DESCRIPTIONS[cls]


def avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2):
    """AVO intercept, gradient, and class for a single interface.

    Args:
        vp1, vs1, rho1: P/S velocity (m/s) and density (g/cc) of the upper medium.
        vp2, vs2, rho2: P/S velocity (m/s) and density (g/cc) of the lower medium.

    Returns:
        dict: intercept (A), gradient (B), avo_class (I/I*/II/IIp/III/IV), and
        avo_class_description.
    """
    require_elastic_medium(vp1, vs1, rho1, "upper medium")
    require_elastic_medium(vp2, vs2, rho2, "lower medium")
    R0, G, _ = _shuey_coefficients(vp1, vs1, rho1, vp2, vs2, rho2)
    cls, desc = _classify_avo(R0, G)
    return {
        "intercept": float(R0),
        "gradient": float(G),
        "avo_class": cls,
        "avo_class_description": desc,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_avo_attributes.py -q`
Expected: 8 passed.

> If any class assertion fails, the chosen velocities don't produce the intended (A,B) signs — adjust the velocities/densities so the sign conditions in the test's first assertion hold (do NOT change `_classify_avo`'s sign rules or the assertions on the class label). The sign-precondition assertions in each test guard against this.

- [ ] **Step 5: Commit**

```bash
git add tools/avo_tools.py tests/test_avo_attributes.py
git commit -m "feat(avo): avo_attributes (intercept/gradient + AVO class)"
```

---

## Task 3: `plot_avo_crossplot`

**Files:**
- Modify: `tools/avo_tools.py` (add after `avo_attributes`)
- Test: `tests/test_avo_attributes.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_avo_attributes.py`:

```python
import os

from tools.avo_tools import plot_avo_crossplot


def test_crossplot_returns_png_path():
    path = plot_avo_crossplot(0.1, -0.2, "I")
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.exists(path)
    os.remove(path)


def test_crossplot_without_class_label():
    path = plot_avo_crossplot(-0.15, -0.1)
    assert os.path.exists(path)
    os.remove(path)


def test_crossplot_origin_point_not_degenerate():
    # A point at the origin must still produce a valid figure (minimum extent).
    path = plot_avo_crossplot(0.0, 0.0, "II")
    assert os.path.exists(path)
    os.remove(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_avo_attributes.py -q`
Expected: FAIL with `ImportError: cannot import name 'plot_avo_crossplot'`.

- [ ] **Step 3: Implement `plot_avo_crossplot`**

In `tools/avo_tools.py`, add at the end of the file:

```python
def plot_avo_crossplot(intercept, gradient, avo_class=None, output_path=None):
    """Plot the AVO intercept-gradient (A-B) crossplot and return the PNG path.

    Quadrants are lightly shaded and labeled by AVO class (I: A>0,B<0; III: A<0,B<0;
    IV: A<0,B>0) with a Class-II band around A=0. The (intercept, gradient) point is
    marked (annotated with avo_class when supplied).
    """
    import tempfile
    import os
    from matplotlib.patches import Rectangle

    A, B = float(intercept), float(gradient)
    # Symmetric extent that always includes the point and the origin, with a floor so
    # a near-origin point doesn't collapse the axes.
    e = max(abs(A), abs(B), 0.05) * 1.3

    fig, ax = plt.subplots(figsize=(7, 7))
    # Quadrant shading (class regions).
    ax.add_patch(Rectangle((0, -e), e, e, color="tab:blue", alpha=0.08))   # I:   A>0, B<0
    ax.add_patch(Rectangle((-e, -e), e, e, color="tab:red", alpha=0.08))   # III: A<0, B<0
    ax.add_patch(Rectangle((-e, 0), e, e, color="tab:green", alpha=0.08))  # IV:  A<0, B>0
    ax.axvspan(-_CLASS_II_INTERCEPT, _CLASS_II_INTERCEPT, color="gray", alpha=0.12)  # II band

    # Class labels at quadrant centers.
    ax.text(0.5 * e, -0.5 * e, "I", ha="center", va="center", fontsize=14, alpha=0.6)
    ax.text(-0.5 * e, -0.5 * e, "III", ha="center", va="center", fontsize=14, alpha=0.6)
    ax.text(-0.5 * e, 0.5 * e, "IV", ha="center", va="center", fontsize=14, alpha=0.6)
    ax.text(0.0, 0.85 * e, "II", ha="center", va="center", fontsize=12, alpha=0.6)

    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
    ax.axvline(0.0, color="k", linewidth=0.6, alpha=0.5)

    ax.plot(A, B, "ko", markersize=9)
    if avo_class:
        ax.annotate(f"Class {avo_class}", (A, B), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold")

    ax.set_xlim(-e, e)
    ax.set_ylim(-e, e)
    ax.set_xlabel("Intercept (A)")
    ax.set_ylabel("Gradient (B)")
    ax.set_title("AVO Intercept-Gradient Crossplot")
    ax.grid(True, alpha=0.3)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_avo_attributes.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/avo_tools.py tests/test_avo_attributes.py
git commit -m "feat(avo): plot_avo_crossplot (quadrant-shaded A-B crossplot)"
```

---

## Task 4: Register tools + chatbot wiring

**Files:**
- Modify: `core/tool_registry.py` (import line 13; add two `ToolSpec`s after the `plot_avo_reflectivity` spec)
- Modify: `core/chatbot_tool_use.py` (lines 73-75 prompt list; `_is_image_output` line 705; `_handle_automatic_chaining` ~line 738; `_update_context` ~line 817)
- Modify: `tests/test_tool_registry.py` (count guard 16 → 18)
- Test: `tests/test_avo_attributes.py` (append)

- [ ] **Step 1: Write the failing registry test**

Append to `tests/test_avo_attributes.py`:

```python
def test_avo_attributes_registered_and_chained():
    from core.tool_registry import REGISTRY_BY_NAME, AUTO_PLOT, TOOL_SCHEMAS

    assert "avo_attributes" in REGISTRY_BY_NAME
    assert "plot_avo_crossplot" in REGISTRY_BY_NAME
    assert AUTO_PLOT.get("avo_attributes") == "plot_avo_crossplot"
    spec = REGISTRY_BY_NAME["avo_attributes"]
    assert set(spec.required) == {"vp1", "vs1", "rho1", "vp2", "vs2", "rho2"}
    assert spec.validator is None
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "avo_attributes" in names and "plot_avo_crossplot" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_avo_attributes.py::test_avo_attributes_registered_and_chained -q`
Expected: FAIL (KeyError / assertion — not registered yet).

- [ ] **Step 3: Register the tools**

In `core/tool_registry.py`, extend the AVO import (line 13):

```python
from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity, plot_avo_reflectivity, avo_attributes, plot_avo_crossplot
```

Add these two `ToolSpec`s to `REGISTRY`, immediately AFTER the existing `plot_avo_reflectivity` spec:

```python
    ToolSpec(
        name="avo_attributes",
        fn=avo_attributes,
        description="Computes AVO interpretation attributes for an interface: intercept (A), gradient (B), and AVO class (I-IV) from the two media's Vp, Vs, and density. Auto-plots the intercept-gradient crossplot.",
        params={
            "vp1": {"type": "number", "description": "P-wave velocity of the upper medium in m/s."},
            "vs1": {"type": "number", "description": "S-wave velocity of the upper medium in m/s."},
            "rho1": {"type": "number", "description": "Density of the upper medium in g/cm³."},
            "vp2": {"type": "number", "description": "P-wave velocity of the lower medium in m/s."},
            "vs2": {"type": "number", "description": "S-wave velocity of the lower medium in m/s."},
            "rho2": {"type": "number", "description": "Density of the lower medium in g/cm³."},
        },
        required=["vp1", "vs1", "rho1", "vp2", "vs2", "rho2"],
        defaults={},
        validator=None,
        auto_plot="plot_avo_crossplot",
    ),
    ToolSpec(
        name="plot_avo_crossplot",
        fn=plot_avo_crossplot,
        description="Plots the AVO intercept-gradient (A-B) crossplot with shaded class regions and the marked point.",
        params={
            "intercept": {"type": "number", "description": "AVO intercept A."},
            "gradient": {"type": "number", "description": "AVO gradient B."},
            "avo_class": {"type": "string", "description": "Optional AVO class label to annotate the point."},
        },
        required=["intercept", "gradient"],
        defaults={},
    ),
```

- [ ] **Step 4: Bump the registry count guard**

In `tests/test_tool_registry.py`, find the assertion `assert len(REGISTRY) == 16` and change `16` to `18`.

- [ ] **Step 5: Run the registry test + count guard**

Run: `python -m pytest tests/test_avo_attributes.py::test_avo_attributes_registered_and_chained tests/test_tool_registry.py -q`
Expected: PASS.

- [ ] **Step 6: Wire the chatbot — `_is_image_output`**

In `core/chatbot_tool_use.py`, in `_is_image_output` (line 705), add `"plot_avo_crossplot"` to the list:

```python
                tool_name in ["plot_ricker", "plot_wedge_model", "plot_avo_reflectivity", "plot_rock_properties", "plot_wedge_gather", "plot_avo_crossplot"])
```

- [ ] **Step 7: Wire the chatbot — auto-chaining**

In `_handle_automatic_chaining`, add a branch BEFORE the final `else: return None` (after the `zoeppritz/shuey` branch around line 741):

```python
            elif tool_name == "avo_attributes":
                if not (isinstance(tool_result, dict) and "intercept" in tool_result):
                    return None
                plot_input = {
                    "intercept": tool_result["intercept"],
                    "gradient": tool_result["gradient"],
                    "avo_class": tool_result.get("avo_class"),
                }
```

- [ ] **Step 8: Wire the chatbot — context**

In `_update_context`, add a branch (after the `zoeppritz/shuey` branch around line 825):

```python
            elif tool_name == "avo_attributes":
                if isinstance(tool_result, dict) and "intercept" in tool_result:
                    self.context_manager.set_context("last_avo_attributes", tool_result)
```

- [ ] **Step 9: Wire the chatbot — system prompt**

In `_create_system_prompt`, after the `- plot_avo_reflectivity: ...` line (line 75), add:

```
- avo_attributes: AVO intercept/gradient + class (I-IV) for an interface, with an intercept-gradient crossplot
```

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (was 137; now 137 + new avo_attributes tests; registry count guard green at 18).

- [ ] **Step 11: Commit**

```bash
git add core/tool_registry.py core/chatbot_tool_use.py tests/test_tool_registry.py tests/test_avo_attributes.py
git commit -m "feat(registry): register avo_attributes + plot_avo_crossplot, wire chaining"
```

---

## Task 5: Document the tools

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add documentation**

In `CLAUDE.md`, find the AVO-related guidance (the "Gotchas specific to this package" bullet that begins "**AVO uses exact Aki-Richards Rpp.**"). Immediately after that bullet, add a new bullet:

```markdown
- **AVO interpretation attributes.** `tools/avo_tools.py::avo_attributes` returns the intercept (A), gradient (B), and AVO class (I/I*/II/IIp/III/IV) for an interface, derived from a shared `_shuey_coefficients` helper (so A/B are identical to `shuey_reflectivity`'s R0/G). Classification follows Rutherford-Williams/Castagna-Swan sign rules with a `|A| <= 0.02` Class-II band. It auto-plots `plot_avo_crossplot` (quadrant-shaded A-B plane). Covered by `tests/test_avo_attributes.py`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document avo_attributes + crossplot"
```

---

## Self-Review Notes

- **Spec coverage:** `_shuey_coefficients` + refactor (Task 1); `avo_attributes` + classification table incl. Class-II band and I*/IIp (Task 2 `_classify_avo`); guards reject unphysical media (Task 2 test); `plot_avo_crossplot` quadrant shading + Class-II band + origin floor (Task 3); registry two specs + auto_plot + count bump (Task 4); chaining/context/`_is_image_output`/system-prompt (Task 4 steps 6-9); docs (Task 5). Analytic-only and scalar (no array/regression mode) honored — none added.
- **Placeholder scan:** none — every code step shows complete code.
- **Type consistency:** `avo_attributes` returns dict keys `intercept, gradient, avo_class, avo_class_description` used identically in the chaining branch (`tool_result["intercept"/"gradient"]`, `.get("avo_class")`) and tests; `_classify_avo` returns `(label, description)`; `_shuey_coefficients` returns `(R0, G, F)` consumed as such in both `shuey_reflectivity` and `avo_attributes`. Registry count 16 → 18 matches adding exactly two tools.
