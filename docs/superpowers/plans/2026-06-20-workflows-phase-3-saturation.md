# Workflows Phase 3 — Saturation Science Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add continuous water saturation (Sw) to the rock-physics chain via cited fluid-mixing laws (Reuss/Wood + Brie), exposed as a new compute tool, a saturation-sweep recipe, an `Sw` EEI target, and an Sw-aware `predict_layer` adapter — closing gap **S1**.

**Architecture:** A private `_effective_fluid(sw, …)` core computes the effective pore-fluid bulk modulus K_fl(Sw) and density ρ_fl(Sw) from a brine end-member and a hydrocarbon end-member under a chosen mixing law. A new `rock_properties_saturation` reuses the existing Han-1986 water-saturated frame from `calculate_rock_properties` and the existing `gassmann_dry`/`gassmann_sat` primitives, swapping in the effective fluid — so the existing `calculate_rock_properties` (and every recipe that depends on its tuple contract) is left untouched. The saturation-sweep recipe, the Sw EEI target, and the Sw-aware `predict_layer` all build on these two functions.

**Tech Stack:** Python 3.9+, NumPy, Matplotlib (existing headless plot convention), pytest. Run from inside `geo-mcp/seismic_chatbot`.

**Spec:** parent design `docs/superpowers/specs/2026-06-18-agentic-workflows-design.md` (Phase 3 row; gap **S1**). No separate Phase-3 design spec — the cited science is inline in this plan (matching the Phase 2a/2c plans).

**Builds on (verified contracts — do not re-derive):**
- `tools/rock_physics_tools.py` module-level constants `_FLUIDS = {'water': (2.2e9, 1.0), 'brine': (2.2e9, 1.0), 'oil': (1.0e9, 0.8), 'gas': (0.05e9, 0.2)}` (K_fl in **Pa**, ρ_fl in **g/cc**), `_K_QUARTZ=37.0e9`, `_K_CLAY=21.0e9` (Pa).
- `gassmann_dry(K_sat, K0, K_fl, phi)` and `gassmann_sat(K_dry, K0, K_fl, phi)` — exact forward/inverse pair (all moduli in Pa).
- `calculate_rock_properties(phit, vclay, fluid_type='water', print_results=True)` → tuple `(vp, vs, rhob, vp_vs_ratio, ai, si)`, **shape-preserving**, **internally clips** phit to [0, 0.35] and vclay to [0, 0.5] (with a warning). The water path returns `rhob = rho_matrix*(1-phit) + 1.0*phit`, so `rho_matrix` is recoverable as `(rhob - phit)/(1 - phit)`.
- `workflows/adapters.py::predict_layer(phit, vclay, fluid="water", *, reduce="mean", label="")` → `Layer`.
- `workflows/recipes/eei_optimal_chi_petro.py::eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine", chi_min=-90.0, chi_max=90.0, chi_step=1.0)` with `_TARGETS = {"vclay", "phit"}`.
- Registry: append a `ToolSpec` to `REGISTRY` (`core/tool_registry.py`); workflows are declared in `workflows/engine.py::WORKFLOW_REGISTRY` and auto-converted. **`REGISTRY` currently has 26 tools.**
- Plot convention: `output_path=None` → `tempfile.mkstemp(suffix=".png")` + `os.close(fd)`; `fig.savefig(output_path, dpi=300, bbox_inches="tight")`; `plt.close(fig)`; `return output_path`.
- Chatbot: `_workflow_image_output` surfaces `{"image_path": ...}` for ANY dict whose `image_path` is a `.png` string; `_update_context` caches `tool_name in WORKFLOW_NAMES`; `_create_system_prompt` has a hardcoded bullet list.

**Working dir for all commands:** `geo-mcp/seismic_chatbot` (its own git repo, branch `stabilize-tool-layer`).

## Global Constraints

- All new science lives in `tools/rock_physics_tools.py` (mixing core + saturated-rock function) and `workflows/` (recipe + adapter/recipe extensions). **Do not modify `calculate_rock_properties`** — it is the contract every existing recipe depends on.
- Cited models only: **Reuss/Wood** (uniform saturation, lower bound) and **Brie et al. 1995** (empirical patchy), with linear (volumetric) fluid-density mixing. Verified against the Rock Physics Handbook (Mavko, Mukerji & Dvorkin).
- Tests are real numeric assertions, no mocks. Endpoint known-answers (Sw=1 ⇒ brine, Sw=0 ⇒ hydrocarbon) pin the science.
- Tool **names differ from function names** only where already established; new tools use the same name for both.
- Registry count moves **26 → 28** (one leaf tool `rock_properties_saturation`, one workflow `saturation_sweep`; the `Sw` EEI target extends the already-registered `eei_optimal_chi_petro` and does not change the count).
- The standalone `test_tool_use.py::test_tool_use_pattern` (stdin) is a KNOWN pre-existing failure; ignore it in full-suite runs.

---

## The science (cited)

For a brine + hydrocarbon mixture at water saturation **Sw ∈ [0, 1]** with end-member
bulk moduli `K_w`, `K_hc` and densities `ρ_w`, `ρ_hc`:

- **Effective density (always linear / volumetric):** `ρ_fl = Sw·ρ_w + (1 − Sw)·ρ_hc`.
- **Reuss / Wood (uniform, fine-scale, isostress — lower bound):**
  `1 / K_fl = Sw / K_w + (1 − Sw) / K_hc`.
- **Brie et al. (1995) (empirical patchy):**
  `K_fl = (K_w − K_hc) · Sw^e + K_hc`, with exponent `e ≈ 3` (default).

Both laws satisfy the end-members exactly: at **Sw=1**, `K_fl = K_w` (brine); at **Sw=0**,
`K_fl = K_hc` (hydrocarbon). Reuss is the harmonic (Wood) lower bound; Brie is an
empirical patchy model, **not** a global upper bound on Reuss — for a strong brine/gas
contrast Brie's `Sw^e` curve dips slightly below Reuss at low Sw (~0–0.17). At moderate
saturation (e.g. Sw=0.5) `K_fl(Reuss) < K_fl(Brie)`, so `Vp(Reuss) < Vp(Brie)` there;
the tests assert this equal-Sw ordering at Sw=0.5, not a global bound.

The partially-saturated rock is then obtained by Gassmann substitution from the
Han-1986 water-saturated frame: invert to the dry frame with the brine fluid, then
forward-substitute the effective fluid `(K_fl, ρ_fl)`. Shear modulus is fluid-independent.

**Known-answer anchors:** `rock_properties_saturation(phit, vclay, sw=1.0)` must equal
`calculate_rock_properties(phit, vclay, 'water')`, and `sw=0.0, hydrocarbon='gas'` must
equal `calculate_rock_properties(phit, vclay, 'gas')` — these pin the mixing core, the
Gassmann path, and the frame reuse together.

---

## File Structure

- `tools/rock_physics_tools.py` — modify: add `_effective_fluid` (core) and `rock_properties_saturation` (compute). **`calculate_rock_properties` untouched.**
- `core/tool_registry.py` — modify: import + one `ToolSpec` for `rock_properties_saturation`.
- `workflows/recipes/saturation_sweep.py` — new: the saturation-sweep recipe + plot.
- `workflows/engine.py` — modify: import + one `WorkflowSpec` for `saturation_sweep`; and extend the existing `eei_optimal_chi_petro` `WorkflowSpec` params (Task 7).
- `workflows/adapters.py` — modify: add `sw`/`law`/`brie_exponent` to `predict_layer`.
- `workflows/recipes/eei_optimal_chi_petro.py` — modify: add `target="sw"` support.
- `core/chatbot_tool_use.py` — modify: system-prompt bullets.
- Tests: `tests/test_saturation.py` (core + saturated rock), `tests/test_saturation_sweep.py` (recipe), `tests/test_saturation_predict_layer.py` (adapter), appends to `tests/test_eei_optimal_chi_petro.py`, `tests/test_tool_registry.py` (26→27→28), `tests/test_workflow_meta_tool.py`, `tests/test_chatbot_workflow.py`.

The 8 tasks: (1) `_effective_fluid` core; (2) `rock_properties_saturation`; (3) register `rock_properties_saturation`; (4) `saturation_sweep` recipe + plot; (5) register `saturation_sweep`; (6) Sw-aware `predict_layer`; (7) `Sw` EEI target (+ re-spec its params); (8) system-prompt bullets + full suite.

---

### Task 1: `_effective_fluid` mixing core

**Files:**
- Modify: `tools/rock_physics_tools.py` (append the private core; reuse the module-level `numpy as np`)
- Create: `tests/test_saturation.py`

**Interfaces:**
- Produces: `_effective_fluid(sw, k_w, rho_w, k_hc, rho_hc, law="reuss", brie_exponent=3.0)` → `(K_fl, rho_fl)`. `k_*` are unit-agnostic (return is in the same unit as the inputs); `rho_*` in g/cc. Vectorized over `sw`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_saturation.py`:

```python
import numpy as np
import pytest

from tools.rock_physics_tools import _effective_fluid

# Batzle-Wang typical end-members (GPa for readability; the core is unit-agnostic in K).
K_W, RHO_W = 2.2, 1.0
K_G, RHO_G = 0.05, 0.2


def test_effective_fluid_endpoints():
    # Sw=1 -> pure brine; Sw=0 -> pure hydrocarbon, for both laws.
    for law in ("reuss", "brie"):
        k1, r1 = _effective_fluid(1.0, K_W, RHO_W, K_G, RHO_G, law=law)
        k0, r0 = _effective_fluid(0.0, K_W, RHO_W, K_G, RHO_G, law=law)
        assert np.isclose(k1, K_W) and np.isclose(r1, RHO_W)
        assert np.isclose(k0, K_G) and np.isclose(r0, RHO_G)


def test_effective_density_is_linear():
    _, rho = _effective_fluid(0.25, K_W, RHO_W, K_G, RHO_G, law="reuss")
    assert np.isclose(rho, 0.25 * RHO_W + 0.75 * RHO_G)


def test_reuss_below_brie_in_between():
    # Reuss is the lower bound: K_fl(reuss) <= K_fl(brie) for 0 < Sw < 1.
    kr, _ = _effective_fluid(0.5, K_W, RHO_W, K_G, RHO_G, law="reuss")
    kb, _ = _effective_fluid(0.5, K_W, RHO_W, K_G, RHO_G, law="brie")
    assert kr < kb < K_W


def test_effective_fluid_vectorized():
    sw = np.linspace(0.0, 1.0, 11)
    k, rho = _effective_fluid(sw, K_W, RHO_W, K_G, RHO_G, law="reuss")
    assert k.shape == sw.shape == rho.shape
    assert np.isclose(k[0], K_G) and np.isclose(k[-1], K_W)


def test_effective_fluid_guards():
    with pytest.raises(ValueError):
        _effective_fluid(1.5, K_W, RHO_W, K_G, RHO_G)          # sw out of [0,1]
    with pytest.raises(ValueError):
        _effective_fluid(0.5, K_W, RHO_W, K_G, RHO_G, law="x")  # bad law
    with pytest.raises(ValueError):
        _effective_fluid(0.5, -1.0, RHO_W, K_G, RHO_G)         # non-positive modulus
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_saturation.py -v`
Expected: FAIL with `ImportError: cannot import name '_effective_fluid' from 'tools.rock_physics_tools'`

- [ ] **Step 3: Implement the core**

Append to `tools/rock_physics_tools.py` (uses the module-level `numpy as np`):

```python
def _effective_fluid(sw, k_w, rho_w, k_hc, rho_hc, law="reuss", brie_exponent=3.0):
    """Effective pore-fluid modulus and density for a brine+hydrocarbon mixture.

    sw: water saturation (fraction, 0-1), scalar or array.
    k_w/k_hc: brine/hydrocarbon bulk moduli (any consistent unit; result matches).
    rho_w/rho_hc: brine/hydrocarbon densities (g/cc).
    law: 'reuss' (uniform/Wood, lower bound) or 'brie' (empirical patchy).
    brie_exponent: Brie exponent e (default 3).

    Returns (K_fl, rho_fl). Density mixes linearly; K_fl per the chosen law.
    Endpoints are exact: Sw=1 -> brine, Sw=0 -> hydrocarbon.
    """
    sw = np.asarray(sw, dtype=float)
    if np.any(sw < 0) or np.any(sw > 1):
        raise ValueError("sw (water saturation) must be within [0, 1]")
    if k_w <= 0 or k_hc <= 0 or rho_w <= 0 or rho_hc <= 0:
        raise ValueError("fluid moduli and densities must be positive")
    if law not in ("reuss", "brie"):
        raise ValueError(f"law must be 'reuss' or 'brie' (got {law!r})")

    rho_fl = sw * rho_w + (1.0 - sw) * rho_hc
    if law == "reuss":
        k_fl = 1.0 / (sw / k_w + (1.0 - sw) / k_hc)
    else:  # brie
        k_fl = (k_w - k_hc) * (sw ** float(brie_exponent)) + k_hc
    return k_fl, rho_fl
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_saturation.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/rock_physics_tools.py tests/test_saturation.py
git commit -m "feat(rock-physics): _effective_fluid core (Reuss/Brie Sw mixing)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `rock_properties_saturation` (partially-saturated rock)

**Files:**
- Modify: `tools/rock_physics_tools.py` (append the compute function; reuses `calculate_rock_properties`, `gassmann_dry`, `gassmann_sat`, `_effective_fluid`, `_FLUIDS`, `_K_QUARTZ`, `_K_CLAY`)
- Modify: `tests/test_saturation.py` (append tests)

**Interfaces:**
- Consumes: `_effective_fluid` (Task 1).
- Produces: `rock_properties_saturation(phit, vclay, sw, hydrocarbon="gas", law="reuss", brie_exponent=3.0, print_results=False)` → tuple `(vp, vs, rhob, vp_vs_ratio, ai, si)`, shape-preserving (same return shape as `calculate_rock_properties`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_saturation.py`:

```python
from tools.rock_physics_tools import rock_properties_saturation, calculate_rock_properties


def test_saturation_endpoint_sw1_matches_water():
    phit, vclay = 0.25, 0.20
    sat = rock_properties_saturation(phit, vclay, sw=1.0, hydrocarbon="gas")
    water = calculate_rock_properties(phit, vclay, "water", print_results=False)
    assert np.allclose(sat, water)


def test_saturation_endpoint_sw0_matches_gas():
    phit, vclay = 0.25, 0.20
    sat = rock_properties_saturation(phit, vclay, sw=0.0, hydrocarbon="gas")
    gas = calculate_rock_properties(phit, vclay, "gas", print_results=False)
    assert np.allclose(sat, gas)


def test_saturation_reuss_vp_below_brie():
    # Reuss is the lower bound: at equal Sw the density mix is identical, so the
    # smaller Reuss K_fl gives a strictly lower Vp than Brie. (The full Vp-Sw curve
    # is NOT monotone vs the gas end-member because density rises with Sw, so only
    # this equal-Sw bound is asserted.)
    phit, vclay = 0.25, 0.20
    vp_r = rock_properties_saturation(phit, vclay, sw=0.5, law="reuss")[0]
    vp_b = rock_properties_saturation(phit, vclay, sw=0.5, law="brie")[0]
    assert vp_r < vp_b


def test_saturation_shape_preserving():
    phit = np.array([0.20, 0.25, 0.30])
    vclay = np.array([0.10, 0.20, 0.30])
    sw = np.array([0.2, 0.5, 0.8])
    vp, vs, rhob, vp_vs, ai, si = rock_properties_saturation(phit, vclay, sw)
    for arr in (vp, vs, rhob, vp_vs, ai, si):
        assert np.asarray(arr).shape == phit.shape


def test_saturation_guards():
    with pytest.raises(ValueError):
        rock_properties_saturation(0.25, 0.20, sw=1.2)               # sw out of range
    with pytest.raises(ValueError):
        rock_properties_saturation(0.25, 0.20, sw=0.5, hydrocarbon="water")  # not a HC
    with pytest.raises(ValueError):
        rock_properties_saturation(0.25, 0.20, sw=0.5, law="bogus")  # bad law
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_saturation.py -k saturation -v`
Expected: FAIL with `ImportError: cannot import name 'rock_properties_saturation'`

- [ ] **Step 3: Implement the function**

Append to `tools/rock_physics_tools.py`:

```python
def rock_properties_saturation(phit, vclay, sw, hydrocarbon="gas",
                               law="reuss", brie_exponent=3.0, print_results=False):
    """Vp, Vs, density, Vp/Vs, AI, SI at a continuous water saturation Sw.

    Predicts the Han-1986 water-saturated frame (via calculate_rock_properties),
    then Gassmann-substitutes the effective brine+hydrocarbon fluid at saturation
    `sw` (Reuss or Brie mixing). Returns the same 6-tuple as
    calculate_rock_properties. `calculate_rock_properties` itself is unchanged.
    """
    hc = str(hydrocarbon).lower()
    if hc not in ("oil", "gas"):
        raise ValueError(f"hydrocarbon must be 'oil' or 'gas' (got {hydrocarbon!r})")
    if law not in ("reuss", "brie"):
        raise ValueError(f"law must be 'reuss' or 'brie' (got {law!r})")

    phit = np.asarray(phit, dtype=float)
    vclay = np.asarray(vclay, dtype=float)
    sw = np.asarray(sw, dtype=float)
    if np.any(sw < 0) or np.any(sw > 1):
        raise ValueError("sw (water saturation) must be within [0, 1]")

    # Water-saturated frame (reuses the Han regressions + internal clipping; DRY).
    vp_w, vs_w, rhob_w, *_ = calculate_rock_properties(
        phit, vclay, "water", print_results=False
    )
    # Recompute the only quantities calculate_rock_properties does not return:
    # mineral modulus K0 (Voigt-Reuss-Hill of quartz/clay) from clipped vclay, and
    # the matrix density (recoverable from rhob_w with rho_fl_water = 1.0 g/cc).
    vclay_c = np.clip(vclay, 0.0, 0.5)
    phit_c = np.clip(phit, 0.0, 0.35)
    k0_voigt = (1 - vclay_c) * _K_QUARTZ + vclay_c * _K_CLAY
    k0_reuss = 1.0 / ((1 - vclay_c) / _K_QUARTZ + vclay_c / _K_CLAY)
    K0 = 0.5 * (k0_voigt + k0_reuss)
    rho_matrix = (rhob_w - 1.0 * phit_c) / (1.0 - phit_c)  # g/cc

    rho_w_si = rhob_w * 1000.0
    mu = rho_w_si * vs_w ** 2                              # Pa, fluid-independent
    K_sat_w = rho_w_si * vp_w ** 2 - (4.0 / 3.0) * mu     # Pa

    K_fl_w, rho_fl_w = _FLUIDS["water"]                   # (2.2e9 Pa, 1.0 g/cc)
    K_hc, rho_hc = _FLUIDS[hc]
    K_fl_eff, rho_fl_eff = _effective_fluid(
        sw, K_fl_w, rho_fl_w, K_hc, rho_hc, law=law, brie_exponent=brie_exponent
    )

    K_dry = gassmann_dry(K_sat_w, K0, K_fl_w, phit_c)
    K_sat_t = gassmann_sat(K_dry, K0, K_fl_eff, phit_c)
    rhob = rho_matrix * (1.0 - phit_c) + rho_fl_eff * phit_c   # g/cc
    rho_t_si = rhob * 1000.0
    vp = np.sqrt((K_sat_t + (4.0 / 3.0) * mu) / rho_t_si)
    vs = np.sqrt(mu / rho_t_si)
    vp_vs_ratio = vp / vs

    rho_kg_m3 = rhob * 1000.0
    ai = rho_kg_m3 * vp / 1e6
    si = rho_kg_m3 * vs / 1e6

    if print_results:
        print("\n=== Rock Properties at Saturation ===")
        print(f"  Sw: {_format_value(sw, 2)}  HC: {hc}  law: {law}")
        print(f"  Vp:  {_format_value(vp, 0)} m/s")
        print(f"  Vs:  {_format_value(vs, 0)} m/s")
        print(f"  Rho: {_format_value(rhob, 3)} g/cc")
        print(f"  Vp/Vs: {_format_value(vp_vs_ratio, 2)}")
        print("=" * 38)

    return vp, vs, rhob, vp_vs_ratio, ai, si
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_saturation.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/rock_physics_tools.py tests/test_saturation.py
git commit -m "feat(rock-physics): rock_properties_saturation (Gassmann at continuous Sw)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Register `rock_properties_saturation` as a leaf tool

**Files:**
- Modify: `core/tool_registry.py` (import + `ToolSpec`)
- Modify: `tests/test_tool_registry.py` (count 26 → 27)
- Create: `tests/test_saturation_tool_registration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_saturation_tool_registration.py`:

```python
import numpy as np

from core import tool_registry as reg
from core.tool_manager import ToolManager


def test_rock_properties_saturation_registered():
    assert "rock_properties_saturation" in reg.REGISTRY_BY_NAME
    assert "rock_properties_saturation" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"rock_properties_saturation"}


def test_rock_properties_saturation_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("rock_properties_saturation", {
        "phit": [0.25], "vclay": [0.20], "sw": [0.5],
    })
    # tuple (vp, vs, rhob, vp_vs, ai, si)
    assert len(res) == 6
    vp = np.asarray(res[0], dtype=float)
    assert np.all(vp > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_saturation_tool_registration.py -v`
Expected: FAIL — `rock_properties_saturation` not in `reg.REGISTRY_BY_NAME`.

- [ ] **Step 3: Register the leaf tool**

In `core/tool_registry.py`, add `rock_properties_saturation` to the existing rock-physics import line (line 14):

```python
from tools.rock_physics_tools import calculate_rock_properties, rock_physics_rag, gassmann_substitution, rock_properties_saturation
```

Then add this `ToolSpec` to the `REGISTRY` list **immediately after the `calculate_rock_properties` spec** (which ends `defaults={"fluid_type": "water"}`) and before the `gassmann_substitution` spec:

```python
    ToolSpec(
        name="rock_properties_saturation",
        fn=rock_properties_saturation,
        description="Calculates Vp, Vs, density, Vp/Vs, acoustic and shear impedance at a continuous water saturation (Sw) from porosity and clay volume. Predicts the water-saturated frame (Han 1986) and applies Gassmann substitution with an effective brine+hydrocarbon pore fluid mixed by the Reuss/Wood (uniform) or Brie (patchy) law. Returns the six values (no plot).",
        params={
            "phit": {"type": "array", "items": {"type": "number"}, "description": "Porosity values (fraction, 0-1)."},
            "vclay": {"type": "array", "items": {"type": "number"}, "description": "Clay volume values (fraction, 0-1)."},
            "sw": {"type": "array", "items": {"type": "number"}, "description": "Water saturation (fraction, 0-1)."},
            "hydrocarbon": {"type": "string", "description": "Hydrocarbon end-member: 'gas' (default) or 'oil'."},
            "law": {"type": "string", "description": "Fluid-mixing law: 'reuss' (uniform/Wood, default) or 'brie' (patchy)."},
            "brie_exponent": {"type": "number", "description": "Brie exponent e (default 3); used only when law='brie'."},
        },
        required=["phit", "vclay", "sw"],
        defaults={"hydrocarbon": "gas", "law": "reuss", "brie_exponent": 3.0},
        auto_plot=None,
    ),
```

- [ ] **Step 4: Bump the registry count**

In `tests/test_tool_registry.py`, change `assert len(reg.REGISTRY) == 26` to:

```python
    assert len(reg.REGISTRY) == 27
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_saturation_tool_registration.py tests/test_tool_registry.py -v`
Expected: PASS. Also confirm `python -c "from core.tool_registry import REGISTRY; print(len(REGISTRY))"` prints 27.

- [ ] **Step 6: Commit**

```bash
git add core/tool_registry.py tests/test_saturation_tool_registration.py tests/test_tool_registry.py
git commit -m "feat(rock-physics): register rock_properties_saturation leaf tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `saturation_sweep` recipe + plot

**Files:**
- Create: `workflows/recipes/saturation_sweep.py`
- Create: `tests/test_saturation_sweep.py`

**Interfaces:**
- Consumes: `rock_properties_saturation` (Task 2).
- Produces: `saturation_sweep(phit, vclay, hydrocarbon="gas", law="reuss", sw_values=None, brie_exponent=3.0)` → dict with `sw`, `vp`, `vs`, `ai`, `vp_vs` curves, `hydrocarbon`, `law`, `image_path`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_saturation_sweep.py`:

```python
import os
import numpy as np
import pytest

from workflows.recipes.saturation_sweep import saturation_sweep


def test_saturation_sweep_structure_and_plot():
    res = saturation_sweep(0.25, 0.20, hydrocarbon="gas", law="reuss")
    assert {"sw", "vp", "vs", "ai", "vp_vs", "hydrocarbon", "law", "image_path"} <= set(res)
    n = len(res["sw"])
    assert len(res["vp"]) == n == len(res["ai"]) > 1
    path = res["image_path"]
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.getsize(path) > 0
    os.remove(path)


def test_saturation_sweep_vp_increases_with_sw():
    # Reuss: Vp at full brine (Sw=1) exceeds Vp at full gas (Sw=0).
    res = saturation_sweep(0.25, 0.20, sw_values=[0.0, 1.0])
    assert res["vp"][1] > res["vp"][0]


def test_saturation_sweep_rejects_bad_law():
    with pytest.raises(ValueError):
        saturation_sweep(0.25, 0.20, law="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_saturation_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'workflows.recipes.saturation_sweep'`

- [ ] **Step 3: Write the recipe**

Create `workflows/recipes/saturation_sweep.py`:

```python
"""saturation_sweep: rock properties vs water saturation (the fluid line).

For a single rock (porosity + clay volume), compute Vp/Vs/AI/(Vp/Vs) across a
range of water saturations Sw using rock_properties_saturation (Reuss or Brie
mixing), and plot the saturation curves. Self-plots via image_path.
"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from tools.rock_physics_tools import rock_properties_saturation


def saturation_sweep(phit, vclay, hydrocarbon="gas", law="reuss",
                     sw_values=None, brie_exponent=3.0):
    """Sweep water saturation for one rock and return Vp/Vs/AI/(Vp/Vs) curves + plot."""
    if law not in ("reuss", "brie"):
        raise ValueError(f"law must be 'reuss' or 'brie' (got {law!r})")
    if sw_values is None:
        sw_values = list(np.linspace(0.0, 1.0, 21))
    sw = np.asarray(sw_values, dtype=float)
    if sw.size == 0:
        raise ValueError("sw_values must contain at least one saturation")

    vp, vs, rhob, vp_vs, ai, si = rock_properties_saturation(
        phit, vclay, sw, hydrocarbon=hydrocarbon, law=law, brie_exponent=brie_exponent
    )
    result = {
        "sw": [float(x) for x in sw],
        "vp": [float(x) for x in np.atleast_1d(vp)],
        "vs": [float(x) for x in np.atleast_1d(vs)],
        "ai": [float(x) for x in np.atleast_1d(ai)],
        "vp_vs": [float(x) for x in np.atleast_1d(vp_vs)],
        "hydrocarbon": hydrocarbon,
        "law": law,
    }
    result["image_path"] = plot_saturation_sweep(
        result["sw"], result["vp"], result["vs"], result["ai"], hydrocarbon, law
    )
    return result


def plot_saturation_sweep(sw, vp, vs, ai, hydrocarbon, law, output_path=None):
    """Two-panel: Vp & Vs vs Sw (left), AI vs Sw (right)."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    sw = np.asarray(sw, dtype=float)

    fig, (ax_v, ax_ai) = plt.subplots(1, 2, figsize=(12, 5))
    ax_v.plot(sw, np.asarray(vp, dtype=float), "b-o", label="Vp")
    ax_v.plot(sw, np.asarray(vs, dtype=float), "r-s", label="Vs")
    ax_v.set_xlabel("Water saturation Sw")
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.set_title("Velocity vs saturation")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    ax_ai.plot(sw, np.asarray(ai, dtype=float), "g-^", label="AI")
    ax_ai.set_xlabel("Water saturation Sw")
    ax_ai.set_ylabel("Acoustic impedance (×10⁶ kg/m²·s)")
    ax_ai.set_title("Impedance vs saturation")
    ax_ai.grid(True, alpha=0.3)
    ax_ai.legend()

    fig.suptitle(f"Saturation sweep ({hydrocarbon}, {law} mixing)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_saturation_sweep.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/saturation_sweep.py tests/test_saturation_sweep.py
git commit -m "feat(workflows): saturation_sweep recipe (rock properties vs Sw)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Register `saturation_sweep` in the workflow engine

**Files:**
- Modify: `workflows/engine.py` (import + `WorkflowSpec`)
- Modify: `tests/test_tool_registry.py` (count 27 → 28)
- Modify: `tests/test_workflow_meta_tool.py` (append a case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workflow_meta_tool.py` (`reg` and `ToolManager` already imported):

```python
def test_saturation_sweep_is_registered_meta_tool():
    assert "saturation_sweep" in reg.REGISTRY_BY_NAME
    assert "saturation_sweep" in reg.TOOL_FUNCTIONS
    assert {t["name"] for t in reg.TOOL_SCHEMAS} >= {"saturation_sweep"}


def test_saturation_sweep_runs_through_tool_manager():
    tm = ToolManager()
    res = tm.execute_tool("saturation_sweep", {"phit": 0.25, "vclay": 0.20})
    assert res["law"] == "reuss"  # default
    assert isinstance(res["image_path"], str) and res["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_meta_tool.py -k saturation_sweep -v`
Expected: FAIL — not in `reg.REGISTRY_BY_NAME`.

- [ ] **Step 3: Register the recipe**

In `workflows/engine.py`, add the import after the existing recipe imports (after `from workflows.recipes.eei_optimal_chi_petro import eei_optimal_chi_petro`):

```python
from workflows.recipes.saturation_sweep import saturation_sweep
```

Add this `WorkflowSpec` to `WORKFLOW_REGISTRY` (after the `eei_optimal_chi_petro` spec, before the closing `]`):

```python
    WorkflowSpec(
        name="saturation_sweep",
        fn=saturation_sweep,
        description=(
            "Saturation (fluid-line) analysis: for a single rock described by porosity "
            "and clay volume, compute Vp, Vs, acoustic impedance and Vp/Vs across a range "
            "of water saturations Sw, using an effective brine+hydrocarbon pore fluid "
            "mixed by the Reuss/Wood (uniform) or Brie (patchy) law. Returns the "
            "saturation curves and a plot. Useful for fluid feasibility / DHI sensitivity."
        ),
        params={
            "phit": {"type": "number", "description": "Porosity (fraction, 0-1)."},
            "vclay": {"type": "number", "description": "Clay volume (fraction, 0-1)."},
            "hydrocarbon": {"type": "string", "description": "Hydrocarbon end-member: 'gas' (default) or 'oil'."},
            "law": {"type": "string", "description": "Fluid-mixing law: 'reuss' (uniform/Wood, default) or 'brie' (patchy)."},
            "sw_values": {"type": "array", "items": {"type": "number"}, "description": "Water saturations to sweep (default 0 to 1 in 21 steps)."},
            "brie_exponent": {"type": "number", "description": "Brie exponent e (default 3); used only when law='brie'."},
        },
        required=["phit", "vclay"],
        defaults={"hydrocarbon": "gas", "law": "reuss", "sw_values": None, "brie_exponent": 3.0},
        auto_plot=None,
    ),
```

- [ ] **Step 4: Bump the registry count**

In `tests/test_tool_registry.py`, change `assert len(reg.REGISTRY) == 27` to:

```python
    assert len(reg.REGISTRY) == 28
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_workflow_meta_tool.py tests/test_tool_registry.py tests/test_workflow_engine.py -v`
Expected: PASS. Also confirm `python -c "from core.tool_registry import REGISTRY; print(len(REGISTRY))"` prints 28.

- [ ] **Step 6: Commit**

```bash
git add workflows/engine.py tests/test_workflow_meta_tool.py tests/test_tool_registry.py
git commit -m "feat(workflows): register saturation_sweep meta-tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Sw-aware `predict_layer` adapter

**Files:**
- Modify: `workflows/adapters.py` (extend `predict_layer`)
- Create: `tests/test_saturation_predict_layer.py`

**Interfaces:**
- Consumes: `rock_properties_saturation` (Task 2).
- Produces: `predict_layer(phit, vclay, fluid="water", *, reduce="mean", label="", sw=None, law="reuss", brie_exponent=3.0)` → `Layer`. When `sw is None`, behavior is unchanged. When `sw` is given, `fluid` names the hydrocarbon end-member and must be `'oil'` or `'gas'`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_saturation_predict_layer.py`:

```python
import numpy as np
import pytest

from workflows.adapters import predict_layer


def test_predict_layer_sw_none_unchanged():
    a = predict_layer(0.25, 0.20, fluid="gas")
    b = predict_layer(0.25, 0.20, fluid="gas", sw=None)
    assert (a.vp, a.vs, a.rho) == (b.vp, b.vs, b.rho)


def test_predict_layer_sw1_matches_brine():
    sat = predict_layer(0.25, 0.20, fluid="gas", sw=1.0)
    brine = predict_layer(0.25, 0.20, fluid="water")
    assert np.isclose(sat.vp, brine.vp) and np.isclose(sat.rho, brine.rho)


def test_predict_layer_sw0_matches_gas():
    sat = predict_layer(0.25, 0.20, fluid="gas", sw=0.0)
    gas = predict_layer(0.25, 0.20, fluid="gas")
    assert np.isclose(sat.vp, gas.vp) and np.isclose(sat.rho, gas.rho)


def test_predict_layer_sw_requires_hydrocarbon_fluid():
    with pytest.raises(ValueError):
        predict_layer(0.25, 0.20, fluid="water", sw=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_saturation_predict_layer.py -v`
Expected: FAIL — `predict_layer() got an unexpected keyword argument 'sw'`.

- [ ] **Step 3: Extend `predict_layer`**

In `workflows/adapters.py`, add the import at the top (after the existing `from tools.rock_physics_tools import ...` line):

```python
from tools.rock_physics_tools import rock_properties_saturation
```

Replace the existing `predict_layer` function (currently lines 36-48) with:

```python
def predict_layer(phit, vclay, fluid="water", *, reduce="mean", label="",
                  sw=None, law="reuss", brie_exponent=3.0):
    """Predict a representative elastic `Layer` from porosity + clay volume (G2 + G3).

    When `sw` is given, `fluid` names the hydrocarbon end-member (must be 'oil' or
    'gas') and the layer is predicted at that water saturation via Reuss/Brie fluid
    mixing; otherwise the discrete `fluid` string is used as before.
    """
    if sw is None:
        vp, vs, rho, *_ = calculate_rock_properties(
            phit, vclay, fluid_type=fluid, print_results=False
        )
    else:
        if str(fluid).lower() not in ("oil", "gas"):
            raise ValueError(
                "when sw is given, fluid must name the hydrocarbon ('oil' or 'gas')"
            )
        vp, vs, rho, *_ = rock_properties_saturation(
            phit, vclay, sw, hydrocarbon=fluid, law=law, brie_exponent=brie_exponent
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_saturation_predict_layer.py tests/test_workflow_adapters.py -v`
Expected: PASS (the new tests plus the existing adapter tests — confirms no regression).

- [ ] **Step 5: Commit**

```bash
git add workflows/adapters.py tests/test_saturation_predict_layer.py
git commit -m "feat(workflows): Sw-aware predict_layer (partial saturation)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `Sw` as an EEI petro target

**Files:**
- Modify: `workflows/recipes/eei_optimal_chi_petro.py` (add `target="sw"`)
- Modify: `workflows/engine.py` (extend the `eei_optimal_chi_petro` `WorkflowSpec` params/defaults)
- Modify: `tests/test_eei_optimal_chi_petro.py` (append tests)

**Interfaces:**
- Consumes: `rock_properties_saturation` (Task 2).
- Produces: `eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine", chi_min=-90.0, chi_max=90.0, chi_step=1.0, sw=None, hydrocarbon="gas", law="reuss")`. New target `"sw"` requires the `sw` log; existing `"vclay"`/`"phit"` paths are unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eei_optimal_chi_petro.py`:

```python
def test_petro_recipe_sw_target():
    phit, vclay = _petro_logs(seed=5)
    n = len(phit)
    sw = list(np.linspace(0.2, 0.9, n))
    res = eei_optimal_chi_petro(phit, vclay, target="sw", sw=sw, hydrocarbon="gas")
    assert res["target"] == "sw"
    assert -90.0 <= res["optimal_chi"] <= 90.0
    os.remove(res["image_path"])


def test_petro_recipe_sw_target_requires_sw():
    phit, vclay = _petro_logs()
    with pytest.raises(ValueError):
        eei_optimal_chi_petro(phit, vclay, target="sw")  # sw missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eei_optimal_chi_petro.py -k sw -v`
Expected: FAIL — `target` `"sw"` rejected by the current `_TARGETS` guard.

- [ ] **Step 3: Extend the recipe**

Replace the contents of `workflows/recipes/eei_optimal_chi_petro.py` with:

```python
"""eei_optimal_chi_petro: EEI optimal-chi from petrophysical logs.

Predict Vp/Vs/density logs from porosity and clay-volume logs (rock physics), then
find the EEI rotation angle chi whose EEI log best correlates with a chosen
petrophysical target (Vclay for lithology, porosity, or water saturation). Wraps
the shared _eei_chi_scan core; self-plots via plot_eei_chi_scan.
"""
import numpy as np

from tools.rock_physics_tools import calculate_rock_properties, rock_properties_saturation
from tools.avo_tools import _eei_chi_scan, plot_eei_chi_scan


_TARGETS = {"vclay", "phit", "sw"}


def eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine",
                          chi_min=-90.0, chi_max=90.0, chi_step=1.0,
                          sw=None, hydrocarbon="gas", law="reuss"):
    """Find optimal EEI chi against a petrophysical target, from porosity/clay logs."""
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {sorted(_TARGETS)} (got {target!r})")

    phit = np.asarray(phit, dtype=float)
    vclay = np.asarray(vclay, dtype=float)

    if target == "sw":
        if sw is None:
            raise ValueError("target='sw' requires the sw log to be provided")
        sw = np.asarray(sw, dtype=float)
        vp, vs, rhob, *_ = rock_properties_saturation(
            phit, vclay, sw, hydrocarbon=hydrocarbon, law=law
        )
        target_log = sw
    else:
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

- [ ] **Step 4: Re-spec the registered params**

In `workflows/engine.py`, update the `eei_optimal_chi_petro` `WorkflowSpec` so the LLM can pass the new params. Change its `description` target clause to mention saturation, add three params, and add their defaults. Replace the spec's `description`, `params`, and `defaults` with:

```python
        description=(
            "EEI optimal-rotation-angle analysis from petrophysics: predict Vp/Vs/density "
            "logs from porosity and clay-volume logs, then find the Extended Elastic "
            "Impedance angle chi whose EEI log best correlates with a chosen target "
            "(Vclay for lithology, porosity, or water saturation Sw). Returns the optimal "
            "chi, the correlation-vs-chi curve, the EEI log at the optimal chi, and a plot."
        ),
        params={
            "phit": {"type": "array", "items": {"type": "number"}, "description": "Porosity log (fraction, 0-1)."},
            "vclay": {"type": "array", "items": {"type": "number"}, "description": "Clay-volume log (fraction, 0-1)."},
            "target": {"type": "string", "description": "Target property to correlate against: 'vclay' (default), 'phit', or 'sw'."},
            "fluid": {"type": "string", "description": "Pore fluid for the vclay/phit prediction: 'brine'/'water', 'oil', or 'gas' (default 'brine')."},
            "chi_min": {"type": "number", "description": "Minimum rotation angle in degrees (default -90)."},
            "chi_max": {"type": "number", "description": "Maximum rotation angle in degrees (default 90)."},
            "chi_step": {"type": "number", "description": "Rotation-angle step in degrees (default 1)."},
            "sw": {"type": "array", "items": {"type": "number"}, "description": "Water-saturation log (fraction, 0-1); required when target='sw'."},
            "hydrocarbon": {"type": "string", "description": "Hydrocarbon end-member for target='sw': 'gas' (default) or 'oil'."},
            "law": {"type": "string", "description": "Fluid-mixing law for target='sw': 'reuss' (default) or 'brie'."},
        },
        required=["phit", "vclay"],
        defaults={"target": "vclay", "fluid": "brine", "chi_min": -90.0, "chi_max": 90.0, "chi_step": 1.0, "sw": None, "hydrocarbon": "gas", "law": "reuss"},
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_eei_optimal_chi_petro.py tests/test_workflow_meta_tool.py tests/test_tool_registry.py -v`
Expected: PASS (the new `sw` tests, the existing vclay/phit tests, and the registry count still 28 — this task adds no new registry entry).

- [ ] **Step 6: Commit**

```bash
git add workflows/recipes/eei_optimal_chi_petro.py workflows/engine.py tests/test_eei_optimal_chi_petro.py
git commit -m "feat(workflows): Sw as an EEI petro target (eei_optimal_chi_petro)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: System-prompt bullets; full-suite check

**Files:**
- Modify: `core/chatbot_tool_use.py` (system-prompt bullets)
- Modify: `tests/test_chatbot_workflow.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chatbot_workflow.py` (reuses the `bot` fixture):

```python
def test_system_prompt_lists_saturation(bot):
    prompt = bot._create_system_prompt()
    assert "- rock_properties_saturation:" in prompt
    assert "- saturation_sweep:" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_workflow.py -k saturation -v`
Expected: FAIL — neither bullet present yet.

- [ ] **Step 3: Add the bullets**

In `core/chatbot_tool_use.py`, in `_create_system_prompt`'s hardcoded "Available tools:" bullet list, add these two bullets after the `eei_optimal_chi_petro` bullet (match the neighboring bullet formatting exactly):

```
- rock_properties_saturation: Computes Vp, Vs, density, Vp/Vs and impedances at a continuous water saturation Sw from porosity and clay volume, via Gassmann substitution with a Reuss (uniform) or Brie (patchy) brine+hydrocarbon fluid mix.
- saturation_sweep: Sweeps water saturation Sw for one rock (porosity & clay volume) and plots the Vp/Vs/AI saturation curves (the fluid line) under Reuss or Brie mixing — useful for fluid-feasibility / DHI sensitivity.
```

(No other chatbot changes: both surface their results via the generic `_workflow_image_output` / dict-return handling; `saturation_sweep` auto-caches via `WORKFLOW_NAMES`. The new `Sw` EEI target rides the existing `eei_optimal_chi_petro` bullet.)

- [ ] **Step 4: Run the targeted tests**

Run: `pytest tests/test_chatbot_workflow.py -v`
Expected: PASS (the prior chatbot-workflow tests plus the new one).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: the new Phase 3 tests pass and nothing else regressed. The standalone
`test_tool_use.py::test_tool_use_pattern` (stdin) is a KNOWN pre-existing failure —
if ONLY that fails, it is expected. If any OTHER test fails, STOP and report BLOCKED.

- [ ] **Step 6: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_workflow.py
git commit -m "feat(workflows): list saturation tools in chatbot system prompt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done when

- `_effective_fluid` returns exact end-members (Sw=1 ⇒ brine, Sw=0 ⇒ hydrocarbon) and Reuss ≤ Brie in between, for both laws and vectorized over Sw.
- `rock_properties_saturation` matches `calculate_rock_properties('water')` at Sw=1 and `calculate_rock_properties('gas')` at Sw=0, is shape-preserving, and gives `Vp(reuss) < Vp(brie)` at equal intermediate Sw (Reuss is the lower bound; the full Vp–Sw curve is not monotone vs the gas end-member because density rises with Sw).
- `rock_properties_saturation` (leaf tool) and `saturation_sweep` (recipe) both run via `ToolManager` and are registered (registry count **28**); `saturation_sweep` returns a dict with `image_path`.
- `predict_layer(..., sw=...)` predicts a partially-saturated `Layer` (matching the brine/gas end-members at Sw=1/0) and rejects a non-hydrocarbon `fluid` when `sw` is set; `sw=None` is byte-for-byte unchanged.
- `eei_optimal_chi_petro(..., target="sw", sw=...)` correlates EEI against the saturation log and rejects `target="sw"` with no `sw`; the existing `vclay`/`phit` targets are unchanged.
- Both new tool names appear in the chatbot system prompt; `calculate_rock_properties` is unmodified.
- Full suite green (modulo the pre-existing stdin-script failure).

## Spec coverage

- Gap **S1** (continuous Sw via cited fluid mixing) → Tasks 1–2 (Reuss/Wood + Brie + Gassmann), with the parent design's "feeding `predict_layer`" goal in Task 6.
- New saturation compute tool (the approved "new tool, leave existing" decision) → Tasks 2–3.
- Saturation-sweep workflow deliverable → Tasks 4–5.
- `Sw` EEI target (deferred from Phase 2b) → Task 7.
- Chatbot exposure → Task 8 (image surfacing + caching come free).
- Cited known-answer + bound-ordering + guard tests → Tasks 1–7.

## Not in this plan

- Voigt upper bound / Voigt-Reuss-Hill *fluid* average (only Reuss + Brie were chosen);
  patchy-vs-uniform beyond Brie; Batzle-Wang pressure/temperature fluid property modeling.
- AVO-vs-Sw composite in `saturation_sweep` (it returns rock-property curves; an AVO
  panel against a fixed shale is a possible later enhancement).
- The generic cross-recipe sweep engine (Phase 4, gap S3) — `saturation_sweep` is a
  single-purpose recipe, not the generic grid runner.
- Refactoring `calculate_rock_properties` to share the water-sat frame helper (kept
  untouched by decision); plus the standing deferred cleanups (carry-over task #14).
```
