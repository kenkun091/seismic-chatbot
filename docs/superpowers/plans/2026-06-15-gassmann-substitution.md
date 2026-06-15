# Gassmann Fluid Substitution Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-facing `gassmann_substitution` tool that takes in-situ `(vp, vs, rho)` + porosity + a fluid swap and returns the Gassmann-substituted elastic properties.

**Architecture:** A new vectorized function in `tools/rock_physics_tools.py` built on the existing, regression-tested `gassmann_sat`/`gassmann_dry` primitives, plus a `_fluid_moduli` helper that resolves preset-or-override fluid properties. Registered as one `ToolSpec` in the registry (no plot, no auto-chain). `calculate_rock_properties` is left untouched — both just share the same two primitives.

**Tech Stack:** Python, NumPy, pytest. All work in `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`); run tests with `python -m pytest tests/ -q` from the package dir.

---

## File Structure

- `tools/rock_physics_tools.py` — add `_fluid_moduli(...)` helper + `gassmann_substitution(...)`. The module already owns `_FLUIDS`, `_K_QUARTZ`, `gassmann_sat`, `gassmann_dry`.
- `core/tool_registry.py` — one new `ToolSpec` + import of `gassmann_substitution`.
- `core/chatbot_tool_use.py` — add the tool name to the system-prompt tool list (cosmetic; the registry is the real contract).
- `tests/test_gassmann_substitution.py` — new test module.
- `CLAUDE.md` — document the tool under rock-physics.

---

## Task 1: `_fluid_moduli` helper + core substitution math

**Files:**
- Modify: `tools/rock_physics_tools.py` (add after `gassmann_dry`, ~line 54)
- Test: `tests/test_gassmann_substitution.py` (create)

- [ ] **Step 1: Write the failing tests (round-trip identity + gas signature)**

```python
# tests/test_gassmann_substitution.py
import numpy as np
import pytest

from tools.rock_physics_tools import gassmann_substitution


def test_roundtrip_identity_same_fluid():
    # Substituting a fluid for itself returns inputs unchanged (phi > 0).
    res = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.25,
        fluid_in="brine", fluid_out="brine", print_results=False,
    )
    assert np.isclose(res["vp"], 3000.0, rtol=1e-6)
    assert np.isclose(res["vs"], 1500.0, rtol=1e-6)
    assert np.isclose(res["rho"], 2.2, rtol=1e-6)


def test_brine_to_gas_signature():
    # Gas vs brine: Vp DOWN, Vs UP (shear modulus fixed, lower density), rho DOWN.
    res = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    assert res["vp"] < 3000.0
    assert res["vs"] > 1500.0
    assert res["rho"] < 2.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gassmann_substitution.py -q`
Expected: FAIL with `ImportError: cannot import name 'gassmann_substitution'`.

- [ ] **Step 3: Implement `_fluid_moduli` + `gassmann_substitution`**

Add to `tools/rock_physics_tools.py` immediately after `gassmann_dry` (around line 54):

```python
def _fluid_moduli(name, k_override=None, rho_override=None):
    """Resolve a fluid's (K_fl in Pa, rho_fl in g/cc) from a preset name and/or
    explicit overrides. Overrides (K in GPa, rho in g/cc) take precedence over the
    preset. Raises ValueError if the fluid is neither a known preset nor fully
    specified by overrides, or if a resolved modulus/density is non-positive.
    """
    K = rho = None
    if name is not None:
        preset = _FLUIDS.get(str(name).lower())
        if preset is not None:
            K, rho = preset
    if k_override is not None:
        K = float(k_override) * 1e9      # GPa -> Pa
    if rho_override is not None:
        rho = float(rho_override)        # g/cc
    if K is None or rho is None:
        raise ValueError(
            f"Unknown fluid '{name}'; use one of {sorted(_FLUIDS)} "
            f"or supply k_fl/rho_fl overrides."
        )
    if K <= 0 or rho <= 0:
        raise ValueError(
            f"fluid modulus and density must be positive (got K={K} Pa, rho={rho} g/cc)"
        )
    return K, rho


def gassmann_substitution(vp, vs, rho, phi, fluid_in, fluid_out,
                          k_mineral=37.0,
                          k_fl_in=None, rho_fl_in=None,
                          k_fl_out=None, rho_fl_out=None,
                          print_results=True):
    """Gassmann fluid substitution from in-situ elastic properties.

    Args:
        vp, vs: in-situ P/S velocities (m/s), scalar or array-like.
        rho: in-situ bulk density (g/cc), scalar or array-like.
        phi: porosity (fraction, 0-1), scalar or array-like.
        fluid_in, fluid_out: 'water'/'brine'/'oil'/'gas' (case-insensitive).
        k_mineral: mineral (grain) bulk modulus in GPa (default 37, quartz).
        k_fl_in/out: optional fluid bulk-modulus override in GPa.
        rho_fl_in/out: optional fluid density override in g/cc.

    Returns:
        dict with substituted 'vp' (m/s), 'vs' (m/s), 'rho' (g/cc), 'vp_vs',
        plus 'k_dry', 'k_sat' (GPa) and 'mu' (GPa, unchanged by substitution).
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # REJECT non-physical inputs.
    if np.any(phi < 0) or np.any(phi > 1):
        raise ValueError("phi (porosity) must be within [0, 1]")
    if np.any(vp <= 0):
        raise ValueError("vp must be positive")
    if np.any(vs <= 0):
        raise ValueError("vs must be positive")
    if np.any(rho <= 0):
        raise ValueError("rho must be positive")
    if k_mineral is None or k_mineral <= 0:
        raise ValueError(f"k_mineral must be positive (got {k_mineral})")

    K0 = float(k_mineral) * 1e9  # GPa -> Pa
    K_fl_in, rho_fl_in_val = _fluid_moduli(fluid_in, k_fl_in, rho_fl_in)
    K_fl_out, rho_fl_out_val = _fluid_moduli(fluid_out, k_fl_out, rho_fl_out)

    # In-situ saturated moduli (SI); density g/cc -> kg/m^3.
    rho_si = rho * 1000.0
    mu = rho_si * vs ** 2                              # Pa, fluid-independent
    K_sat_in = rho_si * vp ** 2 - (4.0 / 3.0) * mu    # Pa

    # Invert to the dry frame, then forward-substitute the new fluid.
    K_dry = gassmann_dry(K_sat_in, K0, K_fl_in, phi)
    if np.any(K_dry < 0) or np.any(K_dry > K0):
        warnings.warn(
            "Inverted dry-frame modulus is non-physical (K_dry < 0 or > K_mineral); "
            "check vp/vs/rho/phi/k_mineral consistency.",
            stacklevel=2,
        )
    K_sat_out = gassmann_sat(K_dry, K0, K_fl_out, phi)

    # Density swap (only the pore fluid changes), g/cc.
    rho_out = rho + phi * (rho_fl_out_val - rho_fl_in_val)
    rho_out_si = rho_out * 1000.0

    vp_out = np.sqrt((K_sat_out + (4.0 / 3.0) * mu) / rho_out_si)
    vs_out = np.sqrt(mu / rho_out_si)
    vp_vs = vp_out / vs_out

    result = {
        "vp": vp_out, "vs": vs_out, "rho": rho_out, "vp_vs": vp_vs,
        "k_dry": K_dry / 1e9, "k_sat": K_sat_out / 1e9, "mu": mu / 1e9,
    }

    if print_results:
        print("\n=== Gassmann Fluid Substitution ===")
        print(f"  {fluid_in} -> {fluid_out}")
        print(f"  Vp:  {_format_value(vp_out, 0)} m/s")
        print(f"  Vs:  {_format_value(vs_out, 0)} m/s")
        print(f"  Rho: {_format_value(rho_out, 3)} g/cc")
        print(f"  Vp/Vs: {_format_value(vp_vs, 2)}")
        print("=" * 35)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gassmann_substitution.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/rock_physics_tools.py tests/test_gassmann_substitution.py
git commit -m "feat(rock-physics): gassmann_substitution core + fluid resolver"
```

---

## Task 2: Cross-check, arrays, overrides, and guards

**Files:**
- Test: `tests/test_gassmann_substitution.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gassmann_substitution.py`:

```python
from tools.rock_physics_tools import calculate_rock_properties


def test_matches_calculate_rock_properties_gas_case():
    # With vclay=0, calculate_rock_properties' mineral modulus K0 == 37 GPa (pure
    # quartz VRH), matching the default k_mineral. Feeding its water-sat output
    # into gassmann_substitution(water->gas) must reproduce its gas-sat output.
    phi = 0.2
    vp_w, vs_w, rhob_w, *_ = calculate_rock_properties(phi, 0.0, "water", print_results=False)
    vp_g, vs_g, rhob_g, *_ = calculate_rock_properties(phi, 0.0, "gas", print_results=False)

    res = gassmann_substitution(
        vp=float(vp_w), vs=float(vs_w), rho=float(rhob_w), phi=phi,
        fluid_in="water", fluid_out="gas", k_mineral=37.0, print_results=False,
    )
    assert np.isclose(res["vp"], float(vp_g), rtol=1e-6)
    assert np.isclose(res["vs"], float(vs_g), rtol=1e-6)
    assert np.isclose(res["rho"], float(rhob_g), rtol=1e-6)


def test_array_inputs_return_arrays():
    res = gassmann_substitution(
        vp=np.array([3000.0, 3200.0]), vs=np.array([1500.0, 1600.0]),
        rho=np.array([2.2, 2.25]), phi=np.array([0.2, 0.18]),
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    assert res["vp"].shape == (2,)
    assert res["vs"].shape == (2,)
    assert res["rho"].shape == (2,)


def test_custom_fluid_override_differs_from_preset():
    base = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    # Override the target fluid with a much stiffer, denser "gas" -> different result.
    override = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas",
        k_fl_out=1.5, rho_fl_out=0.6, print_results=False,
    )
    assert not np.isclose(base["vp"], override["vp"])
    assert override["rho"] > base["rho"]  # denser override fluid -> higher bulk density


def test_guards_reject_bad_inputs():
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=1.5, fluid_in="brine", fluid_out="gas", print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=-0.1, fluid_in="brine", fluid_out="gas", print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(-3000, 1500, 2.2, phi=0.2, fluid_in="brine", fluid_out="gas", print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=0.2, fluid_in="brine", fluid_out="gas", k_mineral=0, print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=0.2, fluid_in="magma", fluid_out="gas", print_results=False)


def test_nonphysical_k_dry_warns_but_returns():
    # Very low Vp at high porosity drives K_dry below zero -> warn, still returns.
    with pytest.warns(UserWarning):
        res = gassmann_substitution(
            vp=1600.0, vs=200.0, rho=2.0, phi=0.35,
            fluid_in="water", fluid_out="gas", print_results=False,
        )
    assert "vp" in res
```

- [ ] **Step 2: Run tests to verify the new ones pass (no new code needed)**

Run: `python -m pytest tests/test_gassmann_substitution.py -q`
Expected: 7 passed. (Task 1's implementation already satisfies these.)

> If `test_nonphysical_k_dry_warns_but_returns` does NOT emit a warning, adjust the input to push `K_dry` negative (lower `vp` toward 1500, raise `phi`), confirming the `K_dry < 0` branch — do not weaken the assertion. If `test_custom_fluid_override_differs_from_preset`'s density assertion is borderline, note that preset gas density is 0.2 g/cc and the override is 0.6 g/cc, so `rho_out` must be strictly higher.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gassmann_substitution.py
git commit -m "test(rock-physics): gassmann_substitution cross-check, arrays, overrides, guards"
```

---

## Task 3: Register the tool

**Files:**
- Modify: `core/tool_registry.py` (import line ~14; add `ToolSpec` in `REGISTRY`)
- Modify: `core/chatbot_tool_use.py` (system-prompt tool list)
- Test: `tests/test_gassmann_substitution.py` (append registry test)

- [ ] **Step 1: Write the failing registry test**

Append to `tests/test_gassmann_substitution.py`:

```python
def test_registered_in_registry():
    from core.tool_registry import REGISTRY_BY_NAME, TOOL_FUNCTIONS, TOOL_SCHEMAS

    assert "gassmann_substitution" in REGISTRY_BY_NAME
    spec = REGISTRY_BY_NAME["gassmann_substitution"]
    assert spec.fn is gassmann_substitution
    assert spec.auto_plot is None
    assert set(spec.required) == {"vp", "vs", "rho", "phi", "fluid_in", "fluid_out"}
    assert TOOL_FUNCTIONS["gassmann_substitution"] is gassmann_substitution
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "gassmann_substitution" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gassmann_substitution.py::test_registered_in_registry -q`
Expected: FAIL with `KeyError: 'gassmann_substitution'`.

- [ ] **Step 3: Add the import and `ToolSpec`**

In `core/tool_registry.py`, extend the rock-physics import (line ~14):

```python
from tools.rock_physics_tools import calculate_rock_properties, rock_physics_rag, gassmann_substitution
```

Then add this `ToolSpec` to the `REGISTRY` list, immediately after the `calculate_rock_properties` spec:

```python
    ToolSpec(
        name="gassmann_substitution",
        fn=gassmann_substitution,
        description="Gassmann fluid substitution: from in-situ Vp, Vs, density and porosity, compute the elastic properties (Vp, Vs, density) after swapping the pore fluid (e.g. brine to gas). Shear modulus is held fluid-independent.",
        params={
            "vp": {"type": "number", "description": "In-situ P-wave velocity in m/s."},
            "vs": {"type": "number", "description": "In-situ S-wave velocity in m/s."},
            "rho": {"type": "number", "description": "In-situ bulk density in g/cm³."},
            "phi": {"type": "number", "description": "Porosity (fraction, 0-1)."},
            "fluid_in": {"type": "string", "description": "In-situ pore fluid: 'water'/'brine', 'oil', or 'gas'."},
            "fluid_out": {"type": "string", "description": "Target pore fluid to substitute in: 'water'/'brine', 'oil', or 'gas'."},
            "k_mineral": {"type": "number", "description": "Mineral (grain) bulk modulus in GPa (default 37, quartz)."},
            "k_fl_in": {"type": "number", "description": "Optional in-situ fluid bulk-modulus override in GPa."},
            "rho_fl_in": {"type": "number", "description": "Optional in-situ fluid density override in g/cm³."},
            "k_fl_out": {"type": "number", "description": "Optional target fluid bulk-modulus override in GPa."},
            "rho_fl_out": {"type": "number", "description": "Optional target fluid density override in g/cm³."},
        },
        required=["vp", "vs", "rho", "phi", "fluid_in", "fluid_out"],
        defaults={"k_mineral": 37.0},
        validator=None,
        auto_plot=None,
    ),
```

- [ ] **Step 4: Run the registry test to verify it passes**

Run: `python -m pytest tests/test_gassmann_substitution.py::test_registered_in_registry -q`
Expected: PASS.

- [ ] **Step 5: Add the tool to the chatbot system-prompt tool list**

In `core/chatbot_tool_use.py`, find the tool listing inside `_create_system_prompt` (the bulleted list of available tools). Add a line consistent with the existing format, e.g.:

```
- gassmann_substitution: Gassmann fluid substitution from in-situ Vp/Vs/density + porosity (e.g. model the gas case of a brine sand).
```

- [ ] **Step 6: Run the full suite to confirm nothing regressed**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (prior 129 + the new gassmann tests).

- [ ] **Step 7: Commit**

```bash
git add core/tool_registry.py core/chatbot_tool_use.py tests/test_gassmann_substitution.py
git commit -m "feat(registry): register gassmann_substitution tool"
```

---

## Task 4: Document the tool

**Files:**
- Modify: `CLAUDE.md` (rock-physics correctness section)

- [ ] **Step 1: Add documentation**

In `CLAUDE.md`, under the "Rock-physics correctness" section, append a short paragraph:

```markdown
`tools/rock_physics_tools.py::gassmann_substitution` exposes Gassmann fluid
substitution as a standalone LLM-facing tool: in-situ `(vp, vs, rho)` + porosity +
a fluid swap (`fluid_in`→`fluid_out`) → substituted `(vp, vs, rho, vp_vs, k_dry,
k_sat, mu)`. It is built on the same verified `gassmann_sat`/`gassmann_dry`
primitives as `calculate_rock_properties` (not a refactor of it). Preset fluids
(`water`/`brine`/`oil`/`gas`) with optional `k_fl_*`/`rho_fl_*` overrides (GPa /
g/cc); `k_mineral` in GPa (quartz default 37); vectorized; no plot. Shear modulus
is held fluid-independent, so brine→gas LOWERS Vp and RAISES Vs. Covered by
`tests/test_gassmann_substitution.py`. (Note: at φ=0 the dry-frame inversion is
degenerate — `K_dry`→`K_mineral` — so the round-trip identity holds only for φ>0.)
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document gassmann_substitution tool"
```

---

## Self-Review Notes

- **Spec coverage:** API (Task 1/3), physics steps 1–5 (Task 1), preset+override fluids (Task 1 `_fluid_moduli`, Task 2 override test), guards reject/warn (Task 1 impl, Task 2 tests), array support (Task 1 `np.asarray`, Task 2 test), registry wiring + system prompt (Task 3), no refactor of `calculate_rock_properties` (honored), no plot (`auto_plot=None`), docs (Task 4). All spec sections map to a task.
- **Spec correction:** the spec's φ=0 edge note ("returns inputs essentially unchanged") is inaccurate — `gassmann_dry` collapses to `K0` at φ=0, so the round-trip identity is tested at φ=0.25 and the φ=0 degeneracy is documented in `CLAUDE.md` instead.
- **Type consistency:** `gassmann_substitution` returns a dict with keys `vp, vs, rho, vp_vs, k_dry, k_sat, mu` everywhere; `_fluid_moduli` returns `(K_Pa, rho_gcc)` consistently; registry `required` list matches the test assertion.
