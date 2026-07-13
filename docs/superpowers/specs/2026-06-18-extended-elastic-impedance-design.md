# Extended Elastic Impedance (EEI) — design

- **Date:** 2026-06-18
- **Status:** Approved (pending spec review)
- **Scope:** One implementation cycle (spec → plan → implement)
- **Roadmap item:** #3-adjacent (Phase 3) of
  `2026-06-15-scientific-completeness-roadmap.md` — the roadmap listed
  Connolly EI; this upgrades it to Whitcombe EEI, which subsumes it.
- **Package:** `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`)

## Problem

The package has acoustic impedance (`calculate_impedance`) and AVO reflectivity, but
no **elastic impedance** — the angle/parameter-dependent impedance used as the
inversion attribute for far-offset stacks. Whitcombe (2002) **Extended Elastic
Impedance (EEI)** generalizes Connolly (1999) EI(θ) by re-parameterizing on a rotation
angle **χ** (−90°…+90°), so a single curve can be tuned to track different elastic
parameters (AI at χ=0; Vp/Vs, λρ, μρ, GR-like trends at other χ).

## Goals

1. `extended_elastic_impedance(...)` — from a layer's `(Vp, Vs, ρ)` and an array of
   rotation angles χ, return EEI(χ).
2. Optional **Whitcombe normalization** via background reference constants.
3. A companion **EEI-vs-χ plot**, auto-chained.

## Non-goals (out of scope)

- Multi-sample EEI **log** generation / EEI inversion workflows (single layer only).
- Automatic selection of the "optimal χ" that best correlates with a target parameter
  (Whitcombe's χ-tuning against a log) — needs a log; deferred.
- A separate incidence-angle EI(θ) tool (EEI subsumes it via `tan χ = sin²θ`; documented,
  not implemented separately).

## Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| Form | **Whitcombe EEI(χ)** (subsumes Connolly EI(θ)) |
| Normalization | **Raw by default; Whitcombe normalization when all three reference constants supplied (all-or-nothing)** |
| Input granularity | **Single layer** (scalar Vp/Vs/ρ), χ as an array |
| Tool name | `extended_elastic_impedance` |
| Plot | EEI vs χ, auto-chained |
| Home module | `tools/avo_tools.py` |

## Physics / definitions

Whitcombe (2002):
```
EEI(χ) = Vp0·ρ0 · (Vp/Vp0)^p · (Vs/Vs0)^q · (ρ/ρ0)^r
   p = cos χ + sin χ
   q = −8K · sin χ
   r = cos χ − 4K · sin χ
   K = (Vs/Vp)²        (background shear/compressional velocity ratio, squared)
```
- χ in **degrees**, converted to radians internally; conventional range −90°…+90°.
- **χ=0 ⇒ p=1, q=0, r=1 ⇒ EEI = Vp·ρ = acoustic impedance** (the correctness anchor),
  regardless of normalization (the reference factors cancel at χ=0).
- **Raw EEI** (no reference): set `Vp0=Vs0=ρ0=1`, i.e. `EEI = Vp^p · Vs^q · ρ^r`.
  This still gives `EEI(0) = Vp·ρ` and varies with χ.
- **Normalized EEI** = raw × `Vp0^(1−p)·Vs0^(−q)·ρ0^(1−r)`; the scale is 1 at χ=0 and
  rescales χ≠0 to keep consistent impedance units across the rotation.

## API

```
extended_elastic_impedance(vp, vs, rho, chi,
                           vp0=None, vs0=None, rho0=None,
                           k=None) -> np.ndarray
```
- `vp, vs, rho`: scalar layer properties (m/s, m/s, g/cc).
- `chi`: array-like rotation angles in degrees.
- `vp0, vs0, rho0`: optional reference constants (same units). **All-or-nothing**:
  - all three `None` → raw EEI (reference = 1);
  - all three supplied → Whitcombe-normalized EEI;
  - some-but-not-all → `ValueError`.
- `k`: optional background `(Vs/Vp)²`; default `(vs/vp)²` from the layer.
- **Returns:** 1-D `np.ndarray` of EEI values, one per χ (same length as `chi`).
  (Mirrors `shuey_reflectivity`/`zoeppritz_reflectivity`, which return an ndarray and
  auto-chain a plot using the angle array from the tool input.)

```
plot_extended_elastic_impedance(chi, eei, output_path=None) -> str
```
- Plots EEI vs χ (line), marks the χ=0 / AI reference (`axvline` at 0), labels axes
  ("Rotation angle χ (deg)", "Extended Elastic Impedance"), title. Saves a temp PNG via
  `tempfile.mkstemp` (matching the module's plot helpers) and returns the path.

## Guards (`tools/physics_guards` + local checks)

- **REJECT** (`ValueError`):
  - `require_elastic_medium(vp, vs, rho)` — vs≥vp, non-positive.
  - any `|χ| > 90`.
  - partial reference constants (some of `vp0/vs0/rho0` set, not all).
  - any supplied reference constant ≤ 0.
- No aliasing/angle-band warnings apply (EEI has no `tan` singularity; χ=±90 is valid).

## Wiring

- **`tools/avo_tools.py`** — add `extended_elastic_impedance` and
  `plot_extended_elastic_impedance`. (Module already imports `numpy as np`,
  `matplotlib.pyplot as plt`, `warnings`, and `require_elastic_medium`.)
- **`core/tool_registry.py`** — two `ToolSpec`s:
  - `extended_elastic_impedance`: params `vp, vs, rho` (required), `chi` (required array),
    `vp0, vs0, rho0, k` (optional numbers); `validator=None` (guards in the function);
    `auto_plot="plot_extended_elastic_impedance"`.
  - `plot_extended_elastic_impedance`: params `chi` (array, required), `eei` (array,
    required); no auto_plot.
  - Bump the `len(REGISTRY)` count guard in `tests/test_tool_registry.py` (18 → 20).
- **`core/chatbot_tool_use.py`** —
  - `_is_image_output`: add `"plot_extended_elastic_impedance"`.
  - `_handle_automatic_chaining`: add a branch for `extended_elastic_impedance` that
    builds `plot_input = {"chi": tool_input["chi"], "eei": tool_result}` (a dedicated
    branch — the input key is `chi`, not `angles`; guard `isinstance(tool_result, np.ndarray) and "chi" in tool_input`).
  - `_update_context`: store `last_eei = {"chi": tool_input["chi"], "eei": tool_result, "parameters": tool_input}` when the result is an ndarray.
  - Add `extended_elastic_impedance` to the system-prompt tool list.

## Data flow

```
LLM → extended_elastic_impedance(vp,vs,rho, chi=[...]) → ndarray EEI(χ)
        → context: last_eei
        → auto-chain plot_extended_elastic_impedance(chi, eei) → {image_path}
```

## Error / edge handling

- `chi` containing `±90` → valid (no singularity).
- Scalar `chi` (single value) → coerced via `np.atleast_1d`; returns length-1 array.
- Partial reference constants → `ValueError` (ambiguous normalization).
- `k` supplied → used as-is (no recomputation from vs/vp).

## Testing (`tests/test_extended_elastic_impedance.py`)

1. **AI anchor:** `extended_elastic_impedance(vp, vs, rho, chi=[0.0])[0]` == `vp*rho`
   (raw), within tolerance.
2. **Closed-form at χ=30°:** value equals `vp**p * vs**q * rho**r` with
   `p=cos+sin, q=−8K sin, r=cos−4K sin`, `K=(vs/vp)²` hand-computed (raw).
3. **Varies with χ:** EEI over `chi=[-45,0,45]` is not constant.
4. **Normalization:** with reference constants ≠ the sample, normalized EEI equals raw
   at χ=0 (within tolerance) but differs at χ=45°.
5. **`k` override:** passing `k` different from `(vs/vp)²` changes the χ≠0 result.
6. **Guards:** `vs≥vp` raises; a χ with `|χ|>90` raises; supplying `vp0` only (partial
   reference) raises; a reference constant ≤0 raises.
7. **Registry + plot:** `extended_elastic_impedance` and
   `plot_extended_elastic_impedance` registered; `AUTO_PLOT["extended_elastic_impedance"]
   == "plot_extended_elastic_impedance"`; the plot returns an existing PNG path.

## Files touched

- `tools/avo_tools.py` — `extended_elastic_impedance`, `plot_extended_elastic_impedance`.
- `core/tool_registry.py` — two new `ToolSpec`s + import.
- `core/chatbot_tool_use.py` — chaining + context + `_is_image_output` + system-prompt.
- `tests/test_extended_elastic_impedance.py` (new); `tests/test_tool_registry.py` (count 18→20).
- `CLAUDE.md` — document under the AVO section.

## Follow-ups (not this cycle)

- Multi-sample EEI **log** + optimal-χ tuning against a target curve (Vp/Vs, λρ, μρ, GR).
- An incidence-angle EI(θ) convenience wrapper (`tan χ = sin²θ`) if users ask for θ.
