# AVO attributes (intercept/gradient + class + crossplot) — design

- **Date:** 2026-06-15
- **Status:** Approved (pending spec review)
- **Scope:** One implementation cycle (spec → plan → implement)
- **Roadmap item:** #2 (Phase 1, AVO interpretation) of
  `2026-06-15-scientific-completeness-roadmap.md`
- **Package:** `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`)

## Problem

The package computes full AVO reflectivity curves (`shuey_reflectivity`,
`zoeppritz_reflectivity`) but never reduces an interface to its standard
interpretation summary: the **intercept** A (normal-incidence reflectivity), the
**gradient** B, and the **AVO class** (I–IV). These three are how AVO behavior is
actually communicated and crossplotted in practice. The intercept and gradient
already exist *inside* `shuey_reflectivity` (its `R0` and `G` terms) but are not
surfaced.

## Goals

1. `avo_attributes(...)` — from the six elastic parameters of an interface, return
   intercept A, gradient B, and the AVO class with a human-readable description.
2. `plot_avo_crossplot(...)` — the A–B crossplot with shaded, labeled class regions
   and the computed point.
3. Guarantee A and B are **identical** to the values `shuey_reflectivity` uses, by
   sharing one coefficient helper (DRY).

## Non-goals (out of scope)

- A second input mode that fits A/B by regression from a passed `(angles, rc)` array
  (decision: **analytic-only** from elastic params; deferred follow-up).
- Vectorized / multi-interface input (an interface's A,B are scalar).
- Background-trend-line fitting, fluid/lithology lines, or χ-angle rotation on the
  crossplot (pedagogical quadrant shading only).
- Any change to `zoeppritz_reflectivity` or the AVO curve math.

## Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| Input | **Analytic only** — A,B computed from `vp1,vs1,rho1,vp2,vs2,rho2` |
| Granularity | **Scalar**, one interface per call |
| Source of A,B | Shared `_shuey_coefficients` helper (same as `shuey_reflectivity`) |
| Crossplot | **Quadrant-shaded** A–B plane with a Class-II band + marked point |

## Physics / definitions

Intercept and gradient are the first two Shuey terms (Shuey 1985), already present in
`shuey_reflectivity`:

```
A (intercept) = R0 = 0.5 * (d_vp/avg_vp + d_rho/avg_rho)
B (gradient)  = 0.5 * d_vp/avg_vp
                - 2 * (avg_vs**2 / avg_vp**2) * (d_rho/avg_rho + 2 * d_vs/avg_vs)
```
where `d_x = x2 - x1`, `avg_x = 0.5*(x1+x2)`. (The third term `F = 0.5*d_vp/avg_vp`
is also returned by the helper but unused by `avo_attributes`.)

**AVO classification** (Rutherford & Williams 1989; Castagna & Swan 1997), on the
signs of A and B with a near-zero-intercept band for Class II. Threshold
`_CLASS_II_INTERCEPT = 0.02` (module constant):

| Condition | Class | Description |
|-----------|-------|-------------|
| `abs(A) <= 0.02` | `II` (or `IIp` if `A < 0`) | Near-zero intercept; phase/polarity reversal with offset |
| `A > 0` and `B < 0` | `I`   | High-impedance contrast; amplitude dims (may reverse) with offset |
| `A < 0` and `B < 0` | `III` | Classic bright spot (e.g. gas sand); brightens with offset |
| `A < 0` and `B > 0` | `IV`  | Bright spot whose magnitude decreases with offset |
| `A > 0` and `B >= 0` | `I*`  | Atypical: positive intercept with non-negative gradient (rising) |

The Class-II band is checked **first** (it spans both signs of B near A≈0). `B == 0`
exactly is treated as the `B >= 0` branch.

## API

```
avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2) -> dict
```
- All six are scalars (m/s, m/s, g/cc). Guards (`tools/physics_guards`):
  `require_elastic_medium(vp1, vs1, rho1, "upper medium")` and
  `require_elastic_medium(vp2, vs2, rho2, "lower medium")` — reject vs≥vp, non-positive.
  No angle parameter, so no angle guard.
- **Returns** a dict:
  - `intercept` (float, A)
  - `gradient` (float, B)
  - `avo_class` (str: `"I"`, `"I*"`, `"II"`, `"IIp"`, `"III"`, `"IV"`)
  - `avo_class_description` (str, the table's description text)

```
plot_avo_crossplot(intercept, gradient, avo_class=None, output_path=None) -> str
```
- Draws the A–B plane: x = intercept, y = gradient. Light quadrant shading with
  labels — Class I (A>0,B<0), Class III (A<0,B<0), Class IV (A<0,B>0) — plus a
  vertical Class-II band at `|A| <= 0.02`. `axhline`/`axvline` at 0. Plots the
  `(intercept, gradient)` point (annotated with `avo_class` if given). Autoscaled so
  the point and origin are both visible (symmetric limits with a margin around the
  point, minimum extent so a near-origin point isn't degenerate). Saves a temp PNG via
  `tempfile.mkstemp` (matching `plot_avo_reflectivity`) and returns the path.

## Wiring

- **`tools/avo_tools.py`** —
  - Add `_shuey_coefficients(vp1, vs1, rho1, vp2, vs2, rho2) -> (R0, G, F)` (pure, no
    guards, no angle).
  - Refactor `shuey_reflectivity` to obtain `R0, G, F` from the helper (keeping its
    existing guards and the `rc = R0 + G*sin²θ + F*(tan²θ − sin²θ)` assembly) — behavior
    unchanged, verified by existing tests.
  - Add `avo_attributes` and `plot_avo_crossplot`.
- **`core/tool_registry.py`** — two `ToolSpec`s:
  - `avo_attributes`: params `vp1,vs1,rho1,vp2,vs2,rho2` (all required, number),
    `defaults={}`, `validator=None` (guards in the function), `auto_plot="plot_avo_crossplot"`.
  - `plot_avo_crossplot`: params `intercept` (number, required), `gradient` (number,
    required), `avo_class` (string, optional). No auto_plot.
  - Bump the `len(REGISTRY)` count guard in `tests/test_tool_registry.py` (16 → 18).
- **`core/chatbot_tool_use.py`** —
  - Auto-chain: after `avo_attributes`, invoke `plot_avo_crossplot` with the cached
    `intercept`/`gradient`/`avo_class` (mirror the existing AVO chaining branch; surfaced
    as `{"image_path": ...}`).
  - `_update_context`: store `last_avo_attributes` from the result.
  - Add `avo_attributes` to the system-prompt tool list.

## Data flow

```
LLM → avo_attributes(vp1..rho2) → {intercept, gradient, avo_class, description}
        → context: last_avo_attributes
        → auto-chain plot_avo_crossplot(A, B, class) → {image_path}
```

## Error / edge handling

- Unphysical medium (`vs≥vp`, non-positive) → `ValueError` (per-medium guard).
- `B == 0` exactly → falls into the `B >= 0` branch (Class I* if A>0; Class II band
  takes precedence when `|A| <= 0.02`).
- Point at the origin (`A=B=0`, e.g. identical media) → Class II; crossplot uses a
  minimum symmetric extent so the plot isn't degenerate.
- `plot_avo_crossplot` with no `avo_class` → plots the point without a class annotation.

## Testing (`tests/test_avo_attributes.py`)

1. **Intercept/gradient correctness:** `avo_attributes` `intercept`/`gradient` equal
   `_shuey_coefficients(...)` `R0`/`G`, and `intercept` equals `shuey_reflectivity(..., [0.0])[0]`
   (the θ=0 limit) within tolerance.
2. **`shuey_reflectivity` unchanged:** a value test pinning `shuey_reflectivity` output
   for a fixed interface/angles is identical before/after the helper refactor (reuse or
   mirror an existing AVO test if present).
3. **Class III (gas sand):** an interface with Vp and ρ both dropping and B<0 →
   `avo_class == "III"`.
4. **Class I (hard event):** A>0, B<0 → `"I"`.
5. **Class II (near-zero intercept):** an interface tuned so `|A| <= 0.02` → `"II"`/`"IIp"`.
6. **Class IV:** A<0, B>0 → `"IV"`.
7. **Guards:** `vs1 >= vp1` raises `ValueError`.
8. **Registry + plot:** `avo_attributes` and `plot_avo_crossplot` are registered;
   `REGISTRY_BY_NAME["avo_attributes"].auto_plot == "plot_avo_crossplot"`;
   `plot_avo_crossplot(0.1, -0.2, "I")` returns an existing PNG path.

(For tests 3–6, the implementer picks concrete velocities/densities that produce the
intended (A,B) signs and verifies the resulting class — the assertion is on the class
label, not hand-computed A/B magnitudes.)

## Files touched

- `tools/avo_tools.py` — `_shuey_coefficients`, `avo_attributes`, `plot_avo_crossplot`;
  refactor `shuey_reflectivity` to use the helper.
- `core/tool_registry.py` — two new `ToolSpec`s + import.
- `core/chatbot_tool_use.py` — chaining + context + system-prompt list.
- `tests/test_avo_attributes.py` (new); `tests/test_tool_registry.py` (count bump).
- `CLAUDE.md` — document under the AVO section.

## Follow-ups (not this cycle)

- Fit A/B by regression from a measured `(angles, rc)` curve (second input mode).
- Background-trend line + fluid/lithology overlays and χ-rotation on the crossplot.
- Crossplotting a *family* of interfaces (multi-point, e.g. from a log) on one plot.
