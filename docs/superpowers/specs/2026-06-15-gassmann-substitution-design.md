# Gassmann fluid substitution tool — design

- **Date:** 2026-06-15
- **Status:** Approved (pending spec review)
- **Scope:** One implementation cycle (spec → plan → implement)
- **Roadmap item:** #1 (Phase 1, AVO interpretation) of
  `2026-06-15-scientific-completeness-roadmap.md`
- **Package:** `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`)

## Problem

Gassmann fluid substitution — "what do these elastic properties look like with a
different pore fluid" — is the core rock-physics step that feeds AVO and wedge modeling
(e.g. modeling the gas case of a brine sand). The forward/inverse Gassmann primitives
(`gassmann_sat`, `gassmann_dry`) already exist and are regression-tested in
`tools/rock_physics_tools.py`, but they are **not LLM-facing**: the only way to reach
them is the bundled `calculate_rock_properties`, which starts from porosity/clay and the
Han(1986) regression rather than from user-supplied in-situ velocities. There is no tool
that takes measured `(vp, vs, rho)` and substitutes a fluid.

## Goals

1. Expose a standalone `gassmann_substitution` tool: in-situ `(vp, vs, rho)` + porosity +
   fluid swap → substituted `(vp, vs, rho)`.
2. Support both the preset fluids (`water`/`brine`/`oil`/`gas`) and **custom-fluid
   overrides** (explicit fluid modulus/density for in and/or out).
3. Reuse the existing, verified `gassmann_sat`/`gassmann_dry` primitives — no new physics.

## Non-goals (out of scope)

- Refactoring `calculate_rock_properties` to call the new function (avoid regression;
  both simply share the same two primitives).
- Any plot/`auto_plot` (per the roadmap, this tool returns values only).
- Mineral mixing from clay volume (a single `k_mineral` parameter suffices; the
  Voigt-Reuss-Hill clay/quartz mix stays internal to `calculate_rock_properties`).
- Temperature/pressure (Batzle-Wang) fluid property modeling — presets + overrides only.

## Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| Input contract | Measured `(vp, vs, rho)` + `phi` + `fluid_in`/`fluid_out` |
| Custom fluids | **Include** optional `k_fl_*`/`rho_fl_*` overrides (in and out) |
| Mineral modulus | Single `k_mineral`, exposed in **GPa** (quartz default 37), converted to Pa internally |
| Array support | Yes — vectorized like `calculate_rock_properties` |
| Plot | None |
| Relationship to `calculate_rock_properties` | Independent; both built on `gassmann_sat`/`gassmann_dry` |

## Physics

Standard Gassmann fluid substitution (Mavko et al., *Rock Physics Handbook*):

1. From in-situ `(vp, vs, rho)`:
   - `mu = rho · vs²` — shear modulus, **fluid-independent** (Gassmann assumption).
   - `K_sat_in = rho · vp² − (4/3) · mu`.
2. Invert to the dry frame: `K_dry = gassmann_dry(K_sat_in, K0, K_fl_in, phi)`.
3. Forward-substitute the new fluid: `K_sat_out = gassmann_sat(K_dry, K0, K_fl_out, phi)`.
4. Density swap (only the pore fluid changes):
   `rho_out = rho_in + phi · (rho_fl_out − rho_fl_in)`.
5. Substituted velocities:
   - `vp_out = √((K_sat_out + (4/3)·mu) / rho_out)`.
   - `vs_out = √(mu / rho_out)`  (mu unchanged; Vs shifts only via density).

`K0 = k_mineral` (GPa→Pa). `K_fl`, `rho_fl` come from the preset `_FLUIDS` table unless
overridden. Density is in g/cc on the API boundary and converted to kg/m³ where the
modulus formulas require SI (matching the existing code's convention).

## API

```
gassmann_substitution(
    vp, vs, rho, phi, fluid_in, fluid_out,
    k_mineral=37.0,                  # GPa (quartz); converted to Pa internally
    k_fl_in=None,  rho_fl_in=None,   # override the in-situ fluid (GPa, g/cc)
    k_fl_out=None, rho_fl_out=None,  # override the target fluid  (GPa, g/cc)
    print_results=True,
) -> dict
```

- `vp`, `vs` in m/s; `rho` in g/cc; `phi` fraction; all scalar **or** array-like
  (broadcast together).
- `fluid_in`/`fluid_out`: one of `water`, `brine`, `oil`, `gas` (case-insensitive). When
  the corresponding `k_fl_*`/`rho_fl_*` override is supplied, it takes precedence over the
  preset for that side; the preset name is then optional/ignored for that fluid's moduli.
- `k_fl_*` overrides are given in **GPa** (consistent with `k_mineral`), `rho_fl_*` in g/cc.

**Returns** a dict:
- `vp`, `vs`, `rho` — substituted properties (same units as input; same shape).
- `vp_vs` — substituted Vp/Vs ratio.
- `k_dry`, `k_sat` — dry-frame and substituted saturated bulk modulus (GPa).
- `mu` — shear modulus (GPa, unchanged by substitution).

## Guards (`tools/physics_guards`)

- **REJECT** (`ValueError`): `phi` outside `[0, 1]`; non-positive `vp`, `vs`, `rho`,
  `k_mineral`, or any supplied `k_fl_*`/`rho_fl_*`; unknown `fluid_in`/`fluid_out` name
  (when no matching override is given).
- **WARN** (`warnings.warn`, proceed): inverted `K_dry` is non-physical
  (`K_dry < 0` or `K_dry > K0`) — signals inconsistent `(vp, vs, rho, phi, k_mineral)`
  inputs; the substitution still returns a value.

Reuse `require_positive` for the positivity checks; porosity-range and fluid-name checks
are explicit (mirroring `calculate_rock_properties`).

## Wiring

- **`tools/rock_physics_tools.py`** — add `gassmann_substitution` built on the existing
  `gassmann_sat`/`gassmann_dry`. Add a small `_fluid_moduli(name, k_override, rho_override)`
  helper that resolves preset-or-override → `(K_fl_Pa, rho_fl_gcc)`.
- **`core/tool_registry.py`** — one `ToolSpec`:
  - `name="gassmann_substitution"`, `fn=gassmann_substitution`.
  - `params`: `vp, vs, rho, phi` (required), `fluid_in, fluid_out` (required string),
    `k_mineral, k_fl_in, rho_fl_in, k_fl_out, rho_fl_out` (optional).
  - `required=["vp","vs","rho","phi","fluid_in","fluid_out"]`.
  - `defaults={"k_mineral": 37.0}`.
  - `validator=None` (guards live in the function), `auto_plot=None`.
  - Derived maps (`TOOL_SCHEMAS`, `TOOL_FUNCTIONS`) update automatically.
- **`core/chatbot_tool_use.py`** — add `gassmann_substitution` to the system-prompt tool
  list. No auto-chaining, no context state (returns values the LLM can pass forward).

## Data flow

```
LLM → gassmann_substitution(vp,vs,rho,phi, fluid_in='brine', fluid_out='gas')
        → {vp', vs', rho', vp_vs, k_dry, k_sat, mu}
        → LLM can feed vp'/vs'/rho' into zoeppritz/shuey/wedge tools
```

## Error / edge handling

- `phi = 0` → no pore fluid; `K_dry == K_sat` and density unchanged. Valid (returns
  inputs essentially unchanged); `gassmann_dry`/`gassmann_sat` are well-defined at φ=0.
- Override given for only one side → that side uses the override, the other uses its
  preset. Valid.
- Array inputs of mismatched shapes that cannot broadcast → NumPy raises (acceptable).
- `K_dry` non-physical → warn, still return.

## Testing (`tests/test_gassmann_substitution.py`)

1. **Round-trip identity:** `fluid_in == fluid_out` (e.g. brine→brine) returns vp/vs/rho
   unchanged (within tolerance).
2. **Gas physics signature:** brine→gas **lowers Vp** and **raises Vs** (mu fixed, lower
   density) and lowers density — the same signature asserted for the rock-physics module.
3. **Consistency cross-check:** for a case matched to `calculate_rock_properties`
   (same K0, fluids, φ, in-situ moduli derived from its water-sat output), the substituted
   Vp/Vs agree within tolerance.
4. **Array inputs:** vector `vp/vs/rho/phi` returns arrays of matching shape.
5. **Custom-fluid override:** supplying `k_fl_out`/`rho_fl_out` overrides the preset
   (e.g. override `gas` moduli → different result than preset gas).
6. **Guards:** `phi > 1` raises; `phi < 0` raises; non-positive `vp`/`rho`/`k_mineral`
   raises; unknown fluid name raises.
7. **Non-physical K_dry warns** (e.g. inconsistent low-Vp input) but still returns.

## Files touched

- `tools/rock_physics_tools.py` — add `gassmann_substitution` + `_fluid_moduli` helper.
- `core/tool_registry.py` — one new `ToolSpec` + import.
- `core/chatbot_tool_use.py` — add to system-prompt tool list.
- `tests/test_gassmann_substitution.py` (new).
- `CLAUDE.md` — document the tool under the rock-physics section.

## Follow-ups (not this cycle)

- AVO modeling of the substituted case as an auto-chain (deferred; the LLM can chain
  manually for now).
- Patchy vs uniform saturation mixing (Brie / Voigt-Reuss bounds on `K_fl`).
