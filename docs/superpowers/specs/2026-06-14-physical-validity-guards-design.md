# Physical-validity guards — design

- **Date:** 2026-06-14
- **Status:** Approved (pending spec review)
- **Scope:** One implementation cycle (spec → plan → implement)
- **Package:** `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`)

## Problem

The compute tools accept physically impossible or out-of-range inputs and produce
confident garbage (NaNs, unphysical velocities, wrong tuning). Concretely:

- `validate_avo` does **presence checks only** — no physical validation. `zoeppritz_reflectivity` / `shuey_reflectivity` accept `vs ≥ vp`, non-positive velocity/density, etc.
- `validate_wedge_model` enforces a hard `1500 ≤ v ≤ 6500 m/s` band that **wrongly rejects** valid media (gas sands, unconsolidated sediments, evaporites), and never checks `vs < vp`.
- `calculate_rock_properties` **silently clips** porosity/clay to the Han range, hiding the fact that the input was out of range.
- No aliasing/Nyquist check anywhere (a coarse `dt` vs a high wavelet frequency silently aliases).
- Dead `_validate_velocity_sequence` / `_validate_density_sequence` in the unused `ParameterValidator` reject velocity/density **inversions** — which are physical and central to AVO. Latent, but a landmine if revived.

## Goals

1. Reject physically impossible inputs with a clear, user-facing error.
2. Warn (and proceed) on inputs that are possible but outside a model's validity / numerically risky.
3. Protect **every** caller: the LLM tool path, direct Python/API calls, internal calls (`wedge → shuey`), and tests.
4. Never reject velocity/density inversions.

## Non-goals (deliberately out of scope)

- Surfacing warnings into the chat UI. Warnings go to logs/stderr (`warnings.warn`) for now; chat-surfacing is a follow-up.
- New physics (angle gathers, anisotropy, Q, multiples, N-layer). Tracked separately.
- Reworking the dead `ParameterValidator` beyond removing the incorrect inversion checks.

## Policy: two-tier

- **REJECT** → raise `ValueError`. On the tool path this propagates through `ToolManager` to the chatbot as a tool error; on direct calls it surfaces to the caller.
- **WARN** → `warnings.warn(...)`, then proceed (clamping only where a model strictly requires it, e.g. Han range).

## Architecture / placement

New module **`tools/physics_guards.py`** holding small, pure, individually-tested helpers. Wired into **both**:

- the registry validators (`tools/parameter_validation.py`) — early, clean reject on the tool path; and
- the compute functions — so direct/internal callers are also protected (single source of truth, no divergent logic).

Double-checking on the tool path (validator + function) is harmless: the predicates are cheap and idempotent, and the function-level guard is the authoritative one.

### `tools/physics_guards.py` API

Reject predicates return an error message string or `None` (so validators can return `(False, msg)`); a convenience wrapper raises.

```
elastic_medium_error(vp, vs, rho, label="medium") -> Optional[str]
    # error if not (vp > 0 and rho > 0 and 0 < vs < vp)

positive_error(value, name) -> Optional[str]
    # error if value is None or value <= 0

angles_error(angles) -> Optional[str]
    # error if any angle < 0 or angle >= 90

require_elastic_medium(vp, vs, rho, label="medium") -> None   # raises ValueError on error
require_positive(value, name) -> None                         # raises ValueError on error

warn_if_aliased(max_content_hz, dt_seconds, label="wavelet") -> None
    # warns if max_content_hz >= nyquist, where nyquist = 0.5 / dt_seconds

warn_if_outside(value, lo, hi, name, unit="") -> None
    # warns if value < lo or value > hi
```

Notes:
- `0 < vs < vp` keeps Poisson ratio in (−1, 0.5) automatically, so no separate Poisson check is needed.
- `dt_seconds`: callers pass seconds. The wedge uses `dt` in **ms**, so it passes `dt/1000`.

### Per-tool wiring

**AVO** (`tools/avo_tools.py`): at the top of `zoeppritz_reflectivity` and `shuey_reflectivity`, call `require_elastic_medium` for medium 1 and medium 2, and reject angles outside `[0, 90)`. Warn if any angle > 45° (linearized/Shuey accuracy degrades; also relevant past critical angle). `validate_avo` delegates to the same predicates for an early tool-path reject.

**Wedge** (`tools/wedge_tools.py` `wedge_model`): reject `max_thickness>0`, `wavelet`-derived `dt>0`, `num_traces≥2`, and each layer as an elastic medium (using the resolved `vs` defaults). Warn via `warn_if_aliased(content, dt/1000)` where content ≈ `3·ricker_freq` (Ricker) or `f4` (Ormsby). `validate_wedge_model`: **replace** the `1500–6500` hard reject with positivity reject + `warn_if_outside(v, 300, 8000, "v{i}", "m/s")`; add per-layer `vs<vp` check when `vs` is supplied.

**Ricker/Ormsby** (`tools/ricker_tools.py`): `create_ricker_wavelet` rejects non-positive `frequency`/`time_length`/`dt` and warns `warn_if_aliased(3·frequency, dt)`. `create_ormsby_wavelet` already enforces `f1<f2<f3<f4` and `f1≥0`; add non-positive `time_length`/`dt` rejects and `warn_if_aliased(f4, dt)`. `validate_make_ricker` keeps its existing bounds (the Nyquist warning lives in the function, since validators are pass/fail and do not warn).

**Rock physics** (`tools/rock_physics_tools.py` `calculate_rock_properties`): reject `phit`/`vclay` outside `[0,1]`; replace the silent clip with `warn_if_outside(phit, 0, 0.35, "phit")` / `warn_if_outside(vclay, 0, 0.5, "vclay")` **then** clip to the Han range. (No registry validator exists for this tool, so guards live in the function.)

**Dead code** (`tools/parameter_validation.py`): remove the monotonic ordering requirement from `_validate_velocity_sequence` / `_validate_density_sequence` so inversions are never rejected (or delete the checks if they have no other purpose).

### Thresholds

| Check | Reject | Warn |
|-------|--------|------|
| Velocity (each medium/layer) | `vp ≤ 0` | outside `300–8000 m/s` |
| S-velocity | not `0 < vs < vp` | — |
| Density | `rho ≤ 0` | — |
| max_thickness / time_length / dt / wavelet_freq | `≤ 0` | — |
| num_traces | `< 2` | — |
| AVO angle | `< 0` or `≥ 90` | any `> 45°` |
| Nyquist | — | `max_content_hz ≥ 0.5/dt_s` |
| Porosity `phit` | `< 0` or `> 1` | `> 0.35` (Han range) → clip |
| Clay `vclay` | `< 0` or `> 1` | `> 0.5` (Han range) → clip |

## Testing

- **`tests/test_physics_guards.py`** — unit tests for each helper: `elastic_medium_error` flags `vs≥vp`/non-positive and passes valid; `angles_error` bounds; `warn_if_aliased` warns above Nyquist and is silent below; `warn_if_outside` boundaries.
- **Per-tool tests** (extend existing files or add `tests/test_input_guards.py`):
  - `zoeppritz_reflectivity` / `shuey_reflectivity` raise `ValueError` on `vs>vp` and on `rho≤0`; raise on angle ≥ 90.
  - `create_wedge_model` raises on negative density / zero thickness; **accepts a velocity inversion** (`v2 < v1`) — regression guard for the AVO use case.
  - `create_wedge_model` with a coarse `dt` and high `wavelet_freq` emits a Nyquist `UserWarning` (`pytest.warns`).
  - `create_ricker_wavelet` raises on `frequency ≤ 0`; warns near Nyquist.
  - `calculate_rock_properties` raises on `phit > 1`; **warns** (not silent) on `phit = 0.45` and still returns a clipped, physical result.
- Full suite must stay green; existing valid-input tests must not start failing (they use `vs<vp`, positive params).

## Files touched

- `tools/physics_guards.py` (new)
- `tools/avo_tools.py`
- `tools/wedge_tools.py`
- `tools/ricker_tools.py`
- `tools/rock_physics_tools.py`
- `tools/parameter_validation.py`
- `tests/test_physics_guards.py` (new), plus guard tests in existing/`tests/test_input_guards.py`
- `CLAUDE.md` (note the guard policy under the existing validation/guards documentation)

## Follow-ups (not this cycle)

- Surface `warnings.warn` messages into the chat UI / tool result so users see them.
- Optional: structured warning objects returned alongside tool results.
