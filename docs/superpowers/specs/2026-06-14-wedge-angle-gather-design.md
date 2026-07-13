# Wedge AVO angle gather — design

- **Date:** 2026-06-14
- **Status:** Approved (pending spec review)
- **Scope:** One implementation cycle (spec → plan → implement)
- **Package:** `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`)

## Problem

The wedge model is single-angle: `create_wedge_model` collapses any angle list to the first angle (with a warning) and returns a 2-D synthetic `(nt × num_traces)`. There is no way to see how the wedge response varies with incidence angle — i.e. no AVO gather. This is the most-requested scientific-completeness gap after the single-angle correctness fixes.

## Goals

1. Produce a **true angle gather**: the wedge synthetic computed independently for each incidence angle, as a 3-D cube `(nt × num_traces × nangles)`.
2. Visualize it as **two curves**: amplitude-vs-thickness per angle (tuning family) and amplitude-vs-angle (AVO) at the isolated top interface.
3. Do all of this **without changing the existing single-angle `wedge_model`** or its 2-D contract — fully backward compatible.

## Non-goals (out of scope)

- Modifying `wedge_model`, `plot_wedge_model`, `analyze_wedge`, or their 2-D contract.
- Exact Zoeppritz in the gather (decision: **Shuey only**, consistent with the existing single-angle wedge; Shuey is already accuracy-warned beyond ~45°).
- New physics beyond per-angle Shuey reflectivity (no anisotropy, Q, multiples, NMO, N-layer).
- Surfacing warnings into the chat UI (separate follow-up).

## Decisions (locked during brainstorming)

| Decision | Choice |
|----------|--------|
| API shape | **New dedicated tools**, single-angle wedge untouched |
| Output | **Both curves**: tuning-per-angle + AVO-at-isolated-top-interface |
| RC method | **Shuey only** |
| Implementation | **Self-contained gather function** (Approach A): build geometry once, loop angles; ~30 lines of geometry setup duplicated from `wedge_model` to keep that function untouched and avoid wasted plot renders |

## Architecture

Three new functions in `tools/wedge_tools.py`, wired into the registry and chatbot like the existing wedge trio.

### Compute: `wedge_avo_gather(...)`

```
wedge_avo_gather(
    max_thickness, v1, v2, v3, rho1, rho2, rho3, angles,
    vs1=None, vs2=None, vs3=None,
    wavelet_freq=30.0, num_traces=61, dt=0.1,
    wv_type='ricker', ormsby_freq=None, zunit='m',
) -> (time_array, gather, parameters)
```

- `angles`: non-empty list of incidence angles (deg).
- `gather`: 3-D `np.ndarray` of shape `(nt, num_traces, nangles)`.
- Geometry (interface times, wavelet via `gen_wavelet`, `t0`, `nt`, `dz`, thickness array) is built **once** — it does not depend on angle. For each angle: compute `rc1`, `rc2` via `shuey_reflectivity` (single-angle call), place into an `(nt × num_traces)` reflectivity model at the interface samples, convolve with the wavelet (`mode='same'`), and store as `gather[:, :, k]`.
- Reuses `vs` defaults (`vs_i = vp_i/2` when not supplied) consistent with `wedge_model`.
- **Guards** (reuse `tools/physics_guards`): `require_positive(max_thickness)`, `require_positive(dt)`, `num_traces >= 2`, per-layer `require_elastic_medium`, `warn_if_outside(vp, 300, 8000)`, `angles_error(angles)` (reject empty list or any angle outside `[0, 90)`), and `warn_if_aliased` (content = `3*wavelet_freq` for ricker, last Ormsby corner otherwise; `dt/1000` since dt is ms).
- `parameters` dict carries: `angles` (list), `v2`, `max_thickness`, `num_traces`, `dt`, `wavelet_freq`, `interface1_t`, `t0`, `zunit`, `wavelet_label`.

### Analyze: `analyze_wedge_gather(gather, parameters)`

Returns a dict:
- `angles`: the angle list.
- `tuning_thickness`: `v2 / (4 * wavelet_freq)` (single value; nominal, independent of angle).
- `per_angle`: list of `{angle, tuning_thickness_observed, tuning_amplitude}` where, for each angle's 2-D panel, `amp_vs_thickness = max(|panel|, axis=0)` (matching `analyze_wedge`'s convention), `tuning_amplitude = max(amp_vs_thickness)`, and `tuning_thickness_observed = thickness[argmax]`.
- `avo`: `{angles, amplitudes}` — the **AVO response**: top-interface amplitude at the **maximum-thickness trace** (last trace, where the top reflection is isolated from tuning interference), per angle. Top-interface amplitude is the peak `|amplitude|` on that trace within a window of **± one dominant period** (`1000/wavelet_freq` ms) around `interface1_t[-1]`, converted to sample indices via `dt`. (At max thickness the base interface lies far outside this window, so it isolates the top reflection.)

### Plot: `plot_wedge_gather(gather, parameters)`

Two stacked panels in one figure (matplotlib `Agg`, temp PNG via `tempfile.mkstemp`):
- Top: amplitude-vs-thickness, one line per angle (legend = angle in deg).
- Bottom: amplitude-vs-angle at the max-thickness trace (the AVO curve), with markers.
Returns the PNG path. Surfaced to the UI as `{"image_path": ...}` by the chatbot.

### Registry wiring (`core/tool_registry.py`)

Add three `ToolSpec`s:
- `wedge_avo_gather` → `wedge_avo_gather`, params mirror `wedge_model` plus `angles` (array, required), `method` omitted (Shuey only). `validator=validate_wedge_model` (positivity) — angle/medium validity is enforced inside the function. `auto_plot="plot_wedge_gather"`.
- `plot_wedge_gather` → `plot_wedge_gather`, params `gather` (3-D array) + `parameters` (object).
- `analyze_wedge_gather` → `analyze_wedge_gather`, same params.

Derived maps (`TOOL_SCHEMAS`, `TOOL_FUNCTIONS`, `AUTO_PLOT`) update automatically from the registry.

### Chatbot wiring (`core/chatbot_tool_use.py`)

- Auto-chaining: after `wedge_avo_gather`, invoke `plot_wedge_gather` using the cached gather + parameters (mirror the `wedge_model` → `plot_wedge_model` branch; pull from `last_wedge_gather`).
- `_update_context`: store `last_wedge_gather = {"gather": gather, "parameters": parameters}` from the `(time_array, gather, parameters)` return.
- Add `wedge_avo_gather` to the system-prompt tool list.

## Data flow

```
LLM → wedge_avo_gather(angles=[...]) → (t, cube[nt,ntr,nang], params)
        → context: last_wedge_gather
        → auto-chain plot_wedge_gather(cube, params) → {image_path}
analyze_wedge_gather(cube, params) → tuning per angle + AVO curve
```

## Error / edge handling

- Empty `angles` list → `ValueError` (via `angles_error` / explicit check).
- Any angle outside `[0, 90)` → `ValueError`.
- Unphysical media (`vs≥vp`, non-positive) → `ValueError` (per-layer guard).
- `nangles == 1` → cube has shape `(nt, num_traces, 1)`; AVO panel still renders (single point); valid.
- Aliasing (coarse dt vs high freq) → `UserWarning`, proceed.

## Testing (`tests/test_wedge_gather.py`)

1. `gather.shape == (nt, num_traces, len(angles))`.
2. A single-angle gather panel `gather[:,:,0]` equals the existing single-angle `wedge_model` synthetic for the same angle (consistency with the established path), within tolerance.
3. Per-angle tuning thickness ≈ `v2/(4*wavelet_freq)` (e.g. v2=3000, f=30 → ~25 m), within one trace spacing.
4. AVO for a gas-sand contrast: amplitudes finite, and the AVO curve varies with angle (not constant) — i.e. `analyze_wedge_gather(...)['avo']['amplitudes']` has non-zero spread.
5. Guards: `vs1>=vp1` raises; an angle `>=90` raises; empty `angles` raises.
6. `plot_wedge_gather` returns an existing PNG path; `analyze_wedge_gather` returns the documented dict keys.
7. Inversion accepted: a velocity/density inversion gas sand runs without raising.

## Files touched

- `tools/wedge_tools.py` — add `wedge_avo_gather`, `analyze_wedge_gather`, `plot_wedge_gather`.
- `core/tool_registry.py` — three new `ToolSpec`s + imports.
- `core/chatbot_tool_use.py` — chaining + context + system-prompt tool list.
- `tests/test_wedge_gather.py` (new).
- `CLAUDE.md` — document the gather tools under the wedge section.

## Follow-ups (not this cycle)

- Optional exact-Zoeppritz `method` for wide-angle accuracy.
- Per-angle VAWIG small-multiple panels as an alternate display.
- Extract shared geometry/synthetic core (Approach B) to de-duplicate `wedge_model` and the gather if more layered-model features are added.
