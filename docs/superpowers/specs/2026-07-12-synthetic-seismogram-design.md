# N-layer synthetic seismogram (`synthetic_seismogram` + `petro_to_synthetic`) — design

- **Date:** 2026-07-12
- **Status:** Approved design (pending spec review)
- **Package:** `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`)
- **Roadmap item:** B1 of `2026-06-15-scientific-completeness-roadmap.md` (the last unbuilt keystone)

## Purpose

Every existing synthetic is locked to the fixed 3-layer / 2-interface wedge geometry.
This arc adds a **general N-layer 1-D convolutional synthetic seismogram** — the
backbone for arbitrary stratigraphy and future well-tie work — plus a petrophysics
recipe so the feature is demo-ready end-to-end (porosity/clay per layer → elastic
stack → synthetic trace) in one conversational turn.

## Decisions locked with the user

1. **Angle scope:** single `angle` parameter. `angle=0` (default) → normal-incidence
   acoustic reflectivity; `angle>0` → per-interface `shuey_reflectivity` (default) or
   exact `zoeppritz_reflectivity` via `method`. An N-layer *angle gather* is a
   deliberate later sibling (as `wedge_avo_gather` was to `wedge_model`).
2. **Thickness domain:** meters only, converted internally to two-way time
   (TWT = 2000·h/vp, ms). No `thickness_units` knob.
3. **Scope:** leaf tool + plot **and** a petro-driven workflow recipe in this arc.
4. **Location:** new focused module `tools/synthetic_tools.py`. Wedge code untouched;
   the shared core is the existing primitives (`gen_wavelet`,
   `shuey_reflectivity`/`zoeppritz_reflectivity`, `physics_guards`), not an extracted
   geometry engine.

## 1. Leaf tool — `tools/synthetic_tools.py`

```python
def create_synthetic_seismogram(
    thickness,            # list[float], meters, length N-1 (basal layer = half-space)
    vp,                   # list[float], m/s, length N (N >= 2)
    rho,                  # list[float], g/cc, length N
    vs=None,              # list[float] | None; None -> vs_i = vp_i / 2 (wedge convention)
    wavelet_freq=30.0,    # Hz (Ricker dominant frequency)
    wv_type="ricker",     # 'ricker' | 'ormsby'
    ormsby_freq=None,     # "f1,f2,f3,f4" string when wv_type='ormsby'
    phase_rot=0.0,        # degrees
    angle=0.0,            # incidence angle, degrees; 0 = normal incidence
    method="shuey",       # 'shuey' | 'zoeppritz'; used only when angle > 0
    dt=0.1,               # ms (wedge parity)
    pad_time=50.0,        # ms of quiet time before first / after last interface
    labels=None,          # list[str] | None -> 'layer 1'...'layer N' (plot annotation)
) -> tuple[np.ndarray, np.ndarray, dict]:   # (time_array, trace, parameters)
```

### Semantics

- **Layer/interface contract:** N = `len(vp)` layers ⇒ N−1 interfaces.
  `len(thickness) == N-1` — the last layer is a terminal half-space and takes no
  thickness. The validator error message states this rule explicitly (it is the one
  place an LLM will stumble): e.g. *"thickness must have len(vp)-1 = 3 entries (one
  per layer above the basal half-space); got 4"*.
- **Reflectivity per interface i** (layer i over layer i+1):
  - `angle == 0` → acoustic: `(Z2−Z1)/(Z2+Z1)`, `Z = vp·rho` (matches `wedge_model`'s
    normal-incidence branch).
  - `angle > 0` → `shuey_reflectivity(...)` or `zoeppritz_reflectivity(...)` with
    `angles=[angle]`, taking element 0. Both already verified in `tools/avo_tools.py`.
- **TWT placement:** `interface_t[i] = pad_time + Σ_{j<=i} 2000·thickness[j]/vp[j]`
  (ms). Time axis starts at `t0 = 0`; `nt = round((interface_t[-1] + pad_time)/dt) + 1`.
  If the wavelet (from `gen_wavelet`, `wavelet_length=256.0`) is longer than the
  modeled window, extend padding the way `wedge_model` does so `mode='same'`
  convolution cannot clip the response.
- **Spike series:** `rc_series[idx] += rc_i` with `idx = round(interface_t[i]/dt)`.
  **`+=`, not `=`** — when thin layers collapse onto one sample the reflectivities
  superpose (deliberate improvement over the wedge's assignment; unit-tested).
- **Convolution:** `scipy.signal.convolve(rc_series, wavelet, mode='same')` — same
  call as `wedge_model`.
- **Wavelet:** reuse `gen_wavelet(dt, wv_type, wavelet_freq, ormsby_freq, '', '',
  phase_rot, wavelet_length=256.0)` from `tools/wedge_tools.py`. Ricker/Ormsby/phase
  rotation come free; `parameters['wavelet_freq']` uses the existing Ormsby dominant-
  frequency rule ((f2+f3)/2), mirroring `wedge_model`.

### Guards (existing two-tier system, `tools/physics_guards.py`)

**All REJECT/WARN rules live in `create_synthetic_seismogram` itself** — the recipe
calls the function directly and bypasses the registry validator, so the function
must be self-defending. The registry `validator` additionally duplicates the cheap
structural checks (lengths, N ≥ 2, enums, angle range) so LLM tool calls fail fast
with crisp messages before execution; it delegates to a shared helper rather than
restating the rules.

- REJECT: length mismatches (thickness N−1; vs/rho length N); N < 2; any
  `require_positive` failure on thickness entries, `dt`, `pad_time`, `wavelet_freq`;
  `require_elastic_medium` per layer (on the effective vs after the vp/2 default);
  `angle` outside [0, 90); unknown `method`/`wv_type`; Ormsby corners not
  `f1<f2<f3<f4` (same rule as `make_ormsby`).
- WARN: `warn_if_outside(vp_i, 300, 8000)` per layer; `warn_if_aliased` (3·f_ricker
  or Ormsby f4 vs `dt/1000`), matching `wedge_model`'s convention.
- Velocity/density **inversions stay allowed** (they are the point of AVO modeling).

### `parameters` dict (JSON-friendly; feeds plot, narration, and context)

`n_layers`, `vp`, `vs`, `rho`, `thickness` (lists), `interface_times` (ms),
`rcs` (per interface), `rc_series` (list, for the plot's stem panel), `time_array`
is returned separately but `t0`, `nt`, `dt`, `pad_time` are included so the plot can
reconstruct the axis from `parameters` alone; `angle`, `method`, `wavelet_freq`,
`wavelet_label`, and the resolved `labels` list.

## 2. Plot — `plot_synthetic_seismogram(trace, parameters, output_path=None) -> str`

Three panels sharing a vertical TWT axis, increasing downward:

1. **Layer model** — stepped acoustic-impedance (vp·rho) profile with horizontal
   interface lines and per-layer labels.
2. **Reflectivity** — stem plot of `rcs` at `interface_times`.
3. **Synthetic trace** — wiggle with positive-lobe fill; title carries
   `wavelet_label` and the angle/method when `angle > 0`.

House pattern throughout: `tempfile.mkstemp(suffix=".png")` + `os.close(fd)`,
`dpi=300`, `bbox_inches="tight"`, `plt.close(fig)`, returns the path.

## 3. Registry & chatbot wiring

- **`core/tool_registry.py`:** two new `ToolSpec`s.
  - `synthetic_seismogram` → `fn=create_synthetic_seismogram`;
    `required=["thickness", "vp", "rho"]`; defaults per the signature; a
    `validator` implementing the REJECT rules above (registry fills defaults
    *before* validating, so the validator sees the full dict);
    `auto_plot="plot_synthetic_seismogram"`.
  - `plot_synthetic_seismogram` → mirrors `plot_wedge_model`'s spec shape
    (`trace` array + `parameters` object, both required). Auto-chain only —
    model-initiated plot calls remain forbidden.
- **`core/chatbot_tool_use.py`:**
  - Result-storage branch: `set_context("last_synthetic", {"time_array": ...,
    "trace": ..., "parameters": ...})` alongside the `last_wedge_model` branch.
  - Auto-plot dispatch: new `elif tool_name == "synthetic_seismogram":` branch
    reading `last_synthetic` → `plot_input = {"trace": ..., "parameters": ...}`.
  - System-prompt tool list: one line for `synthetic_seismogram` (the plot tool is
    not advertised, consistent with the narrated-reply contract).

## 4. Recipe — `workflows/recipes/petro_to_synthetic.py`

```python
def petro_to_synthetic(
    phit,                # list[float], per layer, length N
    vclay,               # list[float], per layer, length N
    thickness,           # list[float], meters, length N-1
    fluids=None,         # list[str] | None -> 'brine' for every layer
    labels=None,         # list[str] | None -> 'layer 1'...'layer N'
    wavelet_freq=30.0,
    angle=0.0,
    method="shuey",
) -> dict
```

- Per-layer chain: `predict_layer(phit[i], vclay[i], fluid=fluids[i],
  label=labels[i])` (Han 1986 → `Layer`, already physics-guarded) → stack arrays →
  `create_synthetic_seismogram(..., labels=labels)` → `plot_synthetic_seismogram`
  (one plot implementation; no recipe-specific plot).
- **Recipe-level early-fail guards** (the hardening Task-14 flagged as missing in
  older recipes — this one ships with them): equal lengths for
  `phit`/`vclay`/`fluids`/`labels` (N), `len(thickness) == N-1`, N ≥ 2, positive
  thicknesses — all raised as `ValueError` with actionable messages *before* any
  rock-physics call.
- **Return dict (JSON-friendly, `run_sweep`-compatible):** `layers`
  (list of `{vp, vs, rho, label, fluid}`), `interface_times`, `rcs`,
  scalar metrics `max_abs_amplitude` (peak |trace|) and `max_abs_rc`,
  `n_layers`, `wavelet_freq`, `angle`, `image_path`.
- **`workflows/engine.py`:** one new `WorkflowSpec` (array-typed params mirroring
  the JSON schema style of `eei_optimal_chi_petro`); registration automatically
  exposes it as a chatbot tool and to `run_sweep`. System-prompt workflow list gains
  a `petro_to_synthetic` line.

## 5. Tests (TDD; suite currently ~330 green)

**`tests/test_synthetic_seismogram.py`**
- Spike placement: 3-layer stack with analytic TWTs → trace extrema at expected
  samples (±1 sample).
- Degenerate 2-layer: single interface; peak sign matches RC sign; amplitude scales
  linearly with RC.
- Thin-layer superposition: two interfaces rounding to one sample → RCs add.
- Angle path: interface RC equals `shuey_reflectivity(...)[0]`;
  `method='zoeppritz'` equals the exact solution; `angle>=90` and unknown `method`
  rejected; `vs=None` → vp/2 default honored.
- **Oracle vs `wedge_model`** (the roadmap's named risk): shale/sand/shale stack at
  a fixed thickness vs the matching wedge trace. The two tools use different time
  references (wedge anchors interface 1 at t_ref = 300 ms; the synthetic uses a
  pad_time-based axis), so the oracle compares **event separation** (Δt between the
  two reflections, within 1 sample, tolerating the wedge's known `idx2+1` shift) and
  **event amplitudes** (within rtol) — not absolute times.
- Guards: every REJECT rule above has a test asserting the message names the
  offending parameter; Nyquist warning fires (`pytest.warns`).
- Plot smoke test: PNG exists, non-empty; temp files cleaned up.
- Registry pins: schema derived into `TOOL_SCHEMAS`, `AUTO_PLOT` maps
  `synthetic_seismogram → plot_synthetic_seismogram`, defaults filled before the
  validator runs.

**`tests/test_petro_to_synthetic.py`**
- Chain correctness: brine stack reproduces `predict_layer` outputs in `layers`.
- Fluid array honored: a gas layer lowers Vp and raises Vs vs the brine case
  (consistent with the Gassmann regression suite).
- Recipe guards: each early-fail rule tested by message.
- `run_sweep` smoke: sweep `wavelet_freq` over a small grid collecting
  `max_abs_amplitude`; coverage report clean.
- Result dict is JSON-serializable (`json.dumps` round-trip).
- Chatbot chaining with `fake_llm_factory`: scripted `petro_to_synthetic` call →
  reply narrated, `image_path` harvested, `last_workflow_result` stored.

## 6. Docs & sync

- `CLAUDE.md`: new section ("N-layer synthetic seismogram") in the style of the
  "Wedge AVO angle gather" section.
- `config/example_prompts.py` **and** `interfaces/web_interface.html` (they drift —
  sync both, verify with a diff): two new prompts, one raw-tool, one recipe, e.g.
  *"Build a 4-layer synthetic: shale over gas sand over brine sand over shale,
  porosities 0.08/0.22/0.20/0.10, clay 0.6/0.1/0.15/0.55, thicknesses 40 m and
  25 m and 30 m, 35 Hz."*
- Roadmap doc: tick B1 as done with a pointer to this spec.

## Out of scope (explicit non-goals for this arc)

- N-layer **angle gather** (natural sibling, own brainstorm → spec cycle).
- Attenuation (Q), multiples, transmission losses, NMO/offset, anisotropy.
- Checkshot/well-tie workflows and LAS/well-log file ingestion.
- Time-domain thickness input (`thickness_units`).
- Surfacing physics warnings into the chat UI (pre-existing, package-wide follow-up).

## Process note

Implementation follows the house cycle: this spec → `writing-plans` →
TDD execution → `finishing-a-development-branch`, all on `stabilize-tool-layer`
committed from inside the package repo.
