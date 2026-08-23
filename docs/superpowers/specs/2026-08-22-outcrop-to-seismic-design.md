# Outcrop photo → seismic section — design

**Date:** 2026-08-22
**Branch:** `stabilize-tool-layer`
**Status:** approved design, pre-implementation

## Purpose

Let a user upload an outcrop photograph and get a synthetic seismic section of
that geology "as if buried": interpret the photo into facies regions, establish
a physical scale, map each facies to elastic properties on a shale background,
and convolve a 2-D reflectivity model with a wavelet. Output is a wiggle or
variable-density image, in time (default) or depth-converted.

This extends the N-layer 1-D synthetic (`tools/synthetic_tools.py`) with a
generic **2-D convolutional model** and adds the first **vision** capability to
the package.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Lithology from photo | Vision LLM, behind a `VisionClient` interface with two backends (Anthropic SDK; OpenAI-compatible vision endpoint) |
| Scale | VLM estimates from references (scale bar / hammer / person / vehicle) with confidence; user may override; if none, the tool asks for a height |
| Geometry | 2-D facies mask (polygons) primary; layer-cake "bands" fallback rasterized through the same path |
| Elastic mapping | lithology → (φ, Vclay, fluid) → Han (1986) + Gassmann via `predict_layer` for clastics; direct literature Vp/Vs/ρ for non-clastics |
| Ingestion | Gradio image upload into a sandboxed dir + `image_path` argument (API/CLI callers pass a server path) |
| Output domain | Time-domain compute; `domain="depth"` returns a depth-converted section |
| Architecture | Staged registry tools with `ContextManager` hand-off plus a one-shot `outcrop_to_seismic` workflow recipe |

## Architecture and data flow

```
photo.jpg ──(Gradio upload / path)──► SEISMIC_UPLOAD_DIR ─► context.last_image
                                                          │
  [1] interpret_outcrop(image_path) ── VisionClient ──► OutcropInterpretation ─► context.last_outcrop
        └─ AUTO_PLOT ─► plot_outcrop_interpretation (photo + polygon overlay + legend)
  [2] outcrop_to_model(height_m?, overrides?) ─────────► EarthModel2D (facies, vp/vs/rho grids) ─► context.last_earth_model
  [3] synthetic_section(wavelet_freq, angle, domain) ──► SeismicSection (nt × nx) ─► context.last_section
        └─ AUTO_PLOT ─► plot_seismic_section (model | wiggle / image)
  outcrop_to_seismic recipe = [1]→[2]→[3]→plot in one call (WorkflowSpec, run_sweep-compatible)
```

**Boundary rule:** only `interpret_outcrop` (and the recipe through it) calls
the VLM. Steps 2–3 are deterministic NumPy, so corrections ("the cliff is
40 m", "make the channel gas-filled") re-run without network or cost.

### New modules

| Module | Responsibility | Depends on |
|---|---|---|
| `core/vision_client.py` | `VisionClient` protocol `interpret_image(image_bytes, mime, prompt, schema) -> dict`; `AnthropicVisionClient`, `OpenAIVisionClient`; `resolve_vision_backend()` (env → backend or clear error) | `anthropic` (lazy import), `openai` |
| `tools/image_safety.py` | `safe_image_path(path, base_dir)` — extension allow-list, size cap, traversal rejection, sandbox confinement; `downscale_for_vision(path, max_edge=1568) -> (bytes, mime)` | Pillow (already pulled by matplotlib/gradio) |
| `tools/outcrop_tools.py` | VLM prompt + JSON schema; `OutcropInterpretation` validation/normalization; `LITHOLOGY_TABLE`; `interpret_outcrop`; `plot_outcrop_interpretation`; `outcrop_to_model` (rasterize + elastic mapping + padding) | `vision_client`, `image_safety`, `workflows.adapters.predict_layer`, `matplotlib.path` |
| `tools/section_tools.py` | `create_synthetic_section` (generic 2-D convolutional model over any elastic grid); `plot_seismic_section`. **No outcrop knowledge.** | `ricker_tools.gen_wavelet`, `avo_tools` (Shuey/Zoeppritz), `physics_guards` |
| `workflows/recipes/outcrop_to_seismic.py` | one-shot chain; registered as `WorkflowSpec` in `workflows/engine.py` | the above |

### Touched existing modules

- `core/tool_registry.py` — `ToolSpec`s for `interpret_outcrop`, `plot_outcrop_interpretation`, `outcrop_to_model`, `synthetic_section`, `plot_seismic_section`; `AUTO_PLOT` entries `interpret_outcrop→plot_outcrop_interpretation`, `synthetic_section→plot_seismic_section`.
- `core/chatbot_tool_use.py` — fill `image_path` from `last_image` when omitted; store `last_outcrop` / `last_earth_model` / `last_section`; auto-chain reads them the way the synthetic chain reads `last_synthetic`; `_compact_tool_result` drops grids; system-prompt bullets describing the staged flow and override vocabulary.
- `core/context_manager.py` — no structural change; new keys live in the existing per-session dict (`last_image`, `last_outcrop`, `last_earth_model`, `last_section`).
- `interfaces/gradio_interface.py` — `gr.Image(type="filepath")` beside the textbox; on change, copy into the session sandbox and set `last_image`; prepend `[image attached: <path>]` to the next user message.
- `config/settings.py` — vision + upload env vars (below).
- `requirements.txt` / `pyproject.toml` — add `anthropic`.
- `config/example_prompts.py` + `interfaces/web_interface.html` (kept in sync) — example prompts.
- `CLAUDE.md` — new section; roadmap tick.
- `interfaces/api_interface.py` — **unchanged** in v1 (callers pass a server-side `image_path`).

## Data contracts

### `OutcropInterpretation` (validated VLM output)

Coordinates are normalized image fractions (x → right, y → down, 0–1).

```json
{
  "regions": [
    {"id": 1, "label": "channel sandstone", "lithology": "sandstone",
     "geometry": {"type": "polygon", "points": [[0.12, 0.40], [0.55, 0.38], [0.50, 0.62]]},
     "porosity": 0.22, "vclay": 0.08,
     "confidence": "medium", "notes": "cross-bedded, lenticular"}
  ],
  "scale": {"estimated_height_m": 35, "reference": "person", "confidence": "low"},
  "background_lithology": "shale",
  "mode": "polygons"
}
```

- `mode: "bands"` (layer-cake fallback): `geometry {"type": "band", "y_top": 0.2, "y_bottom": 0.35}`, rasterized as a full-width rectangle through the same polygon fill.
- `lithology` must be a key of `LITHOLOGY_TABLE`; `porosity`/`vclay` are optional hints overriding table defaults (clipped to Han range with the existing warn-then-clip).
- `lithology: "cover"` (sky, vegetation, talus, scree) is rasterized as background, never as rock.
- `scale.estimated_height_m` may be `null`; `reference` is free text; `confidence ∈ {low, medium, high}`.
- Validation: polygons need ≥ 3 points, all coordinates in [0,1], ids unique. Invalid JSON or schema failure → one retry with the validation error appended to the prompt; second failure → `ValueError("could not interpret image: …")`.
- The VLM prompt contains the image and fixed instructions only; user free text is never injected into it. User guidance flows through `overrides`.

### `LITHOLOGY_TABLE`

Two routes because Han (1986) is a clastic model and must not be applied to carbonates.

| lithology | route | defaults |
|---|---|---|
| `shale`, `mudstone` | Han/Gassmann via `predict_layer` | φ 0.10, Vcl 0.70, brine |
| `siltstone` | Han | φ 0.15, Vcl 0.40, brine |
| `sandstone` | Han | φ 0.20, Vcl 0.10, brine |
| `clean_sandstone` | Han | φ 0.25, Vcl 0.02, brine |
| `conglomerate` | Han | φ 0.15, Vcl 0.05, brine |
| `limestone` | direct | Vp 5000, Vs 2700, ρ 2.55 |
| `dolomite` | direct | Vp 5800, Vs 3200, ρ 2.75 |
| `chalk` | direct | Vp 3500, Vs 1900, ρ 2.20 |
| `salt` | direct | Vp 4500, Vs 2600, ρ 2.10 |
| `coal` | direct | Vp 2400, Vs 1200, ρ 1.40 |
| `basalt` | direct | Vp 5500, Vs 3100, ρ 2.80 |
| `cover` | background | — |

Direct values are literature "typical" numbers (Mavko et al., Rock Physics Handbook; Bourbié et al.) and are documented as such in the table's docstring.

**Overrides** (`outcrop_to_model(overrides={...})`), keyed by region `id` (int) or `label` (str): `lithology`, `fluid` (`water|brine|oil|gas`), `porosity`, `vclay`. `fluid`/`porosity`/`vclay` apply only to Han-route lithologies; on a direct-route rock they raise `ValueError` naming the rock. `background_lithology` is a separate keyword (default `shale`).

### `EarthModel2D` (output of `outcrop_to_model`)

- **Scale resolution order:** explicit `height_m` → `last_outcrop.scale.estimated_height_m` (echoed back with its confidence and reference) → `ValueError("I need the outcrop height in metres …")`. Width = height × image aspect ratio (square pixels, no perspective correction).
- **Grid:** `dz = max(height_m / 400, 0.1)` m (≈ 400 rows); `num_traces` default 101 → `dx = width / (nx − 1)`. Polygons filled with `matplotlib.path.Path.contains_points` on cell centres; later regions overwrite earlier ones (list order = draw order).
- **Padding:** background half-spaces of `pad_m` above and below the image extent; default `pad_m = 1.5 · v_bg / wavelet_freq` (1.5 wavelengths at the background Vp). The time axis starts at 0 at the top of the pad. No depth-dependent compaction (Han is fixed at 40 MPa) — stated assumption.
- **Fields:** `facies` (int grid) + `legend` (id → lithology/label), `vp`, `vs`, `rho` (m/s, m/s, g/cc), `z`, `x` axes (m), `dz`, `dx`, `provenance` (region table with resolved φ/Vcl/fluid or direct values, scale source, overrides applied).

### `create_synthetic_section` (generic 2-D convolutional model)

Inputs: `vp`, `vs`, `rho` grids `(nz × nx)`, `dz`, `dx`, `wavelet_freq=30`, `wavelet_type="ricker"|"ormsby"`, `ormsby_freq`, `dt=0.001` s, `angle=0`, `method="shuey"|"zoeppritz"`, `domain="time"|"depth"`.

Per column: cumulative TWT `t(z) = Σ 2·dz/vp`; an interface wherever adjacent samples differ in any of (vp, vs, rho); RC = acoustic at `angle=0`, else Shuey (default) or exact Zoeppritz from `avo_tools`; interface TWT rounded to the `dt` grid with **superposition** (`+=`, matching the 1-D tool); convolution with `gen_wavelet`. Output `section (nt × nx)`, `time_axis` (ms), `parameters`.

`domain="depth"`: each column is interpolated back through its own `t(z)` onto the model's `z` axis; returns `(section_depth (nz × nx), z_axis)`. The time-domain result is always what is computed; depth is a display conversion.

Guards: `require_elastic_medium` on the grids (reject), `warn_if_aliased` for wavelet vs `dt` (warn). Post-critical Zoeppritz NaNs are replaced by 0 with a warning (documented; the 1-D tool currently propagates NaN — fix there is a separate follow-up).

### `plot_seismic_section`

Panels: facies/AI model (left) and section (right). `display="image"` (variable-density, symmetric seismic colormap, autoscaled to max |amplitude|), `"wiggle"` (VAWIG as in the wedge plot; traces decimated to ≤ 80), or `"both"` (three panels). Axis label follows `domain`. Returns a PNG path.

### `plot_outcrop_interpretation`

Original photo with filled, semi-transparent polygons coloured by lithology, region ids, a legend, and the scale estimate in the title. Returns a PNG path.

## Ingestion

- `gr.Image(type="filepath", label="Outcrop photo")` next to the textbox. On change: `safe_image_path` copies the file into `SEISMIC_UPLOAD_DIR/<session-uuid>/<uuid>.<ext>` (allow-list `.jpg .jpeg .png .webp`; `MAX_IMAGE_MB` default 10; traversal/absolute outside sandbox rejected) and stores the sandboxed path in `context.last_image` (per session).
- On the next send, the handler prepends `[image attached: <path>]` to the user text so the chat LLM knows to call `interpret_outcrop`.
- `interpret_outcrop.image_path` is optional in the schema; the chatbot fills it from `last_image` when omitted. Neither present → "Please upload an outcrop photo first."
- The image is downscaled to ≤ 1568 px on the long edge for the VLM call; overlays use the original.

## Configuration (`config/settings.py`)

| Var | Default | Effect |
|---|---|---|
| `VISION_PROVIDER` | auto | `anthropic` or `openai`. Auto: anthropic if `ANTHROPIC_API_KEY` set, else openai if `VISION_API_KEY`+`VISION_BASE_URL`, else vision disabled |
| `ANTHROPIC_API_KEY` | — | Anthropic backend |
| `VISION_API_KEY`, `VISION_BASE_URL` | — | OpenAI-compatible vision backend (GPT-4o, Databricks-served VLM) |
| `VISION_MODEL` | provider default (`claude-sonnet-5` / `gpt-4o`) | model name |
| `SEISMIC_UPLOAD_DIR` | `<tmpdir>/seismic_uploads` | image sandbox |
| `MAX_IMAGE_MB` | 10 | upload cap |

Vision credentials are optional: missing ⇒ `interpret_outcrop` raises `RuntimeError("vision provider not configured …")` at call time; package import, startup, and every other tool keep working. The `anthropic` backend is imported lazily.

## Error handling

| Failure | Behaviour |
|---|---|
| No image / bad extension / too large / outside sandbox | `ValueError` with a plain-language message, surfaced as the chat reply |
| VLM invalid JSON / schema failure | one retry with the error; then `ValueError("could not interpret image: …")` |
| No scale and no `height_m` | `ValueError` asking for height; `last_outcrop` kept so only step 2 re-runs |
| Fluid/φ/Vcl override on a direct-route lithology | `ValueError` naming the rock and why |
| Non-physical grid from overrides | `physics_guards` reject; aliasing → warn |
| Vision not configured | `RuntimeError` at call time |

`_compact_tool_result` drops `facies`, `vp`, `vs`, `rho`, `section` and returns a summary (region table, scale + confidence, grid shape, peak amplitude) plus `image_path`.

## Testing (offline; `tests/`)

- `conftest.py`: `FakeVisionClient` fixture returning scripted `OutcropInterpretation` JSON (optionally a first invalid response to exercise the retry).
- `test_vision_client.py` — backend selection and fail-fast; schema validation; retry-once; coordinate normalization.
- `test_image_safety.py` — allow-list, size cap, traversal, confinement, downscale.
- `test_outcrop_tools.py` — polygon fill and draw order; bands ≡ rectangle polygons; `cover` → background; Han vs direct routes; overrides and their errors; scale resolution order; padding extent; provenance.
- `test_section_tools.py` — **oracles**: a horizontal 3-band model reproduces `create_synthetic_seismogram` per column (event separation and amplitudes); a single column reproduces `wedge_model`'s max-thickness trace; acoustic/Shuey/Zoeppritz agree at angle 0; depth-conversion round-trip places events at the right depth; wiggle decimation; plot returns a PNG.
- `test_outcrop_to_seismic.py` — recipe end-to-end with `FakeVisionClient`; `WorkflowSpec` registration; `run_sweep` metrics (`max_abs_amplitude`, `n_regions`).
- `test_chatbot_outcrop.py` — with `fake_llm_factory` + `FakeVisionClient`: `image_path` filled from `last_image`; both auto-chains; context keys set; compaction drops grids; `last_image` isolated across sessions.
- `test_outcrop_vision.py` at the package root: real-VLM smoke script, credential-gated, not in the suite.

## Out of scope (YAGNI)

Perspective/orthorectification, depth-dependent compaction, API multipart upload route, interactive polygon editing, multi-photo mosaics, anisotropy / attenuation / multiples, per-region angle gathers.
