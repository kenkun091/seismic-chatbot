# Outcrop → seismic overlay web app (iPad-first) — design

**Date:** 2026-09-01
**Branch:** `reusable-skills` (new work branches from here)
**Status:** approved design, pre-implementation
**Builds on:** `docs/superpowers/specs/2026-08-22-outcrop-to-seismic-design.md`

## Purpose

Split the outcrop-photo → seismic-overlay feature out of the Gradio chat into a
focused, touch/Pencil-first **web app** that runs in Safari on an iPad (and any
desktop browser). Users upload a photo, get the VLM interpretation, **draw lines
and polygons on the photo** to correct or replace the interpreted regions, see
the synthetic wiggle overlay update immediately, sketch free-form notes on top,
talk to the assistant in a side chat that acts on the same session, and save the
whole thing as a project file.

The seismic chatbot package remains the backend and the only physics
implementation. The web app is a client; it never re-implements rock physics,
reflectivity or convolution.

## Platform decision (web app vs. native iPad app)

Decided: **web app (PWA-style), not native.** Rationale recorded so the decision
is revisitable:

- The backend work is identical for either client (the existing API cannot
  reach the outcrop flow at all — see "Current gaps"), so the choice only
  affects the client and distribution.
- The pipeline is online by design: the VLM call must stay server-side
  (provider keys), so native's offline advantage is mostly moot.
- Pointer Events in Safari give Apple Pencil pressure/tilt; palm rejection is an
  iPadOS feature, not an app feature. A polygon/polyline editor over an image is
  a few hundred lines of SVG + TS.
- No Apple developer account, App Store review, or second (Swift) codebase to
  maintain next to the Python one; instant updates; also works on desktop.
- The ratchet is one-way: a PWA can be wrapped (Capacitor) for an App Store
  listing later; a native app cannot cheaply become web.

**Revisit native only if both hold:** users annotate at the outcrop with no
connectivity, *and* LiDAR/ARKit-measured scale is wanted (scale is the weakest
link in the pipeline).

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Meaning of drawn lines | **Two layers.** An *interpretation* layer (polygons / bands / traced contacts → regions) that feeds the model, and a *notes* layer (free strokes, text) that is display-only and never reaches the server |
| Chat | Direct panel-driven UI **plus** a side chat box acting on the same server session |
| State ownership | **Server-owned.** The per-session `ContextManager` is the single source of truth; the client resyncs from it |
| Rendering | Client draws the wiggle overlay itself as SVG from a section array + metric extents; the existing matplotlib PNG remains as an export / Section-tab view |
| Stack | Vanilla TypeScript + Vite, one static bundle mounted by the existing FastAPI app |
| Deployment / auth | Single user or small team, same FastAPI process, existing `X-API-Key` gate; no accounts, no database |
| Persistence | A downloadable **project JSON** (photo + interpretation + notes + parameters); reloadable into a fresh session |

## Current gaps (what the codebase says)

- `POST /chat` (`interfaces/api_interface.py`) takes `{message}` only, returns
  `images` as **server-local file paths**, and creates a **fresh session per
  request**. `interpret_outcrop` needs the session's staged `last_image`, so the
  outcrop flow is unreachable via the API and multi-turn corrections are
  impossible there. No upload, session, or file-serving route exists.
- `core/tool_loop.py::compact_value` strips arrays and paths before anything
  leaves the chat; the interpretation JSON and section arrays never reach a
  client today.
- Geometry is already client-shaped: `OutcropInterpretation.regions[].points`
  are normalized `[x, y]` fractions in `[0, 1]` (x → right, y → down); bands are
  eagerly converted to full-width polygons; `validate_interpretation` is
  idempotent. The text-only `overrides` mechanism cannot change geometry.
- Overlay registration (`tools/section_tools.py::_overlay_figure`) is a metric,
  square-pixel affine map: `y_px/h_px ↔ (z − image_top_m)/height_m`,
  `x_px/w_px ↔ x/width_m`; ≤ `MAX_OVERLAY_TRACES = 40` wiggles.
- There is no drawing UI anywhere; the 2026-08-22 spec deferred "API multipart
  upload route" and "interactive polygon editing".

## Architecture

```
iPad Safari ── /app (static SPA) ──┐
                                   │ same origin, X-API-Key
                                   ▼
FastAPI app (interfaces/api_interface.py)
  ├─ POST /chat                      (unchanged, legacy contract)
  ├─ interfaces/outcrop_api.py       (APIRouter: /sessions/...)
  │     └─ SessionStore (interfaces/sessions.py)
  │           └─ session_id → SeismicChatBotToolUse session
  │                              └─ ContextManager: last_image / last_outcrop /
  │                                 last_earth_model / last_section
  │           every tool route → ToolLoopRunner.execute_call(...)
  └─ StaticFiles(webapp/dist) at /app   (mounted only when the build exists)
```

### Server is the single source of truth

`interfaces/sessions.py::SessionStore` maps `session_id → SessionEntry(bot,
lock, last_used, allowed_files, version)`:

- `create()` calls `base_chatbot.new_session()` (shared heavy components,
  fresh `ContextManager`), sweeps idle sessions first, and refuses beyond the
  cap. `SESSION_TTL_SECONDS` (default 7200) and `MAX_SESSIONS` (default 50) are
  env-configurable. Expiry deletes the session's `SEISMIC_UPLOAD_DIR/<sid>/`
  and its harvested plot files.
- A per-session `threading.Lock` is held for the duration of any request that
  touches the session (the tool loop is not concurrency-safe; the warning
  capture is process-global). A second concurrent request for the same session
  gets `409`.
- `allowed_files` is the only set of paths the file route may serve: the staged
  photo plus every plot path harvested by a chat turn.
- `version` is an integer the store bumps when the object identity of
  `last_outcrop`, `last_earth_model`, or `last_section` differs after a request
  from before it. `ContextManager` is not modified for this; the store compares
  identities around each request. This is how the client learns that a chat
  turn changed something.

### Every tool route goes through `ToolLoopRunner.execute_call`

`execute_call(tool_name, raw_input, collected_images)` is the per-call path
shared by live turns and skill replay: context injection (`last_image` →
`image_path`, `last_outcrop` → `interpretation`, `last_earth_model` → `model`),
warning capture, `tool_call` trace event, context storage, image harvest +
provenance sidecar, auto-plot chaining. The routes reuse it so validators,
physics guards, sandboxes, trace and provenance apply with no parallel code
path.

One change to it: a keyword-only `auto_plot: bool = True` parameter. API tool
routes pass `auto_plot=False` so each call does not render a throwaway PNG.
Default behavior (chat turns, skill replay) is unchanged.

`interfaces/serialize.py` converts tool results to JSON: numpy arrays → nested
lists rounded to 4 significant digits, numpy scalars → Python scalars, paths
dropped unless registered on the session.

## API surface

All `/sessions*` routes use the existing `enforce_chat_policy` dependency
(fail-closed `X-API-Key`, per-client rate limit). `POST /chat` is untouched.

| Route | Body → Response | Notes |
|---|---|---|
| `POST /sessions` | → `{session_id}` | `uuid4`; sweeps expired sessions; `503` when at cap |
| `DELETE /sessions/{id}` | → `204` | removes files |
| `POST /sessions/{id}/image` | multipart `file` → `{width, height, url}` | `tools/image_safety.stage_upload` (`.jpg/.jpeg/.png/.webp`, `MAX_IMAGE_MB`, traversal rejected, Pillow verify) then `session.attach_image`; registers the path in `allowed_files` |
| `GET /sessions/{id}/files/{name}` | → file | **only** names registered on that session; anything else `404` |
| `POST /sessions/{id}/interpret` | `{}` → interpretation | `execute_call("interpret_outcrop", {})`; the one VLM/network hop; `503` when no vision credentials |
| `PUT /sessions/{id}/interpretation` | interpretation → normalized interpretation | **the drawing round-trip**: `validate_interpretation`, then stored as `last_outcrop`. Caps: ≤ 200 regions, ≤ 2000 points per region, ≤ 1 MB body |
| `POST /sessions/{id}/model` | `{height_m?, overrides?, background_lithology?, num_traces?, wavelet_freq?}` → model summary | `execute_call("outcrop_to_model", …)`; response = scalars (`height_m, width_m, image_top_m, dz, dx, nz, nx, pad_m`), `legend`, per-lithology `{vp, vs, rho}`; **no grids** |
| `POST /sessions/{id}/section` | `{wavelet_freq, wv_type, ormsby_freq?, phase_rot, angle, method, dt, pad_time}` → section | `execute_call("synthetic_section", {…, domain: "depth"})`; response `{z, traces, image_top_m, height_m, width_m, dx, max_abs_amplitude, warnings}` where `traces` is `nx` columns of `len(z)` floats (~101 × 600, ≈0.5 MB) |
| `GET /sessions/{id}/plot.png?display=overlay\|image\|wiggle\|both` | → PNG | existing `plot_seismic_section` on `last_section` + `last_earth_model`; export and the Section tab |
| `POST /sessions/{id}/chat` | `{message}` → `{reply, images, trace, version}` | `session.process_single_input`; harvested plot paths are registered and returned as `/files/` URLs |
| `GET /sessions/{id}/state` | → `{version, image, interpretation, model_summary, section_meta}` | client resync; `image` is `{width, height, url}` or `null`; each of the other three is the stored object or `null`; `section_meta` is the section response minus `traces` |

**Errors:** tool/validator `ValueError` → `400 {error}`; missing vision
credentials → `503`; unknown session → `404`; session busy → `409`; body caps →
`413`. Warnings raised by tools are captured (same `warnings.catch_warnings`
pattern as `physics_warning` events) and returned as `warnings: [str]` on
`interpret`/`model`/`section` responses.

## Web client (`webapp/`)

Vite + vanilla TypeScript, no framework. Modules:

- `api.ts` — fetch wrapper (API key and session id from `localStorage`, every
  read/write in try/catch), typed route functions.
- `state.ts` — one store + subscribe; holds image, interpretation (server
  copy + local edits), model summary, section, notes, parameters, `version`,
  `dirty` flag.
- `canvas/` — SVG scene, tools (select, polygon, band, contact, sketch, pan),
  overlay renderer.
- `geometry.ts` — pure functions: normalize/denormalize, Ramer–Douglas–Peucker,
  `linesToBands`, band construction, wiggle path generation.
- `panels/` — Regions, Model, Section, Chat, Notes.
- `project.ts` — project JSON (de)serialization.

### Layout (iPad-first)

The photo canvas fills the main area; a collapsible right panel has tabs
**Regions · Model · Section · Chat · Notes**. A toolbar over the canvas offers
Select · Polygon · Band · Trace contact · Sketch · Pan/zoom, layer toggles
(regions / wiggles / notes) and a wiggle-opacity slider. Pen (`pointerType ===
"pen"`) draws and finger pans by default; a toggle enables finger drawing.
`touch-action: none` on the canvas; palm rejection comes from iPadOS.

### Canvas

One `<svg>` whose `viewBox` is the photo's pixel size; `<image>` at the back,
then `<g id="regions">`, `<g id="wiggles">`, `<g id="notes">`. Pan/zoom is a
CSS transform on a wrapper. **All stored coordinates are normalized `[0, 1]`
(x → right, y → down)** — identical to `OutcropInterpretation.points` — so the
client and server share one representation.

### Drawing model — interpretation layer (feeds the model)

- **Polygon:** tap to add vertices; tap the first vertex to close; ≥ 3 points.
  In Select mode vertices drag, a region can be deleted.
- **Band:** two taps set `y_top` / `y_bottom` → full-width polygon
  `[[0,yt],[1,yt],[1,yb],[0,yb]]` (matches the server's band→polygon rule).
- **Trace contact:** pen stroke → RDP simplification (tolerance ≈ 0.5 % of
  image width) → a contact polyline, kept in client state (not a region).
  **"Lines → bands"** sorts contacts by mean y and builds one region between
  each adjacent pair (plus the image top edge above the first and bottom edge
  below the last), extended to the left/right image edges: for each pair, the
  polygon is upper polyline (left→right) followed by lower polyline
  (right→left). Contacts that cross are rejected with a message.
- Each region row in the Regions tab exposes label, lithology (the 13-term
  enum used by `validate_interpretation`), porosity, vclay, confidence, and a
  VLM-created vs user-drawn badge. New regions default to the background
  lithology with `confidence: "high"` and a note `"user-drawn"`.
- **Commit:** on leaving a drawing tool or tapping Apply, the client `PUT`s the
  whole interpretation and replaces its copy with the normalized response. With
  **auto-update** on (default; debounced ~400 ms) it then re-`POST`s `model`
  and `section`. Both are deterministic and sub-second — no LLM cost.

### Wiggle overlay

From the section response: column centres `x_i = (i + 0.5)·dx`,
`px = x/width_m · w_px`; `py = (z − image_top_m)/height_m · h_px`, cropped to
`[0, h_px]`. Columns are decimated client-side to ≤ 40 (matching
`MAX_OVERLAY_TRACES`), amplitude excursion `0.9 · dx · a/amax` in metres →
pixels, one `<path>` per trace, optional positive-lobe fill. Opacity and
visibility are pure client state. The Section tab shows the server PNG
(`plot.png?display=both`); the client does not render a heatmap.

### Chat tab

Sends to `/sessions/{id}/chat`. Uncommitted region edits are committed (`PUT`)
first so the LLM never works on a stale interpretation. After the reply the
client calls `GET state`; if `version` changed it reloads interpretation, model
summary and section (re-`POST`ing `section` to get traces) into the canvas.
Reply images render inline from `/files/` URLs. The trace headline (tool chain)
is shown under the reply, as the Gradio status line does.

### Notes layer

Free strokes `{points, width, color}` and text labels `{x, y, text}` in
normalized coordinates. **Display-only; the server never receives them.**

### Project save / load

One JSON file:

```json
{"version": 1,
 "image": {"name": "...", "mime": "image/jpeg", "base64": "..."},
 "interpretation": {...}, "notes": {...},
 "model_params": {...}, "section_params": {...}}
```

Saved through a blob download (share sheet on iPad). Load = `POST /sessions` →
upload image → `PUT interpretation` → `POST model` → `POST section`, then
restore notes and parameters locally.

## Security and limits

- All new routes sit behind the existing fail-closed key gate and rate limiter;
  `interpret` and `chat` are the only billed calls.
- File serving is allow-listed per session (staged photo, harvested plots);
  arbitrary paths cannot be requested.
- Uploads go through the existing `tools/image_safety` sandbox unchanged.
- Interpretation body caps (200 regions, 2000 points/region, 1 MB) bound
  rasterization cost; `num_traces`/`nz_target` keep their registry validators.
- Same-origin SPA ⇒ no CORS configuration; `uuid4` session ids; session cap and
  idle TTL bound memory.

## Error handling (client)

Toasts for `4xx/5xx` with the server message. A failed `PUT` keeps local edits
and shows an "unsynced" badge; auto-update pauses until the next successful
commit. A `404` on the session offers "start a new session" (project state is
local, so nothing is lost). Chat errors show inline in the thread.

## Testing

**Server** (`pytest`, `fastapi.testclient.TestClient`, `fake_llm_factory`, a
fake `VisionClient` returning a fixed valid interpretation):

- session lifecycle: create, `404` unknown, TTL expiry sweeps files, cap → `503`,
  busy → `409`;
- upload: allow-list, size cap, traversal rejected, path registered;
- `interpret` with the fake VLM stores `last_outcrop` and returns normalized
  JSON; `503` without credentials;
- `PUT interpretation`: idempotent round-trip, `<3`-point polygon → `400`, caps
  → `413`, stored object becomes `last_outcrop`;
- `model` / `section` response shapes; **oracle:** `traces` equals
  `synthetic_section_from_model(..., domain="depth")` to rounding;
- `plot.png` for each `display`;
- **chat shares context:** `PUT` an edited interpretation, then a scripted LLM
  turn that calls `outcrop_to_model(overrides=…)` → `version` bumps and `state`
  reflects the new model; reply images are served through `/files/`;
- file route refuses an unregistered name;
- `execute_call(auto_plot=False)` produces no PNG; default behavior unchanged;
- legacy `POST /chat` contract (`tests/test_api_chat_contract.py`) unchanged.

**Client** (Vitest, pure functions only): normalize/denormalize, RDP,
`linesToBands` (ordering, edge extension, crossing rejection), band
construction, wiggle path generation (registration formula), project
(de)serialization round-trip. No browser E2E initially.

**Manual iPad checklist** (kept in `webapp/README.md`): Pencil draws / finger
pans, polygon close by tapping first vertex, band via two taps, trace contact +
lines→bands, overlay updates after edit, chat edit changes the canvas, project
save via share sheet and reload, works over LAN from a laptop-hosted server.

## Repository layout

```
interfaces/outcrop_api.py     APIRouter with the /sessions routes
interfaces/sessions.py        SessionStore
interfaces/serialize.py       numpy → JSON
core/tool_loop.py             execute_call(..., auto_plot=True)
webapp/                       package.json, vite.config.ts, src/, README.md
webapp/dist/                  built bundle (gitignored); FastAPI mounts it at /app
tests/test_outcrop_api.py, test_sessions.py, test_serialize.py
```

Docs: a "Outcrop web app" section in `CLAUDE.md` (routes, env vars
`SESSION_TTL_SECONDS` / `MAX_SESSIONS`, build command `npm run build`).

## Delivery phases

1. **Server:** `SessionStore`, all routes, serialization, `auto_plot` opt-out,
   tests.
2. **Client core:** canvas, upload/interpret, polygon/band/contact tools,
   lines→bands, wiggle overlay, Model and Section panels, auto-update.
3. **Chat tab** and state resync.
4. **Notes layer**, project save/load, docs and the iPad checklist.

## Out of scope (YAGNI)

Multi-user accounts or a database; offline operation; native wrapper (Capacitor)
until an App Store listing is wanted; LiDAR scale; perspective correction;
client-side heatmap rendering; editing the earth-model grid directly; free-form
notes reaching the model; browser E2E tests.
