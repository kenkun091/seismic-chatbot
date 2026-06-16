# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope of this file

This is the **canonical seismic-chatbot package** and its **own git repository** (`git rev-parse --show-toplevel` returns *this* directory; remote `kenkun091/seismic-chatbot.git`). Two broader `CLAUDE.md` files load from parent dirs (`wedge-model/`, `wedge-model/geo-mcp/`) and describe the package from the outside. **Where they disagree with this file about the tool layer, this file wins** — the parents predate the registry refactor (see below).

Current working branch: `stabilize-tool-layer` (the parents still say `rag`). Branches: `main`, `rag`, `tool-use`, `tool-use-databricks`, `mcp`, `interactive-db`, `token-tracking`, `stabilize-tool-layer`.

## Run

Imports are top-level absolute (`from config.settings import ...`). Either run from this dir (so it's the cwd / on `PYTHONPATH`), **or** install the package — `pip install -e .` (see `pyproject.toml`) makes the modules importable from any cwd and installs a `seismic-chatbot` console script (= `main:main`).

```bash
pip install -e .               # optional: installable + `seismic-chatbot` entry point
python main.py                 # Gradio tool-use UI (localhost; public tunnel off unless GRADIO_SHARE=1)
python main.py --mode legacy   # interfaces/gradio_interface_legacy.py
python main.py --test          # run example flows (example_tool_use.py) instead of the UI
pytest                         # full suite (rootdir = this dir; .pytest_cache lives here)
pytest tests/test_tool_registry.py -q          # single file
pytest tests/test_tools.py::<name> -q          # single test
```

**Packaging caveat:** the layout is flat (top-level `config`/`core`/`tools`/`knowledge`/`parsing`/`interfaces` are each installed as top-level packages, matching the absolute imports). This installs and containerizes cleanly in an isolated env, but those generic names would collide in a shared site-packages — a future move into a single `seismic_chatbot/` namespace (with relative imports) is the clean fix. Runtime deps live in `requirements.txt` and are pulled into the wheel via `[tool.setuptools.dynamic]`.

`.env` (gitignored, local-only) must define `DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL`. `core/llm_client.py` switches to **Databricks** instead when `DATABRICKS_TOKEN` + `DATABRICKS_BASE_URL` are set. Despite the `openai` SDK and `OPENAI`-ish naming, the default provider is **DeepSeek** (`deepseek-chat`), just an OpenAI-compatible `base_url`. Config constants (model, temperature, RAG params, log level) live in `config/settings.py`.

## Environment variables

`config/settings.py` reads these at import; the rest are read where noted. Missing **LLM credentials now fail fast** — `resolve_llm_credentials` (`core/llm_client.py`) raises a clear `RuntimeError` at startup instead of silently building `OpenAI(api_key=None)`.

**LLM provider** (one set required)
| Var | Purpose |
|-----|---------|
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | Default provider (`deepseek-chat`). Both required together. |
| `DATABRICKS_TOKEN`, `DATABRICKS_BASE_URL` | Alternative provider; **takes precedence** when both are set. |

**Security containment** — these default to the *safe* posture; you must opt in to expose anything.
| Var | Default | Effect |
|-----|---------|--------|
| `GRADIO_SHARE` | unset (off) | `main.py` only opens a public `*.gradio.live` tunnel when set to `1`/`true`/`yes`. Off ⇒ no public, key-billing endpoint. |
| `GRADIO_HOST` | `127.0.0.1` | Gradio bind address (`main.py`). |
| `API_HOST` | `127.0.0.1` | FastAPI/uvicorn bind address (`interfaces/api_interface.py`). |
| `API_PORT` | `8000` | FastAPI port. |
| `API_AUTH_KEY` | unset | Required for the paid `POST /chat` route. **Fails closed**: if unset, `/chat` returns `503`. When set, callers must send a matching `X-API-Key` header (constant-time check). |
| `CHAT_RATE_MAX` | `30` | Max `/chat` requests per client per window. |
| `CHAT_RATE_WINDOW_SECONDS` | `60` | Sliding-window length for the `/chat` rate limiter. |
| `SEISMIC_EXPORT_DIR` | `<tmpdir>/seismic_exports` | Sandbox dir for `wedge_model`'s `export_path` CSV. The LLM-supplied `export_path` is confined here (`tools/path_safety.py`); **absolute paths and `..` traversal raise `ValueError`** — pass a *relative* name. |

**Other**
| Var | Effect |
|-----|--------|
| `DEBUG` | Enables stderr debug prints and a `save.p` pickle of `wedge_model` inputs (`tools/wedge_tools.py`). |

The security primitives (`interfaces/security.py` — `check_api_key`, `RateLimiter`; `tools/path_safety.py` — `safe_export_path`) are dependency-free and unit-tested (`tests/test_security.py`, `test_path_safety.py`, `test_llm_credentials.py`, `test_wedge_export_sandbox.py`).

## Per-session state (multi-user isolation)

Both chatbots (`core/chatbot_tool_use.py::SeismicChatBotToolUse`, `core/chatbot.py::SeismicChatBot`) split **shared, conversation-stateless** components (LLM client, tool manager, knowledge base, input parser) from **per-session** state. The heavy components are injectable via `__init__(llm_client=..., tool_manager=..., knowledge_base=...)`; the `ContextManager` (last-wavelet/wedge/AVO results, last frequency, token counter) is **always fresh** per instance. `new_session()` spawns an isolated session that reuses the shared components.

- Interfaces build the heavy bot **once** and derive a per-connection session: Gradio holds it in `gr.State(None)` and lazily calls `base_bot.new_session()` (`interfaces/gradio_interface.py`, `gradio_interface_legacy.py`); the FastAPI `/chat` route calls `base_chatbot.new_session()` per request (`interfaces/api_interface.py`).
- **When adding state**, put per-conversation data on `ContextManager` (so it stays isolated) and shared/expensive resources on the injected components. Don't add module-level singletons that hold conversation state, and don't store per-user data on the shared components.
- Covered by `tests/test_session_isolation.py` (isolation + shared-component reuse for both chatbots).

## RAG index lifecycle (idempotent population)

The vector store (`knowledge/vector_db.py`) uses **deterministic, content-derived IDs** (`content_id(text, metadata)`) and **upsert**, and `add_document`/`add_documents` **skip already-stored content** (no re-embedding on startup). So the three populate paths that all target the `seismic_knowledge` collection — `knowledge/rag_system.py::populate_knowledge_base`, `tools/rag_tools.py::_get_knowledge_db`, `tools/rock_physics_tools.py::_get_rock_physics_db` — are now idempotent: repeated runs converge instead of accumulating. (This fixed a store that had grown to **2831 rows for ~32 chunks**; a clean rebuild is ~32.) `chroma_db/` is a gitignored, regenerable cache — safe to delete; it rebuilds on next init. When editing `knowledge/topics/*.py`, changed text gets a new ID and is added, but **stale old chunks are not auto-removed** — call `clear_collection()` (or delete `chroma_db/`) to fully rebuild.

**Retrieval semantics:** the collection uses **cosine** space (`hnsw:space: cosine`), so `search()` reports a true cosine similarity (`1 − distance`) and `RAG_SIMILARITY_THRESHOLD = 0.3` is interpretable. 0.3 was calibrated empirically against the rebuilt index: relevant queries score 0.44–0.84, off-topic ≤0.08 (the old `1/(1+L2²)` mapping made 0.7 ≈ 0.79 cosine — pathologically strict). **Changing the embedding model or space requires deleting `chroma_db/` to rebuild.** Revisit the threshold with a real eval set if topics grow.

**No-results grounding:** when retrieval finds nothing above threshold, `_handle_no_rag_results` (`core/chatbot_tool_use.py`) now prompts the LLM to avoid fabricating specific constants/citations and appends a prominent "⚠️ Not from the curated knowledge base" disclaimer (covered by `tests/test_rag_no_results.py`).

## Rock-physics correctness

`tools/rock_physics_tools.py::calculate_rock_properties` was reimplemented on **cited models** (verified against the Rock Physics Handbook): water-saturated Vp/Vs from **Han, Nur & Morgan (1986)** (40 MPa; φ clipped to [0,0.35], C to [0,0.5]); bulk density from mass balance; and **proper Gassmann fluid substitution** (`gassmann_sat`/`gassmann_dry`, forward/inverse) for oil/gas. The previous code's gas branch wrongly *reduced* Vs — Gassmann holds shear modulus fluid-independent, so gas LOWERS Vp but slightly RAISES Vs (lower density); this is now a regression test (`tests/test_rock_physics.py`). Mineral/fluid moduli live in `_K_QUARTZ`/`_K_CLAY`/`_FLUIDS`. The curated `knowledge/topics/rock_physics.py` Gassmann and Nur formulas were corrected (Gassmann ratio+forward forms; Nur is linear in **modulus**, not velocity). **Gardner (0.31 for m/s), RHG, and Wyllie in that file were verified CORRECT and left unchanged** — the original audit had Gardner's units backwards.

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

## Input guards (physical validity)

`tools/physics_guards.py` holds two-tier validity helpers used by both the registry
validators and the compute functions:
- **REJECT** (raise `ValueError`, surfaced to the user): non-physical elastic media
  (`require_elastic_medium`: vp>0, rho>0, 0<vs<vp), non-positive geometry/source
  (`require_positive`), AVO angles outside [0,90), porosity/clay outside [0,1].
- **WARN** (`warnings.warn`, proceed): Nyquist/aliasing (`warn_if_aliased`), unusual
  velocities outside 300-8000 m/s (`warn_if_outside`), and rock-physics inputs beyond
  the Han (1986) range (warn-then-clip).

Velocity/density **inversions are intentionally allowed** (they are the AVO use case).
Warnings currently go to logs/stderr; surfacing them into the chat UI is a follow-up.

## Wavelet/wedge correctness (fixed)

Addressed in `tools/wedge_tools.py` / `tools/avo_tools.py` / `tools/ricker_tools.py` (covered by `tests/test_wedge_correctness.py`, `tests/test_pick_zero_crossings.py`):
- **`num_traces` and `dt` are now honored.** `wedge_model` previously hard-coded `ntraces=61`/`dt=0.1`; both are now parameters threaded from `create_wedge_model`.
- **Ormsby tuning frequency** uses the dominant frequency `(f2+f3)/2` (passband centre), not the low-cut corner `f1`. Stored in `parameters['wavelet_freq']` and read by `analyze_wedge`.
- **Multi-angle wedge no longer averages reflection coefficients** (physically meaningless). It uses the first angle and `warnings.warn`s that you should model each angle separately for a gather.
- **AVO plot** (`plot_avo_reflectivity`) no longer hard-clips y to ±0.3 (which cropped bright spots); it autoscales with an RC=0 reference line.
- **Zero-crossing auto-pick branch repaired and enabled:** `choose_pick_mode` now returns `'zero-crossings'` to match the branch; `pick_zero_crossings` returns its picks; `amp_picks` is filled per-trace (was rebound to a scalar). Verified end-to-end through `make_plot` for a near-zero-contrast (rc1≈0) wedge.
- Dead-code hygiene: `np.alltrue`→`np.all`, `spec.img`→`spec.imag`, and the `analyze_wavelet` spectrum frequency axis (was off by 1000×; the function is still unused).

**Intentionally NOT changed:** `create_ricker_wavelet`'s `time_length`(ms)/`dt`(s) mixed units are an API ergonomics wart, but the frequency math is correct (the ms/kHz juggling cancels to the exact Ricker), so it was left as-is. The AVO math (`zoeppritz_reflectivity` = exact Aki-Richards, `shuey_reflectivity` = Aki-Richards/Wiggins 3-term) was re-verified CORRECT — **do not "fix" it.**

**Still open** (scientific *completeness*, not bugs — see the gap scan): the wedge is single-angle only (no true offset/angle gather), fixed 3-layer/2-interface, no anisotropy / attenuation (Q) / multiples / NMO; and there are still no physical-validity guards (e.g. `vs < vp`, positivity, Nyquist) on the AVO/wedge compute inputs.

## Wedge AVO angle gather

`tools/wedge_tools.py` provides a true angle gather alongside the single-angle wedge:
- `wedge_avo_gather(...)` → `(time_array, gather, parameters)` where `gather` is a 3-D
  cube `(nt × num_traces × nangles)`; per-angle **Shuey** reflectivity, geometry built once.
  The single-angle `wedge_model` (2-D) is untouched.
- `analyze_wedge_gather(gather, parameters)` → per-angle tuning thickness/amplitude plus the
  AVO curve (top-interface amplitude vs angle at the isolated max-thickness trace).
- `plot_wedge_gather(gather, parameters)` → two-panel PNG (tuning curves per angle; AVO vs angle).

Registered in `core/tool_registry.py` (auto-plot `wedge_avo_gather` → `plot_wedge_gather`);
the chatbot stores `last_wedge_gather` and chains to the plot. Covered by `tests/test_wedge_gather.py`.

## The tool layer is registry-driven (the important architecture)

**`core/tool_registry.py` is the single source of truth.** Every LLM-facing tool is declared once as a frozen `ToolSpec` in the `REGISTRY` list (name, `fn`, description, JSON-schema `params`, `required`, `defaults`, optional `validator`, optional `auto_plot`). Everything else is *derived* at import time:

- `TOOL_SCHEMAS` — OpenAI/DeepSeek function schemas (`to_openai_schema`)
- `TOOL_FUNCTIONS` — name → callable map
- `AUTO_PLOT` — compute-tool → plot-tool chaining map
- `REGISTRY_BY_NAME` — name → spec

Consumers: `core/tool_manager.py` builds `self.tools` from `TOOL_FUNCTIONS`, validates/fills defaults/executes from `REGISTRY_BY_NAME`, and serves schemas via `get_tool_schemas()`. `core/chatbot_tool_use.py` reads `AUTO_PLOT` for chaining.

**To add or change a tool, edit `REGISTRY` only.** Do **not** hand-maintain parallel schema/function/validation tables — there are none anymore.

- `config/settings.py::AVAILABLE_TOOLS` was **deleted** (pruned on this branch). The parents' "keep `config/settings.py`, `config/tool_schemas.py`, and `ToolManager.tools` in sync" gotcha is **obsolete**.
- `config/tool_schemas.py` is now just a backward-compat **re-export** of `TOOL_SCHEMAS`/`TOOL_FUNCTIONS` from the registry.
- `ToolManager.tool_configs` is a backward-compat shim that reshapes `REGISTRY_BY_NAME` into the old `{required_params, optional_params}` dict — derived, not stored.

LLM-facing tool **names still differ from function names** (declared in the spec): `make_ricker`→`create_ricker_wavelet`, `make_ormsby`→`create_ormsby_wavelet`, `wedge_model`→`create_wedge_model`, `plot_ricker`→`plot_wavelet`. Registry tools beyond the parents' description include `make_ormsby` (Ormsby bandpass wavelet, 4 corner freqs), `analyze_wedge` (tuning thickness/amplitude, resolution limit), and `shuey_reflectivity`.

`execute_tool` fills `spec.defaults` **before** validating, so a `validator` sees the fully-populated param dict.

## Request flow (`core/chatbot_tool_use.py::SeismicChatBotToolUse`)

1. **Intent split** — `_is_knowledge_question` (LLM, keyword fallback): knowledge question → RAG, else → tool use.
2. **Knowledge path** — `knowledge/` runs RAG over a ChromaDB store (`rag_system.py`, `vector_db.py`; persisted at `chroma_db/`, embeddings `all-MiniLM-L6-v2`) seeded from `knowledge/topics/` (ricker, wedge, seismic properties, rock physics). No-hit → general-knowledge LLM → canned topic text.
3. **Tool path** — LLM emits a tool call; `ToolManager` validates against the spec, fills defaults, executes.
4. **Auto-chaining** — after a compute tool, its `AUTO_PLOT` partner runs automatically; plot tools return a `.png` path surfaced as `{"image_path": ...}`.
5. **Context** — `core/context_manager.py` caches the last wavelet/wedge/AVO/rock-properties result so follow-ups ("now plot it") reuse prior output; also tracks token usage.
6. **Reply parsing** — conversational text is wrapped in `<reply>...</reply>` and extracted via `_extract_reply`.

Tools: `tools/{ricker,wedge,avo,rock_physics}_tools.py` plus `rag_tools.py`, `parameter_validation.py`, `parameter_linking.py`. The seismic math originates in the repo-root `wedge.py` (one dir up, outside this package); `tools/wedge_tools.py` is the maintained reimplementation — fix math here, not in root `wedge.py`.

## Tests

Real pytest suite under `tests/` (the loose `test_*.py` at the package root are standalone scripts, not the suite).

- `tests/conftest.py` provides `fake_llm_factory` → `FakeLLMClient` returning **scripted completions with no network** — use it for anything touching the chatbot/LLM.
- `tests/test_no_dead_code.py` is a **regression guard against resurrecting removed code** (asserts `tools.avo_tools.calculate_AnB` and module `tools.interactive_plotting` stay gone). If you reintroduce a name it bans, that's intentional friction — don't just delete the assertion.
- `test_tool_registry.py` / `test_tool_manager.py` pin the registry-derivation contract; `test_ormsby.py`, `test_wedge_extras.py`, `test_parameter_{validation,linking}.py` cover specific tools.

## Gotchas specific to this package

- **AVO uses exact Aki-Richards Rpp.** `zoeppritz_reflectivity` was corrected to the exact Aki-Richards Rpp (commit `4b79490`); `shuey_reflectivity` is the linearized approximation. Keep them distinct.
- `plot_ricker` requires `time_array` (no longer optional); `make_ormsby` requires `f1<f2<f3<f4`.
- Setting `DEBUG` env var enables stderr debug prints and a `save.p` pickle of `wedge_model` inputs (a `save.p` may already sit in this dir from a prior run).
- Committing from the *outer* repos will not stage this package; commit package work with `git` from inside this dir.
