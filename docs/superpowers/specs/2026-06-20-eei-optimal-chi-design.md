# EEI Optimal-χ (Phase 2b) — Design

**Date:** 2026-06-20
**Branch:** `stabilize-tool-layer`
**Status:** Approved design; implementation plan to follow.
**Parent spec:** `docs/superpowers/specs/2026-06-18-agentic-workflows-design.md` (Phase 2 `eei_optimal_chi` slice, gap **S2**).

## Problem

The codebase computes Extended Elastic Impedance EEI(χ) (Whitcombe 2002) for a
**single layer** across rotation angles (`tools/avo_tools.py::extended_elastic_impedance`,
scalar `vp/vs/rho`). It cannot answer the actual EEI question an interpreter asks:
*which rotation angle χ makes EEI best track a target property (lithology, fluid,
porosity) over a log?* That requires EEI evaluated over a **log** (many depth
samples) at each χ and correlated against a target log — new machinery (gap S2).

This feature adds that, in two modes (per the approved "hybrid" decision): a
raw-logs leaf tool and a petrophysics-driven recipe, over one shared science core.

## The science (cited)

EEI(χ) = scaling · Vp^p · Vs^q · ρ^r, with
`p = cosχ + sinχ`, `q = −8K·sinχ`, `r = cosχ − 4K·sinχ` (Whitcombe 2002).

- **Optimal χ:** `χ* = argmax over χ of |Pearson r(EEI(χ)_log, target_log)|`. Sign-agnostic
  (a strong negative correlation discriminates as well as positive). Pearson r is
  scale/shift-invariant, so **raw (un-normalized) EEI is sufficient** — Whitcombe
  normalization only rescales and does not change r.
- **Background K is a single scalar** = mean of (Vs/Vp)² over the log (overridable
  via `k`), **not** per-sample. χ only has a consistent meaning across the interval
  when K is fixed. This is the correct Whitcombe interval formulation.

## Architecture

One shared science core; two LLM-facing entry points; one shared plot. All new
code for the science lives in `tools/avo_tools.py` (where EEI already lives); the
petro recipe lives in `workflows/recipes/`.

```
raw logs (vp[],vs[],rho[],target[])           petrophysics (phit[],vclay[],fluid)
        │                                                │ calculate_rock_properties (shape-preserving)
        │                                                ▼  vp[],vs[],rho[] logs; target = vclay[] or phit[]
   eei_optimal_chi (leaf tool)                  eei_optimal_chi_petro (recipe)
        └───────────────┬────────────────────────────────┘
                        ▼
            _eei_chi_scan(vp,vs,rho,target,chi,k)   ← core (Whitcombe EEI over log + Pearson r sweep)
                        ▼
            plot_eei_chi_scan(chi,correlation,optimal_chi)  ← shared PNG
        both return a dict INCLUDING image_path
```

### Components

**`_eei_chi_scan(vp, vs, rho, target, chi, k=None)`** — private core (`tools/avo_tools.py`).
- Coerces `vp, vs, rho, target` to equal-length 1-D float arrays.
- `K = mean((vs/vp)**2)` unless `k` is supplied.
- For each χ in `chi`: `p,q,r` from the Whitcombe formulas (scalar K); EEI log =
  `vp**p * vs**q * rho**r`; Pearson r between that log and `target`.
- Returns `{"chi": [...], "correlation": [...], "optimal_chi": float,
  "max_correlation": float (signed, at χ*), "eei_optimal": [...] (EEI log at χ*)}`.
- Vectorized over χ; raw EEI (reference = 1).

**`plot_eei_chi_scan(chi, correlation, optimal_chi, output_path=None)`** — shared plot.
- r-vs-χ curve, χ* marked with its correlation; `tempfile.mkstemp(suffix=".png")` +
  `savefig(dpi=300, bbox_inches="tight")` + `plt.close(fig)` + `return output_path`.

**Leaf tool `eei_optimal_chi(vp, vs, rho, target, chi_min=-90, chi_max=90, chi_step=1, k=None)`**
(`tools/avo_tools.py`) — raw-logs mode.
- Builds `chi = arange(chi_min, chi_max + chi_step, chi_step)`; calls the core;
  calls `plot_eei_chi_scan`; returns the core dict **plus** `image_path`.
- Registered as a leaf `ToolSpec` with `auto_plot=None` (it self-plots).

**Recipe `eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine", chi_min=-90, chi_max=90, chi_step=1)`**
(`workflows/recipes/eei_optimal_chi_petro.py`) — petrophysics mode.
- Predicts Vp/Vs/ρ **logs** via `calculate_rock_properties(phit, vclay, fluid_type=fluid,
  print_results=False)` (shape-preserving → arrays; **not** `predict_layer`, which
  reduces to a scalar).
- Target log: `vclay` → the `vclay[]` input; `phit` → the `phit[]` input. (Sw deferred
  to Phase 3.) Reject any other `target` with `ValueError`.
- Calls the same core + `plot_eei_chi_scan`; returns a dict with the scan results,
  the chosen `target` name, and `image_path`.
- Registered in `workflows/engine.py::WORKFLOW_REGISTRY`.

## Integration (rides existing wiring)

Both entry points **return a dict containing `image_path`**, so:
- The chatbot's generic `_workflow_image_output` surfaces the plot with **no
  per-tool hardcoding** — no `AUTO_PLOT` entry, none of the three hardcoded chatbot
  spots (`_is_image_output` allowlist, `_handle_automatic_chaining`, the per-tool
  `_update_context` branches).
- The recipe auto-caches as `last_workflow_result` via the existing
  `WORKFLOW_NAMES`-keyed branch; the leaf tool is not cached (acceptable — it
  returns its own plot and full results).

Integration cost: one leaf `ToolSpec` for `eei_optimal_chi` (auto_plot=None) +
the recipe in `WORKFLOW_REGISTRY` + two system-prompt bullets. **Registry count
24 → 26.** `core/tool_manager.py` is untouched.

## Data flow (known-answer anchor)

Set `target = Vp·ρ` (acoustic impedance). Since EEI(χ=0) = Vp·ρ exactly, the scan
must return `optimal_chi ≈ 0` and `max_correlation ≈ 1.0`. This single test pins
the EEI formula, the χ sweep, and the Pearson correlation together.

## Error handling

`ValueError` on: mismatched log lengths; non-physical elastic samples (vp≤0, ρ≤0,
vs≤0, or vs≥vp on any sample); **constant target** (zero variance → Pearson
undefined); empty/degenerate χ range; unknown petro `target` (not `vclay`/`phit`).

## Testing

Real numeric assertions, no mocks:
- **AI known-answer:** `target = vp*rho` → `optimal_chi ≈ 0`, `max_correlation ≈ 1.0`.
- **Recovers a planted χ:** build logs whose EEI at a chosen χ is linear in the
  target → the scan returns that χ.
- **Pearson invariance:** scaling/shifting the target does not change `optimal_chi`.
- **Guards:** constant target → ValueError; length mismatch → ValueError;
  non-physical sample → ValueError; bad petro `target` → ValueError.
- **Leaf tool:** returns a dict with `image_path` (.png, nonzero); registered in
  `REGISTRY_BY_NAME`/`TOOL_SCHEMAS`/`TOOL_FUNCTIONS`; runs via `ToolManager`.
- **Petro recipe:** predicts logs, correlates EEI vs Vclay, returns `image_path`;
  registered as a workflow meta-tool; runs via `ToolManager`.
- **Chatbot:** both surface the image via `_workflow_image_output`; both names
  appear in the system prompt; full suite green (modulo the pre-existing stdin
  failure).

## Decisions (made during brainstorming)

1. **Hybrid inputs** → two entry points (leaf tool for raw logs, recipe for
   petrophysics) over a shared core (option A).
2. **Metric:** `argmax |Pearson r|` over χ.
3. **Background K:** single scalar (mean (Vs/Vp)²), overridable; not per-sample.
4. **Petro targets:** `vclay` (lithology) and `phit` (porosity); Sw deferred to Phase 3.
5. **Self-plotting dicts** (both return `image_path`) to ride the generic chatbot
   surfacing and avoid per-tool wiring.

## Out of scope (later / deferred)

- Continuous Sw as a petro target (Phase 3, gap S1).
- Sweeping the scan across many wells/intervals (Phase 4 sweep, gap S3).
- A second plot panel (EEI(χ*)-vs-target scatter) — the r-vs-χ curve with χ*
  marked is sufficient for v1; the scatter is a possible later enhancement.
- LAS/well-log file ingestion — inputs are arrays/params, consistent with the
  parent spec.
- Token economy of returning full `chi`/`correlation`/`eei_optimal` arrays to the
  LLM — consistent with existing recipes; revisit if it becomes a problem.

## Next step

A single implementation plan, in two halves: (1) the science core + `eei_optimal_chi`
leaf tool (+ plot, registration, prompt bullet); (2) the `eei_optimal_chi_petro`
recipe (+ registration, prompt bullet). Via the writing-plans skill.
