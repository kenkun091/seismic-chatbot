# Scientific-completeness tool roadmap — design

- **Date:** 2026-06-15
- **Status:** Approved (pending spec review)
- **Type:** Roadmap (not a single implementation cycle — each tool below gets its own spec → plan → build later)
- **Package:** `geo-mcp/seismic_chatbot` (branch `stabilize-tool-layer`)

## Purpose

The tool layer is now stabilized (registry-driven, guarded, session-isolated) and the
single-angle wedge has gained a true AVO angle gather. This document sequences the
**next** batch of scientific tools that close the largest remaining forward-modeling
gaps, prioritized by user direction toward **AVO interpretation** and **earth-model
generality**.

Each tool is sketched as a registry `ToolSpec` because `core/tool_registry.py` is the
single source of truth — adding a tool means adding one `ToolSpec`; schemas, the
function map, and auto-plot chaining derive automatically.

## Current coverage (for context)

Wavelets (Ricker, Ormsby); single-angle wedge + AVO angle gather; exact Zoeppritz +
Shuey AVO; Han(1986)/Gassmann rock physics; tuning analysis; two-tier physical-validity
guards. 15 registry tools.

## Prioritized phases

### Phase 1 — AVO interpretation (fast, low-risk, high-leverage)

Both reuse already-verified math and make the *existing* AVO tools interpretable rather
than adding new physics.

#### A1. `gassmann_substitution` — fluid substitution as a first-class tool

- **Why:** Gassmann fluid substitution is the core "what does the gas/oil/brine case
  look like" workflow that feeds every AVO and wedge model. The forward/inverse math
  (`gassmann_sat`/`gassmann_dry`) is already implemented and regression-tested inside
  `tools/rock_physics_tools.py`, but it is **not LLM-facing** — only reachable through
  the bundled `calculate_rock_properties`.
- **Shape:** `gassmann_substitution(vp, vs, rho, phi, k_mineral, mu_mineral, fluid_in, fluid_out, ...) → {vp, vs, rho}` (the substituted elastic properties).
- **Output:** dict of substituted `vp`, `vs`, `rho` (plus the intermediate dry-frame
  modulus for transparency). No `auto_plot`.
- **Value** high · **Effort** low (wrap + expose existing verified math) · **Risk** low.
- **Depends on:** nothing new.

#### A2. `avo_attributes` — intercept/gradient + AVO class + crossplot

- **Why:** Intercept (A / R0) and gradient (B) and the AVO class (I–IV) are the standard
  interpretation summary of an interface's AVO behavior. We compute reflectivity curves
  but never reduce them to these attributes.
- **Shape:** `avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2) → {intercept, gradient, avo_class, ...}`.
  Intercept/gradient come from the Shuey two-term we already have
  (`A = ½(Δvp/v̄p + Δρ/ρ̄)`, `B` = Shuey gradient); class from the (A, B) quadrant /
  sign rules (I: A>0, B<0 steep; II: A≈0; III: A<0, B<0; IV: A<0, B>0).
- **Output + plot:** `auto_plot=plot_avo_crossplot` — the A–B plane with class-region
  backgrounds and the computed point/trend marked.
- **Value** high · **Effort** medium (attribute math reuses Shuey; new crossplot render)
  · **Risk** low.
- **Depends on:** the existing Shuey machinery in `tools/avo_tools.py`.

### Phase 2 — Earth-model generality (keystone)

#### B1. `synthetic_seismogram` — general N-layer 1-D convolutional model

- **Why:** Every existing synthetic is locked to a fixed 3-layer / 2-interface geometry
  (`wedge_model`). A general N-layer 1-D convolutional model removes that ceiling and
  becomes the backbone for well-tie synthetics and arbitrary stratigraphy.
- **Shape:** `synthetic_seismogram(thickness[], vp[], vs[], rho[], wavelet_freq, wv_type='ricker', angle=0, dt=..., ...) → (time_array, trace, parameters)`.
  Build per-interface reflectivity (normal-incidence, or angle-dependent via
  Shuey/Zoeppritz), convert layer thicknesses to two-way time, place reflectivity at
  interface samples, convolve with the wavelet.
- **Output + plot:** `auto_plot=plot_synthetic_seismogram` — model/velocity track +
  reflectivity series + synthetic trace.
- **Value** very high · **Effort** med–high (two-way-time placement, angle reflectivity)
  · **Risk** medium (correctness of TWT placement and interface sampling — needs an
  oracle test against `wedge_model` for the 3-layer case).
- **Depends on:** reuses `avo_tools` reflectivity and `ricker_tools` wavelets; should
  share a geometry/convolution core with `wedge_tools` rather than re-deriving it.

### Phase 3 — later (in roadmap, lower priority)

- **A3. `elastic_impedance`** (Connolly 1999): EI(θ) closed form for inversion/interp.
  Value medium · Effort low–med · Risk low.
- **`ruger_reflectivity`** (VTI anisotropic AVO, Thomsen δ/ε): the Rüger (1997) formula
  was **already adversarially verified** in a prior session (reduces to Shuey at
  δ=ε=0; anisotropic increment `½Δδ·sin²θ + ½Δε·sin²θ·tan²θ`), so it is a low-risk
  **pull-forward** whenever anisotropy is wanted. Composes with the existing angle
  gather (`method='ruger'`).
- **Constant-Q attenuation** → **NMO / offset moveout** (turns the angle gather into an
  offset gather). These add propagation realism once the interpretation + model-
  generality foundation is in place.

## Out of scope (YAGNI — explicitly cut)

- **Multiples** (free-surface / internal): high implementation cost, low immediate
  interpretive value for a forward-modeling teaching/assistant tool. Revisit only if a
  concrete use case appears.
- **Backus averaging** (thin-layer upscaling to effective anisotropic medium):
  speculative until anisotropy is actually in use; defer.
- **Full reflectivity/wavenumber synthetics**, anisotropy beyond VTI, attenuation
  dispersion modeling beyond constant-Q.

## Build order summary

1. `gassmann_substitution` (A1) — quick win, unblocks fluid-case AVO modeling.
2. `avo_attributes` + `plot_avo_crossplot` (A2) — headline AVO interpretation.
3. `synthetic_seismogram` + `plot_synthetic_seismogram` (B1) — N-layer keystone.
4. Then, as wanted: `elastic_impedance`, pull-forward `ruger_reflectivity`, Q + NMO.

## Process note

Each numbered item re-enters the standard cycle: `brainstorming` → spec in
`docs/superpowers/specs/` → `writing-plans` → `subagent-driven-development` →
`finishing-a-development-branch`. This roadmap only fixes priority and sequence; it does
not pre-approve any individual tool's design.
