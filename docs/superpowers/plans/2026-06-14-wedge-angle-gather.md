# Wedge AVO Angle Gather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a true wedge AVO angle gather — the synthetic wedge computed per incidence angle as a 3-D cube — with its own plot and analysis tools, leaving the single-angle `wedge_model` untouched.

**Architecture:** Three new self-contained functions in `tools/wedge_tools.py` (`wedge_avo_gather`, `analyze_wedge_gather`, `plot_wedge_gather`). The gather builds the wedge geometry once and loops over angles, computing Shuey reflectivity per angle and convolving into per-angle panels stacked into `(nt × num_traces × nangles)`. Registered in `core/tool_registry.py` and wired into `core/chatbot_tool_use.py` (context + auto-plot chaining) like the existing wedge trio.

**Tech Stack:** Python 3.9, NumPy, SciPy, matplotlib (Agg), pytest. Run tests from inside `geo-mcp/seismic_chatbot/` with `python -m pytest`.

---

## Working-tree note

The branch `stabilize-tool-layer` is clean and all prior session work is committed. The guard helpers (`tools/physics_guards.py`) and the single-angle wedge (`tools/wedge_tools.py::wedge_model`) are in place and tested. The gather reuses `gen_wavelet`, `shuey_reflectivity`, and the physics guards — all already importable in `tools/wedge_tools.py`.

## File Structure

- `tools/wedge_tools.py` — append three functions (compute, analyze, plot). Add `angles_error` to the existing `physics_guards` import.
- `core/tool_registry.py` — three new `ToolSpec`s + imports.
- `core/chatbot_tool_use.py` — context storage, auto-plot chaining, image detection, system-prompt entry.
- `tests/test_wedge_gather.py` (new) — all gather tests.
- `CLAUDE.md` — document the gather tools.

---

### Task 1: `wedge_avo_gather` compute function

**Files:**
- Modify: `tools/wedge_tools.py` (the `from .physics_guards import ...` line; append new function at end of file)
- Test: `tests/test_wedge_gather.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_wedge_gather.py`:

```python
import os

import numpy as np
import pytest

from tools.wedge_tools import wedge_avo_gather, create_wedge_model

GKW = dict(max_thickness=60, v1=2500, v2=3000, v3=3500, rho1=2.2, rho2=2.3, rho3=2.4)


def test_gather_shape():
    angles = [0, 10, 20, 30]
    t, cube, params = wedge_avo_gather(angles=angles, **GKW)
    assert cube.ndim == 3
    assert cube.shape == (len(t), 61, len(angles))
    assert params["num_traces"] == 61
    assert params["angles"] == angles


def test_single_angle_panel_matches_wedge_model():
    # At a non-zero angle both paths use Shuey, so the gather panel must equal the
    # single-angle wedge_model synthetic exactly (same geometry + same RC).
    ang = 10
    _, cube, _ = wedge_avo_gather(angles=[ang], **GKW)
    _, _, synth, _ = create_wedge_model(incident_angle=ang, **GKW)
    synth = np.asarray(synth)
    assert cube.shape[:2] == synth.shape
    assert np.allclose(cube[:, :, 0], synth, atol=1e-9)


def test_gather_accepts_velocity_inversion():
    _, cube, _ = wedge_avo_gather(
        angles=[10, 20], max_thickness=50,
        v1=3000, v2=2300, v3=3200, rho1=2.4, rho2=2.0, rho3=2.4,
    )
    assert cube.shape[2] == 2
    assert np.all(np.isfinite(cube))


def test_gather_rejects_vs_ge_vp():
    with pytest.raises(ValueError):
        wedge_avo_gather(angles=[10], vs1=3000, **GKW)  # vs1>=vp1=2500


def test_gather_rejects_bad_angle():
    with pytest.raises(ValueError):
        wedge_avo_gather(angles=[95], **GKW)


def test_gather_rejects_empty_angles():
    with pytest.raises(ValueError):
        wedge_avo_gather(angles=[], **GKW)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wedge_gather.py -q`
Expected: ImportError (`wedge_avo_gather` does not exist).

- [ ] **Step 3: Write minimal implementation**

In `tools/wedge_tools.py`, update the physics_guards import to add `angles_error`:

```python
from .physics_guards import require_elastic_medium, require_positive, warn_if_aliased, warn_if_outside, angles_error
```

Append this function at the END of `tools/wedge_tools.py`:

```python
def wedge_avo_gather(
    max_thickness, v1, v2, v3, rho1, rho2, rho3, angles,
    vs1=None, vs2=None, vs3=None,
    wavelet_freq=30.0, num_traces=61, dt=0.1,
    wv_type='ricker', ormsby_freq=None,
    phase_rot=0.0, plotpadtime=50.0, zunit='m',
):
    """Wedge AVO angle gather: the synthetic wedge computed independently per
    incidence angle (Shuey reflectivity), returned as a 3-D cube
    (nt x num_traces x nangles). The single-angle wedge_model is left untouched.

    Returns (time_array, gather, parameters).
    """
    angles = list(angles)
    if not angles:
        raise ValueError("angles must be a non-empty list of incidence angles (deg).")

    vp_layers = [v1, v2, v3]
    rho_layers = [rho1, rho2, rho3]
    vs_layers = [vs1 if vs1 is not None else v1 / 2.0,
                 vs2 if vs2 is not None else v2 / 2.0,
                 vs3 if vs3 is not None else v3 / 2.0]

    # --- Physical-validity guards (shared helpers) ---
    require_positive(max_thickness, "max_thickness")
    require_positive(dt, "dt")
    if num_traces < 2:
        raise ValueError(f"num_traces must be >= 2 (got {num_traces})")
    for _i in range(3):
        require_elastic_medium(vp_layers[_i], vs_layers[_i], rho_layers[_i], f"layer {_i + 1}")
        warn_if_outside(vp_layers[_i], 300, 8000, f"vp layer {_i + 1}", "m/s")
    _ang_err = angles_error(angles)
    if _ang_err:
        raise ValueError(_ang_err)
    if wv_type == 'ormsby' and ormsby_freq:
        _content_hz = float(ormsby_freq.split(',')[-1])
    elif wavelet_freq:
        _content_hz = 3.0 * wavelet_freq
    else:
        _content_hz = 0.0
    warn_if_aliased(_content_hz, dt / 1000.0, "wedge gather wavelet")

    # --- Geometry (built once; independent of angle; mirrors wedge_model) ---
    z_min = 0
    z_max = max_thickness
    ntraces = num_traces
    t, wavelet, wavelet_label = gen_wavelet(
        dt, wv_type, wavelet_freq, ormsby_freq, '', '', phase_rot, wavelet_length=256.0)
    wavelet_length = t[-1] - t[0] + dt
    pad_time = plotpadtime
    model_time = 2 * pad_time + 2000 * (z_max - z_min) / vp_layers[1]
    if model_time < wavelet_length:
        pad_time += (wavelet_length - model_time) / 2.0 + dt
    thickness = np.linspace(z_min, z_max, ntraces)
    t_ref = 300
    interface1_t = t_ref + thickness * 0
    interface2_t = t_ref + thickness * 2000 / vp_layers[1]
    max_interface_time = max(interface1_t.max(), interface2_t.max())
    min_interface_time = min(interface1_t.min(), interface2_t.min())
    required_time_range = max_interface_time - min_interface_time + 2 * pad_time
    nt = int(round(2 * pad_time + 2000 * (z_max - z_min) / vp_layers[1] / dt))
    nt = max(nt, int(round(required_time_range / dt)) + 100)
    t0 = min_interface_time - pad_time

    # --- Per-angle reflectivity + convolution ---
    nangles = len(angles)
    gather = np.zeros((nt, ntraces, nangles))
    for k, ang in enumerate(angles):
        rc1 = shuey_reflectivity(
            vp1=vp_layers[0], vs1=vs_layers[0], rho1=rho_layers[0],
            vp2=vp_layers[1], vs2=vs_layers[1], rho2=rho_layers[1], angles=[ang])[0]
        rc2 = shuey_reflectivity(
            vp1=vp_layers[1], vs1=vs_layers[1], rho1=rho_layers[1],
            vp2=vp_layers[2], vs2=vs_layers[2], rho2=rho_layers[2], angles=[ang])[0]
        rc_model = np.zeros((nt, ntraces))
        for itr in range(ntraces):
            idx1 = int(round((interface1_t[itr] - t0) / dt))
            idx2 = int(round((interface2_t[itr] - t0) / dt)) + 1
            if 0 <= idx1 < nt:
                rc_model[idx1, itr] = rc1
            if 0 <= idx2 < nt:
                rc_model[idx2, itr] = rc2
        gather[:, :, k] = np.apply_along_axis(
            lambda _tr: scipy.signal.convolve(_tr, wavelet, mode='same'), axis=0, arr=rc_model)

    parameters = {
        'angles': angles,
        'v2': v2,
        'max_thickness': max_thickness,
        'num_traces': ntraces,
        'dt': dt,
        'wavelet_freq': wavelet_freq,
        'interface1_t': interface1_t,
        't0': t0,
        'nt': nt,
        'zunit': zunit,
        'wavelet_label': wavelet_label,
    }
    return t, gather, parameters
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wedge_gather.py -q`
Expected: 6 passed. If `test_single_angle_panel_matches_wedge_model` fails, the geometry diverged from `wedge_model` — diff the geometry block above against `wedge_model` and align (that test is the oracle for geometry correctness).

- [ ] **Step 5: Commit**

```bash
git add tools/wedge_tools.py tests/test_wedge_gather.py
git commit -m "feat(wedge): add wedge_avo_gather (3-D Shuey angle gather)"
```

---

### Task 2: `analyze_wedge_gather`

**Files:**
- Modify: `tools/wedge_tools.py` (append function)
- Test: `tests/test_wedge_gather.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wedge_gather.py`:

```python
from tools.wedge_tools import analyze_wedge_gather


def test_analyze_gather_tuning_and_avo_keys():
    angles = [0, 15, 30]
    _, cube, params = wedge_avo_gather(angles=angles, **GKW)  # v2=3000, f=30 -> tuning ~25 m
    out = analyze_wedge_gather(cube, params)
    assert abs(out["tuning_thickness"] - 25.0) < 1e-6
    assert len(out["per_angle"]) == 3
    assert out["per_angle"][0]["angle"] == 0
    assert set(out["avo"].keys()) == {"angles", "amplitudes"}
    assert len(out["avo"]["amplitudes"]) == 3


def test_analyze_gather_avo_varies_with_angle():
    # Gas-sand contrast -> AVO amplitude must vary across angles (not constant).
    angles = [0, 15, 30, 40]
    _, cube, params = wedge_avo_gather(
        angles=angles, max_thickness=60,
        v1=3000, v2=2300, v3=3200, rho1=2.4, rho2=2.0, rho3=2.4)
    out = analyze_wedge_gather(cube, params)
    amps = out["avo"]["amplitudes"]
    assert max(amps) - min(amps) > 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wedge_gather.py -k analyze -q`
Expected: ImportError (`analyze_wedge_gather` does not exist).

- [ ] **Step 3: Write minimal implementation**

Append to `tools/wedge_tools.py`:

```python
def analyze_wedge_gather(gather, parameters):
    """Analyze a wedge AVO gather: per-angle tuning thickness/amplitude and the
    AVO response (top-interface amplitude vs angle at the isolated max-thickness
    trace). Returns a dict; no plotting."""
    gather = np.asarray(gather, dtype=float)
    nt, ntraces, nangles = gather.shape
    v2 = parameters["v2"]
    freq = parameters["wavelet_freq"]
    if freq <= 0:
        raise ValueError(f"wavelet_freq must be positive, got {freq}")
    angles = list(parameters["angles"])
    max_thickness = parameters["max_thickness"]
    thickness = np.linspace(0, max_thickness, ntraces)
    tuning_thickness = v2 / (4.0 * freq)

    per_angle = []
    for k, ang in enumerate(angles):
        amp_vs_thickness = np.max(np.abs(gather[:, :, k]), axis=0)
        idx = int(np.argmax(amp_vs_thickness))
        per_angle.append({
            "angle": ang,
            "tuning_thickness_observed": float(thickness[idx]),
            "tuning_amplitude": float(amp_vs_thickness[idx]),
        })

    # AVO at the isolated top interface (max-thickness trace), windowed +/- one
    # dominant period around interface1_t so the base reflection is excluded.
    dt = parameters["dt"]
    t0 = parameters["t0"]
    interface1_t = np.asarray(parameters["interface1_t"], dtype=float)
    top_t = float(interface1_t[-1])
    i_center = int(round((top_t - t0) / dt))
    i_half = int(round((1000.0 / freq) / dt))
    i_lo = max(0, i_center - i_half)
    i_hi = min(nt, i_center + i_half + 1)
    avo_amps = [float(np.max(np.abs(gather[i_lo:i_hi, -1, k]))) for k in range(nangles)]

    return {
        "angles": angles,
        "tuning_thickness": float(tuning_thickness),
        "per_angle": per_angle,
        "avo": {"angles": angles, "amplitudes": avo_amps},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wedge_gather.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/wedge_tools.py tests/test_wedge_gather.py
git commit -m "feat(wedge): analyze_wedge_gather (per-angle tuning + AVO curve)"
```

---

### Task 3: `plot_wedge_gather`

**Files:**
- Modify: `tools/wedge_tools.py` (append function)
- Test: `tests/test_wedge_gather.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wedge_gather.py`:

```python
from tools.wedge_tools import plot_wedge_gather


def test_plot_gather_returns_png():
    _, cube, params = wedge_avo_gather(angles=[0, 20, 40], **GKW)
    path = plot_wedge_gather(cube, params)
    assert isinstance(path, str) and path.endswith(".png")
    assert os.path.exists(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wedge_gather.py -k plot -q`
Expected: ImportError (`plot_wedge_gather` does not exist).

- [ ] **Step 3: Write minimal implementation**

Append to `tools/wedge_tools.py`:

```python
def plot_wedge_gather(gather, parameters, figsize=None):
    """Plot a wedge AVO gather: amplitude-vs-thickness per angle (top panel) and
    amplitude-vs-angle at the isolated max-thickness trace (bottom panel).
    Returns the PNG path."""
    import tempfile

    gather = np.asarray(gather, dtype=float)
    nt, ntraces, nangles = gather.shape
    angles = list(parameters["angles"])
    max_thickness = parameters["max_thickness"]
    zunit = parameters.get("zunit", "m")
    thickness = np.linspace(0, max_thickness, ntraces)
    analysis = analyze_wedge_gather(gather, parameters)

    if figsize is None:
        figsize = (10, 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    for k, ang in enumerate(angles):
        amp_vs_thickness = np.max(np.abs(gather[:, :, k]), axis=0)
        ax1.plot(thickness, amp_vs_thickness, label=f"{ang}°")
    ax1.set_xlabel(f"Thickness ({zunit})")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Tuning curves by incidence angle")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="angle")

    ax2.plot(analysis["avo"]["angles"], analysis["avo"]["amplitudes"], "o-")
    ax2.set_xlabel("Incidence angle (deg)")
    ax2.set_ylabel("Top-interface amplitude")
    ax2.set_title("AVO at maximum thickness (isolated top interface)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_fd, fig_fname = tempfile.mkstemp(suffix=".png")
    os.close(fig_fd)
    plt.savefig(fig_fname)
    plt.close()
    return fig_fname
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wedge_gather.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/wedge_tools.py tests/test_wedge_gather.py
git commit -m "feat(wedge): plot_wedge_gather (tuning-per-angle + AVO two-panel)"
```

---

### Task 4: Register the gather tools

**Files:**
- Modify: `core/tool_registry.py`
- Test: `tests/test_wedge_gather.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wedge_gather.py`:

```python
def test_gather_tools_registered():
    from core.tool_registry import REGISTRY_BY_NAME, TOOL_FUNCTIONS, AUTO_PLOT
    assert "wedge_avo_gather" in REGISTRY_BY_NAME
    assert "plot_wedge_gather" in TOOL_FUNCTIONS
    assert "analyze_wedge_gather" in TOOL_FUNCTIONS
    assert AUTO_PLOT.get("wedge_avo_gather") == "plot_wedge_gather"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wedge_gather.py -k registered -q`
Expected: FAIL (`wedge_avo_gather` not in `REGISTRY_BY_NAME`).

- [ ] **Step 3: Write minimal implementation**

In `core/tool_registry.py`, change the wedge import line:

```python
from tools.wedge_tools import create_wedge_model, plot_wedge_model, analyze_wedge, wedge_avo_gather, plot_wedge_gather, analyze_wedge_gather
```

Add these three `ToolSpec`s to the `REGISTRY` list (place them immediately after the existing `analyze_wedge` ToolSpec):

```python
    ToolSpec(
        name="wedge_avo_gather",
        fn=wedge_avo_gather,
        description="Builds a wedge AVO angle gather: the synthetic wedge computed per incidence angle (Shuey), returned as a 3-D cube (time x thickness x angle).",
        params={
            "max_thickness": {"type": "number", "description": "Maximum thickness of the wedge layer in meters."},
            "v1": {"type": "number", "description": "P-wave velocity of the first layer in m/s."},
            "v2": {"type": "number", "description": "P-wave velocity of the second (wedge) layer in m/s."},
            "v3": {"type": "number", "description": "P-wave velocity of the third layer in m/s."},
            "rho1": {"type": "number", "description": "Density of the first layer in g/cm³."},
            "rho2": {"type": "number", "description": "Density of the second (wedge) layer in g/cm³."},
            "rho3": {"type": "number", "description": "Density of the third layer in g/cm³."},
            "angles": {"type": "array", "items": {"type": "number"}, "description": "Incidence angles in degrees (one synthetic panel per angle)."},
            "vs1": {"type": "number", "description": "S-wave velocity of layer 1 in m/s (optional, defaults to v1/2)."},
            "vs2": {"type": "number", "description": "S-wave velocity of layer 2 in m/s (optional, defaults to v2/2)."},
            "vs3": {"type": "number", "description": "S-wave velocity of layer 3 in m/s (optional, defaults to v3/2)."},
            "wavelet_freq": {"type": "number", "description": "Ricker wavelet frequency in Hz (default 30)."},
            "num_traces": {"type": "integer", "description": "Number of thickness traces (default 61)."},
        },
        required=["max_thickness", "v1", "v2", "v3", "rho1", "rho2", "rho3", "angles"],
        defaults={"wavelet_freq": 30.0, "num_traces": 61},
        validator=validate_wedge_model,
        auto_plot="plot_wedge_gather",
    ),
    ToolSpec(
        name="plot_wedge_gather",
        fn=plot_wedge_gather,
        description="Plots a wedge AVO gather: amplitude-vs-thickness per angle and amplitude-vs-angle (AVO) at maximum thickness.",
        params={
            "gather": {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}, "description": "3-D gather cube (time x thickness x angle) from wedge_avo_gather."},
            "parameters": {"type": "object", "description": "Parameters returned by wedge_avo_gather."},
        },
        required=["gather", "parameters"],
        defaults={},
    ),
    ToolSpec(
        name="analyze_wedge_gather",
        fn=analyze_wedge_gather,
        description="Analyzes a wedge AVO gather: per-angle tuning thickness/amplitude and the AVO curve at maximum thickness.",
        params={
            "gather": {"type": "array", "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}, "description": "3-D gather cube (time x thickness x angle) from wedge_avo_gather."},
            "parameters": {"type": "object", "description": "Parameters returned by wedge_avo_gather."},
        },
        required=["gather", "parameters"],
        defaults={},
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wedge_gather.py tests/test_tool_registry.py tests/test_tool_manager.py -q`
Expected: all pass (registry derivation tests still green; gather registered).

- [ ] **Step 5: Commit**

```bash
git add core/tool_registry.py tests/test_wedge_gather.py
git commit -m "feat(registry): register wedge_avo_gather + plot/analyze gather tools"
```

---

### Task 5: Wire chaining, context, and system prompt

**Files:**
- Modify: `core/chatbot_tool_use.py`
- Test: `tests/test_wedge_gather.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wedge_gather.py`:

```python
def _bot():
    from core.chatbot_tool_use import SeismicChatBotToolUse
    return SeismicChatBotToolUse(llm_client=object(), knowledge_base=object())


def test_gather_context_and_chaining():
    bot = _bot()
    t, cube, params = wedge_avo_gather(angles=[0, 20], **GKW)
    result = (t, cube, params)
    bot._update_context("wedge_avo_gather", {"angles": [0, 20]}, result)
    stored = bot.context_manager.get_context("last_wedge_gather")
    assert stored is not None and "gather" in stored and "parameters" in stored

    chained = bot._handle_automatic_chaining("wedge_avo_gather", {"angles": [0, 20]}, result)
    assert chained is not None and "image_path" in chained
    assert chained["image_path"].endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wedge_gather.py -k context_and_chaining -q`
Expected: FAIL — `last_wedge_gather` is None (no context branch) and chaining returns None.

- [ ] **Step 3: Write minimal implementation**

In `core/chatbot_tool_use.py`:

(a) Add `"plot_wedge_gather"` to the `_is_image_output` tool list (the line `tool_name in ["plot_ricker", "plot_wedge_model", "plot_avo_reflectivity", "plot_rock_properties"]`):

```python
                tool_name in ["plot_ricker", "plot_wedge_model", "plot_avo_reflectivity", "plot_rock_properties", "plot_wedge_gather"])
```

(b) In `_handle_automatic_chaining`, add an `elif` branch after the `wedge_model` branch (after the line `plot_input = {"synthetic_data": last["synthetic"], "parameters": last["parameters"]}`):

```python
            elif tool_name == "wedge_avo_gather":
                last = self.context_manager.get_context("last_wedge_gather")
                if not (last and "gather" in last and "parameters" in last):
                    return None
                plot_input = {"gather": last["gather"], "parameters": last["parameters"]}
```

(c) In `_update_context`, add an `elif` branch after the `wedge_model` branch (after its `set_context("last_wedge_model", {...})` block):

```python
            elif tool_name == "wedge_avo_gather":
                # Store gather data for automatic plotting (3-tuple return)
                if isinstance(tool_result, tuple) and len(tool_result) == 3:
                    time_array, gather, parameters = tool_result
                    self.context_manager.set_context("last_wedge_gather", {
                        "time_array": time_array,
                        "gather": gather,
                        "parameters": parameters,
                        "input_params": tool_input
                    })
```

(d) In `_create_system_prompt`, add a line to the tool list (after `- plot_wedge_model: Plots wedge model results`):

```python
- wedge_avo_gather: Builds an AVO angle gather (synthetic wedge per incidence angle) and plots tuning-vs-angle + AVO
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wedge_gather.py tests/test_session_isolation.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_wedge_gather.py
git commit -m "feat(chatbot): context + auto-plot chaining for wedge_avo_gather"
```

---

### Task 6: Document the gather tools

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add documentation**

Add this subsection to `CLAUDE.md` near the wedge documentation (e.g. after the "Wavelet/wedge correctness (fixed)" section):

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document wedge AVO angle gather tools"
```

---

### Final verification

- [ ] **Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (115 pre-existing + ~12 new gather tests). Investigate any pre-existing test that regresses (the gather adds new tools; it must not alter existing tool behavior).

---

## Self-Review

**Spec coverage:**
- New dedicated tools, wedge_model untouched → Tasks 1-3 (append-only; no edits to `wedge_model`). ✓
- 3-D cube `(nt × num_traces × nangles)` → Task 1 + `test_gather_shape`. ✓
- Shuey per angle → Task 1 (`shuey_reflectivity` per angle). ✓
- Both curves (tuning per angle + AVO at isolated top interface) → Tasks 2 & 3. ✓
- Guards reused (vs<vp, positivity, angles, Nyquist) → Task 1 + reject tests. ✓
- Single-angle equivalence oracle → `test_single_angle_panel_matches_wedge_model`. ✓
- AVO varies with angle (gas sand) → `test_analyze_gather_avo_varies_with_angle`. ✓
- Registry + chaining + context + system prompt → Tasks 4 & 5 + tests. ✓
- Docs → Task 6. ✓
- Empty/invalid angles, inversion accepted → Task 1 tests. ✓

**Placeholder scan:** No TBD/TODO; every code step has full code. ✓

**Type/name consistency:** `wedge_avo_gather` returns a 3-tuple `(time_array, gather, parameters)` consistently across Tasks 1, 2, 3, 5. `parameters` keys (`angles`, `v2`, `wavelet_freq`, `max_thickness`, `num_traces`, `dt`, `interface1_t`, `t0`, `nt`, `zunit`) are produced in Task 1 and consumed in Tasks 2 (`analyze_wedge_gather`) and 3 (`plot_wedge_gather`). Context key `last_wedge_gather` and chaining input `{"gather", "parameters"}` match between Task 4 (ToolSpec param names `gather`/`parameters`) and Task 5 (chaining/context). ✓
