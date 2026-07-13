# N-Layer Synthetic Seismogram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a general N-layer 1-D convolutional synthetic seismogram (`synthetic_seismogram` leaf tool + plot) and a petrophysics-driven `petro_to_synthetic` workflow recipe, per `docs/superpowers/specs/2026-07-12-synthetic-seismogram-design.md`.

**Architecture:** New focused module `tools/synthetic_tools.py` reuses the existing verified primitives — `gen_wavelet` (tools/wedge_tools.py), `shuey_reflectivity`/`zoeppritz_reflectivity` (tools/avo_tools.py), and the two-tier guards (tools/physics_guards.py). Registry (`core/tool_registry.py`) exposes it with auto-plot chaining; the chatbot stores `last_synthetic` context; the recipe chains `predict_layer` per layer and registers as a `WorkflowSpec` in `workflows/engine.py`.

**Tech Stack:** Python, numpy, scipy.signal, matplotlib, pytest. No new dependencies.

## Global Constraints

- **Working directory is the package root** (`geo-mcp/seismic_chatbot`) — imports are top-level absolute (`from tools.synthetic_tools import ...`); run `pytest` from here.
- **Git:** this directory is its own repo; commit from inside it, on branch `stabilize-tool-layer`. End every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Units:** `dt` and all times in **ms**; thickness in **meters**; vp/vs in m/s; rho in g/cc. TWT through a layer = `2000 * h / vp` (ms).
- **Layer contract:** N = `len(vp)` layers (N ≥ 2) ⇒ N−1 interfaces; `len(thickness) == N-1` (basal layer is a half-space). `vs=None` → `vs_i = vp_i / 2`.
- **Registry validators return `(bool, str)`** — `(True, "")` on success, `(False, message)` on failure (see `tools/parameter_validation.py`). They do NOT raise.
- **All REJECT/WARN guards live in the compute function** (the recipe bypasses the registry validator); the registry validator wraps the same shared helper for fast LLM-facing errors.
- **Plot house pattern:** `tempfile.mkstemp(suffix=".png")` + `os.close(fd)`, `dpi=300`, `bbox_inches="tight"`, `plt.close(fig)`, return the path.
- **`TOOL_SCHEMAS` entries are flat dicts** `{"name", "description", "parameters"}` (not nested under `"function"`).
- Velocity/density inversions between layers are **allowed** (that is the AVO use case) — never guard against them.
- Do not modify `tools/wedge_tools.py`, `tools/avo_tools.py`, or any existing recipe.

---

### Task 1: Input validation core (`validate_synthetic_inputs`)

**Files:**
- Create: `tools/synthetic_tools.py`
- Create: `tests/test_synthetic_seismogram.py`

**Interfaces:**
- Consumes: `require_positive(value, name)`, `require_elastic_medium(vp, vs, rho, label)` from `tools/physics_guards.py`.
- Produces: `validate_synthetic_inputs(thickness, vp, rho, vs=None, angle=0.0, method="shuey", wv_type="ricker", ormsby_freq=None, dt=0.1, pad_time=50.0, wavelet_freq=30.0) -> list[float]` — returns the **effective vs list** (vp/2 default applied), raises `ValueError` on any REJECT rule. Tasks 2, 3, and 6 rely on this exact signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_synthetic_seismogram.py`:

```python
"""Tests for tools/synthetic_tools.py — N-layer 1-D convolutional synthetic."""
import numpy as np
import pytest

from tools.synthetic_tools import validate_synthetic_inputs

VP3 = [3000.0, 2500.0, 3200.0]
RHO3 = [2.4, 2.2, 2.5]
TH2 = [50.0, 50.0]


class TestValidateSyntheticInputs:
    def test_valid_inputs_return_vs_default(self):
        vs_eff = validate_synthetic_inputs(TH2, VP3, RHO3)
        assert vs_eff == [1500.0, 1250.0, 1600.0]

    def test_explicit_vs_is_returned(self):
        vs = [1600.0, 1300.0, 1700.0]
        assert validate_synthetic_inputs(TH2, VP3, RHO3, vs=vs) == vs

    def test_fewer_than_two_layers_rejected(self):
        with pytest.raises(ValueError, match=r"at least 2 layers"):
            validate_synthetic_inputs([], [3000.0], [2.4])

    def test_thickness_length_rule_names_the_contract(self):
        with pytest.raises(ValueError, match=r"len\(vp\)-1 = 2 .*basal half-space.*got 3"):
            validate_synthetic_inputs([10.0, 10.0, 10.0], VP3, RHO3)

    def test_rho_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match=r"rho must have 3"):
            validate_synthetic_inputs(TH2, VP3, [2.4, 2.2])

    def test_vs_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match=r"vs must have 3"):
            validate_synthetic_inputs(TH2, VP3, RHO3, vs=[1500.0, 1250.0])

    def test_non_positive_thickness_rejected(self):
        with pytest.raises(ValueError, match=r"thickness\[1\]"):
            validate_synthetic_inputs([50.0, -5.0], VP3, RHO3)

    def test_non_positive_dt_rejected(self):
        with pytest.raises(ValueError, match="dt"):
            validate_synthetic_inputs(TH2, VP3, RHO3, dt=0.0)

    def test_angle_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 90\)"):
            validate_synthetic_inputs(TH2, VP3, RHO3, angle=90.0)
        with pytest.raises(ValueError, match=r"\[0, 90\)"):
            validate_synthetic_inputs(TH2, VP3, RHO3, angle=-5.0)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="method"):
            validate_synthetic_inputs(TH2, VP3, RHO3, method="aki")

    def test_unknown_wv_type_rejected(self):
        with pytest.raises(ValueError, match="wv_type"):
            validate_synthetic_inputs(TH2, VP3, RHO3, wv_type="klauder")

    def test_ormsby_requires_corners(self):
        with pytest.raises(ValueError, match="ormsby_freq is required"):
            validate_synthetic_inputs(TH2, VP3, RHO3, wv_type="ormsby")

    def test_ormsby_corners_must_increase(self):
        with pytest.raises(ValueError, match="four increasing corners"):
            validate_synthetic_inputs(TH2, VP3, RHO3, wv_type="ormsby",
                                      ormsby_freq="5,40,10,50")

    def test_non_elastic_layer_rejected(self):
        # vs >= vp is non-physical (require_elastic_medium)
        with pytest.raises(ValueError):
            validate_synthetic_inputs(TH2, VP3, RHO3, vs=[3000.0, 1250.0, 1600.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic_seismogram.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'tools.synthetic_tools'` (or ImportError).

- [ ] **Step 3: Write the implementation**

Create `tools/synthetic_tools.py`:

```python
"""N-layer 1-D convolutional synthetic seismogram.

General stratigraphic synthetic: N layers -> N-1 interfaces; reflectivity is
placed at interface two-way times and convolved with a Ricker/Ormsby wavelet.
Reuses gen_wavelet (tools/wedge_tools.py) and the verified Shuey/Zoeppritz
reflectivity (tools/avo_tools.py). Geometry here is a single 1-D trace — the
wedge's 2-D trace fan stays in tools/wedge_tools.py.

All REJECT/WARN guards live in this module (not only the registry validator):
workflow recipes call these functions directly and bypass registry validation.
"""
import os
import tempfile

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from tools.wedge_tools import gen_wavelet
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity
from tools.physics_guards import (
    require_elastic_medium,
    require_positive,
    warn_if_aliased,
    warn_if_outside,
)


def _ormsby_corners(ormsby_freq):
    """Parse and validate an 'f1,f2,f3,f4' Ormsby corner string."""
    try:
        corners = [float(x) for x in str(ormsby_freq).split(",")]
    except ValueError:
        corners = []
    if len(corners) != 4 or not (corners[0] < corners[1] < corners[2] < corners[3]):
        raise ValueError(
            f"ormsby_freq must be four increasing corners 'f1,f2,f3,f4' "
            f"(got {ormsby_freq!r})"
        )
    return corners


def validate_synthetic_inputs(thickness, vp, rho, vs=None, angle=0.0,
                              method="shuey", wv_type="ricker", ormsby_freq=None,
                              dt=0.1, pad_time=50.0, wavelet_freq=30.0):
    """REJECT-tier validation shared by the compute function and the registry.

    Returns the effective vs list (vp/2 default applied) or raises ValueError.
    """
    vp = list(vp)
    rho = list(rho)
    thickness = list(thickness)
    n = len(vp)
    if n < 2:
        raise ValueError(f"need at least 2 layers (got {n})")
    if len(rho) != n:
        raise ValueError(f"rho must have {n} entries to match vp (got {len(rho)})")
    if len(thickness) != n - 1:
        raise ValueError(
            f"thickness must have len(vp)-1 = {n - 1} entries (one per layer "
            f"above the basal half-space); got {len(thickness)}"
        )
    if vs is None:
        vs_eff = [v / 2.0 for v in vp]
    else:
        vs_eff = list(vs)
        if len(vs_eff) != n:
            raise ValueError(f"vs must have {n} entries to match vp (got {len(vs_eff)})")
    for i, h in enumerate(thickness):
        require_positive(h, f"thickness[{i}]")
    require_positive(dt, "dt")
    require_positive(pad_time, "pad_time")
    require_positive(wavelet_freq, "wavelet_freq")
    if not (0 <= angle < 90):
        raise ValueError(f"angle must be in [0, 90) degrees (got {angle})")
    if method not in ("shuey", "zoeppritz"):
        raise ValueError(f"method must be 'shuey' or 'zoeppritz' (got {method!r})")
    if wv_type not in ("ricker", "ormsby"):
        raise ValueError(f"wv_type must be 'ricker' or 'ormsby' (got {wv_type!r})")
    if wv_type == "ormsby":
        if not ormsby_freq:
            raise ValueError("ormsby_freq is required when wv_type='ormsby'")
        _ormsby_corners(ormsby_freq)
    for i in range(n):
        require_elastic_medium(vp[i], vs_eff[i], rho[i], f"layer {i + 1}")
    return vs_eff
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthetic_seismogram.py -q`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/synthetic_tools.py tests/test_synthetic_seismogram.py
git commit -m "feat(synthetic): validate_synthetic_inputs — REJECT-tier N-layer input rules

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Compute core — `create_synthetic_seismogram` (normal incidence)

**Files:**
- Modify: `tools/synthetic_tools.py` (append after `validate_synthetic_inputs`)
- Test: `tests/test_synthetic_seismogram.py` (append)

**Interfaces:**
- Consumes: `validate_synthetic_inputs` (Task 1); `gen_wavelet(dt, wv_type, ricker_freq, ormsby_freq, wavelet_str, wavelet_fname, phase_rot, wavelet_length=...)` → `(t, wavelet, wavelet_label)` from `tools/wedge_tools.py`.
- Produces: `create_synthetic_seismogram(thickness, vp, rho, vs=None, wavelet_freq=30.0, wv_type="ricker", ormsby_freq=None, phase_rot=0.0, angle=0.0, method="shuey", dt=0.1, pad_time=50.0, labels=None) -> (time_array: np.ndarray, trace: np.ndarray, parameters: dict)`. `parameters` keys: `n_layers, vp, vs, rho, thickness, labels, interface_times, rcs, rc_series, t0, nt, dt, pad_time, angle, method, wavelet_freq, wavelet_label` (all lists/floats/str — JSON-friendly). Tasks 3–9 rely on this exact contract.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synthetic_seismogram.py` (extend the top import to `from tools.synthetic_tools import validate_synthetic_inputs, create_synthetic_seismogram`):

```python
class TestCreateSyntheticSeismogram:
    def test_return_shapes_and_parameter_keys(self):
        t, trace, p = create_synthetic_seismogram(TH2, VP3, RHO3)
        assert t.shape == trace.shape == (p["nt"],)
        for key in ("n_layers", "vp", "vs", "rho", "thickness", "labels",
                    "interface_times", "rcs", "rc_series", "t0", "nt", "dt",
                    "pad_time", "angle", "method", "wavelet_freq", "wavelet_label"):
            assert key in p
        assert p["n_layers"] == 3
        assert p["labels"] == ["layer 1", "layer 2", "layer 3"]

    def test_interface_twt_placement(self):
        t, trace, p = create_synthetic_seismogram(TH2, VP3, RHO3, dt=0.1, pad_time=50.0)
        t1 = 50.0 + 2000.0 * 50.0 / 3000.0          # 83.3333 ms
        t2 = t1 + 2000.0 * 50.0 / 2500.0            # 123.3333 ms
        assert np.allclose(p["interface_times"], [t1, t2])
        rc_series = np.asarray(p["rc_series"])
        idx = np.flatnonzero(rc_series)
        assert list(idx) == [round(t1 / 0.1), round(t2 / 0.1)]

    def test_acoustic_rc_values(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3)
        z = [v * r for v, r in zip(VP3, RHO3)]
        rc1 = (z[1] - z[0]) / (z[1] + z[0])
        rc2 = (z[2] - z[1]) / (z[2] + z[1])
        assert np.allclose(p["rcs"], [rc1, rc2])

    def test_event_sign_matches_rc(self):
        t, trace, p = create_synthetic_seismogram(TH2, VP3, RHO3, dt=0.1)
        i0 = round(p["interface_times"][0] / 0.1)
        win = trace[i0 - 100:i0 + 100]
        peak = win[np.argmax(np.abs(win))]
        assert np.sign(peak) == np.sign(p["rcs"][0])  # negative contrast here

    def test_thin_layers_superpose_on_one_sample(self):
        # 1 mm middle layer: both interfaces round to the same time sample,
        # so the reflection coefficients must ADD (not overwrite).
        _, _, p = create_synthetic_seismogram([50.0, 0.001], VP3, RHO3, dt=0.1)
        rc_series = np.asarray(p["rc_series"])
        idx = np.flatnonzero(rc_series)
        assert len(idx) == 1
        assert np.isclose(rc_series[idx[0]], p["rcs"][0] + p["rcs"][1])

    def test_amplitude_proportional_to_rc(self):
        # A lone spike convolved with the wavelet: signed peak / rc is the
        # wavelet peak — identical across models.
        _, tr_a, pa = create_synthetic_seismogram([50.0], [3000.0, 2500.0], [2.4, 2.2])
        _, tr_b, pb = create_synthetic_seismogram([50.0], [3000.0, 2000.0], [2.4, 2.0])
        peak_a = tr_a[np.argmax(np.abs(tr_a))]
        peak_b = tr_b[np.argmax(np.abs(tr_b))]
        assert np.isclose(peak_a / pa["rcs"][0], peak_b / pb["rcs"][0], rtol=1e-9)

    def test_ormsby_dominant_frequency_rule(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, wv_type="ormsby",
                                              ormsby_freq="5,10,40,50")
        assert p["wavelet_freq"] == 25.0             # (f2+f3)/2

    def test_labels_override_and_length_check(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3,
                                              labels=["shale", "sand", "shale"])
        assert p["labels"] == ["shale", "sand", "shale"]
        with pytest.raises(ValueError, match="labels must have 3"):
            create_synthetic_seismogram(TH2, VP3, RHO3, labels=["a", "b"])

    def test_unusual_velocity_warns(self):
        with pytest.warns(UserWarning):
            create_synthetic_seismogram(TH2, [100.0, 2500.0, 3200.0], RHO3,
                                        vs=[50.0, 1250.0, 1600.0])

    def test_aliasing_warns(self):
        # dt=1.0 ms -> Nyquist 500 Hz; 3 * 200 Hz Ricker content exceeds it.
        with pytest.warns(UserWarning):
            create_synthetic_seismogram(TH2, VP3, RHO3, dt=1.0, wavelet_freq=200.0)

    def test_parameters_json_friendly(self):
        import json
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3)
        json.dumps(p)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic_seismogram.py -q`
Expected: ImportError on `create_synthetic_seismogram` (Task 1 tests still pass once the import is split, so expected outcome is a collection error until Step 3).

- [ ] **Step 3: Write the implementation**

Append to `tools/synthetic_tools.py`:

```python
def create_synthetic_seismogram(thickness, vp, rho, vs=None, wavelet_freq=30.0,
                                wv_type="ricker", ormsby_freq=None, phase_rot=0.0,
                                angle=0.0, method="shuey", dt=0.1, pad_time=50.0,
                                labels=None):
    """Build an N-layer 1-D convolutional synthetic seismogram.

    N = len(vp) layers, thickness has N-1 entries (basal layer is a
    half-space). Reflectivity: acoustic at angle=0, Shuey/Zoeppritz at
    angle>0. Returns (time_array, trace, parameters); times in ms.
    """
    vs_eff = validate_synthetic_inputs(
        thickness, vp, rho, vs=vs, angle=angle, method=method, wv_type=wv_type,
        ormsby_freq=ormsby_freq, dt=dt, pad_time=pad_time, wavelet_freq=wavelet_freq,
    )
    vp = [float(v) for v in vp]
    rho = [float(r) for r in rho]
    thickness = [float(h) for h in thickness]
    n = len(vp)

    if labels is None:
        labels = [f"layer {i + 1}" for i in range(n)]
    elif len(labels) != n:
        raise ValueError(f"labels must have {n} entries to match vp (got {len(labels)})")

    # WARN tier (mirrors wedge_model's conventions)
    for i in range(n):
        warn_if_outside(vp[i], 300, 8000, f"vp layer {i + 1}", "m/s")
    if wv_type == "ormsby":
        corners = _ormsby_corners(ormsby_freq)
        content_hz = corners[3]
        dominant_freq = (corners[1] + corners[2]) / 2.0
    else:
        content_hz = 3.0 * wavelet_freq
        dominant_freq = wavelet_freq
    warn_if_aliased(content_hz, dt / 1000.0, "synthetic wavelet")

    # Interface two-way times: top of layer 1 sits at pad_time.
    twt = pad_time + np.cumsum(
        [2000.0 * thickness[j] / vp[j] for j in range(n - 1)]
    )

    # Per-interface reflectivity.
    rcs = []
    for i in range(n - 1):
        if angle == 0:
            z1 = vp[i] * rho[i]
            z2 = vp[i + 1] * rho[i + 1]
            rcs.append((z2 - z1) / (z2 + z1))
        else:
            refl = shuey_reflectivity if method == "shuey" else zoeppritz_reflectivity
            rc = refl(vp1=vp[i], vs1=vs_eff[i], rho1=rho[i],
                      vp2=vp[i + 1], vs2=vs_eff[i + 1], rho2=rho[i + 1],
                      angles=[angle])
            rcs.append(float(np.asarray(rc).ravel()[0]))

    _, wavelet, wavelet_label = gen_wavelet(
        dt, wv_type, wavelet_freq, ormsby_freq, "", "", phase_rot,
        wavelet_length=256.0,
    )

    nt = int(round((twt[-1] + pad_time) / dt)) + 1
    nt = max(nt, wavelet.size)  # mode='same' must never clip the response
    time_array = np.arange(nt) * dt

    rc_series = np.zeros(nt)
    for t_i, rc in zip(twt, rcs):
        idx = int(round(t_i / dt))
        if 0 <= idx < nt:
            rc_series[idx] += rc  # thin layers superpose (deliberate: not '=')

    trace = scipy.signal.convolve(rc_series, wavelet, mode="same")

    parameters = {
        "n_layers": n,
        "vp": vp,
        "vs": [float(v) for v in vs_eff],
        "rho": rho,
        "thickness": thickness,
        "labels": list(labels),
        "interface_times": [float(t) for t in twt],
        "rcs": [float(r) for r in rcs],
        "rc_series": rc_series.tolist(),
        "t0": 0.0,
        "nt": int(nt),
        "dt": float(dt),
        "pad_time": float(pad_time),
        "angle": float(angle),
        "method": method,
        "wavelet_freq": float(dominant_freq),
        "wavelet_label": wavelet_label,
    }
    return time_array, trace, parameters
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthetic_seismogram.py -q`
Expected: 25 passed (14 from Task 1 + 11 new). If `test_aliasing_warns` fails because no warning fires, check `warn_if_aliased` receives `dt / 1000.0` (seconds), not `dt`.

- [ ] **Step 5: Commit**

```bash
git add tools/synthetic_tools.py tests/test_synthetic_seismogram.py
git commit -m "feat(synthetic): create_synthetic_seismogram — N-layer normal-incidence convolutional trace

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Angle-dependent reflectivity path

**Files:**
- Modify: `tools/synthetic_tools.py` (the angle branch already exists from Task 2 — this task pins it with tests; fix only if a test fails)
- Test: `tests/test_synthetic_seismogram.py` (append)

**Interfaces:**
- Consumes: `shuey_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, angles)` and `zoeppritz_reflectivity(...)` from `tools/avo_tools.py` (both return an array over `angles`).
- Produces: nothing new — verifies the `angle`/`method` contract of `create_synthetic_seismogram`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synthetic_seismogram.py` (add `from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity` at the top):

```python
class TestAnglePath:
    VS3 = [1500.0, 1100.0, 1600.0]

    def test_rc_matches_shuey_at_angle(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3, angle=20.0)
        expected = shuey_reflectivity(
            vp1=VP3[0], vs1=self.VS3[0], rho1=RHO3[0],
            vp2=VP3[1], vs2=self.VS3[1], rho2=RHO3[1], angles=[20.0])
        assert np.isclose(p["rcs"][0], float(np.asarray(expected).ravel()[0]))

    def test_rc_matches_zoeppritz_when_requested(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3,
                                              angle=20.0, method="zoeppritz")
        expected = zoeppritz_reflectivity(
            vp1=VP3[0], vs1=self.VS3[0], rho1=RHO3[0],
            vp2=VP3[1], vs2=self.VS3[1], rho2=RHO3[1], angles=[20.0])
        assert np.isclose(p["rcs"][0], float(np.asarray(expected).ravel()[0]))

    def test_shuey_and_zoeppritz_differ_at_high_angle(self):
        # Sanity: the exact solution and the linearization diverge at 40 deg,
        # proving the method switch actually switches implementations.
        _, _, ps = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3, angle=40.0)
        _, _, pz = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3,
                                               angle=40.0, method="zoeppritz")
        assert not np.isclose(ps["rcs"][0], pz["rcs"][0], rtol=1e-6)

    def test_vs_default_used_in_angle_path(self):
        # vs omitted -> vp/2; result must equal explicitly passing vp/2.
        _, _, p_default = create_synthetic_seismogram(TH2, VP3, RHO3, angle=15.0)
        _, _, p_explicit = create_synthetic_seismogram(
            TH2, VP3, RHO3, vs=[v / 2.0 for v in VP3], angle=15.0)
        assert np.allclose(p_default["rcs"], p_explicit["rcs"])

    def test_angle_zero_is_acoustic_not_shuey(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3, angle=0.0)
        z = [v * r for v, r in zip(VP3, RHO3)]
        assert np.isclose(p["rcs"][0], (z[1] - z[0]) / (z[1] + z[0]))
```

- [ ] **Step 2: Run tests to verify current behavior**

Run: `pytest tests/test_synthetic_seismogram.py::TestAnglePath -v`
Expected: PASS if Task 2's angle branch is correct (the branch was written in Task 2; these tests pin it). If any fail, fix the branch in `create_synthetic_seismogram` until green — the reference behavior is defined by these tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_synthetic_seismogram.py
git commit -m "test(synthetic): pin Shuey/Zoeppritz angle path against avo_tools

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Oracle test against `wedge_model`

**Files:**
- Test: `tests/test_synthetic_seismogram.py` (append; test-only task — its purpose is to catch geometry/convolution discrepancies; fix `tools/synthetic_tools.py` only if it fails)

**Interfaces:**
- Consumes: `create_wedge_model(max_thickness, v1, v2, v3, rho1, rho2, rho3, num_traces, dt, wavelet_freq, ...) -> (time_array, model, synthetic, parameters)` from `tools/wedge_tools.py`; wedge `parameters` carries `t0` (ms). Wedge anchors interface 1 at 300 ms; interface 2 at `300 + 2000*h/v2`; its `synthetic` is `(nt, num_traces)` with trace column i at thickness `linspace(0, max_thickness, num_traces)[i]`.
- Produces: nothing new — cross-validates the two implementations.

- [ ] **Step 1: Write the oracle test**

Append to `tests/test_synthetic_seismogram.py` (add `from tools.wedge_tools import create_wedge_model` at the top):

```python
class TestOracleAgainstWedge:
    def test_event_separation_and_amplitudes_match_wedge(self):
        """3-layer stack vs the matching wedge trace.

        The two tools use different time references (wedge anchors interface 1
        at 300 ms; the synthetic uses a pad_time axis), so compare the event
        SEPARATION and event AMPLITUDES, not absolute times. The wedge places
        its second interface one sample late (known idx2+1 quirk) -> allow a
        2-sample separation tolerance.
        """
        vp, rho = [3000.0, 2500.0, 3200.0], [2.4, 2.2, 2.5]
        h, dt = 50.0, 0.1

        _, syn_trace, sp = create_synthetic_seismogram(
            [60.0, h], vp, rho, dt=dt, wavelet_freq=30.0, pad_time=60.0)

        _, _, wedge_synth, wp = create_wedge_model(
            max_thickness=100.0, v1=vp[0], v2=vp[1], v3=vp[2],
            rho1=rho[0], rho2=rho[1], rho3=rho[2],
            num_traces=101, dt=dt, wavelet_freq=30.0)
        wtrace = wedge_synth[:, 50]  # linspace(0,100,101)[50] == 50 m == h
        wtime = wp["t0"] + np.arange(wedge_synth.shape[0]) * dt

        syn_time = np.arange(sp["nt"]) * dt

        def event(trace, time, t_expect, half_win=15.0):
            m = (time >= t_expect - half_win) & (time <= t_expect + half_win)
            seg, tseg = trace[m], time[m]
            k = int(np.argmax(np.abs(seg)))
            return tseg[k], seg[k]

        t1s, a1s = event(syn_trace, syn_time, sp["interface_times"][0])
        t2s, a2s = event(syn_trace, syn_time, sp["interface_times"][1])
        t1w, a1w = event(wtrace, wtime, 300.0)
        t2w, a2w = event(wtrace, wtime, 300.0 + 2000.0 * h / vp[1])

        assert abs((t2s - t1s) - (t2w - t1w)) <= 2 * dt + 1e-9
        assert np.isclose(a1s, a1w, rtol=0.05)
        assert np.isclose(a2s, a2w, rtol=0.05)
```

- [ ] **Step 2: Run the oracle**

Run: `pytest tests/test_synthetic_seismogram.py::TestOracleAgainstWedge -v`
Expected: PASS. If it fails, the discrepancy is in `create_synthetic_seismogram` (TWT placement, spike indexing, or wavelet call) — `wedge_model` is the pinned reference; do NOT modify `tools/wedge_tools.py`. A multi-angle/aliasing `UserWarning` from wedge internals is fine; only assertion failures matter.

- [ ] **Step 3: Commit**

```bash
git add tests/test_synthetic_seismogram.py
git commit -m "test(synthetic): oracle — event separation + amplitudes match wedge_model 3-layer case

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Plot — `plot_synthetic_seismogram`

**Files:**
- Modify: `tools/synthetic_tools.py` (append)
- Test: `tests/test_synthetic_seismogram.py` (append)

**Interfaces:**
- Consumes: the `parameters` dict contract from Task 2 (notably `rc_series`, `interface_times`, `rcs`, `vp`, `rho`, `labels`, `t0`, `nt`, `dt`).
- Produces: `plot_synthetic_seismogram(trace, parameters, output_path=None) -> str` (PNG path). Must accept `trace` as list OR ndarray (registry JSON delivers lists). Tasks 6–8 rely on this signature.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synthetic_seismogram.py` (extend the synthetic_tools import with `plot_synthetic_seismogram`, and add `import os` at the top):

```python
class TestPlotSyntheticSeismogram:
    def _make(self):
        return create_synthetic_seismogram(TH2, VP3, RHO3,
                                           labels=["shale", "sand", "shale"])

    def test_creates_png_at_given_path(self, tmp_path):
        _, trace, p = self._make()
        out = plot_synthetic_seismogram(trace, p, output_path=str(tmp_path / "syn.png"))
        assert out == str(tmp_path / "syn.png")
        assert os.path.getsize(out) > 0

    def test_default_tempfile_path(self):
        _, trace, p = self._make()
        out = plot_synthetic_seismogram(trace, p)
        try:
            assert out.endswith(".png") and os.path.getsize(out) > 0
        finally:
            os.remove(out)

    def test_accepts_list_trace(self, tmp_path):
        _, trace, p = self._make()
        out = plot_synthetic_seismogram(list(trace), p,
                                        output_path=str(tmp_path / "syn2.png"))
        assert os.path.getsize(out) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic_seismogram.py::TestPlotSyntheticSeismogram -q`
Expected: ImportError / AttributeError — `plot_synthetic_seismogram` not defined.

- [ ] **Step 3: Write the implementation**

Append to `tools/synthetic_tools.py`:

```python
def plot_synthetic_seismogram(trace, parameters, output_path=None):
    """3-panel synthetic display: impedance model | reflectivity | trace.

    Shared vertical TWT axis, increasing downward. Returns the PNG path.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

    trace = np.asarray(trace, dtype=float)
    dt = float(parameters["dt"])
    t0 = float(parameters.get("t0", 0.0))
    nt = int(parameters["nt"])
    time = t0 + np.arange(nt) * dt
    interface_times = [float(x) for x in parameters["interface_times"]]
    rcs = [float(x) for x in parameters["rcs"]]
    vp = parameters["vp"]
    rho = parameters["rho"]
    labels = parameters.get("labels") or [f"layer {i+1}" for i in range(len(vp))]
    ai = [v * r for v, r in zip(vp, rho)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True)

    # Panel 1: stepped acoustic-impedance profile with layer labels.
    ax = axes[0]
    bounds = [time[0]] + interface_times + [time[-1]]
    for i, a in enumerate(ai):
        ax.fill_betweenx([bounds[i], bounds[i + 1]], 0, a, alpha=0.35)
        ax.text(0.03 * max(ai), (bounds[i] + bounds[i + 1]) / 2.0, labels[i],
                va="center", fontsize=9)
    for it in interface_times:
        ax.axhline(it, color="k", lw=0.8)
    ax.set_xlabel("AI (m/s·g/cc)")
    ax.set_ylabel("TWT (ms)")
    ax.set_title("Layer model")

    # Panel 2: reflectivity stems.
    ax = axes[1]
    ax.axvline(0, color="k", lw=0.5)
    for it, rc in zip(interface_times, rcs):
        ax.plot([0, rc], [it, it], "b-", lw=1.5)
        ax.plot(rc, it, "bo", ms=4)
    ax.set_xlabel("Reflection coefficient")
    ax.set_title("Reflectivity")

    # Panel 3: synthetic wiggle with positive fill.
    ax = axes[2]
    ax.plot(trace, time, "k-", lw=0.8)
    ax.fill_betweenx(time, 0, trace, where=trace > 0, color="k", alpha=0.6)
    ax.set_xlabel("Amplitude")
    title = "Synthetic"
    if parameters.get("angle", 0):
        title += f" ({parameters['angle']:g}°, {parameters['method']})"
    ax.set_title(title)

    axes[0].invert_yaxis()  # sharey -> all panels flip together
    fig.suptitle(f"N-layer synthetic — {parameters.get('wavelet_label', '')}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_synthetic_seismogram.py -q`
Expected: all pass (34 by now: 14 validation + 11 compute + 5 angle + 1 oracle + 3 plot).

- [ ] **Step 5: Commit**

```bash
git add tools/synthetic_tools.py tests/test_synthetic_seismogram.py
git commit -m "feat(synthetic): plot_synthetic_seismogram — model | reflectivity | trace panels

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Registry wiring + `(bool, str)` validator adapter

**Files:**
- Modify: `tools/parameter_validation.py` (append validator)
- Modify: `core/tool_registry.py` (imports + two `ToolSpec`s after the `analyze_wedge_gather` entry)
- Test: `tests/test_synthetic_seismogram.py` (append)

**Interfaces:**
- Consumes: `validate_synthetic_inputs` (Task 1), `create_synthetic_seismogram` (Task 2), `plot_synthetic_seismogram` (Task 5), `ToolSpec` dataclass (`core/tool_registry.py`).
- Produces: registry names `"synthetic_seismogram"` (with `auto_plot="plot_synthetic_seismogram"`) and `"plot_synthetic_seismogram"`; `validate_synthetic_seismogram(params) -> tuple[bool, str]` in `tools/parameter_validation.py`. Task 7 relies on these names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_synthetic_seismogram.py`:

```python
class TestRegistryWiring:
    def test_registered_with_auto_plot(self):
        from core.tool_registry import (AUTO_PLOT, REGISTRY_BY_NAME,
                                        TOOL_FUNCTIONS, TOOL_SCHEMAS)
        assert "synthetic_seismogram" in TOOL_FUNCTIONS
        assert "plot_synthetic_seismogram" in TOOL_FUNCTIONS
        assert AUTO_PLOT["synthetic_seismogram"] == "plot_synthetic_seismogram"
        spec = REGISTRY_BY_NAME["synthetic_seismogram"]
        assert spec.required == ["thickness", "vp", "rho"]
        assert any(s["name"] == "synthetic_seismogram" for s in TOOL_SCHEMAS)

    def test_validator_tuple_contract(self):
        from tools.parameter_validation import validate_synthetic_seismogram
        ok, msg = validate_synthetic_seismogram(
            {"thickness": [50.0], "vp": [3000.0, 2500.0], "rho": [2.4, 2.2]})
        assert ok is True and msg == ""
        ok, msg = validate_synthetic_seismogram(
            {"thickness": [50.0, 50.0], "vp": [3000.0, 2500.0], "rho": [2.4, 2.2]})
        assert ok is False and "thickness" in msg

    def test_tool_manager_executes_end_to_end(self):
        from core.tool_manager import ToolManager
        tm = ToolManager()
        result = tm.process_tool_call(
            "synthetic_seismogram",
            {"thickness": [50.0, 50.0], "vp": [3000.0, 2500.0, 3200.0],
             "rho": [2.4, 2.2, 2.5]})
        assert isinstance(result, tuple) and len(result) == 3
        assert result[2]["n_layers"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_synthetic_seismogram.py::TestRegistryWiring -q`
Expected: FAIL — `KeyError: 'synthetic_seismogram'` / ImportError for the validator.

- [ ] **Step 3: Add the validator**

Append to `tools/parameter_validation.py`:

```python
def validate_synthetic_seismogram(params):
    """Registry adapter for the N-layer synthetic: (bool, str) contract.

    Wraps tools.synthetic_tools.validate_synthetic_inputs so the REJECT rules
    stay defined in exactly one place.
    """
    from tools.synthetic_tools import validate_synthetic_inputs

    try:
        validate_synthetic_inputs(
            thickness=params.get("thickness") or [],
            vp=params.get("vp") or [],
            rho=params.get("rho") or [],
            vs=params.get("vs"),
            angle=params.get("angle", 0.0),
            method=params.get("method", "shuey"),
            wv_type=params.get("wv_type", "ricker"),
            ormsby_freq=params.get("ormsby_freq"),
            dt=params.get("dt", 0.1),
            pad_time=params.get("pad_time", 50.0),
            wavelet_freq=params.get("wavelet_freq", 30.0),
        )
    except ValueError as exc:
        return False, str(exc)
    return True, ""
```

(The lazy import avoids a module-level cycle if `synthetic_tools` ever imports validation helpers.)

- [ ] **Step 4: Register the two ToolSpecs**

In `core/tool_registry.py`:

1. Extend the tools import block:

```python
from tools.synthetic_tools import create_synthetic_seismogram, plot_synthetic_seismogram
```

2. Extend the `parameter_validation` import line to include `validate_synthetic_seismogram`.

3. Insert after the `analyze_wedge_gather` `ToolSpec` (keeps wedge/synthetic tools grouped):

```python
    ToolSpec(
        name="synthetic_seismogram",
        fn=create_synthetic_seismogram,
        description=(
            "Builds a general N-layer 1-D convolutional synthetic seismogram: "
            "per-layer thickness (meters, one per layer above the basal "
            "half-space), Vp, density and optional Vs; reflectivity at each "
            "interface (acoustic at normal incidence, Shuey or exact Zoeppritz "
            "at an incidence angle) convolved with a Ricker or Ormsby wavelet."
        ),
        params={
            "thickness": {"type": "array", "items": {"type": "number"},
                          "description": "Layer thicknesses in meters, length N-1 — one per layer above the basal half-space (the last layer needs no thickness)."},
            "vp": {"type": "array", "items": {"type": "number"},
                   "description": "P-wave velocity per layer in m/s (length N, N >= 2)."},
            "rho": {"type": "array", "items": {"type": "number"},
                    "description": "Density per layer in g/cc (length N)."},
            "vs": {"type": "array", "items": {"type": "number"},
                   "description": "Optional S-wave velocity per layer in m/s (length N). Defaults to Vp/2."},
            "wavelet_freq": {"type": "number",
                             "description": "Ricker dominant frequency in Hz (default 30)."},
            "wv_type": {"type": "string",
                        "description": "Wavelet type: 'ricker' (default) or 'ormsby'."},
            "ormsby_freq": {"type": "string",
                            "description": "Four increasing Ormsby corner frequencies 'f1,f2,f3,f4'; required when wv_type='ormsby'."},
            "phase_rot": {"type": "number",
                          "description": "Wavelet phase rotation in degrees (default 0)."},
            "angle": {"type": "number",
                      "description": "Incidence angle in degrees, 0 <= angle < 90 (default 0 = normal incidence)."},
            "method": {"type": "string",
                       "description": "Angle-dependent reflectivity: 'shuey' (default) or 'zoeppritz'; used when angle > 0."},
            "dt": {"type": "number",
                   "description": "Time sampling interval in ms (default 0.1)."},
            "pad_time": {"type": "number",
                         "description": "Quiet time in ms before the first and after the last interface (default 50)."},
            "labels": {"type": "array", "items": {"type": "string"},
                       "description": "Optional layer names for the plot (length N)."},
        },
        required=["thickness", "vp", "rho"],
        defaults={"vs": None, "wavelet_freq": 30.0, "wv_type": "ricker",
                  "ormsby_freq": None, "phase_rot": 0.0, "angle": 0.0,
                  "method": "shuey", "dt": 0.1, "pad_time": 50.0, "labels": None},
        validator=validate_synthetic_seismogram,
        auto_plot="plot_synthetic_seismogram",
    ),
    ToolSpec(
        name="plot_synthetic_seismogram",
        fn=plot_synthetic_seismogram,
        description="Plots an N-layer synthetic seismogram: layer impedance model, reflectivity series, and the synthetic trace.",
        params={
            "trace": {"type": "array", "items": {"type": "number"},
                      "description": "Synthetic trace samples."},
            "parameters": {"type": "object",
                           "description": "Parameters dict returned by synthetic_seismogram."},
        },
        required=["trace", "parameters"],
        defaults={},
    ),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_synthetic_seismogram.py -q && pytest tests/test_tool_registry.py tests/test_tool_manager.py -q`
Expected: all pass (registry derivation tests must stay green — they pin the derivation contract, and the new specs flow through it).

- [ ] **Step 6: Commit**

```bash
git add tools/parameter_validation.py core/tool_registry.py tests/test_synthetic_seismogram.py
git commit -m "feat(registry): synthetic_seismogram + plot_synthetic_seismogram ToolSpecs with shared validator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Chatbot wiring — context, auto-plot chain, system prompt

**Files:**
- Modify: `core/chatbot_tool_use.py` (three insertions, anchors below)
- Create: `tests/test_chatbot_synthetic.py`

**Interfaces:**
- Consumes: registry names from Task 6; `create_synthetic_seismogram` 3-tuple return; `fake_llm_factory` fixture (`tests/conftest.py`, call as `fake_llm_factory([])`).
- Produces: context key `"last_synthetic"` = `{"time_array", "trace", "parameters", "input_params"}`; auto-chain branch producing `{"image_path": ...}`; system-prompt bullet `- synthetic_seismogram: ...`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chatbot_synthetic.py`:

```python
"""Chatbot wiring for the N-layer synthetic: context, auto-plot chain, prompt."""
import os

import pytest

from core.chatbot_tool_use import SeismicChatBotToolUse
from tools.synthetic_tools import create_synthetic_seismogram


@pytest.fixture
def bot(fake_llm_factory):
    return SeismicChatBotToolUse(llm_client=fake_llm_factory([]))


@pytest.fixture
def synthetic_result():
    return create_synthetic_seismogram(
        [50.0, 50.0], [3000.0, 2500.0, 3200.0], [2.4, 2.2, 2.5])


def test_update_context_stores_last_synthetic(bot, synthetic_result):
    bot._update_context("synthetic_seismogram", {"vp": [3000.0, 2500.0, 3200.0]},
                        synthetic_result)
    stored = bot.context_manager.get_context("last_synthetic")
    assert stored is not None
    assert stored["parameters"]["n_layers"] == 3
    assert stored["input_params"] == {"vp": [3000.0, 2500.0, 3200.0]}


def test_auto_chain_plots_from_context(bot, synthetic_result):
    bot._update_context("synthetic_seismogram", {}, synthetic_result)
    chained = bot._handle_automatic_chaining("synthetic_seismogram", {},
                                             synthetic_result)
    assert chained is not None and chained["image_path"].endswith(".png")
    assert os.path.getsize(chained["image_path"]) > 0
    os.remove(chained["image_path"])


def test_auto_chain_without_context_returns_none(bot, synthetic_result):
    chained = bot._handle_automatic_chaining("synthetic_seismogram", {},
                                             synthetic_result)
    assert chained is None


def test_system_prompt_lists_synthetic(bot):
    assert "- synthetic_seismogram:" in bot._create_system_prompt()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chatbot_synthetic.py -q`
Expected: 3 failures (`last_synthetic` is None; chain returns None; prompt bullet missing). `test_auto_chain_without_context_returns_none` may already pass — fine.

- [ ] **Step 3: Implement the three insertions in `core/chatbot_tool_use.py`**

(a) In `_update_context`, after the `elif tool_name == "wedge_avo_gather":` block:

```python
            elif tool_name == "synthetic_seismogram":
                # Store synthetic trace for automatic plotting (3-tuple return)
                if isinstance(tool_result, tuple) and len(tool_result) == 3:
                    time_array, trace, parameters = tool_result
                    self.context_manager.set_context("last_synthetic", {
                        "time_array": time_array,
                        "trace": trace,
                        "parameters": parameters,
                        "input_params": tool_input
                    })
```

(b) In `_handle_automatic_chaining`, after the `elif tool_name == "wedge_avo_gather":` branch:

```python
            elif tool_name == "synthetic_seismogram":
                last = self.context_manager.get_context("last_synthetic")
                if not (last and "trace" in last and "parameters" in last):
                    return None
                plot_input = {"trace": last["trace"], "parameters": last["parameters"]}
```

(c) In `_create_system_prompt`'s "Available tools:" list, after the `- analyze_wedge:` line:

```
- synthetic_seismogram: Builds a general N-layer synthetic seismogram from per-layer thickness (m), Vp, density and optional Vs — reflectivity at each interface (acoustic, or Shuey/Zoeppritz at an incidence angle) convolved with a Ricker/Ormsby wavelet, with a layer-model/reflectivity/trace plot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chatbot_synthetic.py tests/test_chatbot_workflow.py tests/test_chatbot_narration.py -q`
Expected: all pass (narration/harvest tests confirm no regression in the reply contract).

- [ ] **Step 5: Commit**

```bash
git add core/chatbot_tool_use.py tests/test_chatbot_synthetic.py
git commit -m "feat(chatbot): synthetic_seismogram — last_synthetic context, auto-plot chain, prompt bullet

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Recipe — `petro_to_synthetic`

**Files:**
- Create: `workflows/recipes/petro_to_synthetic.py`
- Create: `tests/test_petro_to_synthetic.py`

**Interfaces:**
- Consumes: `predict_layer(phit, vclay, fluid="water", *, reduce="mean", label="") -> Layer` (`workflows/adapters.py`; `Layer` has `.vp/.vs/.rho/.label` floats); `create_synthetic_seismogram` and `plot_synthetic_seismogram` (Tasks 2/5).
- Produces: `petro_to_synthetic(phit, vclay, thickness, fluids=None, labels=None, wavelet_freq=30.0, angle=0.0, method="shuey") -> dict` with keys `layers` (list of `{vp, vs, rho, label, fluid}`), `interface_times`, `rcs`, `max_abs_rc`, `max_abs_amplitude`, `n_layers`, `wavelet_freq`, `angle`, `image_path`. Task 9 registers exactly this callable.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_petro_to_synthetic.py`:

```python
"""petro_to_synthetic recipe: per-layer petrophysics -> N-layer synthetic."""
import json
import os

import pytest

from workflows.recipes.petro_to_synthetic import petro_to_synthetic

PHIT = [0.10, 0.25, 0.10]
VCLAY = [0.55, 0.10, 0.55]
TH = [30.0, 20.0]


def _cleanup(result):
    if os.path.exists(result.get("image_path", "")):
        os.remove(result["image_path"])


def test_end_to_end_brine_stack():
    res = petro_to_synthetic(PHIT, VCLAY, TH)
    try:
        assert res["n_layers"] == 3
        assert len(res["layers"]) == 3
        assert len(res["interface_times"]) == 2
        assert len(res["rcs"]) == 2
        assert res["max_abs_amplitude"] > 0
        assert res["max_abs_rc"] > 0
        assert all(ly["fluid"] == "brine" for ly in res["layers"])
        assert res["layers"][0]["label"] == "layer 1"
        assert os.path.getsize(res["image_path"]) > 0
    finally:
        _cleanup(res)


def test_layers_match_predict_layer():
    from workflows.adapters import predict_layer
    res = petro_to_synthetic(PHIT, VCLAY, TH)
    try:
        expected = predict_layer(PHIT[1], VCLAY[1], fluid="brine", label="layer 2")
        assert res["layers"][1]["vp"] == pytest.approx(expected.vp)
        assert res["layers"][1]["vs"] == pytest.approx(expected.vs)
        assert res["layers"][1]["rho"] == pytest.approx(expected.rho)
    finally:
        _cleanup(res)


def test_gas_layer_lowers_vp_and_raises_vs():
    brine = petro_to_synthetic(PHIT, VCLAY, TH)
    gas = petro_to_synthetic(PHIT, VCLAY, TH, fluids=["brine", "gas", "brine"])
    try:
        assert gas["layers"][1]["vp"] < brine["layers"][1]["vp"]
        assert gas["layers"][1]["vs"] > brine["layers"][1]["vs"]  # Gassmann: mu fluid-independent, rho drops
    finally:
        _cleanup(brine)
        _cleanup(gas)


def test_custom_labels_flow_through():
    res = petro_to_synthetic(PHIT, VCLAY, TH, labels=["shale", "sand", "shale"])
    try:
        assert [ly["label"] for ly in res["layers"]] == ["shale", "sand", "shale"]
    finally:
        _cleanup(res)


def test_result_is_json_serializable():
    res = petro_to_synthetic(PHIT, VCLAY, TH)
    try:
        json.dumps(res)
    finally:
        _cleanup(res)


class TestRecipeGuards:
    def test_fewer_than_two_layers(self):
        with pytest.raises(ValueError, match="at least 2 layers"):
            petro_to_synthetic([0.2], [0.1], [])

    def test_vclay_length_mismatch(self):
        with pytest.raises(ValueError, match="vclay must have 3"):
            petro_to_synthetic(PHIT, [0.5, 0.1], TH)

    def test_thickness_length_rule(self):
        with pytest.raises(ValueError, match=r"len\(phit\)-1 = 2"):
            petro_to_synthetic(PHIT, VCLAY, [30.0])

    def test_fluids_length_mismatch(self):
        with pytest.raises(ValueError, match="fluids must have 3"):
            petro_to_synthetic(PHIT, VCLAY, TH, fluids=["brine"])

    def test_labels_length_mismatch(self):
        with pytest.raises(ValueError, match="labels must have 3"):
            petro_to_synthetic(PHIT, VCLAY, TH, labels=["a"])

    def test_non_positive_thickness(self):
        with pytest.raises(ValueError, match=r"thickness\[0\]"):
            petro_to_synthetic(PHIT, VCLAY, [-5.0, 20.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_petro_to_synthetic.py -q`
Expected: collection error — `ModuleNotFoundError: workflows.recipes.petro_to_synthetic`.

- [ ] **Step 3: Write the recipe**

Create `workflows/recipes/petro_to_synthetic.py`:

```python
"""petro_to_synthetic: N-layer synthetic seismogram from petrophysics.

Predict each layer's elastic properties from porosity, clay volume and pore
fluid (Han 1986 / Gassmann via predict_layer), stack them, and build the
N-layer convolutional synthetic with a model/reflectivity/trace plot.
"""
import numpy as np

from workflows.adapters import predict_layer
from tools.synthetic_tools import (create_synthetic_seismogram,
                                   plot_synthetic_seismogram)


def petro_to_synthetic(phit, vclay, thickness, fluids=None, labels=None,
                       wavelet_freq=30.0, angle=0.0, method="shuey"):
    """N-layer petro-to-synthetic. Returns a JSON-friendly dict (see tests).

    Early-fail guards run before any rock-physics call so a malformed request
    costs nothing and the error names the offending parameter.
    """
    phit = list(phit)
    vclay = list(vclay)
    thickness = list(thickness)
    n = len(phit)
    if n < 2:
        raise ValueError(f"need at least 2 layers (got {n})")
    if len(vclay) != n:
        raise ValueError(f"vclay must have {n} entries to match phit (got {len(vclay)})")
    if len(thickness) != n - 1:
        raise ValueError(
            f"thickness must have len(phit)-1 = {n - 1} entries (one per layer "
            f"above the basal half-space); got {len(thickness)}"
        )
    if fluids is None:
        fluids = ["brine"] * n
    elif len(fluids) != n:
        raise ValueError(f"fluids must have {n} entries to match phit (got {len(fluids)})")
    if labels is None:
        labels = [f"layer {i + 1}" for i in range(n)]
    elif len(labels) != n:
        raise ValueError(f"labels must have {n} entries to match phit (got {len(labels)})")
    for i, h in enumerate(thickness):
        if not (isinstance(h, (int, float)) and h > 0):
            raise ValueError(f"thickness[{i}] must be positive (got {h})")

    layers = [predict_layer(phit[i], vclay[i], fluid=fluids[i], label=labels[i])
              for i in range(n)]

    _, trace, parameters = create_synthetic_seismogram(
        thickness=thickness,
        vp=[ly.vp for ly in layers],
        rho=[ly.rho for ly in layers],
        vs=[ly.vs for ly in layers],
        wavelet_freq=wavelet_freq,
        angle=angle,
        method=method,
        labels=labels,
    )
    image_path = plot_synthetic_seismogram(trace, parameters)

    return {
        "layers": [
            {"vp": ly.vp, "vs": ly.vs, "rho": ly.rho,
             "label": ly.label, "fluid": fluids[i]}
            for i, ly in enumerate(layers)
        ],
        "interface_times": parameters["interface_times"],
        "rcs": parameters["rcs"],
        "max_abs_rc": max(abs(r) for r in parameters["rcs"]),
        "max_abs_amplitude": float(np.max(np.abs(trace))),
        "n_layers": n,
        "wavelet_freq": float(wavelet_freq),
        "angle": float(angle),
        "image_path": image_path,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_petro_to_synthetic.py -q`
Expected: 11 passed. (Han-range warnings from `calculate_rock_properties` for low-porosity shales are fine.)

- [ ] **Step 5: Commit**

```bash
git add workflows/recipes/petro_to_synthetic.py tests/test_petro_to_synthetic.py
git commit -m "feat(workflows): petro_to_synthetic recipe — per-layer petrophysics to N-layer synthetic

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: WorkflowSpec registration + run_sweep compatibility

**Files:**
- Modify: `workflows/engine.py` (import + one `WorkflowSpec` appended to `WORKFLOW_REGISTRY`)
- Modify: `core/chatbot_tool_use.py` (one workflow bullet in the system prompt)
- Test: `tests/test_petro_to_synthetic.py` (append)

**Interfaces:**
- Consumes: `petro_to_synthetic` (Task 8); `WorkflowSpec(name, fn, description, params, required, defaults, auto_plot)`; `run_sweep(recipe, grid, metric, fixed=None)` returning `{"rows", "stats", "coverage", "image_path", ...}`.
- Produces: workflow name `"petro_to_synthetic"` in `WORKFLOW_NAMES` (auto-exposed as a chatbot tool and to `run_sweep` via the registry conversion — no `core/tool_registry.py` edit needed).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_petro_to_synthetic.py`:

```python
class TestWorkflowRegistration:
    def test_in_workflow_and_tool_registries(self):
        from workflows.engine import WORKFLOW_NAMES, WORKFLOW_REGISTRY_BY_NAME
        from core.tool_registry import TOOL_FUNCTIONS
        assert "petro_to_synthetic" in WORKFLOW_NAMES
        assert "petro_to_synthetic" in TOOL_FUNCTIONS
        spec = WORKFLOW_REGISTRY_BY_NAME["petro_to_synthetic"]
        assert spec.required == ["phit", "vclay", "thickness"]

    def test_engine_run_fills_defaults(self):
        from workflows.engine import WorkflowEngine
        res = WorkflowEngine().run("petro_to_synthetic",
                                   {"phit": PHIT, "vclay": VCLAY, "thickness": TH})
        try:
            assert res["wavelet_freq"] == 30.0 and res["angle"] == 0.0
        finally:
            _cleanup(res)

    def test_system_prompt_lists_recipe(self, fake_llm_factory):
        from core.chatbot_tool_use import SeismicChatBotToolUse
        bot = SeismicChatBotToolUse(llm_client=fake_llm_factory([]))
        assert "- petro_to_synthetic:" in bot._create_system_prompt()

    def test_run_sweep_over_wavelet_freq(self):
        from workflows.sweep import run_sweep
        res = run_sweep(
            "petro_to_synthetic",
            grid={"wavelet_freq": [20.0, 40.0]},
            metric="max_abs_amplitude",
            fixed={"phit": PHIT, "vclay": VCLAY, "thickness": TH},
        )
        try:
            assert res["coverage"] == {"total": 2, "ran": 2, "failed": 0,
                                       "failures": []}
            assert res["stats"]["kind"] == "numeric"
            assert len(res["rows"]) == 2
        finally:
            _cleanup(res)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_petro_to_synthetic.py::TestWorkflowRegistration -q`
Expected: FAIL — `petro_to_synthetic` not in `WORKFLOW_NAMES`; prompt bullet missing.

- [ ] **Step 3: Register the workflow**

In `workflows/engine.py`, add the import next to the other recipe imports:

```python
from workflows.recipes.petro_to_synthetic import petro_to_synthetic
```

Append to `WORKFLOW_REGISTRY` (after the `run_sweep` entry):

```python
    WorkflowSpec(
        name="petro_to_synthetic",
        fn=petro_to_synthetic,
        description=(
            "N-layer synthetic seismogram from petrophysics: predict each "
            "layer's elastic properties (Vp, Vs, density) from porosity, clay "
            "volume and pore fluid (Han 1986 / Gassmann), stack the layers with "
            "their thicknesses, and build the 1-D convolutional synthetic "
            "(acoustic at normal incidence, Shuey/Zoeppritz at an angle). "
            "Returns the per-layer properties, interface times and reflection "
            "coefficients, amplitude metrics, and a layer-model/reflectivity/"
            "trace plot."
        ),
        params={
            "phit": {"type": "array", "items": {"type": "number"},
                     "description": "Porosity per layer (fraction, 0-1), length N (N >= 2, top to bottom)."},
            "vclay": {"type": "array", "items": {"type": "number"},
                      "description": "Clay volume per layer (fraction, 0-1), length N."},
            "thickness": {"type": "array", "items": {"type": "number"},
                          "description": "Layer thicknesses in meters, length N-1 (the basal layer is a half-space)."},
            "fluids": {"type": "array", "items": {"type": "string"},
                       "description": "Pore fluid per layer ('brine'/'water', 'oil', or 'gas'), length N. Default: all 'brine'."},
            "labels": {"type": "array", "items": {"type": "string"},
                       "description": "Optional layer names for the plot, length N."},
            "wavelet_freq": {"type": "number",
                             "description": "Ricker dominant frequency in Hz (default 30)."},
            "angle": {"type": "number",
                      "description": "Incidence angle in degrees, 0 <= angle < 90 (default 0)."},
            "method": {"type": "string",
                       "description": "Angle-dependent reflectivity: 'shuey' (default) or 'zoeppritz'."},
        },
        required=["phit", "vclay", "thickness"],
        defaults={"fluids": None, "labels": None, "wavelet_freq": 30.0,
                  "angle": 0.0, "method": "shuey"},
        auto_plot=None,
    ),
```

In `core/chatbot_tool_use.py`, add to the system prompt's workflow bullets (after the `- run_sweep:` line):

```
- petro_to_synthetic: N-layer synthetic seismogram from petrophysics — predicts each layer's elastic properties from porosity/clay/fluid (Han 1986 + Gassmann), stacks them with their thicknesses, and returns per-layer properties, interface reflectivities, amplitude metrics, and a layer-model/reflectivity/trace plot.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_petro_to_synthetic.py tests/test_chatbot_workflow.py tests/test_sweep.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add workflows/engine.py core/chatbot_tool_use.py tests/test_petro_to_synthetic.py
git commit -m "feat(workflows): register petro_to_synthetic — chatbot tool + run_sweep compatible

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Docs, example-prompt sync, roadmap tick, full suite

**Files:**
- Modify: `CLAUDE.md` (new section after "## Wedge AVO angle gather")
- Modify: `config/example_prompts.py` (two entries appended to `"Workflows & Advanced Analysis"`)
- Modify: `interfaces/web_interface.html` (same two entries in the JS `"Workflows & Advanced Analysis"` array, ~line 338)
- Modify: `docs/superpowers/specs/2026-06-15-scientific-completeness-roadmap.md` (tick B1)

**Interfaces:**
- Consumes: everything shipped in Tasks 1–9.
- Produces: user-facing docs; no code.

- [ ] **Step 1: Add the CLAUDE.md section**

Insert after the "## Wedge AVO angle gather" section:

```markdown
## N-layer synthetic seismogram

`tools/synthetic_tools.py` provides the general (non-wedge) 1-D convolutional model:
- `create_synthetic_seismogram(thickness, vp, rho, vs=None, ...)` → `(time_array,
  trace, parameters)`. N = len(vp) layers, `thickness` has **N−1** entries (basal
  layer is a half-space); meters in, TWT = 2000·h/vp ms internally. `angle=0` →
  acoustic RC; `angle>0` → Shuey (default) or exact Zoeppritz per interface
  (`method`). `vs=None` defaults to vp/2 (wedge convention). Thin layers that round
  to one sample **superpose** (`+=`, deliberately unlike the wedge's assignment).
  Guards live in the function itself (recipes bypass the registry validator);
  `validate_synthetic_inputs` is shared with the registry's
  `validate_synthetic_seismogram` (bool/str contract).
- `plot_synthetic_seismogram(trace, parameters)` → 3-panel PNG (AI layer model |
  reflectivity stems | wiggle trace), auto-chained via `AUTO_PLOT`; the chatbot
  stores `last_synthetic`.
- `workflows/recipes/petro_to_synthetic.py`: per-layer porosity/clay/fluid →
  `predict_layer` each → the synthetic; registered as a `WorkflowSpec`
  (`run_sweep`-compatible metrics `max_abs_amplitude`, `max_abs_rc`), with
  recipe-level early-fail length/geometry guards.
- Oracle-tested against `wedge_model`'s 3-layer case on event separation and
  amplitudes (the two tools use different absolute time references). Covered by
  `tests/test_synthetic_seismogram.py`, `test_petro_to_synthetic.py`,
  `test_chatbot_synthetic.py`.
```

- [ ] **Step 2: Sync example prompts (both sources)**

Append to the `"Workflows & Advanced Analysis"` list in `config/example_prompts.py`:

```python
        {
            "title": "N-layer synthetic seismogram",
            "prompt": "Build a synthetic seismogram for a 4-layer stack: Vp 3000, 2500, 2800, 3200 m/s, density 2.40, 2.20, 2.30, 2.50 g/cc, thicknesses 60, 40 and 30 m, with a 35 Hz Ricker wavelet",
            "description": "General N-layer convolutional synthetic — layer model, reflectivity, and trace"
        },
        {
            "title": "Synthetic from petrophysics",
            "prompt": "Build a 3-layer synthetic from petrophysics: shale (porosity 0.10, clay 0.55) over gas sand (porosity 0.25, clay 0.10) over shale (porosity 0.10, clay 0.55), thicknesses 40 and 25 m, fluids brine, gas, brine",
            "description": "petro_to_synthetic workflow: Han (1986)/Gassmann layers stacked into a synthetic trace"
        },
```

Mirror the **same two objects** (same titles/prompts/descriptions, JS object syntax) at the end of the `"Workflows & Advanced Analysis"` array in `interfaces/web_interface.html` (anchor: line ~338).

Verify the two sources stay in sync:

```bash
python - <<'EOF'
from config.example_prompts import EXAMPLE_PROMPTS
html = open("interfaces/web_interface.html").read()
for cat in ("Workflows & Advanced Analysis",):
    for item in EXAMPLE_PROMPTS[cat]:
        assert item["title"] in html, f"missing in html: {item['title']}"
        assert item["prompt"] in html, f"prompt drift: {item['title']}"
print("example prompts in sync")
EOF
```

Expected output: `example prompts in sync`.

- [ ] **Step 3: Tick the roadmap**

In `docs/superpowers/specs/2026-06-15-scientific-completeness-roadmap.md`, change the build-order line

```
3. `synthetic_seismogram` + `plot_synthetic_seismogram` (B1) — N-layer keystone.
```

to

```
3. `synthetic_seismogram` + `plot_synthetic_seismogram` (B1) — N-layer keystone. **DONE 2026-07-12** (see `2026-07-12-synthetic-seismogram-design.md`).
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: everything green except the one pre-existing known failure (`test_tool_use_pattern` stdin-capture conflict, if still present). No new failures, no new warnings-as-errors.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md config/example_prompts.py interfaces/web_interface.html docs/superpowers/specs/2026-06-15-scientific-completeness-roadmap.md
git commit -m "docs(synthetic): CLAUDE.md section, example prompts (py+html sync), roadmap B1 tick

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Verification (whole-feature)

After Task 10, exercise the feature end-to-end once (per the verify house rule):

```bash
python - <<'EOF'
from workflows.recipes.petro_to_synthetic import petro_to_synthetic
res = petro_to_synthetic(
    [0.10, 0.25, 0.10], [0.55, 0.10, 0.55], [40.0, 25.0],
    fluids=["brine", "gas", "brine"], labels=["shale", "gas sand", "shale"])
print("image:", res["image_path"])
print("rcs:", res["rcs"])          # top-of-gas-sand RC should be negative
print("layers:", [(l["label"], round(l["vp"])) for l in res["layers"]])
EOF
```

Open the printed PNG (Read tool) and confirm three panels: the labeled AI profile,
a negative stem at the top of the gas sand, and the wiggle trace with the
corresponding trough. Delete the PNG afterwards.
