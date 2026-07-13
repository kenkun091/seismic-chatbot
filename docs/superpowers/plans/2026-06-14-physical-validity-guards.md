# Physical-validity Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two-tier physical-validity guards (reject impossible inputs, warn on out-of-range) across the AVO, wedge, wavelet, and rock-physics tools so the package stops producing confident garbage from bad inputs.

**Architecture:** A new pure helper module `tools/physics_guards.py` holds small predicates (return error string or `None`), raisers, and warn helpers. These are wired into both the registry validators (`tools/parameter_validation.py`, tool path) and the compute functions (direct/internal callers). REJECT → `ValueError`; WARN → `warnings.warn`.

**Tech Stack:** Python 3.9, NumPy, pytest, Python `warnings`. Run tests from inside `geo-mcp/seismic_chatbot/` with `python -m pytest`.

---

## Working-tree note (read before executing)

This branch (`stabilize-tool-layer`) has **uncommitted changes from a prior session** in several of the files this plan touches (`tools/avo_tools.py`, `tools/wedge_tools.py`, `tools/ricker_tools.py`, `tools/rock_physics_tools.py`, etc.). The `git add <file>` steps below will therefore stage those pre-existing edits too. That is acceptable (it is all this session's work on the same branch), but if you want the guard commits isolated, commit or stash the existing working-tree changes **before** starting Task 1.

## File Structure

- **Create** `tools/physics_guards.py` — pure validity predicates + raisers + warn helpers. One responsibility: deciding what is physically valid.
- **Modify** `tools/avo_tools.py` — guard `zoeppritz_reflectivity` / `shuey_reflectivity`.
- **Modify** `tools/wedge_tools.py` — guard `wedge_model`.
- **Modify** `tools/ricker_tools.py` — guard `create_ricker_wavelet` / `create_ormsby_wavelet`.
- **Modify** `tools/rock_physics_tools.py` — guard `calculate_rock_properties`.
- **Modify** `tools/parameter_validation.py` — strengthen `validate_avo`, slim `validate_wedge_model` (drop the over-tight band), and remove the dead inversion-ordering checks.
- **Create** `tests/test_physics_guards.py` — unit tests for the helpers.
- **Create** `tests/test_input_guards.py` — per-tool guard behavior tests.
- **Modify** `CLAUDE.md` — document the guard policy.

---

### Task 1: `physics_guards` helper module

**Files:**
- Create: `tools/physics_guards.py`
- Test: `tests/test_physics_guards.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_physics_guards.py`:

```python
import warnings

import pytest

from tools.physics_guards import (
    elastic_medium_error, positive_error, angles_error,
    require_elastic_medium, require_positive, warn_if_aliased, warn_if_outside,
)


def test_elastic_medium_valid_returns_none():
    assert elastic_medium_error(2500, 1200, 2.2, "m") is None


def test_elastic_medium_rejects_vs_ge_vp():
    assert elastic_medium_error(2500, 2600, 2.2, "m") is not None
    assert elastic_medium_error(2500, 2500, 2.2, "m") is not None


def test_elastic_medium_rejects_nonpositive():
    assert elastic_medium_error(0, 1200, 2.2) is not None
    assert elastic_medium_error(2500, 1200, 0) is not None
    assert elastic_medium_error(2500, 0, 2.2) is not None


def test_positive_error():
    assert positive_error(-1, "x") is not None
    assert positive_error(0, "x") is not None
    assert positive_error(5, "x") is None


def test_angles_error_bounds():
    assert angles_error([0, 30, 45]) is None
    assert angles_error([90]) is not None
    assert angles_error([-1]) is not None


def test_require_helpers_raise():
    with pytest.raises(ValueError):
        require_elastic_medium(2500, 2600, 2.2)
    with pytest.raises(ValueError):
        require_positive(0, "dt")


def test_warn_if_aliased_warns_above_nyquist():
    with pytest.warns(UserWarning):
        warn_if_aliased(6000, 1e-4)  # nyquist = 5000 Hz


def test_warn_if_aliased_silent_below_nyquist():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_aliased(90, 1e-4)  # nyquist = 5000 Hz -> silent


def test_warn_if_outside_warns_and_is_silent():
    with pytest.warns(UserWarning):
        warn_if_outside(9000, 300, 8000, "v", "m/s")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_outside(2500, 300, 8000, "v", "m/s")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics_guards.py -q`
Expected: collection error / ImportError — `tools.physics_guards` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `tools/physics_guards.py`:

```python
"""Physical-validity guards for seismic compute inputs.

Two tiers:
- REJECT physically impossible inputs (raise ValueError via the require_* helpers).
- WARN on possible-but-out-of-range / numerically risky inputs (warnings.warn).

Pure and dependency-free (no numpy) so it is trivially testable and importable
from any tool. Callers pass plain floats / iterables of floats.
"""
import warnings
from typing import Optional


def elastic_medium_error(vp, vs, rho, label="medium") -> Optional[str]:
    """Return an error string if (vp, vs, rho) is not a physical elastic solid.

    Requires vp > 0, rho > 0, and 0 < vs < vp (which also keeps Poisson's ratio
    in (-1, 0.5)). Returns None when valid.
    """
    if vp is None or vp <= 0:
        return f"{label}: vp must be positive (got {vp})"
    if rho is None or rho <= 0:
        return f"{label}: density must be positive (got {rho})"
    if vs is None or vs <= 0:
        return f"{label}: vs must be positive (got {vs})"
    if vs >= vp:
        return f"{label}: vs must be less than vp (got vs={vs}, vp={vp})"
    return None


def positive_error(value, name) -> Optional[str]:
    """Return an error string if value is missing or non-positive, else None."""
    if value is None or value <= 0:
        return f"{name} must be positive (got {value})"
    return None


def angles_error(angles) -> Optional[str]:
    """Return an error string if any incidence angle is outside [0, 90) deg."""
    for a in angles:
        if a < 0 or a >= 90:
            return f"incidence angle must be in [0, 90) degrees (got {a})"
    return None


def require_elastic_medium(vp, vs, rho, label="medium") -> None:
    err = elastic_medium_error(vp, vs, rho, label)
    if err:
        raise ValueError(err)


def require_positive(value, name) -> None:
    err = positive_error(value, name)
    if err:
        raise ValueError(err)


def warn_if_aliased(max_content_hz, dt_seconds, label="wavelet") -> None:
    """Warn if frequency content reaches/exceeds Nyquist (0.5 / dt_seconds)."""
    if dt_seconds and dt_seconds > 0:
        nyquist = 0.5 / dt_seconds
        if max_content_hz >= nyquist:
            warnings.warn(
                f"{label}: frequency content (~{max_content_hz:g} Hz) reaches or exceeds "
                f"the Nyquist frequency ({nyquist:g} Hz) for dt={dt_seconds:g} s; "
                f"results may be aliased.",
                stacklevel=2,
            )


def warn_if_outside(value, lo, hi, name, unit="") -> None:
    """Warn (and proceed) if value is outside [lo, hi]."""
    if value < lo or value > hi:
        u = f" {unit}" if unit else ""
        warnings.warn(
            f"{name}={value:g}{u} is outside the expected range [{lo:g}, {hi:g}]{u}.",
            stacklevel=2,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics_guards.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/physics_guards.py tests/test_physics_guards.py
git commit -m "feat(guards): add physics_guards helpers (reject/warn predicates)"
```

---

### Task 2: AVO guards

**Files:**
- Modify: `tools/avo_tools.py` (top of `zoeppritz_reflectivity` and `shuey_reflectivity`; add imports)
- Modify: `tools/parameter_validation.py` (`validate_avo`)
- Test: `tests/test_input_guards.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_input_guards.py`:

```python
import warnings

import numpy as np
import pytest

from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity


def test_zoeppritz_rejects_vs_ge_vp():
    with pytest.raises(ValueError):
        zoeppritz_reflectivity(vp1=2500, vs1=2600, rho1=2.2,
                               vp2=3000, vs2=1500, rho2=2.4, angles=[0, 10])


def test_shuey_rejects_nonpositive_density():
    with pytest.raises(ValueError):
        shuey_reflectivity(vp1=2500, vs1=1200, rho1=0,
                           vp2=3000, vs2=1500, rho2=2.4, angles=[0, 10])


def test_avo_rejects_angle_ge_90():
    with pytest.raises(ValueError):
        zoeppritz_reflectivity(vp1=2500, vs1=1200, rho1=2.2,
                               vp2=3000, vs2=1500, rho2=2.4, angles=[95])


def test_avo_valid_still_works():
    rc = shuey_reflectivity(vp1=2500, vs1=1200, rho1=2.2,
                            vp2=3000, vs2=1500, rho2=2.4, angles=[0, 10, 20])
    assert np.all(np.isfinite(rc))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_input_guards.py -q`
Expected: FAIL — `test_zoeppritz_rejects_vs_ge_vp`, `test_shuey_rejects_nonpositive_density`, `test_avo_rejects_angle_ge_90` do not raise (they currently return arrays, possibly with NaN).

- [ ] **Step 3: Write minimal implementation**

In `tools/avo_tools.py`, change the top imports:

```python
# Tools for AVO analysis
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tools.physics_guards import require_elastic_medium, angles_error
```

Replace the first line of `zoeppritz_reflectivity` (currently `angles = np.radians(np.asarray(angles, dtype=float))`) with the guard block + conversion:

```python
    require_elastic_medium(vp1, vs1, rho1, "upper medium")
    require_elastic_medium(vp2, vs2, rho2, "lower medium")
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    _ang_err = angles_error(angles)
    if _ang_err:
        raise ValueError(_ang_err)
    if np.any(angles > 45):
        warnings.warn("AVO: incidence angles > 45 deg; results may be less reliable.", stacklevel=2)
    angles = np.radians(angles)
```

Replace the first line of `shuey_reflectivity` (currently `angles = np.radians(np.asarray(angles))`) with the same guard block:

```python
    require_elastic_medium(vp1, vs1, rho1, "upper medium")
    require_elastic_medium(vp2, vs2, rho2, "lower medium")
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    _ang_err = angles_error(angles)
    if _ang_err:
        raise ValueError(_ang_err)
    if np.any(angles > 45):
        warnings.warn("AVO: incidence angles > 45 deg; results may be less reliable.", stacklevel=2)
    angles = np.radians(angles)
```

In `tools/parameter_validation.py`, add an import near the other per-tool-validator imports (just below `from typing import Tuple as _Tuple, ...`):

```python
from tools.physics_guards import elastic_medium_error as _elastic_medium_error
```

Replace `validate_avo` with:

```python
def validate_avo(params: _Dict[str, _Any]) -> _Tuple[bool, str]:
    for p in ["vp1", "vs1", "rho1", "vp2", "vs2", "rho2", "angles"]:
        if p not in params:
            return False, f"Missing required parameter: {p}"
    err = (_elastic_medium_error(params["vp1"], params["vs1"], params["rho1"], "upper medium")
           or _elastic_medium_error(params["vp2"], params["vs2"], params["rho2"], "lower medium"))
    if err:
        return False, err
    return True, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_input_guards.py tests/test_tools.py -q`
Expected: PASS (new guard tests pass; existing `test_tools.py` AVO tests still pass — they use `vs<vp`, positive params, angles 0-6).

- [ ] **Step 5: Commit**

```bash
git add tools/avo_tools.py tools/parameter_validation.py tests/test_input_guards.py
git commit -m "feat(guards): reject unphysical media/angles in AVO tools"
```

---

### Task 3: Wedge guards

**Files:**
- Modify: `tools/wedge_tools.py` (`wedge_model`, after the layer arrays at line ~619; add imports)
- Modify: `tools/parameter_validation.py` (`validate_wedge_model` — drop the over-tight velocity band)
- Test: `tests/test_input_guards.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_input_guards.py`:

```python
from tools.wedge_tools import create_wedge_model

_WEDGE = dict(max_thickness=50, v1=2500, v2=3000, v3=3500, rho1=2.2, rho2=2.3, rho3=2.4)


def test_wedge_rejects_negative_density():
    args = dict(_WEDGE)
    args["rho2"] = -1
    with pytest.raises(ValueError):
        create_wedge_model(**args)


def test_wedge_rejects_vs_ge_vp_when_supplied():
    with pytest.raises(ValueError):
        create_wedge_model(vs1=3000, **_WEDGE)  # vs1=3000 >= vp1=2500


def test_wedge_accepts_velocity_inversion():
    # gas sand: v2 < v1 and rho2 < rho1 are physical and must NOT be rejected
    _, _, synth, _ = create_wedge_model(
        max_thickness=50, v1=3000, v2=2300, v3=3200, rho1=2.4, rho2=2.0, rho3=2.4
    )
    assert np.asarray(synth).ndim == 2


def test_wedge_warns_on_aliasing():
    with pytest.warns(UserWarning):
        create_wedge_model(wavelet_freq=200, dt=4.0, **_WEDGE)  # nyquist=125 Hz, content=600 Hz
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_input_guards.py -q`
Expected: FAIL — `test_wedge_rejects_negative_density` and `test_wedge_rejects_vs_ge_vp_when_supplied` do not raise; `test_wedge_warns_on_aliasing` emits no warning.

- [ ] **Step 3: Write minimal implementation**

In `tools/wedge_tools.py`, add to the relative imports block (next to `from .path_safety import safe_export_path`):

```python
from .physics_guards import require_elastic_medium, require_positive, warn_if_aliased, warn_if_outside
```

In `wedge_model`, immediately after the `vs_layers = [...]` assignment (the block ending at line ~619), insert:

```python
    # --- Physical-validity guards ---
    require_positive(max_thickness, "max_thickness")
    require_positive(dt, "dt")
    if num_traces < 2:
        raise ValueError(f"num_traces must be >= 2 (got {num_traces})")
    for _i in range(3):
        require_elastic_medium(vp_layers[_i], vs_layers[_i], rho_layers[_i], f"layer {_i + 1}")
        warn_if_outside(vp_layers[_i], 300, 8000, f"vp layer {_i + 1}", "m/s")
    # Nyquist / aliasing warning (dt is in ms here -> convert to seconds)
    if wv_type == 'ormsby' and ormsby_freq:
        _content_hz = float(ormsby_freq.split(',')[-1])
    else:
        _content_hz = 3.0 * ricker_freq
    warn_if_aliased(_content_hz, dt / 1000.0, "wedge wavelet")
```

In `tools/parameter_validation.py`, replace `validate_wedge_model` with a version that drops the over-restrictive 1500-6500 band (positivity only; the unusual-velocity warning lives in `wedge_model`):

```python
def validate_wedge_model(params: _Dict[str, _Any]) -> _Tuple[bool, str]:
    thickness = params.get("max_thickness")
    if not thickness or thickness <= 0:
        return False, "Maximum thickness must be positive"
    for i in range(1, 4):
        v = params.get(f"v{i}")
        if not v or v <= 0:
            return False, f"Velocity v{i} must be positive"
    for i in range(1, 4):
        rho = params.get(f"rho{i}")
        if not rho or rho <= 0:
            return False, f"Density rho{i} must be positive"
    return True, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_input_guards.py tests/test_tools.py tests/test_wedge_correctness.py tests/test_wedge_extras.py -q`
Expected: PASS (guard tests pass; existing wedge tests still pass — they use valid, positive params with `vs` defaulting to `vp/2 < vp`).

- [ ] **Step 5: Commit**

```bash
git add tools/wedge_tools.py tools/parameter_validation.py tests/test_input_guards.py
git commit -m "feat(guards): validate wedge layers; drop over-tight velocity band; Nyquist warn"
```

---

### Task 4: Ricker / Ormsby guards

**Files:**
- Modify: `tools/ricker_tools.py` (`create_ricker_wavelet`, `create_ormsby_wavelet`; add import)
- Test: `tests/test_input_guards.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_input_guards.py`:

```python
from tools.ricker_tools import create_ricker_wavelet, create_ormsby_wavelet


def test_ricker_rejects_nonpositive_frequency():
    with pytest.raises(ValueError):
        create_ricker_wavelet(frequency=0)


def test_ricker_warns_near_nyquist():
    with pytest.warns(UserWarning):
        create_ricker_wavelet(frequency=300, dt=0.002)  # nyquist=250 Hz, content=900 Hz


def test_ormsby_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        create_ormsby_wavelet(5, 10, 40, 50, dt=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_input_guards.py -k "ricker or ormsby" -q`
Expected: FAIL — `create_ricker_wavelet(frequency=0)` does not raise (produces a degenerate array); no Nyquist warning; `dt=0` does not raise.

- [ ] **Step 3: Write minimal implementation**

In `tools/ricker_tools.py`, add after the existing top imports:

```python
from tools.physics_guards import require_positive, warn_if_aliased
```

In `create_ricker_wavelet`, insert as the first statements of the body (before `f0 = frequency/1000.`):

```python
    require_positive(frequency, "frequency")
    require_positive(time_length, "time_length")
    require_positive(dt, "dt")
    warn_if_aliased(3.0 * frequency, dt, "ricker wavelet")
```

In `create_ormsby_wavelet`, after the existing `if not (f1 < f2 < f3 < f4): ...` / `if f1 < 0: ...` checks and before the `from tools.wedge_tools import ormsby` call, insert:

```python
    require_positive(time_length, "time_length")
    require_positive(dt, "dt")
    warn_if_aliased(f4, dt, "ormsby wavelet")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_input_guards.py tests/test_ormsby.py tests/test_wedge_extras.py -q`
Expected: PASS (existing wavelet tests use positive params and dt=0.001 s with low frequencies, so no spurious Nyquist warning).

- [ ] **Step 5: Commit**

```bash
git add tools/ricker_tools.py tests/test_input_guards.py
git commit -m "feat(guards): validate wavelet params; Nyquist warn for ricker/ormsby"
```

---

### Task 5: Rock-physics guards

**Files:**
- Modify: `tools/rock_physics_tools.py` (`calculate_rock_properties`; add `import warnings`)
- Test: `tests/test_input_guards.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_input_guards.py`:

```python
from tools.rock_physics_tools import calculate_rock_properties


def test_rock_rejects_porosity_gt_1():
    with pytest.raises(ValueError):
        calculate_rock_properties(1.5, 0.2, print_results=False)


def test_rock_warns_outside_han_range_and_clips():
    with pytest.warns(UserWarning):
        vp, vs, rhob, *_ = calculate_rock_properties(0.45, 0.2, print_results=False)
    assert float(vp) > float(vs) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_input_guards.py -k "rock" -q`
Expected: FAIL — `phit=1.5` does not raise (it is silently clipped); `phit=0.45` is silently clipped with no warning.

- [ ] **Step 3: Write minimal implementation**

In `tools/rock_physics_tools.py`, add `import warnings` to the top imports (below `import numpy as np`).

Replace the input-coercion/clip block at the start of `calculate_rock_properties`:

```python
    phit = np.asarray(phit, dtype=float)
    vclay = np.asarray(vclay, dtype=float)

    # Clip to the Han (1986) validity range (documented; avoids unphysical extrapolation).
    phit = np.clip(phit, 0.0, 0.35)
    vclay = np.clip(vclay, 0.0, 0.5)
```

with:

```python
    phit = np.asarray(phit, dtype=float)
    vclay = np.asarray(vclay, dtype=float)

    # REJECT physically impossible fractions.
    if np.any(phit < 0) or np.any(phit > 1):
        raise ValueError("phit (porosity) must be within [0, 1]")
    if np.any(vclay < 0) or np.any(vclay > 1):
        raise ValueError("vclay (clay volume) must be within [0, 1]")

    # WARN (not silent) when outside the Han (1986) validity range, then clip.
    if np.any(phit > 0.35):
        warnings.warn("phit beyond the Han (1986) validity range (>0.35); clipping to 0.35.", stacklevel=2)
    if np.any(vclay > 0.5):
        warnings.warn("vclay beyond the Han (1986) validity range (>0.5); clipping to 0.5.", stacklevel=2)
    phit = np.clip(phit, 0.0, 0.35)
    vclay = np.clip(vclay, 0.0, 0.5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_input_guards.py tests/test_rock_physics.py tests/test_tools.py -q`
Expected: PASS (existing rock-physics tests use in-range inputs, so no spurious warning/raise).

- [ ] **Step 5: Commit**

```bash
git add tools/rock_physics_tools.py tests/test_input_guards.py
git commit -m "feat(guards): reject impossible porosity/clay; warn-then-clip outside Han range"
```

---

### Task 6: Remove dead inversion-ordering checks

**Files:**
- Modify: `tools/parameter_validation.py` (`_validate_velocity_sequence`, `_validate_density_sequence`)
- Test: `tests/test_input_guards.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_input_guards.py`:

```python
from tools.parameter_validation import ParameterValidator


def test_validator_allows_velocity_and_density_inversion():
    v = ParameterValidator()
    ok_v, _ = v._validate_velocity_sequence({"v1": 3000, "v2": 2300, "v3": 3200})
    ok_r, _ = v._validate_density_sequence({"rho1": 2.4, "rho2": 2.0, "rho3": 2.4})
    assert ok_v is True
    assert ok_r is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_input_guards.py -k "inversion" -q`
Expected: FAIL — current `_validate_velocity_sequence` returns `False` for `v1 > v2` (it requires `v1 <= v2 <= v3`).

- [ ] **Step 3: Write minimal implementation**

In `tools/parameter_validation.py`, replace `_validate_velocity_sequence` and `_validate_density_sequence` (lines ~160-176) with positivity-only checks (ordering/inversions are physical and allowed):

```python
    def _validate_velocity_sequence(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Velocities must be positive; ordering (inversions) is allowed."""
        for k in ('v1', 'v2', 'v3'):
            v = params.get(k, 0)
            if v <= 0:
                return False, f"Velocity {k} must be positive (got {v})"
        return True, ""

    def _validate_density_sequence(self, params: Dict[str, float]) -> Tuple[bool, str]:
        """Densities must be positive; ordering (inversions) is allowed."""
        for k in ('rho1', 'rho2', 'rho3'):
            r = params.get(k, 0)
            if r <= 0:
                return False, f"Density {k} must be positive (got {r})"
        return True, ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_input_guards.py tests/test_parameter_validation.py -q`
Expected: PASS. If `tests/test_parameter_validation.py` contains a test asserting that an inversion is *rejected*, update it to assert acceptance (an inversion is physical) and note the change in the commit message.

- [ ] **Step 5: Commit**

```bash
git add tools/parameter_validation.py tests/test_input_guards.py
git commit -m "fix(guards): stop rejecting physical velocity/density inversions"
```

---

### Task 7: Document the guard policy

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add documentation**

Add a short subsection to `CLAUDE.md` near the existing validation/guards documentation:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document physical-validity input guards"
```

---

### Final verification

- [ ] **Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all tests pass (the 90 pre-existing + the new guard tests). Investigate any pre-existing valid-input test that now raises/warns — it indicates a threshold set too tight; adjust the threshold, not the guard's intent.

---

## Self-Review

**Spec coverage:**
- Two-tier policy (reject/warn) → Tasks 1-5. ✓
- Shared helper wired into validators + functions → Task 1 (helper), Tasks 2-5 (both call sites). ✓
- AVO elastic-medium + angle reject, >45° warn → Task 2. ✓
- Wedge per-layer medium, positivity, Nyquist warn, drop 1500-6500 band, vs<vp → Task 3. ✓
- Ricker/Ormsby positivity + Nyquist → Task 4. ✓
- Rock-physics reject [0,1] + warn-then-clip Han range → Task 5. ✓
- Remove dead inversion-ordering checks → Task 6. ✓
- Tests incl. inversion-accepted regression → Tasks 3 & 6. ✓
- Docs → Task 7. ✓
- Non-goal (no chat-UI warning surfacing) → noted in Task 7 doc. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type/name consistency:** Helper names (`elastic_medium_error`, `positive_error`, `angles_error`, `require_elastic_medium`, `require_positive`, `warn_if_aliased`, `warn_if_outside`) are used identically across Tasks 1-5. `validate_avo`/`validate_wedge_model` signatures unchanged (`(params) -> (bool, str)`). `wedge_model` uses already-existing `num_traces`/`dt` params and `vp_layers`/`vs_layers`/`rho_layers`. ✓
