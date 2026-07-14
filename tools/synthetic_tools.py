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
