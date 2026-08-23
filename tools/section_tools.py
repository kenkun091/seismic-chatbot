"""Generic 2-D convolutional synthetic over an elastic grid.

Input: vp / vs / rho grids (nz x nx) on a regular (dz, dx) mesh — any gridded
earth model (outcrop rasterization, hand-built, future imports). Per column:
depth -> TWT through that column's velocities, reflectivity at every property
change (acoustic at normal incidence; Shuey or exact Zoeppritz at an angle),
interfaces rounded onto the dt grid with superposition, then convolution with
a Ricker/Ormsby wavelet (tools/wedge_tools.gen_wavelet — same as the 1-D tool).

This module knows nothing about outcrops or photos.
"""
import os
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from tools.wedge_tools import gen_wavelet, plot_vawig
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity
from tools.synthetic_tools import _ormsby_corners
from tools.physics_guards import angles_error, warn_if_aliased

WAVELET_LENGTH_MS = 256.0   # identical to create_synthetic_seismogram


def validate_section_inputs(vp, vs, rho, dz, dx, angle, method, wv_type, ormsby_freq,
                            dt, pad_time, wavelet_freq, domain) -> None:
    """REJECT tier for the 2-D section (raises ValueError)."""
    vp = np.asarray(vp, dtype=float); vs = np.asarray(vs, dtype=float); rho = np.asarray(rho, dtype=float)
    if vp.ndim != 2:
        raise ValueError(f"vp/vs/rho must be 2-D (nz x nx) grids; got {vp.ndim}-D")
    if not (vp.shape == vs.shape == rho.shape):
        raise ValueError(f"vp, vs and rho must share one shape; got {vp.shape}, {vs.shape}, {rho.shape}")
    if vp.shape[0] < 2 or vp.shape[1] < 1:
        raise ValueError(f"grid needs at least 2 rows and 1 column; got shape {vp.shape}")
    for name, arr in (("vp", vp), ("vs", vs), ("rho", rho)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} grid must be finite everywhere")
    if np.any(vp <= 0):
        raise ValueError("vp must be positive everywhere")
    if np.any(rho <= 0):
        raise ValueError("rho must be positive everywhere")
    if np.any(vs <= 0) or np.any(vs >= vp):
        raise ValueError("vs must satisfy 0 < vs < vp everywhere")
    for name, val in (("dz", dz), ("dx", dx), ("dt", dt), ("wavelet_freq", wavelet_freq)):
        if not (isinstance(val, (int, float)) and np.isfinite(val) and val > 0):
            raise ValueError(f"{name} must be a positive number (got {val!r})")
    if not (isinstance(pad_time, (int, float)) and pad_time >= 0):
        raise ValueError(f"pad_time must be >= 0 ms (got {pad_time!r})")
    err = angles_error(np.atleast_1d(float(angle)))
    if err:
        raise ValueError(f"angle: {err}")
    if method not in ("shuey", "zoeppritz"):
        raise ValueError("method must be 'shuey' or 'zoeppritz'")
    if wv_type not in ("ricker", "ormsby"):
        raise ValueError("wv_type must be 'ricker' or 'ormsby'")
    if wv_type == "ormsby":
        if not ormsby_freq:
            raise ValueError("ormsby_freq ('f1,f2,f3,f4') is required when wv_type='ormsby'")
        _ormsby_corners(ormsby_freq)   # raises on malformed corners
    if domain not in ("time", "depth"):
        raise ValueError("domain must be 'time' or 'depth'")


def _interface_rc(vp1, vs1, rho1, vp2, vs2, rho2, angle, method, cache):
    key = (vp1, vs1, rho1, vp2, vs2, rho2)
    if key in cache:
        return cache[key]
    if angle == 0:
        z1, z2 = vp1 * rho1, vp2 * rho2
        rc = (z2 - z1) / (z2 + z1)
    else:
        fn = shuey_reflectivity if method == "shuey" else zoeppritz_reflectivity
        rc = float(np.asarray(fn(vp1=vp1, vs1=vs1, rho1=rho1, vp2=vp2, vs2=vs2,
                                 rho2=rho2, angles=[angle])).ravel()[0])
    cache[key] = rc
    return rc


def create_synthetic_section(vp, vs, rho, dz, dx, wavelet_freq=30.0, wv_type="ricker",
                             ormsby_freq=None, phase_rot=0.0, angle=0.0, method="shuey",
                             dt=1.0, pad_time=50.0, domain="time"
                             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """2-D convolutional synthetic. Returns (axis, section, parameters).

    domain='time'  -> axis is TWT in ms (nt), section is nt x nx.
    domain='depth' -> axis is depth in m (nz), section is the depth-converted
                      nt x nx result (computed in time, then mapped column by
                      column through that column's own t(z)).
    """
    validate_section_inputs(vp, vs, rho, dz, dx, angle, method, wv_type, ormsby_freq,
                            dt, pad_time, wavelet_freq, domain)
    vp = np.asarray(vp, dtype=float); vs = np.asarray(vs, dtype=float); rho = np.asarray(rho, dtype=float)
    nz, nx = vp.shape
    angle = float(angle)

    if wv_type == "ormsby":
        corners = _ormsby_corners(ormsby_freq)
        content_hz = corners[3]
        dominant_freq = (corners[1] + corners[2]) / 2.0
    else:
        content_hz = 3.0 * wavelet_freq
        dominant_freq = float(wavelet_freq)
    warn_if_aliased(content_hz, dt / 1000.0, "section wavelet")

    # TWT at the BOTTOM of every cell, per column (ms); top of grid at pad_time.
    twt_bottom = pad_time + np.cumsum(2000.0 * dz / vp, axis=0)
    total_twt = twt_bottom[-1, :].max()

    _, wavelet, wavelet_label = gen_wavelet(dt, wv_type, wavelet_freq, ormsby_freq, "", "",
                                            phase_rot, wavelet_length=WAVELET_LENGTH_MS)
    nt = int(round((total_twt + pad_time) / dt)) + 1
    nt = max(nt, wavelet.size)
    time_array = np.arange(nt) * dt

    rc_series = np.zeros((nt, nx))
    cache: Dict[tuple, float] = {}
    n_interfaces = 0
    n_postcritical = 0
    for j in range(nx):
        col_vp, col_vs, col_rho = vp[:, j], vs[:, j], rho[:, j]
        change = np.where((col_vp[1:] != col_vp[:-1]) | (col_vs[1:] != col_vs[:-1])
                          | (col_rho[1:] != col_rho[:-1]))[0]
        for k in change:
            rc = _interface_rc(col_vp[k], col_vs[k], col_rho[k],
                               col_vp[k + 1], col_vs[k + 1], col_rho[k + 1],
                               angle, method, cache)
            n_interfaces += 1
            if not np.isfinite(rc):
                n_postcritical += 1
                rc = 0.0
            idx = int(round(twt_bottom[k, j] / dt))
            if 0 <= idx < nt:
                rc_series[idx, j] += rc   # superpose thin layers (same as the 1-D tool)
    if n_postcritical:
        warnings.warn(f"{n_postcritical} post-critical Zoeppritz interface(s) at {angle:g} deg "
                      f"were set to zero reflectivity", stacklevel=2)

    section = scipy.signal.convolve(rc_series, wavelet[:, None], mode="same")

    parameters = {
        "nt": int(nt), "dt": float(dt), "nx": int(nx), "dx": float(dx),
        "nz": int(nz), "dz": float(dz), "pad_time": float(pad_time),
        "angle": angle, "method": method,
        "wavelet_freq": float(dominant_freq), "wavelet_label": wavelet_label,
        "domain": domain, "n_interfaces": int(n_interfaces),
        "max_abs_amplitude": float(np.max(np.abs(section))) if section.size else 0.0,
        "n_postcritical_zeroed": int(n_postcritical),
    }
    if domain == "depth":
        z_axis = (np.arange(nz) + 0.5) * dz
        return z_axis, depth_convert(section, time_array, vp, dz, pad_time), parameters
    return time_array, section, parameters


def depth_convert(section, time_array, vp, dz, pad_time) -> np.ndarray:
    """Map a time section (nt x nx) onto the model's depth cells (nz x nx).

    Each column is interpolated at the TWT of its own cell centres, so the
    result registers with the elastic grid (and the photo) column by column.
    """
    section = np.asarray(section, dtype=float); vp = np.asarray(vp, dtype=float)
    nz, nx = vp.shape
    out = np.zeros((nz, nx))
    for j in range(nx):
        cell_twt = 2000.0 * dz / vp[:, j]
        t_center = pad_time + np.cumsum(cell_twt) - 0.5 * cell_twt
        out[:, j] = np.interp(t_center, time_array, section[:, j], left=0.0, right=0.0)
    return out
