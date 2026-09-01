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
