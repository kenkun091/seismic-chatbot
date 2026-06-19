"""Adapters that glue rock-physics outputs to AVO/wedge tool inputs.

Closes the mechanical gaps between the leaf tools:
- G2: unpack `calculate_rock_properties` (a positional tuple) / `gassmann_substitution`
  (a dict) into a typed `Layer`.
- G3: reduce a log of samples (arrays) to one representative scalar layer.
- G1: assemble layers into the {vp1, vs1, ...} / {v1, v2, v3, ...} dicts that the
  AVO and wedge tools require as inputs.

The leaf tools themselves are not modified.
"""
import numpy as np

from tools.rock_physics_tools import calculate_rock_properties, gassmann_substitution
from tools.physics_guards import require_elastic_medium
from workflows.types import Layer


def _reduce(values, reduce="mean"):
    """Reduce a scalar/array to one float.

    `reduce` is 'mean', 'median', or an integer sample index.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("cannot reduce an empty array to a scalar Layer")
    if reduce == "mean":
        return float(np.mean(arr))
    if reduce == "median":
        return float(np.median(arr))
    if isinstance(reduce, int) and not isinstance(reduce, bool):
        return float(arr[reduce])
    raise ValueError(f"unknown reduce mode: {reduce!r}")


def predict_layer(phit, vclay, fluid="water", *, reduce="mean", label=""):
    """Predict a representative elastic `Layer` from porosity + clay volume (G2 + G3)."""
    vp, vs, rho, *_ = calculate_rock_properties(
        phit, vclay, fluid_type=fluid, print_results=False
    )
    layer = Layer(
        vp=_reduce(vp, reduce),
        vs=_reduce(vs, reduce),
        rho=_reduce(rho, reduce),
        label=label,
    )
    require_elastic_medium(layer.vp, layer.vs, layer.rho, label=label or "layer")
    return layer
