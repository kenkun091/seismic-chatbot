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


def build_interface(upper: Layer, lower: Layer) -> dict:
    """Assemble two layers into the AVO interface dict the reflectivity tools expect.

    Returns {vp1, vs1, rho1, vp2, vs2, rho2} (G1). Rejects non-physical layers.
    """
    require_elastic_medium(upper.vp, upper.vs, upper.rho, label=upper.label or "upper")
    require_elastic_medium(lower.vp, lower.vs, lower.rho, label=lower.label or "lower")
    return {
        "vp1": upper.vp, "vs1": upper.vs, "rho1": upper.rho,
        "vp2": lower.vp, "vs2": lower.vs, "rho2": lower.rho,
    }


def build_earth_model(layers) -> dict:
    """Assemble exactly 3 layers into the wedge-model input dict (G1).

    Returns {v1, v2, v3, rho1, rho2, rho3, vs1, vs2, vs3} keyed for create_wedge_model.
    """
    layers = list(layers)
    if len(layers) != 3:
        raise ValueError(f"build_earth_model expects exactly 3 layers (got {len(layers)})")
    for i, ly in enumerate(layers, start=1):
        require_elastic_medium(ly.vp, ly.vs, ly.rho, label=ly.label or f"layer{i}")
    l1, l2, l3 = layers
    return {
        "v1": l1.vp, "v2": l2.vp, "v3": l3.vp,
        "rho1": l1.rho, "rho2": l2.rho, "rho3": l3.rho,
        "vs1": l1.vs, "vs2": l2.vs, "vs3": l3.vs,
    }


def layer_from_gassmann(vp, vs, rho, phi, fluid_in, fluid_out,
                        *, reduce="mean", label="", **kwargs) -> Layer:
    """Run Gassmann fluid substitution and adapt the result dict into a `Layer` (G2).

    Extra kwargs (k_mineral, k_fl_in, rho_fl_in, k_fl_out, rho_fl_out) pass through
    to gassmann_substitution.
    """
    res = gassmann_substitution(
        vp=vp, vs=vs, rho=rho, phi=phi,
        fluid_in=fluid_in, fluid_out=fluid_out,
        print_results=False, **kwargs,
    )
    layer = Layer(
        vp=_reduce(res["vp"], reduce),
        vs=_reduce(res["vs"], reduce),
        rho=_reduce(res["rho"], reduce),
        label=label,
    )
    require_elastic_medium(layer.vp, layer.vs, layer.rho, label=label or "substituted")
    return layer


def predict_elastic_layer(phit, vclay, fluid="water", reduce="mean") -> dict:
    """LLM-facing leaf tool: representative elastic properties of one layer as a dict.

    Thin wrapper over predict_layer that returns a JSON-friendly dict (no Layer type),
    so it can be registered as a standard leaf tool.
    """
    layer = predict_layer(phit, vclay, fluid=fluid, reduce=reduce)
    return {"vp": layer.vp, "vs": layer.vs, "rho": layer.rho, "vp_vs": layer.vp / layer.vs}
