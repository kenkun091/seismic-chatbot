"""petro_to_avo: AVO feasibility from petrophysics.

Predict elastic properties of a sand and an overlying shale from porosity and
clay volume (Han 1986), assemble the shale-over-sand interface, and model the
AVO reflectivity curve plus interpretation attributes (intercept, gradient,
AVO class). The composite plot is added in Task 2.
"""
import numpy as np

from workflows.adapters import predict_layer, build_interface
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity, avo_attributes


def petro_to_avo(phit_sand, vclay_sand, phit_shale, vclay_shale, angles,
                 fluid_sand="brine", fluid_shale="water", method="shuey"):
    """Run the petrophysics -> elastic -> interface -> AVO chain.

    Returns a JSON-friendly dict with the two layers, the reflectivity-vs-angle
    curve, and the AVO attributes.
    """
    if method not in ("shuey", "zoeppritz"):
        raise ValueError(f"method must be 'shuey' or 'zoeppritz' (got {method!r})")
    upper = predict_layer(phit_shale, vclay_shale, fluid=fluid_shale, label="shale")
    lower = predict_layer(phit_sand, vclay_sand, fluid=fluid_sand, label="sand")
    iface = build_interface(upper, lower)
    rc_fn = shuey_reflectivity if method == "shuey" else zoeppritz_reflectivity
    rc = np.asarray(rc_fn(**iface, angles=angles), dtype=float)
    attrs = avo_attributes(**iface)
    return {
        "upper": {"vp": upper.vp, "vs": upper.vs, "rho": upper.rho, "label": upper.label},
        "lower": {"vp": lower.vp, "vs": lower.vs, "rho": lower.rho, "label": lower.label},
        "angles": [float(a) for a in angles],
        "rc": [float(x) for x in rc],
        "intercept": float(attrs["intercept"]),
        "gradient": float(attrs["gradient"]),
        "avo_class": attrs["avo_class"],
        "avo_class_description": attrs["avo_class_description"],
        "method": method,
    }
