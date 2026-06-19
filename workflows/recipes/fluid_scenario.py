"""fluid_scenario: AVO fluid-substitution scenarios (brine vs gas vs ...).

Predict the in-situ sand and overlying shale from petrophysics, then use Gassmann
fluid substitution to model the AVO response for each scenario fluid. This mirrors
the interpretation workflow: log-derived in-situ properties, substituted to
alternate fluids to test the AVO / DHI response. The per-fluid sand layers are
organized in a Scenario. The composite plot is added in Task 2.
"""
import numpy as np

from workflows.types import Scenario
from workflows.adapters import predict_layer, build_interface, layer_from_gassmann
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity, avo_attributes


def fluid_scenario(phit_sand, vclay_sand, phit_shale, vclay_shale, angles,
                   fluids=None, fluid_in="brine", method="shuey"):
    """Model AVO for a sand under shale across several pore fluids (Gassmann).

    Returns a JSON-friendly dict with the shale layer and, per fluid case, the
    substituted sand layer, the reflectivity-vs-angle curve, and AVO attributes.
    """
    if method not in ("shuey", "zoeppritz"):
        raise ValueError(f"method must be 'shuey' or 'zoeppritz' (got {method!r})")
    if fluids is None:
        fluids = ["brine", "gas"]

    rc_fn = shuey_reflectivity if method == "shuey" else zoeppritz_reflectivity
    shale = predict_layer(phit_shale, vclay_shale, fluid="water", label="shale")
    in_situ = predict_layer(phit_sand, vclay_sand, fluid=fluid_in, label="sand")

    # Build the per-fluid sand layers into a Scenario (in-situ used as-is; others
    # via Gassmann substitution from the in-situ state).
    sand_layers = {}
    for f in fluids:
        if f == fluid_in:
            sand_layers[f] = in_situ
        else:
            sand_layers[f] = layer_from_gassmann(
                in_situ.vp, in_situ.vs, in_situ.rho, phi=phit_sand,
                fluid_in=fluid_in, fluid_out=f, label=f"sand-{f}",
            )
    scenario = Scenario(name="fluid", cases=sand_layers)

    cases = {}
    for f, sand_f in scenario.cases.items():
        iface = build_interface(shale, sand_f)
        rc = np.asarray(rc_fn(**iface, angles=angles), dtype=float)
        attrs = avo_attributes(**iface)
        cases[f] = {
            "layer": {"vp": sand_f.vp, "vs": sand_f.vs, "rho": sand_f.rho, "label": sand_f.label},
            "rc": [float(x) for x in rc],
            "intercept": float(attrs["intercept"]),
            "gradient": float(attrs["gradient"]),
            "avo_class": attrs["avo_class"],
            "avo_class_description": attrs["avo_class_description"],
        }

    return {
        "shale": {"vp": shale.vp, "vs": shale.vs, "rho": shale.rho, "label": shale.label},
        "fluids": list(fluids),
        "fluid_in": fluid_in,
        "cases": cases,
        "angles": [float(a) for a in angles],
        "method": method,
    }
