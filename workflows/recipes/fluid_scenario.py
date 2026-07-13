"""fluid_scenario: AVO fluid-substitution scenarios (brine vs gas vs ...).

Predict the in-situ sand and overlying shale from petrophysics, then use Gassmann
fluid substitution to model the AVO response for each scenario fluid. This mirrors
the interpretation workflow: log-derived in-situ properties, substituted to
alternate fluids to test the AVO / DHI response. The per-fluid sand layers are
organized in a Scenario, and an overlaid composite plot (R(theta) and the
intercept-gradient points per fluid) is returned via image_path.
"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

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
    if not fluids:
        raise ValueError("fluids must contain at least one pore fluid")

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

    image_path = plot_fluid_scenario(shale, angles, cases, method)

    return {
        "shale": {"vp": shale.vp, "vs": shale.vs, "rho": shale.rho, "label": shale.label},
        "fluids": list(fluids),
        "fluid_in": fluid_in,
        "cases": cases,
        "angles": [float(a) for a in angles],
        "method": method,
        "image_path": image_path,
    }


def plot_fluid_scenario(shale, angles, cases, method, output_path=None):
    """Overlaid composite: R(theta) per fluid (left) and A-B points per fluid (right)."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    angles = np.asarray(angles, dtype=float)

    fig, (ax_rc, ax_ab) = plt.subplots(1, 2, figsize=(12, 5))

    for f, res in cases.items():
        ax_rc.plot(angles, np.asarray(res["rc"], dtype=float), "o-", label=f)
        ax_ab.plot([res["intercept"]], [res["gradient"]], "s", markersize=10,
                   label=f"{f} ({res['avo_class']})")

    ax_rc.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax_rc.set_xlabel("Incidence angle (deg)")
    ax_rc.set_ylabel("Reflection coefficient")
    ax_rc.set_title(f"AVO by fluid ({method})")
    ax_rc.legend()
    ax_rc.grid(True, alpha=0.3)

    vals = [abs(res["intercept"]) for res in cases.values()]
    vals += [abs(res["gradient"]) for res in cases.values()]
    lim = max(0.1, max(vals, default=0.0) * 1.5)
    ax_ab.axhline(0.0, color="grey", lw=0.8)
    ax_ab.axvline(0.0, color="grey", lw=0.8)
    ax_ab.set_xlim(-lim, lim)
    ax_ab.set_ylim(-lim, lim)
    ax_ab.set_xlabel("Intercept A")
    ax_ab.set_ylabel("Gradient B")
    ax_ab.set_title("Intercept-Gradient by fluid")
    ax_ab.legend()

    fig.suptitle(f"Fluid scenarios: sand below {shale.label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
