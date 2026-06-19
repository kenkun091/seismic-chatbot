"""petro_to_avo: AVO feasibility from petrophysics.

Predict elastic properties of a sand and an overlying shale from porosity and
clay volume (Han 1986), assemble the shale-over-sand interface, and model the
AVO reflectivity curve plus interpretation attributes (intercept, gradient,
AVO class). The composite plot is added in Task 2.
"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

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
    image_path = plot_petro_to_avo(upper, lower, angles, rc, attrs)
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
        "image_path": image_path,
    }


def plot_petro_to_avo(upper, lower, angles, rc, attrs, output_path=None):
    """Composite plot: model/attribute summary, R(theta) curve, A-B point."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    angles = np.asarray(angles, dtype=float)
    rc = np.asarray(rc, dtype=float)

    fig, (ax_tbl, ax_rc, ax_ab) = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: layer + attribute summary (monospace text)
    ax_tbl.axis("off")
    lines = [
        f"{'':10s}{'Vp':>8s}{'Vs':>8s}{'rho':>7s}",
        f"{upper.label:10s}{upper.vp:8.0f}{upper.vs:8.0f}{upper.rho:7.2f}",
        f"{lower.label:10s}{lower.vp:8.0f}{lower.vs:8.0f}{lower.rho:7.2f}",
        "",
        f"Intercept A = {attrs['intercept']:.4f}",
        f"Gradient  B = {attrs['gradient']:.4f}",
        f"AVO class   = {attrs['avo_class']}",
    ]
    ax_tbl.text(0.0, 0.95, "\n".join(lines), family="monospace", va="top", fontsize=11)
    ax_tbl.set_title("Model & AVO attributes")

    # Panel 2: R(theta)
    ax_rc.plot(angles, rc, "o-", color="C0")
    ax_rc.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax_rc.set_xlabel("Incidence angle (deg)")
    ax_rc.set_ylabel("Reflection coefficient")
    ax_rc.set_title("AVO reflectivity")
    ax_rc.grid(True, alpha=0.3)

    # Panel 3: intercept-gradient point
    A = float(attrs["intercept"])
    B = float(attrs["gradient"])
    lim = max(0.1, abs(A) * 1.5, abs(B) * 1.5)
    ax_ab.axhline(0.0, color="grey", lw=0.8)
    ax_ab.axvline(0.0, color="grey", lw=0.8)
    ax_ab.plot([A], [B], "s", color="C3", markersize=10)
    ax_ab.annotate(attrs["avo_class"], (A, B),
                   textcoords="offset points", xytext=(8, 8))
    ax_ab.set_xlim(-lim, lim)
    ax_ab.set_ylim(-lim, lim)
    ax_ab.set_xlabel("Intercept A")
    ax_ab.set_ylabel("Gradient B")
    ax_ab.set_title("Intercept-Gradient")

    fig.suptitle(f"Petro -> AVO: {lower.label} below {upper.label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
