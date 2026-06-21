"""saturation_sweep: rock properties vs water saturation (the fluid line).

For a single rock (porosity + clay volume), compute Vp/Vs/AI/(Vp/Vs) across a
range of water saturations Sw using rock_properties_saturation (Reuss or Brie
mixing), and plot the saturation curves. Self-plots via image_path.
"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from tools.rock_physics_tools import rock_properties_saturation


def saturation_sweep(phit, vclay, hydrocarbon="gas", law="reuss",
                     sw_values=None, brie_exponent=3.0):
    """Sweep water saturation for one rock and return Vp/Vs/AI/(Vp/Vs) curves + plot."""
    if law not in ("reuss", "brie"):
        raise ValueError(f"law must be 'reuss' or 'brie' (got {law!r})")
    if sw_values is None:
        sw_values = list(np.linspace(0.0, 1.0, 21))
    sw = np.asarray(sw_values, dtype=float)
    if sw.size == 0:
        raise ValueError("sw_values must contain at least one saturation")

    vp, vs, rhob, vp_vs, ai, si = rock_properties_saturation(
        phit, vclay, sw, hydrocarbon=hydrocarbon, law=law, brie_exponent=brie_exponent
    )
    result = {
        "sw": [float(x) for x in sw],
        "vp": [float(x) for x in np.atleast_1d(vp)],
        "vs": [float(x) for x in np.atleast_1d(vs)],
        "ai": [float(x) for x in np.atleast_1d(ai)],
        "vp_vs": [float(x) for x in np.atleast_1d(vp_vs)],
        "hydrocarbon": hydrocarbon,
        "law": law,
    }
    result["image_path"] = plot_saturation_sweep(
        result["sw"], result["vp"], result["vs"], result["ai"], hydrocarbon, law
    )
    return result


def plot_saturation_sweep(sw, vp, vs, ai, hydrocarbon, law, output_path=None):
    """Two-panel: Vp & Vs vs Sw (left), AI vs Sw (right)."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    sw = np.asarray(sw, dtype=float)

    fig, (ax_v, ax_ai) = plt.subplots(1, 2, figsize=(12, 5))
    ax_v.plot(sw, np.asarray(vp, dtype=float), "b-o", label="Vp")
    ax_v.plot(sw, np.asarray(vs, dtype=float), "r-s", label="Vs")
    ax_v.set_xlabel("Water saturation Sw")
    ax_v.set_ylabel("Velocity (m/s)")
    ax_v.set_title("Velocity vs saturation")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    ax_ai.plot(sw, np.asarray(ai, dtype=float), "g-^", label="AI")
    ax_ai.set_xlabel("Water saturation Sw")
    ax_ai.set_ylabel("Acoustic impedance (×10⁶ kg/m²·s)")
    ax_ai.set_title("Impedance vs saturation")
    ax_ai.grid(True, alpha=0.3)
    ax_ai.legend()

    fig.suptitle(f"Saturation sweep ({hydrocarbon}, {law} mixing)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
