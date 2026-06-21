"""tuning: wedge tuning-thickness / vertical-resolution analysis from petrophysics.

Predict a sand and an encasing shale from porosity and clay volume, build a sand
wedge between two shale layers, convolve to a synthetic gather, and analyze the
amplitude-vs-thickness curve for tuning thickness and resolution limit, and
return a composite plot.
"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from workflows.adapters import predict_layer, build_earth_model
from tools.wedge_tools import create_wedge_model, analyze_wedge


def tuning(phit_sand, vclay_sand, phit_shale, vclay_shale, max_thickness,
           wavelet_freq=30.0, num_traces=61, fluid_sand="brine"):
    """Tuning-wedge analysis for a sand encased in shale, predicted from petrophysics.

    Returns a JSON-friendly dict with the sand/shale layers, the tuning thickness,
    tuning amplitude, resolution limit, and the amplitude-vs-thickness curve.
    """
    sand = predict_layer(phit_sand, vclay_sand, fluid=fluid_sand, label="sand")
    shale = predict_layer(phit_shale, vclay_shale, fluid="water", label="shale")
    earth = build_earth_model([shale, sand, shale])

    time_array, model, synthetic, parameters = create_wedge_model(
        max_thickness=max_thickness, wavelet_freq=wavelet_freq,
        num_traces=num_traces, **earth,
    )
    analysis = analyze_wedge(synthetic, parameters)
    image_path = plot_tuning(analysis, wavelet_freq)

    return {
        "sand": {"vp": sand.vp, "vs": sand.vs, "rho": sand.rho, "label": sand.label},
        "shale": {"vp": shale.vp, "vs": shale.vs, "rho": shale.rho, "label": shale.label},
        "tuning_thickness": float(analysis["tuning_thickness"]),
        "tuning_amplitude": float(analysis["tuning_amplitude"]),
        "resolution_limit": float(analysis["resolution_limit"]),
        "thicknesses": [float(t) for t in analysis["thicknesses"]],
        "max_amplitudes": [float(a) for a in analysis["max_amplitudes"]],
        "wavelet_freq": float(wavelet_freq),
        "max_thickness": float(max_thickness),
        "image_path": image_path,
    }


def plot_tuning(analysis, wavelet_freq, output_path=None):
    """Amplitude-vs-thickness curve with tuning-thickness and resolution-limit markers."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    thicknesses = np.asarray(analysis["thicknesses"], dtype=float)
    max_amplitudes = np.asarray(analysis["max_amplitudes"], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thicknesses, max_amplitudes, "b-", label="Max amplitude")
    ax.axvline(analysis["tuning_thickness"], color="r", ls="--",
               label=f"Tuning thickness: {analysis['tuning_thickness']:.2f} m")
    ax.axvline(analysis["resolution_limit"], color="g", ls="--",
               label=f"Resolution limit: {analysis['resolution_limit']:.2f} m")
    ax.set_xlabel("Thickness (m)")
    ax.set_ylabel("Maximum amplitude")
    ax.set_title(f"Wedge tuning curve ({wavelet_freq:g} Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
