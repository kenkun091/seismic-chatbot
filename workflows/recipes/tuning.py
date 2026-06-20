"""tuning: wedge tuning-thickness / vertical-resolution analysis from petrophysics.

Predict a sand and an encasing shale from porosity and clay volume, build a sand
wedge between two shale layers, convolve to a synthetic gather, and analyze the
amplitude-vs-thickness curve for tuning thickness and resolution limit. The
composite plot is added in Task 2.
"""
import numpy as np

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
    }
