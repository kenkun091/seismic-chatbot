"""outcrop_to_seismic: outcrop photo -> interpretation -> 2-D model -> seismic section.

One-shot chain of the staged tools (interpret_outcrop -> outcrop_to_model ->
synthetic_section -> plots). The chatbot stores the intermediate results so
follow-up corrections can re-run only the offline steps.
"""
import os
import tempfile

import numpy as np

from tools.outcrop_tools import (interpret_outcrop, plot_outcrop_interpretation,
                                 outcrop_to_model)
from tools.section_tools import synthetic_section_from_model, plot_seismic_section


def outcrop_to_seismic(image_path=None, height_m=None, overrides=None,
                       background_lithology=None, wavelet_freq=30.0, angle=0.0,
                       method="shuey", domain="time", display="image",
                       num_traces=101, vision_client=None):
    """Returns a JSON-friendly dict (see tests). Only interpret_outcrop calls the VLM."""
    interp = interpret_outcrop(image_path, vision_client=vision_client)
    overlay = plot_outcrop_interpretation(interp)
    model = outcrop_to_model(interp, height_m=height_m, overrides=overrides,
                             background_lithology=background_lithology,
                             num_traces=num_traces, wavelet_freq=wavelet_freq)
    axis, section, parameters = synthetic_section_from_model(
        model, wavelet_freq=wavelet_freq, angle=angle, method=method, domain=domain)
    png = plot_seismic_section(section, parameters, axis=axis, model=model, display=display)
    return {
        "interpretation": interp,
        "model": model,
        "section": {"axis": axis, "section": section, "parameters": parameters},
        "regions": model["regions"],
        "scale": {"height_m": model["height_m"], "source": model["scale_source"],
                  "confidence": model["scale_confidence"]},
        "grid_shape": [int(model["nz"]), int(model["nx"])],
        "n_regions": sum(1 for r in model["regions"] if r["route"] != "background"),
        "n_interfaces": parameters["n_interfaces"],
        "max_abs_amplitude": parameters["max_abs_amplitude"],
        "wavelet_freq": float(wavelet_freq),
        "angle": float(angle),
        "domain": domain,
        "image_path": png,
        "extra_image_paths": [overlay],
    }
