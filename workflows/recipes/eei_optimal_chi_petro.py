"""eei_optimal_chi_petro: EEI optimal-chi from petrophysical logs.

Predict Vp/Vs/density logs from porosity and clay-volume logs (rock physics), then
find the EEI rotation angle chi whose EEI log best correlates with a chosen
petrophysical target (Vclay for lithology, porosity, or water saturation). Wraps
the shared _eei_chi_scan core; self-plots via plot_eei_chi_scan.
"""
import numpy as np

from tools.rock_physics_tools import calculate_rock_properties, rock_properties_saturation
from tools.avo_tools import _eei_chi_scan, plot_eei_chi_scan


_TARGETS = {"vclay", "phit", "sw"}


def eei_optimal_chi_petro(phit, vclay, target="vclay", fluid="brine",
                          chi_min=-90.0, chi_max=90.0, chi_step=1.0,
                          sw=None, hydrocarbon="gas", law="reuss"):
    """Find optimal EEI chi against a petrophysical target, from porosity/clay logs."""
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {sorted(_TARGETS)} (got {target!r})")

    phit = np.asarray(phit, dtype=float)
    vclay = np.asarray(vclay, dtype=float)

    if target == "sw":
        if sw is None:
            raise ValueError("target='sw' requires the sw log to be provided")
        sw = np.asarray(sw, dtype=float)
        vp, vs, rhob, *_ = rock_properties_saturation(
            phit, vclay, sw, hydrocarbon=hydrocarbon, law=law
        )
        target_log = sw
    else:
        vp, vs, rhob, *_ = calculate_rock_properties(
            phit, vclay, fluid_type=fluid, print_results=False
        )
        target_log = vclay if target == "vclay" else phit

    chi = np.arange(chi_min, chi_max + chi_step, chi_step)
    result = _eei_chi_scan(vp, vs, rhob, target_log, chi)
    result["target"] = target
    result["fluid"] = fluid
    result["image_path"] = plot_eei_chi_scan(
        result["chi"], result["correlation"], result["optimal_chi"]
    )
    return result
