import numpy as np
import pytest

from tools.rock_physics_tools import gassmann_substitution


def test_roundtrip_identity_same_fluid():
    # Substituting a fluid for itself returns inputs unchanged (phi > 0).
    res = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.25,
        fluid_in="brine", fluid_out="brine", print_results=False,
    )
    assert np.isclose(res["vp"], 3000.0, rtol=1e-6)
    assert np.isclose(res["vs"], 1500.0, rtol=1e-6)
    assert np.isclose(res["rho"], 2.2, rtol=1e-6)


def test_brine_to_gas_signature():
    # Gas vs brine: Vp DOWN, Vs UP (shear modulus fixed, lower density), rho DOWN.
    res = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    assert res["vp"] < 3000.0
    assert res["vs"] > 1500.0
    assert res["rho"] < 2.2
