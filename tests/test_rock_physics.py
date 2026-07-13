"""Rock physics correctness: Han (1986) water-saturated velocities, mass-balance
density, and proper Gassmann fluid substitution (shear modulus fluid-independent)."""
import numpy as np

from tools.rock_physics_tools import calculate_rock_properties, gassmann_sat, gassmann_dry


def test_gassmann_forward_inverse_roundtrip():
    K0, K_fl, phi, K_dry = 31.16e9, 2.2e9, 0.2, 1.418e10
    K_sat = gassmann_sat(K_dry, K0, K_fl, phi)
    assert K_sat > K_dry  # saturating with fluid stiffens the rock
    recovered = gassmann_dry(K_sat, K0, K_fl, phi)
    assert abs(recovered - K_dry) / K_dry < 1e-6


def test_water_saturated_velocities_match_han():
    # phi=0.20, C=0.30 -> Han: Vp=3.55 km/s, Vs=1.971 km/s
    vp, vs, rhob, vpvs, ai, si = calculate_rock_properties(
        0.20, 0.30, fluid_type="water", print_results=False
    )
    assert abs(float(vp) - 3550.0) < 5.0
    assert abs(float(vs) - 1971.0) < 5.0
    assert 1.4 < float(vpvs) < 2.6


def test_gas_lowers_vp_but_raises_vs_relative_to_water():
    """The crux: Gassmann holds shear modulus fluid-independent, so replacing
    brine with lighter gas LOWERS Vp (soft fluid) but slightly RAISES Vs (lower
    density). The old code wrongly reduced Vs for gas."""
    vp_w, vs_w, rho_w, *_ = calculate_rock_properties(0.25, 0.20, "water", print_results=False)
    vp_g, vs_g, rho_g, *_ = calculate_rock_properties(0.25, 0.20, "gas", print_results=False)
    assert float(vp_g) < float(vp_w)          # gas softens K -> Vp down
    assert float(vs_g) > float(vs_w)           # mu constant, density down -> Vs up
    assert float(rho_g) < float(rho_w)         # gas is lighter
    assert (float(vs_g) - float(vs_w)) / float(vs_w) < 0.10  # the Vs rise is modest


def test_outputs_are_physical():
    vp, vs, rhob, vpvs, ai, si = calculate_rock_properties(0.15, 0.10, "oil", print_results=False)
    assert float(vp) > float(vs) > 0
    assert float(rhob) > 0
    assert float(ai) > float(si) > 0


def test_unknown_fluid_raises():
    import pytest
    with pytest.raises(ValueError):
        calculate_rock_properties(0.2, 0.2, fluid_type="plasma", print_results=False)
