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


from tools.rock_physics_tools import calculate_rock_properties


def test_matches_calculate_rock_properties_gas_case():
    # With vclay=0, calculate_rock_properties' mineral modulus K0 == 37 GPa (pure
    # quartz VRH), matching the default k_mineral. Feeding its water-sat output
    # into gassmann_substitution(water->gas) must reproduce its gas-sat output.
    phi = 0.2
    vp_w, vs_w, rhob_w, *_ = calculate_rock_properties(phi, 0.0, "water", print_results=False)
    vp_g, vs_g, rhob_g, *_ = calculate_rock_properties(phi, 0.0, "gas", print_results=False)

    res = gassmann_substitution(
        vp=float(vp_w), vs=float(vs_w), rho=float(rhob_w), phi=phi,
        fluid_in="water", fluid_out="gas", k_mineral=37.0, print_results=False,
    )
    assert np.isclose(res["vp"], float(vp_g), rtol=1e-6)
    assert np.isclose(res["vs"], float(vs_g), rtol=1e-6)
    assert np.isclose(res["rho"], float(rhob_g), rtol=1e-6)


def test_array_inputs_return_arrays():
    res = gassmann_substitution(
        vp=np.array([3000.0, 3200.0]), vs=np.array([1500.0, 1600.0]),
        rho=np.array([2.2, 2.25]), phi=np.array([0.2, 0.18]),
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    assert res["vp"].shape == (2,)
    assert res["vs"].shape == (2,)
    assert res["rho"].shape == (2,)


def test_custom_fluid_override_differs_from_preset():
    base = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    # Override the target fluid with a much stiffer, denser "gas" -> different result.
    override = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas",
        k_fl_out=1.5, rho_fl_out=0.6, print_results=False,
    )
    assert not np.isclose(base["vp"], override["vp"])
    assert override["rho"] > base["rho"]  # denser override fluid -> higher bulk density


def test_guards_reject_bad_inputs():
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=1.5, fluid_in="brine", fluid_out="gas", print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=-0.1, fluid_in="brine", fluid_out="gas", print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(-3000, 1500, 2.2, phi=0.2, fluid_in="brine", fluid_out="gas", print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=0.2, fluid_in="brine", fluid_out="gas", k_mineral=0, print_results=False)
    with pytest.raises(ValueError):
        gassmann_substitution(3000, 1500, 2.2, phi=0.2, fluid_in="magma", fluid_out="gas", print_results=False)


def test_nonphysical_k_dry_warns_but_returns():
    # Very low Vp at high porosity drives K_dry below zero -> warn, still returns.
    with pytest.warns(UserWarning):
        res = gassmann_substitution(
            vp=1600.0, vs=200.0, rho=2.0, phi=0.35,
            fluid_in="water", fluid_out="gas", print_results=False,
        )
    assert "vp" in res
