import numpy as np
import pytest

from tools.rock_physics_tools import gassmann_substitution, calculate_rock_properties


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
    vp = np.array([3000.0, 3200.0])
    vs = np.array([1500.0, 1600.0])
    rho = np.array([2.2, 2.25])
    phi = np.array([0.2, 0.18])
    res = gassmann_substitution(
        vp=vp, vs=vs, rho=rho, phi=phi,
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    assert res["vp"].shape == (2,)
    assert res["vs"].shape == (2,)
    assert res["rho"].shape == (2,)
    # The array path must match the scalar path element-wise.
    scalar0 = gassmann_substitution(
        vp=3000.0, vs=1500.0, rho=2.2, phi=0.2,
        fluid_in="brine", fluid_out="gas", print_results=False,
    )
    assert np.isclose(res["vp"][0], scalar0["vp"])
    assert np.isclose(res["vs"][0], scalar0["vs"])
    assert np.isclose(res["rho"][0], scalar0["rho"])


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
    # Very low Vp at high porosity drives K_dry below zero. The function warns
    # (about the non-physical dry frame) and still returns a structurally complete
    # dict. Vs stays finite (mu>0, rho_out>0); Vp is NaN because the substituted
    # saturated modulus goes negative under sqrt — pin that as the documented
    # consequence of inconsistent inputs rather than leaving it incidental.
    with pytest.warns(UserWarning, match="non-physical"):
        res = gassmann_substitution(
            vp=1600.0, vs=200.0, rho=2.0, phi=0.35,
            fluid_in="water", fluid_out="gas", print_results=False,
        )
    assert set(res) >= {"vp", "vs", "rho", "vp_vs", "k_dry", "k_sat", "mu"}
    assert np.isfinite(res["vs"])
    assert np.isnan(res["vp"])


def test_registered_in_registry():
    from core.tool_registry import REGISTRY_BY_NAME, TOOL_FUNCTIONS, TOOL_SCHEMAS

    assert "gassmann_substitution" in REGISTRY_BY_NAME
    spec = REGISTRY_BY_NAME["gassmann_substitution"]
    assert spec.fn is gassmann_substitution
    assert spec.auto_plot is None
    assert set(spec.required) == {"vp", "vs", "rho", "phi", "fluid_in", "fluid_out"}
    assert TOOL_FUNCTIONS["gassmann_substitution"] is gassmann_substitution
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "gassmann_substitution" in names
