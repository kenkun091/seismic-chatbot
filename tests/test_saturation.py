import numpy as np
import pytest

from tools.rock_physics_tools import _effective_fluid

# Batzle-Wang typical end-members (GPa for readability; the core is unit-agnostic in K).
K_W, RHO_W = 2.2, 1.0
K_G, RHO_G = 0.05, 0.2


def test_effective_fluid_endpoints():
    # Sw=1 -> pure brine; Sw=0 -> pure hydrocarbon, for both laws.
    for law in ("reuss", "brie"):
        k1, r1 = _effective_fluid(1.0, K_W, RHO_W, K_G, RHO_G, law=law)
        k0, r0 = _effective_fluid(0.0, K_W, RHO_W, K_G, RHO_G, law=law)
        assert np.isclose(k1, K_W) and np.isclose(r1, RHO_W)
        assert np.isclose(k0, K_G) and np.isclose(r0, RHO_G)


def test_effective_density_is_linear():
    _, rho = _effective_fluid(0.25, K_W, RHO_W, K_G, RHO_G, law="reuss")
    assert np.isclose(rho, 0.25 * RHO_W + 0.75 * RHO_G)


def test_reuss_below_brie_in_between():
    # At Sw=0.5 (mid saturation) Reuss < Brie < K_w. NOTE: Reuss is the harmonic
    # (Wood) lower bound, but Brie is an empirical patchy model, NOT a global upper
    # bound on Reuss — for a strong brine/gas contrast Brie dips slightly below
    # Reuss at low Sw (~0-0.17). The ordering asserted here holds at moderate Sw.
    kr, _ = _effective_fluid(0.5, K_W, RHO_W, K_G, RHO_G, law="reuss")
    kb, _ = _effective_fluid(0.5, K_W, RHO_W, K_G, RHO_G, law="brie")
    assert kr < kb < K_W


def test_effective_fluid_vectorized():
    sw = np.linspace(0.0, 1.0, 11)
    k, rho = _effective_fluid(sw, K_W, RHO_W, K_G, RHO_G, law="reuss")
    assert k.shape == sw.shape == rho.shape
    assert np.isclose(k[0], K_G) and np.isclose(k[-1], K_W)


def test_effective_fluid_guards():
    with pytest.raises(ValueError):
        _effective_fluid(1.5, K_W, RHO_W, K_G, RHO_G)          # sw out of [0,1]
    with pytest.raises(ValueError):
        _effective_fluid(0.5, K_W, RHO_W, K_G, RHO_G, law="x")  # bad law
    with pytest.raises(ValueError):
        _effective_fluid(0.5, -1.0, RHO_W, K_G, RHO_G)         # non-positive modulus


from tools.rock_physics_tools import rock_properties_saturation, calculate_rock_properties


def test_saturation_endpoint_sw1_matches_water():
    phit, vclay = 0.25, 0.20
    sat = rock_properties_saturation(phit, vclay, sw=1.0, hydrocarbon="gas")
    water = calculate_rock_properties(phit, vclay, "water", print_results=False)
    assert np.allclose(sat, water)


def test_saturation_endpoint_sw0_matches_gas():
    phit, vclay = 0.25, 0.20
    sat = rock_properties_saturation(phit, vclay, sw=0.0, hydrocarbon="gas")
    gas = calculate_rock_properties(phit, vclay, "gas", print_results=False)
    assert np.allclose(sat, gas)


def test_saturation_endpoint_sw0_matches_oil():
    phit, vclay = 0.25, 0.20
    sat = rock_properties_saturation(phit, vclay, sw=0.0, hydrocarbon="oil")
    oil = calculate_rock_properties(phit, vclay, "oil", print_results=False)
    assert np.allclose(sat, oil)


def test_saturation_reuss_vp_below_brie():
    # Reuss is the lower bound: at equal Sw the density mix is identical, so the
    # smaller Reuss K_fl gives a strictly lower Vp than Brie. (The full Vp-Sw curve
    # is NOT monotone vs the gas end-member because density rises with Sw, so only
    # this equal-Sw bound is asserted.)
    phit, vclay = 0.25, 0.20
    vp_r = rock_properties_saturation(phit, vclay, sw=0.5, law="reuss")[0]
    vp_b = rock_properties_saturation(phit, vclay, sw=0.5, law="brie")[0]
    assert vp_r < vp_b


def test_saturation_shape_preserving():
    phit = np.array([0.20, 0.25, 0.30])
    vclay = np.array([0.10, 0.20, 0.30])
    sw = np.array([0.2, 0.5, 0.8])
    vp, vs, rhob, vp_vs, ai, si = rock_properties_saturation(phit, vclay, sw)
    for arr in (vp, vs, rhob, vp_vs, ai, si):
        assert np.asarray(arr).shape == phit.shape


def test_saturation_guards():
    with pytest.raises(ValueError):
        rock_properties_saturation(0.25, 0.20, sw=1.2)               # sw out of range
    with pytest.raises(ValueError):
        rock_properties_saturation(0.25, 0.20, sw=0.5, hydrocarbon="water")  # not a HC
    with pytest.raises(ValueError):
        rock_properties_saturation(0.25, 0.20, sw=0.5, law="bogus")  # bad law


def test_effective_fluid_rejects_each_nonpositive():
    bad = [(-1.0, 1.0, 0.05, 0.2), (2.2, -1.0, 0.05, 0.2),
           (2.2, 1.0, -0.1, 0.2), (2.2, 1.0, 0.05, -0.2)]
    for k_w, rho_w, k_hc, rho_hc in bad:
        with pytest.raises(ValueError):
            _effective_fluid(0.5, k_w, rho_w, k_hc, rho_hc)
