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
    # Reuss is the lower bound: K_fl(reuss) <= K_fl(brie) for 0 < Sw < 1.
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
