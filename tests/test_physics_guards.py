import warnings

import pytest

from tools.physics_guards import (
    elastic_medium_error, positive_error, angles_error,
    require_elastic_medium, require_positive, warn_if_aliased, warn_if_outside,
)


def test_elastic_medium_valid_returns_none():
    assert elastic_medium_error(2500, 1200, 2.2, "m") is None


def test_elastic_medium_rejects_vs_ge_vp():
    assert elastic_medium_error(2500, 2600, 2.2, "m") is not None
    assert elastic_medium_error(2500, 2500, 2.2, "m") is not None


def test_elastic_medium_rejects_nonpositive():
    assert elastic_medium_error(0, 1200, 2.2) is not None
    assert elastic_medium_error(2500, 1200, 0) is not None
    assert elastic_medium_error(2500, 0, 2.2) is not None


def test_positive_error():
    assert positive_error(-1, "x") is not None
    assert positive_error(0, "x") is not None
    assert positive_error(5, "x") is None


def test_angles_error_bounds():
    assert angles_error([0, 30, 45]) is None
    assert angles_error([90]) is not None
    assert angles_error([-1]) is not None


def test_require_helpers_raise():
    with pytest.raises(ValueError):
        require_elastic_medium(2500, 2600, 2.2)
    with pytest.raises(ValueError):
        require_positive(0, "dt")


def test_warn_if_aliased_warns_above_nyquist():
    with pytest.warns(UserWarning):
        warn_if_aliased(6000, 1e-4)  # nyquist = 5000 Hz


def test_warn_if_aliased_silent_below_nyquist():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_aliased(90, 1e-4)  # nyquist = 5000 Hz -> silent


def test_warn_if_outside_warns_and_is_silent():
    with pytest.warns(UserWarning):
        warn_if_outside(9000, 300, 8000, "v", "m/s")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_outside(2500, 300, 8000, "v", "m/s")
