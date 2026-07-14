"""Tests for tools/synthetic_tools.py — N-layer 1-D convolutional synthetic."""
import numpy as np
import pytest

from tools.synthetic_tools import validate_synthetic_inputs

VP3 = [3000.0, 2500.0, 3200.0]
RHO3 = [2.4, 2.2, 2.5]
TH2 = [50.0, 50.0]


class TestValidateSyntheticInputs:
    def test_valid_inputs_return_vs_default(self):
        vs_eff = validate_synthetic_inputs(TH2, VP3, RHO3)
        assert vs_eff == [1500.0, 1250.0, 1600.0]

    def test_explicit_vs_is_returned(self):
        vs = [1600.0, 1300.0, 1700.0]
        assert validate_synthetic_inputs(TH2, VP3, RHO3, vs=vs) == vs

    def test_fewer_than_two_layers_rejected(self):
        with pytest.raises(ValueError, match=r"at least 2 layers"):
            validate_synthetic_inputs([], [3000.0], [2.4])

    def test_thickness_length_rule_names_the_contract(self):
        with pytest.raises(ValueError, match=r"len\(vp\)-1 = 2 .*basal half-space.*got 3"):
            validate_synthetic_inputs([10.0, 10.0, 10.0], VP3, RHO3)

    def test_rho_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match=r"rho must have 3"):
            validate_synthetic_inputs(TH2, VP3, [2.4, 2.2])

    def test_vs_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match=r"vs must have 3"):
            validate_synthetic_inputs(TH2, VP3, RHO3, vs=[1500.0, 1250.0])

    def test_non_positive_thickness_rejected(self):
        with pytest.raises(ValueError, match=r"thickness\[1\]"):
            validate_synthetic_inputs([50.0, -5.0], VP3, RHO3)

    def test_non_positive_dt_rejected(self):
        with pytest.raises(ValueError, match="dt"):
            validate_synthetic_inputs(TH2, VP3, RHO3, dt=0.0)

    def test_angle_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 90\)"):
            validate_synthetic_inputs(TH2, VP3, RHO3, angle=90.0)
        with pytest.raises(ValueError, match=r"\[0, 90\)"):
            validate_synthetic_inputs(TH2, VP3, RHO3, angle=-5.0)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="method"):
            validate_synthetic_inputs(TH2, VP3, RHO3, method="aki")

    def test_unknown_wv_type_rejected(self):
        with pytest.raises(ValueError, match="wv_type"):
            validate_synthetic_inputs(TH2, VP3, RHO3, wv_type="klauder")

    def test_ormsby_requires_corners(self):
        with pytest.raises(ValueError, match="ormsby_freq is required"):
            validate_synthetic_inputs(TH2, VP3, RHO3, wv_type="ormsby")

    def test_ormsby_corners_must_increase(self):
        with pytest.raises(ValueError, match="four increasing corners"):
            validate_synthetic_inputs(TH2, VP3, RHO3, wv_type="ormsby",
                                      ormsby_freq="5,40,10,50")

    def test_non_elastic_layer_rejected(self):
        # vs >= vp is non-physical (require_elastic_medium)
        with pytest.raises(ValueError):
            validate_synthetic_inputs(TH2, VP3, RHO3, vs=[3000.0, 1250.0, 1600.0])
