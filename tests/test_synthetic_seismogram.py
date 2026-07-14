"""Tests for tools/synthetic_tools.py — N-layer 1-D convolutional synthetic."""
import numpy as np
import pytest

from tools.synthetic_tools import validate_synthetic_inputs, create_synthetic_seismogram

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


class TestCreateSyntheticSeismogram:
    def test_return_shapes_and_parameter_keys(self):
        t, trace, p = create_synthetic_seismogram(TH2, VP3, RHO3)
        assert t.shape == trace.shape == (p["nt"],)
        for key in ("n_layers", "vp", "vs", "rho", "thickness", "labels",
                    "interface_times", "rcs", "rc_series", "t0", "nt", "dt",
                    "pad_time", "angle", "method", "wavelet_freq", "wavelet_label"):
            assert key in p
        assert p["n_layers"] == 3
        assert p["labels"] == ["layer 1", "layer 2", "layer 3"]

    def test_interface_twt_placement(self):
        t, trace, p = create_synthetic_seismogram(TH2, VP3, RHO3, dt=0.1, pad_time=50.0)
        t1 = 50.0 + 2000.0 * 50.0 / 3000.0          # 83.3333 ms
        t2 = t1 + 2000.0 * 50.0 / 2500.0            # 123.3333 ms
        assert np.allclose(p["interface_times"], [t1, t2])
        rc_series = np.asarray(p["rc_series"])
        idx = np.flatnonzero(rc_series)
        assert list(idx) == [round(t1 / 0.1), round(t2 / 0.1)]

    def test_acoustic_rc_values(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3)
        z = [v * r for v, r in zip(VP3, RHO3)]
        rc1 = (z[1] - z[0]) / (z[1] + z[0])
        rc2 = (z[2] - z[1]) / (z[2] + z[1])
        assert np.allclose(p["rcs"], [rc1, rc2])

    def test_event_sign_matches_rc(self):
        t, trace, p = create_synthetic_seismogram(TH2, VP3, RHO3, dt=0.1)
        i0 = round(p["interface_times"][0] / 0.1)
        win = trace[i0 - 100:i0 + 100]
        peak = win[np.argmax(np.abs(win))]
        assert np.sign(peak) == np.sign(p["rcs"][0])  # negative contrast here

    def test_thin_layers_superpose_on_one_sample(self):
        # 1 mm middle layer: both interfaces round to the same time sample,
        # so the reflection coefficients must ADD (not overwrite).
        _, _, p = create_synthetic_seismogram([50.0, 0.001], VP3, RHO3, dt=0.1)
        rc_series = np.asarray(p["rc_series"])
        idx = np.flatnonzero(rc_series)
        assert len(idx) == 1
        assert np.isclose(rc_series[idx[0]], p["rcs"][0] + p["rcs"][1])

    def test_amplitude_proportional_to_rc(self):
        # A lone spike convolved with the wavelet: signed peak / rc is the
        # wavelet peak — identical across models.
        _, tr_a, pa = create_synthetic_seismogram([50.0], [3000.0, 2500.0], [2.4, 2.2])
        _, tr_b, pb = create_synthetic_seismogram([50.0], [3000.0, 2000.0], [2.4, 2.0])
        peak_a = tr_a[np.argmax(np.abs(tr_a))]
        peak_b = tr_b[np.argmax(np.abs(tr_b))]
        assert np.isclose(peak_a / pa["rcs"][0], peak_b / pb["rcs"][0], rtol=1e-9)

    def test_ormsby_dominant_frequency_rule(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, wv_type="ormsby",
                                              ormsby_freq="5,10,40,50")
        assert p["wavelet_freq"] == 25.0             # (f2+f3)/2

    def test_labels_override_and_length_check(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3,
                                              labels=["shale", "sand", "shale"])
        assert p["labels"] == ["shale", "sand", "shale"]
        with pytest.raises(ValueError, match="labels must have 3"):
            create_synthetic_seismogram(TH2, VP3, RHO3, labels=["a", "b"])

    def test_unusual_velocity_warns(self):
        with pytest.warns(UserWarning):
            create_synthetic_seismogram(TH2, [100.0, 2500.0, 3200.0], RHO3,
                                        vs=[50.0, 1250.0, 1600.0])

    def test_aliasing_warns(self):
        # dt=1.0 ms -> Nyquist 500 Hz; 3 * 200 Hz Ricker content exceeds it.
        with pytest.warns(UserWarning):
            create_synthetic_seismogram(TH2, VP3, RHO3, dt=1.0, wavelet_freq=200.0)

    def test_parameters_json_friendly(self):
        import json
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3)
        json.dumps(p)  # must not raise
