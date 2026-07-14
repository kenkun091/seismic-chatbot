"""Tests for tools/synthetic_tools.py — N-layer 1-D convolutional synthetic."""
import os
import numpy as np
import pytest

from tools.synthetic_tools import (
    validate_synthetic_inputs,
    create_synthetic_seismogram,
    plot_synthetic_seismogram,
)
from tools.avo_tools import shuey_reflectivity, zoeppritz_reflectivity
from tools.wedge_tools import create_wedge_model

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


class TestAnglePath:
    VS3 = [1500.0, 1100.0, 1600.0]

    def test_rc_matches_shuey_at_angle(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3, angle=20.0)
        expected = shuey_reflectivity(
            vp1=VP3[0], vs1=self.VS3[0], rho1=RHO3[0],
            vp2=VP3[1], vs2=self.VS3[1], rho2=RHO3[1], angles=[20.0])
        assert np.isclose(p["rcs"][0], float(np.asarray(expected).ravel()[0]))

    def test_rc_matches_zoeppritz_when_requested(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3,
                                              angle=20.0, method="zoeppritz")
        expected = zoeppritz_reflectivity(
            vp1=VP3[0], vs1=self.VS3[0], rho1=RHO3[0],
            vp2=VP3[1], vs2=self.VS3[1], rho2=RHO3[1], angles=[20.0])
        assert np.isclose(p["rcs"][0], float(np.asarray(expected).ravel()[0]))

    def test_shuey_and_zoeppritz_differ_at_high_angle(self):
        # Sanity: the exact solution and the linearization diverge at 40 deg,
        # proving the method switch actually switches implementations.
        _, _, ps = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3, angle=40.0)
        _, _, pz = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3,
                                               angle=40.0, method="zoeppritz")
        assert not np.isclose(ps["rcs"][0], pz["rcs"][0], rtol=1e-6)

    def test_vs_default_used_in_angle_path(self):
        # vs omitted -> vp/2; result must equal explicitly passing vp/2.
        _, _, p_default = create_synthetic_seismogram(TH2, VP3, RHO3, angle=15.0)
        _, _, p_explicit = create_synthetic_seismogram(
            TH2, VP3, RHO3, vs=[v / 2.0 for v in VP3], angle=15.0)
        assert np.allclose(p_default["rcs"], p_explicit["rcs"])

    def test_angle_zero_is_acoustic_not_shuey(self):
        _, _, p = create_synthetic_seismogram(TH2, VP3, RHO3, vs=self.VS3, angle=0.0)
        z = [v * r for v, r in zip(VP3, RHO3)]
        assert np.isclose(p["rcs"][0], (z[1] - z[0]) / (z[1] + z[0]))


class TestOracleAgainstWedge:
    def test_event_separation_and_amplitudes_match_wedge(self):
        """3-layer stack vs the matching wedge trace.

        The two tools use different time references (wedge anchors interface 1
        at 300 ms; the synthetic uses a pad_time axis), so compare the event
        SEPARATION and event AMPLITUDES, not absolute times. The wedge places
        its second interface one sample late (known idx2+1 quirk) -> allow a
        2-sample separation tolerance.
        """
        vp, rho = [3000.0, 2500.0, 3200.0], [2.4, 2.2, 2.5]
        h, dt = 50.0, 0.1

        _, syn_trace, sp = create_synthetic_seismogram(
            [60.0, h], vp, rho, dt=dt, wavelet_freq=30.0, pad_time=60.0)

        _, _, wedge_synth, wp = create_wedge_model(
            max_thickness=100.0, v1=vp[0], v2=vp[1], v3=vp[2],
            rho1=rho[0], rho2=rho[1], rho3=rho[2],
            num_traces=101, dt=dt, wavelet_freq=30.0)
        wtrace = wedge_synth[:, 50]  # linspace(0,100,101)[50] == 50 m == h
        wtime = wp["t0"] + np.arange(wedge_synth.shape[0]) * dt

        syn_time = np.arange(sp["nt"]) * dt

        def event(trace, time, t_expect, half_win=15.0):
            m = (time >= t_expect - half_win) & (time <= t_expect + half_win)
            seg, tseg = trace[m], time[m]
            k = int(np.argmax(np.abs(seg)))
            return tseg[k], seg[k]

        t1s, a1s = event(syn_trace, syn_time, sp["interface_times"][0])
        t2s, a2s = event(syn_trace, syn_time, sp["interface_times"][1])
        t1w, a1w = event(wtrace, wtime, 300.0)
        t2w, a2w = event(wtrace, wtime, 300.0 + 2000.0 * h / vp[1])

        assert abs((t2s - t1s) - (t2w - t1w)) <= 2 * dt + 1e-9
        assert np.isclose(a1s, a1w, rtol=0.05)
        assert np.isclose(a2s, a2w, rtol=0.05)


class TestPlotSyntheticSeismogram:
    def _make(self):
        return create_synthetic_seismogram(TH2, VP3, RHO3,
                                           labels=["shale", "sand", "shale"])

    def test_creates_png_at_given_path(self, tmp_path):
        _, trace, p = self._make()
        out = plot_synthetic_seismogram(trace, p, output_path=str(tmp_path / "syn.png"))
        assert out == str(tmp_path / "syn.png")
        assert os.path.getsize(out) > 0

    def test_default_tempfile_path(self):
        _, trace, p = self._make()
        out = plot_synthetic_seismogram(trace, p)
        try:
            assert out.endswith(".png") and os.path.getsize(out) > 0
        finally:
            os.remove(out)

    def test_accepts_list_trace(self, tmp_path):
        _, trace, p = self._make()
        out = plot_synthetic_seismogram(list(trace), p,
                                        output_path=str(tmp_path / "syn2.png"))
        assert os.path.getsize(out) > 0
