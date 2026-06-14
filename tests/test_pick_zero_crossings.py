"""The zero-crossing auto-pick branch was dead (string mismatch) and broken
(pick_zero_crossings returned None; amp_picks rebound to a scalar). These tests
pin the repaired behavior."""
import numpy as np

from tools.wedge_tools import pick_zero_crossings, pick_interface_and_amp, create_wedge_model


def _sine_gather(nt, ntraces, zero_sample=50, period=40):
    i = np.arange(nt)
    col = np.sin(2 * np.pi * (i - zero_sample) / float(period))  # zeros every period/2
    return np.stack([col for _ in range(ntraces)], axis=1)


def test_pick_zero_crossings_returns_array_near_crossing():
    nt, ntraces, dt, t0 = 200, 6, 1.0, 0.0
    data = _sine_gather(nt, ntraces)
    ref = np.full(ntraces, 50.0)
    top = np.full(ntraces, 20.0)
    base = np.full(ntraces, 80.0)
    picks = pick_zero_crossings(data, ref, top, base, t0, dt)
    assert picks is not None                      # function used to return None
    assert picks.shape == (ntraces,)
    assert np.all(picks >= top) and np.all(picks <= base)
    assert np.all(np.abs(picks - 50.0) <= 2.0)    # near the true crossing


def test_zero_crossing_branch_produces_valid_arrays():
    nt, ntraces, dt, t0 = 200, 8, 1.0, 0.0
    data = _sine_gather(nt, ntraces)               # ~0 amplitude at interface1 -> zero-crossing mode
    interface1_t = np.full(ntraces, 50.0)
    interface2_t = np.linspace(70.0, 120.0, ntraces)
    hor1, hor2, hor3, amp = pick_interface_and_amp(data, interface1_t, interface2_t, t0, nt, dt)
    assert hor3 is not None                         # zero-crossing branch was reached
    for arr in (hor1, hor2, hor3, amp):
        a = np.asarray(arr)
        assert a.shape == (ntraces,)               # amp_picks used to be a scalar
        assert np.all(np.isfinite(a))


def test_full_wedge_pipeline_runs_in_zero_crossing_mode():
    # imp1 == imp2 (2500*2.4 == 3000*2.0 == 6000) -> rc1 ~ 0 -> zero-crossing
    # pick mode -> exercises the (newly enabled) branch through make_plot.
    _, _, synth, _ = create_wedge_model(
        max_thickness=50, v1=2500, v2=3000, v3=3500,
        rho1=2.4, rho2=2.0, rho3=2.4,
    )
    synth = np.asarray(synth)
    assert synth.ndim == 2 and synth.shape[1] == 61
