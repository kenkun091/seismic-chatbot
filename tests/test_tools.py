import numpy as np
from tools.ricker_tools import create_ricker_wavelet
from tools.wedge_tools import create_wedge_model
from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity
from tools.rock_physics_tools import calculate_rock_properties


def test_ricker_zero_mean_and_symmetric():
    t, w = create_ricker_wavelet(frequency=30, time_length=256, dt=0.001)
    assert abs(np.mean(w)) < 1e-2
    # peak at center (allow off-by-one due to nt//2 centering)
    assert abs(np.argmax(w) - len(w) // 2) <= 1
    # symmetric about center
    assert np.allclose(w, w[::-1], atol=1e-6)


def test_wedge_model_shapes():
    t, model, synthetic, params = create_wedge_model(
        max_thickness=50, v1=2500, v2=3000, v3=3500,
        rho1=2.2, rho2=2.3, rho3=2.4, num_traces=31,
    )
    synthetic = np.asarray(synthetic)
    # num_traces is honored; synthetic shape is (nsamples, ntraces).
    assert 31 in synthetic.shape
    assert synthetic.shape[1] == 31
    assert "wavelet_label" in params


def test_zoeppritz_vs_shuey_small_angles():
    args = dict(vp1=2500, vs1=1200, rho1=2.2, vp2=3000, vs2=1500, rho2=2.4)
    angles = [0, 2, 4, 6]
    rz = zoeppritz_reflectivity(angles=angles, **args)
    rs = shuey_reflectivity(angles=angles, **args)
    assert np.allclose(rz, rs, atol=0.02)


def test_zoeppritz_normal_incidence_sign():
    # impedance increases downward -> positive RC at 0 deg
    rc = zoeppritz_reflectivity(vp1=2500, vs1=1200, rho1=2.2,
                                vp2=3000, vs2=1500, rho2=2.4, angles=[0])
    assert rc[0] > 0


def test_rock_properties_ranges():
    vp, vs, rhob, vpvs, ai, si = calculate_rock_properties(
        phit=[0.1, 0.2], vclay=[0.1, 0.3], fluid_type="water", print_results=False
    )
    vp = np.asarray(vp)
    assert (vp > 1000).all() and (vp < 7000).all()
