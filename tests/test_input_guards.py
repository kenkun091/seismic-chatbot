import warnings

import numpy as np
import pytest

from tools.avo_tools import zoeppritz_reflectivity, shuey_reflectivity


def test_zoeppritz_rejects_vs_ge_vp():
    with pytest.raises(ValueError):
        zoeppritz_reflectivity(vp1=2500, vs1=2600, rho1=2.2,
                               vp2=3000, vs2=1500, rho2=2.4, angles=[0, 10])


def test_shuey_rejects_nonpositive_density():
    with pytest.raises(ValueError):
        shuey_reflectivity(vp1=2500, vs1=1200, rho1=0,
                           vp2=3000, vs2=1500, rho2=2.4, angles=[0, 10])


def test_avo_rejects_angle_ge_90():
    with pytest.raises(ValueError):
        zoeppritz_reflectivity(vp1=2500, vs1=1200, rho1=2.2,
                               vp2=3000, vs2=1500, rho2=2.4, angles=[95])


def test_avo_valid_still_works():
    rc = shuey_reflectivity(vp1=2500, vs1=1200, rho1=2.2,
                            vp2=3000, vs2=1500, rho2=2.4, angles=[0, 10, 20])
    assert np.all(np.isfinite(rc))


from tools.wedge_tools import create_wedge_model

_WEDGE = dict(max_thickness=50, v1=2500, v2=3000, v3=3500, rho1=2.2, rho2=2.3, rho3=2.4)


def test_wedge_rejects_negative_density():
    args = dict(_WEDGE)
    args["rho2"] = -1
    with pytest.raises(ValueError):
        create_wedge_model(**args)


def test_wedge_rejects_vs_ge_vp_when_supplied():
    with pytest.raises(ValueError):
        create_wedge_model(vs1=3000, **_WEDGE)  # vs1=3000 >= vp1=2500


def test_wedge_accepts_velocity_inversion():
    # gas sand: v2 < v1 and rho2 < rho1 are physical and must NOT be rejected
    _, _, synth, _ = create_wedge_model(
        max_thickness=50, v1=3000, v2=2300, v3=3200, rho1=2.4, rho2=2.0, rho3=2.4
    )
    assert np.asarray(synth).ndim == 2


def test_wedge_warns_on_aliasing():
    with pytest.warns(UserWarning):
        create_wedge_model(wavelet_freq=200, dt=4.0, **_WEDGE)  # nyquist=125 Hz, content=600 Hz


from tools.ricker_tools import create_ricker_wavelet, create_ormsby_wavelet


def test_ricker_rejects_nonpositive_frequency():
    with pytest.raises(ValueError):
        create_ricker_wavelet(frequency=0)


def test_ricker_warns_near_nyquist():
    with pytest.warns(UserWarning):
        create_ricker_wavelet(frequency=300, dt=0.002)  # nyquist=250 Hz, content=900 Hz


def test_ormsby_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        create_ormsby_wavelet(5, 10, 40, 50, dt=0)
