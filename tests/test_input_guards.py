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
