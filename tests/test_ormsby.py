import numpy as np
from tools.ricker_tools import create_ormsby_wavelet


def test_ormsby_shape_and_peak():
    t, w = create_ormsby_wavelet(5, 10, 40, 50, time_length=256, dt=0.001)
    assert len(t) == len(w)
    assert len(w) > 50
    assert np.isfinite(w).all()
    # Ormsby is normalized to a unit peak in wedge_tools.ormsby
    assert abs(np.max(w) - 1.0) < 1e-6
