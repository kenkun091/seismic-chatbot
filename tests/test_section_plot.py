"""Depth conversion, the model adapter, and plot_seismic_section."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tools.section_tools import (create_synthetic_section, synthetic_section_from_model,
                                 plot_seismic_section, _build_section_figure, MAX_WIGGLE_TRACES)


def _single_interface(nx=3, nz=200, dz=0.5, z_int=60.0):
    vp = np.full((nz, nx), 2500.0); vp[int(z_int / dz):] = 3500.0
    vs = vp / 2; rho = np.full((nz, nx), 2.3); rho[int(z_int / dz):] = 2.6
    return vp, vs, rho


def _model(nx=3, nz=200, dz=0.5):
    vp, vs, rho = _single_interface(nx, nz, dz)
    facies = np.zeros((nz, nx), int); facies[int(60.0 / dz):] = 1
    return {"vp": vp, "vs": vs, "rho": rho, "dz": dz, "dx": 2.0, "facies": facies,
            "legend": {0: {"lithology": "shale", "label": "background"},
                       1: {"lithology": "limestone", "label": "lime"}},
            "z": (np.arange(nz) + 0.5) * dz, "x": np.arange(nx) * 2.0}


def test_depth_domain_peak_sits_at_interface_depth():
    vp, vs, rho = _single_interface()
    z, sec, par = create_synthetic_section(vp, vs, rho, 0.5, 2.0, domain="depth")
    assert par["domain"] == "depth" and sec.shape == vp.shape and z.shape == (vp.shape[0],)
    peak_z = z[np.argmax(np.abs(sec[:, 0]))]
    assert abs(peak_z - 60.0) <= 0.5 + 1e-9          # within one cell (zero-phase wavelet)


def test_time_domain_peak_sits_at_interface_time():
    vp, vs, rho = _single_interface()
    t, sec, par = create_synthetic_section(vp, vs, rho, 0.5, 2.0, pad_time=50.0)
    expected = 50.0 + 2000.0 * 60.0 / 2500.0
    assert abs(t[np.argmax(np.abs(sec[:, 0]))] - expected) <= par["dt"]


def test_model_adapter_matches_direct_call():
    m = _model()
    a1, s1, p1 = synthetic_section_from_model(m, wavelet_freq=25.0)
    a2, s2, p2 = create_synthetic_section(m["vp"], m["vs"], m["rho"], m["dz"], m["dx"], wavelet_freq=25.0)
    np.testing.assert_allclose(s1, s2); np.testing.assert_allclose(a1, a2)
    # the adapter additionally stamps parameters["display"] (default "image") for the
    # auto-plot chain; everything else must match create_synthetic_section exactly.
    assert p1.pop("display") == "image"
    assert p1 == p2


def test_model_adapter_requires_model():
    with pytest.raises(ValueError, match="earth model first"):
        synthetic_section_from_model(None)


@pytest.mark.parametrize("display", ["image", "wiggle", "both"])
def test_plot_modes_write_png(display):
    m = _model()
    axis, sec, par = synthetic_section_from_model(m)
    png = plot_seismic_section(sec, par, axis=axis, model=m, display=display)
    try:
        assert png.endswith(".png") and os.path.getsize(png) > 0
    finally:
        os.remove(png)


def test_plot_without_model_and_without_axis():
    m = _model()
    axis, sec, par = synthetic_section_from_model(m, domain="depth")
    png = plot_seismic_section(sec, par)      # axis reconstructed from parameters
    try:
        assert os.path.getsize(png) > 0
    finally:
        os.remove(png)


def test_plot_bad_display_rejected():
    m = _model()
    axis, sec, par = synthetic_section_from_model(m)
    with pytest.raises(ValueError, match="display"):
        plot_seismic_section(sec, par, display="hologram")


def test_wiggle_decimation_step():
    from tools.section_tools import _wiggle_step
    assert _wiggle_step(50) == 1 and _wiggle_step(MAX_WIGGLE_TRACES) == 1
    assert _wiggle_step(MAX_WIGGLE_TRACES + 1) == 2 and _wiggle_step(401) == 6


def _padded_model_with_image(nx=3, dz=1.0, npad=300, height=20.0, dx=25.0):
    """A model shaped like an outcrop_to_model result: a thin 'photographed'
    image section (height_m tall) sandwiched between large background pads,
    so the model panel/section is dominated by padding — exactly the case
    #3's cropping is for."""
    nz_img = int(round(height / dz))
    nz = nz_img + 2 * npad
    vp = np.full((nz, nx), 2000.0); vp[npad:npad + nz_img] = 3000.0
    vs = vp / 2.0
    rho = np.full((nz, nx), 2.2); rho[npad:npad + nz_img] = 2.4
    z = (np.arange(nz) + 0.5) * dz
    x = np.arange(nx) * dx
    return {"vp": vp, "vs": vs, "rho": rho, "z": z, "x": x, "dz": dz, "dx": dx,
            "image_top_m": float(npad * dz), "height_m": float(height)}


@pytest.mark.parametrize("domain", ["depth", "time"])
def test_section_axis_crops_to_outcrop_extent(domain):
    model = _padded_model_with_image()
    axis, sec, par = synthetic_section_from_model(model, wavelet_freq=30.0, domain=domain)
    fig, axes_by_kind = _build_section_figure(sec, par, axis=axis, model=model, display="image")
    try:
        ylim = axes_by_kind["image"].get_ylim()
        cropped_span = abs(ylim[0] - ylim[1])
        full_span = abs(axis[-1] - axis[0])
        assert cropped_span < 0.4 * full_span
    finally:
        plt.close(fig)


def test_model_lacking_image_extent_keeps_full_axis():
    """A plain grid (no image_top_m/height_m) must not be cropped."""
    m = _model()
    axis, sec, par = synthetic_section_from_model(m, domain="depth")
    fig, axes_by_kind = _build_section_figure(sec, par, axis=axis, model=m, display="image")
    try:
        ylim = axes_by_kind["image"].get_ylim()
        assert sorted(ylim) == pytest.approx(sorted((axis[0], axis[-1])))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("domain", ["depth", "time"])
@pytest.mark.parametrize("display", ["image", "wiggle", "both"])
def test_axes_orient_downward_and_model_x_extent(domain, display):
    """Every section/model axis must read depth/time increasing downward
    (ylim[0] > ylim[1], matplotlib's (bottom, top) convention), and the
    model panel's x-extent must span the model's own x array — for every
    display mode and domain, with the outcrop-extent crop from #3 active."""
    model = _padded_model_with_image()
    axis, sec, par = synthetic_section_from_model(model, wavelet_freq=30.0, domain=domain)
    fig, axes_by_kind = _build_section_figure(sec, par, axis=axis, model=model, display=display)
    try:
        for kind, ax in axes_by_kind.items():
            ylim = ax.get_ylim()
            assert ylim[0] > ylim[1], f"{kind} panel ylim not downward: {ylim}"
        model_ax = axes_by_kind["model"]
        xlim = model_ax.get_xlim()
        x = model["x"]
        assert xlim == pytest.approx((x[0], x[-1]))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Photo overlay modes ("overlay" = wiggles on the outcrop photo,
# "overlay_image" = translucent color section on the photo)
# ---------------------------------------------------------------------------

def _overlay_model(image_path, nx=5, dz=1.0, npad=100, height=20.0, dx=10.0):
    """A padded outcrop-style model that also carries the photo keys the
    overlay needs (image_path, width_m), like a real outcrop_to_model result."""
    m = _padded_model_with_image(nx=nx, dz=dz, npad=npad, height=height, dx=dx)
    m["image_path"] = image_path
    m["width_m"] = float((nx - 1) * dx)
    return m


@pytest.mark.parametrize("display", ["overlay", "overlay_image"])
@pytest.mark.parametrize("domain", ["time", "depth"])
def test_overlay_modes_write_png_from_both_domains(outcrop_image, display, domain):
    m = _overlay_model(outcrop_image)
    axis, sec, par = synthetic_section_from_model(m, domain=domain, display=display)
    png = plot_seismic_section(sec, par, axis=axis, model=m, display=display)
    try:
        assert png.endswith(".png") and os.path.getsize(png) > 0
    finally:
        os.remove(png)


def test_overlay_axes_registered_to_photo_extent(outcrop_image):
    m = _overlay_model(outcrop_image)
    axis, sec, par = synthetic_section_from_model(m, display="overlay")
    fig, axes_by_kind = _build_section_figure(sec, par, axis=axis, model=m, display="overlay")
    try:
        ax = axes_by_kind["overlay"]
        top, h, w = m["image_top_m"], m["height_m"], m["width_m"]
        assert ax.get_ylim() == (top + h, top)        # depth increases downward
        assert ax.get_xlim() == (0.0, w)              # full photo width
        assert "model" not in axes_by_kind            # photo IS the background panel
    finally:
        plt.close(fig)


def test_overlay_requires_photo_model():
    m = _model()                                       # no image_path / extent keys
    axis, sec, par = synthetic_section_from_model(m, display="image")
    with pytest.raises(ValueError, match="photo"):
        plot_seismic_section(sec, par, axis=axis, model=m, display="overlay")
    with pytest.raises(ValueError, match="photo"):
        plot_seismic_section(sec, par, axis=axis, model=None, display="overlay_image")


def test_overlay_display_flows_through_adapter_and_tool_manager(outcrop_image):
    m = _overlay_model(outcrop_image)
    _, _, par = synthetic_section_from_model(m, display="overlay_image")
    assert par["display"] == "overlay_image"
    from core.tool_manager import ToolManager
    _, _, par2 = ToolManager().process_tool_call(
        "synthetic_section", {"model": m, "display": "overlay"})
    assert par2["display"] == "overlay"
