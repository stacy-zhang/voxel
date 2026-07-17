"""Colormap and transfer-function helpers for the VTK render path.

STAGE 3 (RENDER) turns a dense volume into pixels. These pure helpers build the
VTK color/opacity transfer functions and lookup tables from vtkplotlib
colormaps, and provide the log-compression + robust-percentile utilities that
frame the transfer functions over a meaningful data range (matching napari).
They hold no scene state, so new visualization capabilities can reuse them
without touching the render orchestration in ``voxel/app/server.py``.
"""

from typing import Optional, Tuple

import matplotlib
import matplotlib.cm as _mpl_cm

# vtkplotlib 2.1.1 still calls the long-deprecated ``matplotlib.cm.get_cmap``,
# which was removed in matplotlib 3.9+. When the pixi environment was rebuilt it
# pulled a newer matplotlib, so ``vpl.colors.as_vtk_cmap`` started raising and
# the colormap lookup silently fell back to grayscale. Restore a compatible
# ``get_cmap`` shim so the original vtkplotlib colormaps keep working.
if not hasattr(_mpl_cm, "get_cmap"):

    def _get_cmap_compat(name=None, lut=None):
        cmap = matplotlib.colormaps[name] if name is not None \
            else matplotlib.colormaps[matplotlib.rcParams["image.cmap"]]
        if lut is not None:
            cmap = cmap.resampled(lut)
        return cmap

    _mpl_cm.get_cmap = _get_cmap_compat

import vtkplotlib as vpl  # noqa: E402  (import after the shim above)
import numpy as np  # noqa: E402
from vtkmodules.vtkCommonCore import vtkLookupTable  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction  # noqa: E402
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction  # noqa: E402


def _cmap_rgb(colormap: str, n: int = 256) -> Optional[np.ndarray]:
    """Sample an ``(n, 3)`` RGB array from a vtkplotlib colormap.

    Uses vtkplotlib's ``colors.as_vtk_cmap`` to resolve the named colormap into
    a ``vtkLookupTable`` and reads the RGB entries back out. Centralizing the
    colormap lookup here means both the volume transfer function and the
    slice/probe lookup tables share vtkplotlib's colormapping instead of
    sampling matplotlib colormaps directly. Returns ``None`` when the colormap
    can't be resolved so callers can fall back to grayscale.
    """
    try:
        lut = vpl.colors.as_vtk_cmap(colormap, cache=False)
    except Exception:
        return None
    count = int(lut.GetNumberOfTableValues())
    if count <= 0:
        return None
    idx = np.linspace(0.0, count - 1, n).round().astype(int)
    rgb = np.empty((n, 3), dtype=float)
    for j, i in enumerate(idx):
        r, g, b, _a = lut.GetTableValue(int(i))
        rgb[j] = (r, g, b)
    return rgb


def _apply_color_transfer_function(
    color_tf: vtkColorTransferFunction,
    colormap: str,
    value_range: Optional[Tuple[float, float]],
) -> None:
    color_tf.RemoveAllPoints()
    if value_range is None or value_range[0] == value_range[1]:
        color_tf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(1.0, 1.0, 1.0, 1.0)
        return

    lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:
        hi = lo + 1.0

    colors = _cmap_rgb(colormap, 256)
    if colors is not None:
        # Sample the colormap densely (matches _make_lookup_table and napari,
        # which use the full 256-entry LUT). The previous 8-point sampling made
        # VTK linearly interpolate in RGB between widely spaced control points,
        # which visibly distorts perceptual colormaps like viridis/turbo.
        n = len(colors)
        for idx, rgb in enumerate(colors):
            t = lo + (hi - lo) * idx / (n - 1)
            color_tf.AddRGBPoint(float(t), float(rgb[0]), float(rgb[1]), float(rgb[2]))
    else:
        color_tf.AddRGBPoint(lo, 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(hi, 1.0, 1.0, 1.0)


def _apply_opacity_function(
    opacity_tf: vtkPiecewiseFunction,
    value_range: Optional[Tuple[float, float]],
    opacity_scale: float,
) -> None:
    opacity_tf.RemoveAllPoints()
    if value_range is None or value_range[0] == value_range[1]:
        opacity_tf.AddPoint(0.0, 0.0)
        opacity_tf.AddPoint(1.0, 1.0)
        return

    lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:
        hi = lo + 1.0
    opacity_scale = max(0.0, min(float(opacity_scale), 4.0))
    # Match napari's image-volume rendering. napari maps each voxel through the
    # colormap + contrast limits and does NOT apply any custom opacity ramp, so
    # the value->appearance relationship is linear across the contrast range:
    # intensities at the low limit fade out and intensities at the high limit
    # are shown fully. A single linear opacity ramp from lo->hi reproduces that.
    # The previous steep ramp forced everything below 35% of the range fully
    # transparent and muted the 35-80% mid-range, which hid the diffuse and
    # mid-intensity structure that napari clearly renders.
    opacity_tf.AddPoint(lo, 0.0)
    opacity_tf.AddPoint(hi, min(1.0, opacity_scale))


def _apply_opacity_points(
    opacity_tf: vtkPiecewiseFunction,
    value_range: Optional[Tuple[float, float]],
    points,
    opacity_scale: float,
) -> None:
    """Build the opacity transfer function from ParaView-style control points.

    ``points`` is a list of ``[x, y]`` pairs where ``x`` is the *normalized*
    position across the contrast window ``value_range`` (0 -> lo, 1 -> hi) and
    ``y`` is the opacity (0..1). This lets the right-panel graph editor shape an
    arbitrary piecewise-linear opacity ramp instead of the single linear ramp in
    ``_apply_opacity_function``. Falls back to that linear ramp when no usable
    points are supplied so existing behaviour is preserved.
    """
    cleaned = []
    if points:
        for p in points:
            try:
                x = float(p[0])
                y = float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            cleaned.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    if len(cleaned) < 2:
        _apply_opacity_function(opacity_tf, value_range, opacity_scale)
        return

    opacity_tf.RemoveAllPoints()
    if value_range is None or value_range[0] == value_range[1]:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(value_range[0]), float(value_range[1])
        if hi <= lo:
            hi = lo + 1.0
    scale = max(0.0, min(float(opacity_scale), 4.0))
    cleaned.sort(key=lambda q: q[0])
    for x, y in cleaned:
        val = lo + (hi - lo) * x
        opacity_tf.AddPoint(float(val), min(1.0, y * scale))


def cmap_css_gradient(colormap: str, n: int = 24) -> str:
    """Return a CSS ``linear-gradient(...)`` string sampling ``colormap``.

    Used to paint the background of the right-panel transfer-function editor so
    the graph's x-axis reads as the actual color range (as in ParaView). Samples
    the same vtkplotlib colormap the render path uses, so the swatch matches the
    volume. Falls back to a black->white ramp when the colormap can't resolve.
    """
    colors = _cmap_rgb(colormap, n)
    stops = []
    if colors is not None:
        for i, rgb in enumerate(colors):
            pct = i / (n - 1) * 100.0
            r, g, b = (int(round(c * 255)) for c in rgb)
            stops.append(f"rgb({r},{g},{b}) {pct:.1f}%")
    else:
        stops = ["rgb(0,0,0) 0%", "rgb(255,255,255) 100%"]
    return "linear-gradient(to right, " + ", ".join(stops) + ")"


def _log1p_clip(a: np.ndarray) -> np.ndarray:
    """Log-compress the volume for display (mirrors the napari path).

    RSM intensities span many orders of magnitude, so a raw linear mapping
    leaves everything but the brightest Bragg peak fully transparent. napari
    shows the data via log1p(max(a, 0)); we do the same before handing the
    volume to VTK.
    """
    return np.log1p(np.maximum(np.asarray(a, dtype=np.float32), 0.0))


def _robust_percentiles(
    a: np.ndarray, prc: Tuple[float, float] = (1.0, 99.8)
) -> Tuple[float, float]:
    """Robust (lo, hi) contrast limits, ignoring non-finite values.

    Matches napari's contrast-limit computation so the web app frames the
    transfer functions over the meaningful data range instead of the full
    [min, max], which a single outlier voxel would otherwise dominate.
    """
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(a, prc)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(a)), float(np.max(a))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _make_lookup_table(colormap: str, value_range: Tuple[float, float]) -> vtkLookupTable:
    """Build a vtkLookupTable for slice/probe actors from a vtkplotlib colormap."""
    lut = vtkLookupTable()
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:
        hi = lo + 1.0
    n = 256
    lut.SetNumberOfTableValues(n)  # specify total number of colors in the gradient
    lut.SetRange(lo, hi)  # define min/max data values to map
    colors = _cmap_rgb(colormap, n)
    if colors is not None:
        for i in range(n):
            r, g, b = colors[i]
            lut.SetTableValue(i, float(r), float(g), float(b), 1.0)  # A=1.0 (opaque)
    else:
        for i in range(n):
            t = i / (n - 1)
            lut.SetTableValue(i, t, t, t, 1.0)  # grayscale fallback
    lut.Build()
    return lut


def attenuated_mip_image(
    volume: np.ndarray,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    near_pts: np.ndarray,
    far_pts: np.ndarray,
    value_range: Tuple[float, float],
    colormap: str,
    *,
    attenuation: float = 0.05,
    n_samples: int = 256,
) -> np.ndarray:
    """CPU emulation of napari's attenuated maximum-intensity projection.

    This is the faithful substitute for napari's ``attenuated_mip`` rendering
    that VTK cannot express on this host (its smart volume mapper has no
    attenuated-MIP blend mode, and the GPU shader-replacement API is missing
    from the installed build; the render window also falls back to the
    ``llvmpipe`` software rasteriser, so there is no GPU to lean on). Instead of
    asking VTK to render, we cast one ray per output pixel through the volume in
    NumPy and reproduce napari's exact formula:

        along each ray (front -> back), normalise each sample to the contrast
        window [lo, hi], then weight it by ``exp(-attenuation * sum_in_front)``
        and keep the running maximum.

    The exponential falloff dims deeper/weaker samples so only strong peaks
    survive, which is what makes napari's view read as clean, high-contrast
    peaks on a dark background instead of VTK's flat MIP where background haze
    bleeds over everything.

    Parameters
    ----------
    volume : (nx, ny, nz) float32
        The display-scaled (e.g. log-compressed), NaN-free volume, laid out
        exactly as it is shown -- i.e. read straight back from the displayed
        ``vtkImageData`` so ``origin``/``spacing`` place it correctly.
    origin, spacing : length-3
        World placement of ``volume[0, 0, 0]`` and the per-axis voxel size, as
        reported by the displayed ``vtkImageData``.
    near_pts, far_pts : (H, W, 3) float
        World-space ray entry (near plane) and exit (far plane) points for each
        output pixel, with row 0 at the top of the image. The caller builds
        these from the live VTK camera so the projection matches the 3D view.
    value_range : (lo, hi)
        Contrast window (same one driving the VTK transfer functions).
    colormap : str
        Colormap name (resolved through the shared ``_cmap_rgb`` helper so the
        snapshot matches the volume's colors).
    attenuation : float
        napari's attenuation coefficient (default 0.05), applied to values
        normalised into [0, 1].
    n_samples : int
        Number of samples taken along each ray between the near and far planes.

    Returns
    -------
    (H, W, 3) uint8 RGB image.
    """
    vol = np.ascontiguousarray(volume, dtype=np.float32)
    nx, ny, nz = vol.shape
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    spacing = np.where(np.abs(spacing) < 1e-12, 1.0, spacing)

    H, W = near_pts.shape[:2]
    P0 = near_pts.reshape(-1, 3).astype(np.float64)
    P1 = far_pts.reshape(-1, 3).astype(np.float64)
    npx = P0.shape[0]

    lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:
        hi = lo + 1.0
    inv_span = 1.0 / (hi - lo)

    out_val = np.full(npx, -np.inf, dtype=np.float32)  # running attenuated max
    acc = np.zeros(npx, dtype=np.float32)              # normalised intensity in front

    step = P1 - P0
    for s in range(n_samples):
        t = s / (n_samples - 1) if n_samples > 1 else 0.0
        P = P0 + step * t
        # world -> continuous voxel index
        fx = (P[:, 0] - origin[0]) / spacing[0]
        fy = (P[:, 1] - origin[1]) / spacing[1]
        fz = (P[:, 2] - origin[2]) / spacing[2]
        valid = (
            (fx >= 0) & (fx <= nx - 1)
            & (fy >= 0) & (fy <= ny - 1)
            & (fz >= 0) & (fz <= nz - 1)
        )
        nval = np.zeros(npx, dtype=np.float32)
        if valid.any():
            xf = fx[valid]
            yf = fy[valid]
            zf = fz[valid]
            x0 = np.floor(xf).astype(np.int32)
            y0 = np.floor(yf).astype(np.int32)
            z0 = np.floor(zf).astype(np.int32)
            x1 = np.minimum(x0 + 1, nx - 1)
            y1 = np.minimum(y0 + 1, ny - 1)
            z1 = np.minimum(z0 + 1, nz - 1)
            dx = (xf - x0).astype(np.float32)
            dy = (yf - y0).astype(np.float32)
            dz = (zf - z0).astype(np.float32)
            # trilinear interpolation of the 8 surrounding voxels
            c00 = vol[x0, y0, z0] * (1 - dx) + vol[x1, y0, z0] * dx
            c10 = vol[x0, y1, z0] * (1 - dx) + vol[x1, y1, z0] * dx
            c01 = vol[x0, y0, z1] * (1 - dx) + vol[x1, y0, z1] * dx
            c11 = vol[x0, y1, z1] * (1 - dx) + vol[x1, y1, z1] * dx
            c0 = c00 * (1 - dy) + c10 * dy
            c1 = c01 * (1 - dy) + c11 * dy
            interp = c0 * (1 - dz) + c1 * dz
            # normalise into the contrast window, matching napari's clim texture
            nval[valid] = np.clip((interp - lo) * inv_span, 0.0, 1.0)
        # attenuate by the intensity already accumulated in front (excludes the
        # current sample), then keep the running maximum
        weight = np.exp(-attenuation * acc)
        cand = np.where(valid, nval * weight, -np.inf)
        out_val = np.maximum(out_val, cand)
        acc += nval

    norm = np.where(np.isfinite(out_val), out_val, 0.0)
    norm = np.clip(norm, 0.0, 1.0)
    idx = np.clip((norm * 255.0).astype(np.int32), 0, 255)

    colors = _cmap_rgb(colormap, 256)
    if colors is None:
        rgb = np.stack([norm, norm, norm], axis=-1)
    else:
        rgb = colors[idx]
    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(H, W, 3)
    return rgb8
