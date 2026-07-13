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
