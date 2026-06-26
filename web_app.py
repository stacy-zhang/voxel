import os
import asyncio
import math
import time
from pathlib import Path
import pathlib
from typing import Optional, Tuple
import importlib
import sys
import types
import re

import numpy as np
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints # maps raw scalar data to colors (RGBA)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkImageData,
    vtkLine,
    vtkPiecewiseFunction,
    vtkPolyData,
    vtkPolyLine,
)
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkProbeFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkCylinderSource, vtkRegularPolygonSource, vtkSphereSource # generates a cylinder or sphere mesh (for cylindrical/spherical slicing)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkCoordinate,  # converts between viewport/world coordinate systems
    vtkImageSlice, 
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkInteractionStyle import (  # camera vs. locked styles for the ROI selector
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleUser,
)
from vtkmodules.vtkRenderingImage import vtkImageResliceMapper # maps a 3D volume to a 2D slice plane
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.util import numpy_support

from trame.app import get_server
from trame.ui.html import DivLayout
from trame.widgets import html
from trame.widgets.vtk import VtkRemoteView

PACKAGE_NAME = "napari_resview"
PACKAGE_PATH = Path(__file__).resolve().parent / PACKAGE_NAME

if PACKAGE_NAME not in sys.modules:
    package_module = types.ModuleType(PACKAGE_NAME)
    package_module.__path__ = [str(PACKAGE_PATH)]
    sys.modules[PACKAGE_NAME] = package_module

_data_io = importlib.import_module("napari_resview.data_io")
_rsm3d = importlib.import_module("napari_resview.rsm3d")

RSMDataLoader_ISR = _data_io.RSMDataLoader_ISR
RSMDataloader_CMS = _data_io.RSMDataloader_CMS
write_rsm_volume_to_vtr = _data_io.write_rsm_volume_to_vtr
RSMBuilder = _rsm3d.RSMBuilder

try:
    import matplotlib
    import matplotlib.cm as mpl_cm
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

COLORMAP_NAMES = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "gray",
]


def _float(value: Optional[object], default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_path(value: Optional[str]) -> str:
    return str(value).strip() if value is not None else ""


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

    if _HAS_MATPLOTLIB and colormap in matplotlib.colormaps:
        cmap = mpl_cm.get_cmap(colormap)
        colors = cmap(np.linspace(0.0, 1.0, 8))[:, :3]
        for idx, rgb in enumerate(colors):
            t = lo + (hi - lo) * idx / (len(colors) - 1)
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
    span = hi - lo
    opacity_tf.AddPoint(lo, 0.0)
    opacity_tf.AddPoint(lo + 0.05 * span, 0.02 * opacity_scale)
    opacity_tf.AddPoint(lo + 0.25 * span, 0.12 * opacity_scale)
    opacity_tf.AddPoint(lo + 0.75 * span, 0.35 * opacity_scale)
    opacity_tf.AddPoint(hi, min(1.0, 0.8 * opacity_scale))


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
    """Build a vtkLookupTable for slice/probe actors from a matplotlib colormap."""
    lut = vtkLookupTable()
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi <= lo:
        hi = lo + 1.0
    n = 256
    lut.SetNumberOfTableValues(n) # specify total number of colors in the gradient
    lut.SetRange(lo, hi) # define min/max data values to map
    if _HAS_MATPLOTLIB and colormap in matplotlib.colormaps:
        cmap = mpl_cm.get_cmap(colormap)
        colors = cmap(np.linspace(0.0, 1.0, n))
        for i in range(n):
            r, g, b, _ = colors[i]
            lut.SetTableValue(i, float(r), float(g), float(b), 1.0) # i is the index, A (transparency) is set to 1.0 for full opacity
    else:
        for i in range(n):
            t = i / (n - 1)
            lut.SetTableValue(i, t, t, t, 1.0) # this creates a grayscale gradient if matplotlib is not available or the colormap is invalid
            # when R, G, and B, are equal, the color is a shade of gray, and the value of t determines how light/dark that shade is
    lut.Build()
    return lut


def _crop_window_from_state(state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if not getattr(state, "crop_enabled", False):
        return None

    try:
        r0 = int(state.crop_row_min)
        r1 = int(state.crop_row_max)
        c0 = int(state.crop_col_min)
        c1 = int(state.crop_col_max)
    except (TypeError, ValueError):
        return None

    if r0 < 0 or c0 < 0 or r1 <= r0 or c1 <= c0:
        return None
    return ((r0, r1), (c0, c1))


def _adjust_setup_for_crop(setup, crop_window: Tuple[Tuple[int, int], Tuple[int, int]]):
    (r0, _), (c0, _) = crop_window
    setup.xcenter = max(0, int(setup.xcenter) - c0)
    setup.ycenter = max(0, int(setup.ycenter) - r0)
    setup.xpixels = max(1, int(setup.xpixels) - c0)
    setup.ypixels = max(1, int(setup.ypixels) - r0)


def _crop_dataframe_intensity(df, crop_window):
    if crop_window is None:
        return df
    (r0, r1), (c0, c1) = crop_window
    if r1 <= r0 or c1 <= c0:
        return df

    df = df.copy()
    df["intensity"] = [frame[r0:r1, c0:c1] for frame in df["intensity"]]
    return df


# Default number of frames to pre-select when a TIFF directory is chosen. The
# scan-range inputs are seeded so the first DEFAULT_FRAME_COUNT frames load.
DEFAULT_FRAME_COUNT = 362


def _scan_numbers_in_dir(tiff_dir: Optional[str]) -> list:
    """Return the scan numbers parsed from TIFF filenames in ``tiff_dir``.

    Mirrors ``RSMDataloader_CMS``'s filename parsing (the original napari
    behavior) so the Data-tab scan-range inputs reference the real scan ids
    embedded in the file names rather than positional frame indices. The list
    is sorted ascending, matching the order the CMS loader presents frames.
    """
    directory = Path(_ensure_path(tiff_dir)).expanduser()
    if not directory.is_dir():
        return []
    pattern = re.compile(RSMDataloader_CMS.SCAN_REGEX)
    scans = []
    for path in sorted(
        list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    ):
        match = pattern.search(path.name)
        if match:
            scans.append(int(match.group(1)))
    return sorted(scans)


def _parse_scan_list(text: Optional[str]) -> list:
    """Parse a scan-range string like "17-20, 30" into a sorted list of ints.

    Accepts comma/whitespace-separated single scans ("30") and inclusive
    ranges ("17-20"). Mirrors the napari widget's parse_scan_list so the web
    app accepts the same syntax. Raises ValueError on malformed input.
    """
    text = _ensure_path(text)
    if not text:
        return []
    out: set = set()
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                raise ValueError(f"Bad scan range: '{part}'")
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"Bad scan id: '{part}'")
    return sorted(out)


# partly copied from resview_widget.py
DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# The bundled defaults YAML lives inside the napari_resview package, not next to
# web_app.py. Use PACKAGE_PATH so the auto-filled setup path points at the file
# that actually exists (web_app.py sits one directory above the package, so
# Path(__file__).with_name(...) would resolve to a non-existent root-level file).
os.environ.setdefault(
    DEFAULTS_ENV,
    # str(pathlib.Path(__file__).with_name("rsm3d_defaults.yaml").resolve()),
    str(PACKAGE_PATH / "rsm3d_defaults.yaml"),
)
def yaml_path() -> str: # Allow override of defaults path via environment variable, else use ~/.rsm3d_defaults.yaml
    p = os.environ.get(DEFAULTS_ENV, "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")

def create_server():
    server = get_server(name="napari_resview_web", client_type="vue3")
    state, ctrl = server.state, server.controller

    # Data tab
    state.setdefault("loader_mode", "CMS")
    state.setdefault("setup_path", yaml_path())
    state.setdefault("tiff_dir", "")
    state.setdefault("spec_path", "")
    state.setdefault("cms_angle_step", 0.50) # range from 0 to 360, default 0.50 (every half-degree)
    # Scan-number selection, parsed from the TIFF filenames. Accepts a single
    # scan ("30") or an inclusive range / list ("17-20, 30"); empty = all.
    # Auto-seeded to the first DEFAULT_FRAME_COUNT frames when a TIFF directory
    # is selected so not all (e.g. 962) frames load at once.
    state.setdefault("scan_range", "")
    state.setdefault("crop_enabled", False)
    state.setdefault("crop_row_min", 0)
    state.setdefault("crop_row_max", 0)
    state.setdefault("crop_col_min", 0)
    state.setdefault("crop_col_max", 0)
    ## Experimental setup
    state.setdefault("exp_distance", 0.0)
    state.setdefault("exp_pitch", 0.0)
    state.setdefault("exp_det_h", 0)
    state.setdefault("exp_det_w", 0)
    state.setdefault("exp_bc_h", 0)
    state.setdefault("exp_bc_w", 0)
    state.setdefault("exp_energy", 0.0)
    state.setdefault("exp_wavelength", 0.0)

    # Build tab
    state.setdefault("space", "q")
    state.setdefault("grid_size", 90)
    state.setdefault("normalize", "mean")
    state.setdefault("ub_includes_2pi", True)
    state.setdefault("1_based_center", True)
    state.setdefault("fuzzy_gridder", False)
    state.setdefault("width_fuzzy", 0.01) # range from 0 to 999999999

    # View tab
    state.setdefault("log_view", True)
    state.setdefault("rendering", "attenuated_mip")
    state.setdefault("contrast_lo", 1.0)
    state.setdefault("contrast_hi", 99.8)
    state.setdefault("export_path", str(Path.cwd() / "rsm_output.vtr"))

    # Analysis tab
    ## orthogonal slicing (positions are 0-100)
    state.setdefault("slice_x_show", False)
    state.setdefault("slice_y_show", False)
    state.setdefault("slice_z_show", False)
    state.setdefault("slice_x_pos", 50)
    state.setdefault("slice_y_pos", 50)
    state.setdefault("slice_z_pos", 50)
    state.setdefault("slice_opacity", 0.8)
    state.setdefault("slice_cmap", "turbo")
    state.setdefault("slice_show_border", True)

    ## cylindrical slicing (Q space only)
    state.setdefault("cyl_show", False)
    state.setdefault("cyl_radius", 1.0)
    state.setdefault("cyl_samples", 64)
    state.setdefault("cyl_opacity", 0.7)
    state.setdefault("cyl_cmap", "plasma")

    ## spherical slicing (Q space only)
    state.setdefault("sph_show", False)
    state.setdefault("sph_radius", 1.0)
    state.setdefault("sph_samples", 64)
    state.setdefault("sph_opacity", 0.7)
    state.setdefault("sph_cmap", "viridis")

    state.setdefault("blend_mode", 0)
    state.setdefault("shade", True)
    state.setdefault("opacity_scale", 1.0)
    state.setdefault("colormap", "viridis")
    state.setdefault("status", "Ready")
    state.setdefault("status_log", ["Ready"])
    state.setdefault("scalar_range", "—")
    state.setdefault("volume_dims", "—")
    # Intensity frame viewer (View Intensity): the bottom slider scrubs through
    # frames [0, intensity_frame_max] of the currently loaded range.
    state.setdefault("intensity_slider_show", False)
    state.setdefault("intensity_frame_index", 0)
    state.setdefault("intensity_frame_max", 0)
    # When True, an adjustable ROI rectangle is drawn over the intensity frame
    # viewer; dragging/resizing it auto-fills the Crop row/col inputs.
    state.setdefault("roi_show", True)
    # True while the frame slider is auto-advancing (Play). Drives the
    # play/stop button label and the playback loop's keep-running check.
    state.setdefault("intensity_playing", False)
    state.setdefault("ambient", 0.2)
    state.setdefault("diffuse", 0.7)
    state.setdefault("specular", 0.3)
    state.setdefault("specular_power", 10.0)

    # File browser dialog state
    state.setdefault("fb_show", False)
    state.setdefault("fb_cwd", str(Path.home()))
    state.setdefault("fb_items", [])
    state.setdefault("fb_target", "")
    state.setdefault("fb_mode", "file")
    state.setdefault("fb_title", "Select a file")

    # Sidebar visibility
    state.setdefault("sidebar_open", True)

    # Which tab is expanded ("" = all collapsed; one of data/build/view/analysis)
    state.setdefault("open_tab", "")

    renderer = vtkRenderer()
    renderer.SetBackground(0.10, 0.10, 0.12)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1024, 768)
    render_window.SetOffScreenRendering(1)

    # trame-vtk's push_image() calls render_window.GetInteractor().EnableRenderOff(),
    # so the render window must have an interactor attached even though we render
    # off-screen and never start an event loop on it.
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()

    volume_mapper = vtkSmartVolumeMapper()
    volume_mapper.SetBlendModeToComposite()

    color_tf = vtkColorTransferFunction()
    opacity_tf = vtkPiecewiseFunction()

    volume_property = vtkVolumeProperty()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetColor(color_tf)
    volume_property.SetScalarOpacity(opacity_tf)
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.2)
    volume_property.SetDiffuse(0.7)
    volume_property.SetSpecular(0.3)
    volume_property.SetSpecularPower(10.0)

    volume_actor = vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(volume_property)
    volume_actor.VisibilityOff()
    renderer.AddVolume(volume_actor)
    renderer.ResetCamera()

    # --- Analysis: orthogonal slice actors (one per axis) ------------------
    # Each axis gets a vtkImageSlice backed by a reslice mapper. They are
    # hidden until the user enables them in the Analysis tab.
    slice_actors = {}
    slice_mappers = {}
    for axis in ("x", "y", "z"):
        mapper = vtkImageResliceMapper()
        mapper.SliceFacesCameraOff()
        mapper.SliceAtFocalPointOff()
        actor = vtkImageSlice()
        actor.SetMapper(mapper)
        actor.VisibilityOff()
        renderer.AddViewProp(actor)
        slice_mappers[axis] = mapper
        slice_actors[axis] = actor

    # --- Analysis: cylindrical / spherical probe actors --------------------
    cyl_actor = vtkActor()
    cyl_actor.VisibilityOff()
    renderer.AddActor(cyl_actor)
    sph_actor = vtkActor()
    sph_actor.VisibilityOff()
    renderer.AddActor(sph_actor)

    # --- Intensity frame viewer: a single 2D image slice scrubbed by the
    # bottom slider (View Intensity). Hidden until the user enables it.
    intensity_mapper = vtkImageResliceMapper()
    intensity_mapper.SliceFacesCameraOff()
    intensity_mapper.SliceAtFocalPointOff()
    intensity_actor = vtkImageSlice()
    intensity_actor.SetMapper(intensity_mapper)
    intensity_actor.VisibilityOff()
    renderer.AddViewProp(intensity_actor)

    # --- Intensity ROI selector -------------------------------------------
    # An adjustable rectangle drawn over the intensity frame. It lives in image
    # (world) coordinates at z>0 (just in front of the frame), so its corners
    # map directly to (col, row) pixel indices that fill the Crop inputs
    # (mirrors the napari "Crop from ROI" workflow). The user can drag the body
    # to move it, drag the corner/edge disks to resize, and drag the top disk to
    # rotate; the axis-aligned bounding box of the (possibly rotated) corners
    # defines the crop, exactly like napari.
    roi_state = {
        "active": False,
        "cx": 0.0, "cy": 0.0,        # center (image coords)
        "hw": 0.0, "hh": 0.0,        # half width / half height
        "angle": 0.0,                # radians, CCW
        "handle_r": 0.5,             # handle disk radius (world units)
        "z": 1.0,                    # small +z offset so the gizmo draws on top
        "grab": None,                # which part is being dragged
        "grab_offset": (0.0, 0.0),
        "fixed": (0.0, 0.0),         # fixed reference point during a resize
    }

    # Rectangle outline (closed polyline through the 4 corners).
    roi_outline_pts = vtkPoints()
    roi_outline_pts.SetNumberOfPoints(4)
    roi_outline_poly = vtkPolyData()
    roi_outline_poly.SetPoints(roi_outline_pts)
    _roi_outline_cells = vtkCellArray()
    _roi_outline_line = vtkPolyLine()
    _roi_outline_line.GetPointIds().SetNumberOfIds(5)
    for _i in range(4):
        _roi_outline_line.GetPointIds().SetId(_i, _i)
    _roi_outline_line.GetPointIds().SetId(4, 0)
    _roi_outline_cells.InsertNextCell(_roi_outline_line)
    roi_outline_poly.SetLines(_roi_outline_cells)
    roi_outline_mapper = vtkPolyDataMapper()
    roi_outline_mapper.SetInputData(roi_outline_poly)
    roi_outline_actor = vtkActor()
    roi_outline_actor.SetMapper(roi_outline_mapper)
    roi_outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    roi_outline_actor.GetProperty().SetLineWidth(1.5)
    roi_outline_actor.GetProperty().LightingOff()
    roi_outline_actor.PickableOff()
    roi_outline_actor.VisibilityOff()
    renderer.AddActor(roi_outline_actor)

    # Eight resize handles (4 corners + 4 edge midpoints) as filled disks.
    roi_handle_pts = vtkPoints()
    roi_handle_pts.SetNumberOfPoints(8)
    roi_handle_poly = vtkPolyData()
    roi_handle_poly.SetPoints(roi_handle_pts)
    roi_handle_src = vtkRegularPolygonSource()
    roi_handle_src.SetNumberOfSides(24)
    roi_handle_src.GeneratePolygonOn()
    roi_handle_src.SetNormal(0.0, 0.0, 1.0)
    roi_handle_src.SetRadius(0.5)
    roi_handle_glyph = vtkGlyph3D()
    roi_handle_glyph.SetInputData(roi_handle_poly)
    roi_handle_glyph.SetSourceConnection(roi_handle_src.GetOutputPort())
    roi_handle_glyph.SetScaleModeToDataScalingOff()
    roi_handle_glyph.SetScaleFactor(1.0)
    roi_handle_mapper = vtkPolyDataMapper()
    roi_handle_mapper.SetInputConnection(roi_handle_glyph.GetOutputPort())
    roi_handle_mapper.ScalarVisibilityOff()
    roi_handle_actor = vtkActor()
    roi_handle_actor.SetMapper(roi_handle_mapper)
    roi_handle_actor.GetProperty().SetColor(1.0, 1.0, 1.0) 
    roi_handle_actor.GetProperty().LightingOff()
    roi_handle_actor.PickableOff()
    roi_handle_actor.VisibilityOff()
    renderer.AddActor(roi_handle_actor)

    # Rotation handle (a disk above the top edge) plus a connector line.
    roi_rot_src = vtkRegularPolygonSource()
    roi_rot_src.SetNumberOfSides(24)
    roi_rot_src.GeneratePolygonOn()
    roi_rot_src.SetNormal(0.0, 0.0, 1.0)
    roi_rot_src.SetRadius(0.5)
    roi_rot_mapper = vtkPolyDataMapper()
    roi_rot_mapper.SetInputConnection(roi_rot_src.GetOutputPort())
    roi_rot_mapper.ScalarVisibilityOff()
    roi_rot_actor = vtkActor()
    roi_rot_actor.SetMapper(roi_rot_mapper)
    roi_rot_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    roi_rot_actor.GetProperty().LightingOff()
    roi_rot_actor.PickableOff()
    roi_rot_actor.VisibilityOff()
    renderer.AddActor(roi_rot_actor)

    roi_rotline_pts = vtkPoints()
    roi_rotline_pts.SetNumberOfPoints(2)
    roi_rotline_poly = vtkPolyData()
    roi_rotline_poly.SetPoints(roi_rotline_pts)
    _roi_rotline_cells = vtkCellArray()
    _roi_rotline_seg = vtkLine()
    _roi_rotline_seg.GetPointIds().SetId(0, 0)
    _roi_rotline_seg.GetPointIds().SetId(1, 1)
    _roi_rotline_cells.InsertNextCell(_roi_rotline_seg)
    roi_rotline_poly.SetLines(_roi_rotline_cells)
    roi_rotline_mapper = vtkPolyDataMapper()
    roi_rotline_mapper.SetInputData(roi_rotline_poly)
    roi_rotline_actor = vtkActor()
    roi_rotline_actor.SetMapper(roi_rotline_mapper)
    roi_rotline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    roi_rotline_actor.GetProperty().SetLineWidth(1.5)
    roi_rotline_actor.GetProperty().LightingOff()
    roi_rotline_actor.PickableOff()
    roi_rotline_actor.VisibilityOff()
    renderer.AddActor(roi_rotline_actor)

    # Interactor styles: the default trackball drives the 3D volume; a no-op
    # style locks the camera straight-on while the 2D ROI selector is active so
    # the screen<->image coordinate mapping stays exact.
    roi_trackball_style = vtkInteractorStyleTrackballCamera()
    roi_lock_style = vtkInteractorStyleUser()
    interactor.SetInteractorStyle(roi_trackball_style)

    current_volume = None
    current_axes = None
    current_builder = None
    # Loaded experiment state (populated by Load Data, consumed by Build/Regrid).
    current_setup = None
    current_ub = None
    current_df = None
    current_frames = None
    # Pixel dimensions (cols, rows) of the currently displayed intensity frame;
    # used to clamp ROI-derived crop indices. None until a frame is shown.
    intensity_nx = None
    intensity_ny = None
    # Regridded volume awaiting display (View RSM renders it).
    regrid_volume = None
    regrid_axes = None
    # The vtkImageData built for display (log/contrast-scaled) — referenced by
    # the slicing/probe helpers in the Analysis tab.
    current_image = None
    # Robust display range (log-scaled) used to build the transfer functions.
    render_range = None
    # In-flight asyncio task, tracked so the Stop button can cancel it.
    current_task = None
    # In-flight frame-playback task (Play button); tracked so it can be cancelled.
    play_task = None
    # Defer VtkRemoteView creation until the UI context is built
    remote_view = None
    # how to instantiate remote_view: it needs the render_window, but that needs to be created after the trame server is running. So we create it here as None, and then assign it inside the DivLayout context manager where we have access to the server.

    def _set_status(message: str):
        """Append a message to the Status history and push it to the client.

        The log is reassigned (not mutated in place) so trame detects the
        change, and flushed immediately so the message appears live; otherwise
        trame only sends state changes to the browser once the event handler
        returns.
        """
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        state.status_log = list(state.status_log or []) + [entry]
        state.status = entry  # most recent message (kept for convenience)
        state.flush()

    def _update_rendering():
        if current_volume is None:
            return
        state.scalar_range = state.scalar_range or "—"
        volume_property.SetShade(bool(state.shade))
        volume_property.SetAmbient(_float(state.ambient if hasattr(state, "ambient") else 0.2, 0.2))
        volume_property.SetDiffuse(_float(state.diffuse if hasattr(state, "diffuse") else 0.7, 0.7))
        volume_property.SetSpecular(_float(state.specular if hasattr(state, "specular") else 0.3, 0.3))
        volume_property.SetSpecularPower(_float(state.specular_power if hasattr(state, "specular_power") else 10.0, 10.0))
        
        # Map the napari-style rendering choice to a VTK blend mode:
        #   mip -> maximum intensity; everything else -> composite.
        rendering = _ensure_path(getattr(state, "rendering", "")) or "attenuated_mip"
        if rendering == "mip" or int(_float(state.blend_mode, 0)) == 1:
            volume_mapper.SetBlendModeToMaximumIntensity()
        else:
            volume_mapper.SetBlendModeToComposite()

        # Use the robust log-scaled display range computed in _set_volume_data
        # (falls back to the raw min/max if unavailable).
        scalar_range = render_range
        if scalar_range is None and current_volume is not None:
            scalar_range = (
                float(np.nanmin(current_volume)),
                float(np.nanmax(current_volume)),
            )

        _apply_color_transfer_function(color_tf, _ensure_path(state.colormap), scalar_range)
        _apply_opacity_function(opacity_tf, scalar_range, _float(state.opacity_scale, 1.0))
        render_window.Render()
        # remote_view is created later, inside the UI layout. Guard against it
        # being None if a render is somehow triggered before the UI is built.
        if remote_view is not None:
            remote_view.update()

    def _set_volume_data(volume: np.ndarray, axes: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        nonlocal current_volume, current_axes, render_range, current_image
        # Switching to a volume view supersedes the 2D intensity-frame viewer.
        intensity_actor.VisibilityOff()
        _roi_set_active(False)  # the ROI selector only applies to the intensity view
        renderer.GetActiveCamera().ParallelProjectionOff()  # restore 3D perspective
        state.intensity_slider_show = False
        state.intensity_playing = False  # halt any in-progress frame playback
        current_volume = np.asarray(volume, dtype=np.float32)
        current_axes = axes

        # Log-compress for display so the high-dynamic-range RSM is visible,
        # and derive the transfer-function range from user-inputted contrast percentiles.
        # The raw current_volume is kept untouched for VTR export.
        if bool(getattr(state, "log_view", True)):
            display_volume = _log1p_clip(current_volume)
        else:
            display_volume = np.maximum(current_volume, 0.0)
        lo = _float(getattr(state, "contrast_lo", 1.0), 1.0)
        hi = _float(getattr(state, "contrast_hi", 99.8), 99.8)
        if not (0.0 <= lo < hi <= 100.0):
            lo, hi = 1.0, 99.8
        render_range = _robust_percentiles(display_volume, (lo, hi))

        image = vtkImageData()
        nx, ny, nz = display_volume.shape
        image.SetDimensions(nx, ny, nz)

        spacings = []
        origin = []
        for axis_values in axes:
            axis_arr = np.asarray(axis_values, dtype=float)
            if axis_arr.size > 1:
                spacing = float(axis_arr[1] - axis_arr[0])
            else:
                spacing = 1.0
            spacings.append(spacing)
            origin.append(float(axis_arr[0]))

        image.SetSpacing(*spacings)
        image.SetOrigin(*origin)

        vtk_array = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(display_volume, dtype=np.float32).ravel(order="F"),
            deep=True,
            array_type=numpy_support.get_vtk_array_type(np.float32),
        )
        vtk_array.SetName("intensity")
        image.GetPointData().SetScalars(vtk_array)
        volume_mapper.SetInputData(image)
        volume_actor.VisibilityOn()
        current_image = image
        renderer.ResetCamera()
        _update_rendering()
        _update_all_slices()
        return current_volume, current_axes
        # ---------------------------------------------------------------------
    # Analysis: orthogonal / cylindrical / spherical slicing helpers
    # ---------------------------------------------------------------------
    def _axis_bounds(image, axis_index):
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        dims = image.GetDimensions()
        lo = origin[axis_index]
        hi = origin[axis_index] + spacing[axis_index] * max(0, dims[axis_index] - 1)
        return float(lo), float(hi)

    def _update_ortho_slice(axis):
        if current_image is None or render_range is None:
            slice_actors[axis].VisibilityOff()
            return
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        show = bool(getattr(state, f"slice_{axis}_show", False))
        if not show:
            slice_actors[axis].VisibilityOff()
            return
        pos_pct = _float(getattr(state, f"slice_{axis}_pos", 50), 50.0)
        lo, hi = _axis_bounds(current_image, axis_index)
        coord = lo + (hi - lo) * max(0.0, min(pos_pct, 100.0)) / 100.0

        normal = [0.0, 0.0, 0.0]
        normal[axis_index] = 1.0
        origin = [0.0, 0.0, 0.0]
        center = current_image.GetCenter()
        origin[0], origin[1], origin[2] = center
        origin[axis_index] = coord

        mapper = slice_mappers[axis]
        mapper.SetInputData(current_image)
        plane = mapper.GetSlicePlane()
        plane.SetOrigin(*origin)
        plane.SetNormal(*normal)

        lut = _make_lookup_table(_ensure_path(state.slice_cmap), render_range)
        prop = slice_actors[axis].GetProperty()
        prop.SetLookupTable(lut)
        prop.UseLookupTableScalarRangeOn()
        prop.SetOpacity(_float(state.slice_opacity, 0.8))
        slice_actors[axis].VisibilityOn()

    def _probe_surface(actor, source_output, colormap, opacity, show):
        if not show or current_image is None or render_range is None:
            actor.VisibilityOff()
            return
        probe = vtkProbeFilter()
        probe.SetInputData(source_output)
        probe.SetSourceData(current_image)
        probe.Update()
        lut = _make_lookup_table(colormap, render_range)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(probe.GetOutput())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(render_range[0], render_range[1])
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("intensity")
        mapper.SetColorModeToMapScalars()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(_float(opacity, 0.7))
        actor.VisibilityOn()

    def _update_cylinder():
        if current_image is None:
            cyl_actor.VisibilityOff()
            return
        show = bool(state.cyl_show)
        if not show:
            cyl_actor.VisibilityOff()
            return
        center = current_image.GetCenter()
        _, zhi = _axis_bounds(current_image, 2)
        zlo, _ = _axis_bounds(current_image, 2)
        height = max(1e-6, zhi - zlo)
        radius = _float(state.cyl_radius, 1.0)
        samples = max(8, int(_float(state.cyl_samples, 64)))
        src = vtkCylinderSource()
        src.SetRadius(radius)
        src.SetHeight(height)
        src.SetResolution(samples)
        src.CappingOff()
        src.Update()
        # vtkCylinderSource is aligned to Y; rotate so its axis lies along Z and
        # translate to the volume center.
        tf = vtkTransform()
        tf.Translate(center[0], center[1], center[2])
        tf.RotateX(90.0)
        tpd = vtkTransformPolyDataFilter()
        tpd.SetTransform(tf)
        tpd.SetInputData(src.GetOutput())
        tpd.Update()
        _probe_surface(
            cyl_actor, tpd.GetOutput(), _ensure_path(state.cyl_cmap),
            _float(state.cyl_opacity, 0.7), show,
        )

    def _update_sphere():
        if current_image is None:
            sph_actor.VisibilityOff()
            return
        show = bool(state.sph_show)
        if not show:
            sph_actor.VisibilityOff()
            return
        center = current_image.GetCenter()
        radius = _float(state.sph_radius, 1.0)
        samples = max(8, int(_float(state.sph_samples, 64)))
        src = vtkSphereSource()
        src.SetCenter(center[0], center[1], center[2])
        src.SetRadius(radius)
        src.SetThetaResolution(samples)
        src.SetPhiResolution(samples)
        src.Update()
        _probe_surface(
            sph_actor, src.GetOutput(), _ensure_path(state.sph_cmap),
            _float(state.sph_opacity, 0.7), show,
        )

    def _update_all_slices():
        try:
            for axis in ("x", "y", "z"):
                _update_ortho_slice(axis)
            _update_cylinder()
            _update_sphere()
        except Exception as exc:  # never let slicing break the main render
            _set_status(f"Slice update skipped: {exc}")

    @ctrl.set("update_slices")
    def update_slices(**kwargs):
        _update_all_slices()
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    # ---------------------------------------------------------------------
    # Data loading helpers
    # ---------------------------------------------------------------------
    def _selected_scans_from_state(tiff_dir):
        """Resolve the Data-tab scan-range text to an explicit scan list.

        Parses the scan ids from the TIFF filenames and keeps those matching
        the user's scan-range string (e.g. "17-20, 30"). An empty string loads
        everything. Raises ValueError on malformed input.
        """
        scans = _scan_numbers_in_dir(tiff_dir)
        if not scans:
            return None
        requested = _parse_scan_list(getattr(state, "scan_range", ""))
        if not requested:
            return None
        available = set(scans)
        selected = sorted(s for s in requested if s in available)
        return selected or None

    def _load_experiment(loader_mode):
        setup_path = Path(_ensure_path(state.setup_path)).expanduser()
        tiff_dir = Path(_ensure_path(state.tiff_dir)).expanduser()
        selected_scans = _selected_scans_from_state(str(tiff_dir))
        if loader_mode == "ISR":
            spec_path = Path(_ensure_path(state.spec_path)).expanduser()
            loader = RSMDataLoader_ISR(
                str(spec_path),
                str(setup_path),
                str(tiff_dir),
                use_dask=False,
                selected_scans=selected_scans,
            )
            setup, ub, df = loader.load()
        else:
            loader = RSMDataloader_CMS(
                str(setup_path),
                str(tiff_dir),
                angle_step=_float(state.cms_angle_step, 1.0),
                selected_scans=selected_scans,
            )
            setup, ub, df = loader.load()
        frames = None
        if df is not None and "intensity" in getattr(df, "columns", []):
            frames = list(df["intensity"])
        return setup, ub, df, frames

    def _populate_setup_fields(setup, frames):
        """Reflect a loaded ExperimentSetup into the Data-tab fields."""
        if setup is not None:
            state.exp_distance = float(getattr(setup, "distance", 0.0) or 0.0)
            state.exp_pitch = float(getattr(setup, "pitch", 0.0) or 0.0)
            state.exp_det_h = int(getattr(setup, "ypixels", 0) or 0)
            state.exp_det_w = int(getattr(setup, "xpixels", 0) or 0)
            state.exp_bc_h = int(getattr(setup, "ycenter", 0) or 0)
            state.exp_bc_w = int(getattr(setup, "xcenter", 0) or 0)
            state.exp_energy = float(getattr(setup, "energy", 0.0) or 0.0)
            state.exp_wavelength = float(getattr(setup, "wavelength", 0.0) or 0.0)
        # Detector dimensions from the actual frames take precedence.
        if frames:
            first = frames[0]
            if hasattr(first, "shape") and len(first.shape) >= 2:
                h, w = int(first.shape[-2]), int(first.shape[-1])
                state.exp_det_h = h
                state.exp_det_w = w

    def _apply_setup_overrides(setup):
        """Push user-edited Data-tab values onto the setup before building."""
        if setup is None:
            return
        try:
            setup.distance = float(state.exp_distance) or setup.distance
            setup.pitch = float(state.exp_pitch) or setup.pitch
            setup.ypixels = int(state.exp_det_h) or setup.ypixels
            setup.xpixels = int(state.exp_det_w) or setup.xpixels
            setup.ycenter = int(state.exp_bc_h)
            setup.xcenter = int(state.exp_bc_w)
            if float(state.exp_energy) > 0:
                setup.energy = float(state.exp_energy)
            if float(state.exp_wavelength) > 0:
                setup.wavelength = float(state.exp_wavelength)
        except (TypeError, ValueError) as exc:
            _set_status(f"Setup override skipped: {exc}")

    def _compute_builder(setup, ub, df):
        builder = RSMBuilder(setup, ub, df, ub_includes_2pi=True)
        builder.compute_full(verbose=False)
        return builder

    def _regrid_volume(builder, grid_size):
        return builder.regrid_xu(
            space=_ensure_path(state.space) or "q",
            grid_shape=(grid_size, grid_size, grid_size),
            normalize=_ensure_path(state.normalize) or "mean",
        )

    def _track(coro):
        """Run a coroutine as a cancellable task (for the Stop button)."""
        nonlocal current_task
        loop = asyncio.get_event_loop()
        current_task = loop.create_task(coro)
        return current_task

    # ---------------------------------------------------------------------
    # Pipeline actions
    # ---------------------------------------------------------------------
    async def _do_load_data():
        nonlocal current_setup, current_ub, current_df, current_frames
        nonlocal current_builder, regrid_volume, regrid_axes
        tiff_dir = Path(_ensure_path(state.tiff_dir)).expanduser()
        setup_path = Path(_ensure_path(state.setup_path)).expanduser()
        loader_mode = _ensure_path(state.loader_mode).upper() or "CMS"

        if not setup_path.is_file():
            _set_status("Missing YAML setup file.")
            return
        if not tiff_dir.is_dir():
            _set_status("Missing TIFF directory.")
            return
        if loader_mode == "ISR":
            spec_path = Path(_ensure_path(state.spec_path)).expanduser()
            if not spec_path.is_file():
                _set_status("Missing SPEC file for ISR mode.")
                return

        loop = asyncio.get_event_loop()
        try:
            _set_status("Loading experiment and TIFF frames...")
            setup, ub, df, frames = await loop.run_in_executor(
                None, _load_experiment, loader_mode
            )

            current_setup, current_ub, current_df, current_frames = setup, ub, df, frames
            current_builder = None
            regrid_volume = None
            regrid_axes = None
            state.intensity_slider_show = False
            _populate_setup_fields(setup, frames)
            n = len(frames) if frames else 0
            _set_status(f"Data loaded ({n} frame(s)). Ready to build.")
        except asyncio.CancelledError:
            _set_status("Load cancelled.")
        except Exception as exc:
            _set_status(f"Load error: {exc}")

    @ctrl.set("load_data")
    def load_data(**kwargs):
        _track(_do_load_data())
        
    async def _do_build_rsm():
        nonlocal current_builder
        if current_setup is None or current_df is None:
            _set_status("Load data first.")
            return
        _apply_setup_overrides(current_setup)
        loop = asyncio.get_event_loop()
        try:
            _set_status("Computing Q/HKL mapping...")
            current_builder = await loop.run_in_executor(
                None, _compute_builder, current_setup, current_ub, current_df
            )
            _set_status("RSM map built. Ready to regrid.")
        except asyncio.CancelledError:
            _set_status("Build cancelled.")
        except Exception as exc:
            _set_status(f"Build error: {exc}")

    @ctrl.set("build_rsm")
    def build_rsm(**kwargs):
        _track(_do_build_rsm())

    async def _do_regrid():
        nonlocal regrid_volume, regrid_axes
        if current_builder is None:
            _set_status("Build the RSM map first.")
            return
        loop = asyncio.get_event_loop()
        try:
            grid_size = max(16, int(_float(state.grid_size, 90)))
            _set_status(f"Regridding to {grid_size}³ volume...")
            volume, axes = await loop.run_in_executor(
                None, _regrid_volume, current_builder, grid_size
            )
            regrid_volume, regrid_axes = volume, axes
            state.scalar_range = (
                f"{float(np.nanmin(volume)):.4g} … {float(np.nanmax(volume)):.4g}"
            )
            state.volume_dims = (
                f"{volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]}"
            )
            _set_status("Regrid complete. Use View RSM to display.")
        except asyncio.CancelledError:
            _set_status("Regrid cancelled.")
        except Exception as exc:
            _set_status(f"Regrid error: {exc}")

    @ctrl.set("regrid")
    def regrid(**kwargs):
        _track(_do_regrid())

    @ctrl.set("view_rsm")
    def view_rsm(**kwargs):
        if regrid_volume is None or regrid_axes is None:
            _set_status("Regrid first.")
            return
        _set_status("Updating 3D view...")
        _set_volume_data(regrid_volume, regrid_axes)
        _set_status("RSM volume displayed.")

    # ---------------------------------------------------------------------
    # Intensity ROI selector helpers
    # ---------------------------------------------------------------------
    MIN_HALF = 1.0  # smallest allowed half-extent (pixels) when resizing

    def _roi_axes():
        a = roi_state["angle"]
        return (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))

    def _roi_corners():
        cx, cy = roi_state["cx"], roi_state["cy"]
        hw, hh = roi_state["hw"], roi_state["hh"]
        (ux, uy), (vx, vy) = _roi_axes()
        return [
            (cx - hw * ux - hh * vx, cy - hw * uy - hh * vy),  # c0: -u -v
            (cx + hw * ux - hh * vx, cy + hw * uy - hh * vy),  # c1: +u -v
            (cx + hw * ux + hh * vx, cy + hw * uy + hh * vy),  # c2: +u +v
            (cx - hw * ux + hh * vx, cy - hw * uy + hh * vy),  # c3: -u +v
        ]

    def _roi_edge_mids():
        cx, cy = roi_state["cx"], roi_state["cy"]
        hw, hh = roi_state["hw"], roi_state["hh"]
        (ux, uy), (vx, vy) = _roi_axes()
        return {
            "eR": (cx + hw * ux, cy + hw * uy),
            "eT": (cx + hh * vx, cy + hh * vy),
            "eL": (cx - hw * ux, cy - hw * uy),
            "eB": (cx - hh * vx, cy - hh * vy),
        }

    def _roi_rot_pos():
        cx, cy = roi_state["cx"], roi_state["cy"]
        hh = roi_state["hh"]
        (_ux, _uy), (vx, vy) = _roi_axes()
        pad = roi_state["handle_r"] * 3.0
        return (cx + (hh + pad) * vx, cy + (hh + pad) * vy)

    def _roi_refresh_actors():
        z = roi_state["z"]
        corners = _roi_corners()
        for i, (px, py) in enumerate(corners):
            roi_outline_pts.SetPoint(i, px, py, z)
        roi_outline_pts.Modified()
        mids = _roi_edge_mids()
        handle_pts = corners + [mids["eR"], mids["eT"], mids["eL"], mids["eB"]]
        for i, (px, py) in enumerate(handle_pts):
            roi_handle_pts.SetPoint(i, px, py, z)
        roi_handle_pts.Modified()
        roi_handle_glyph.Modified()
        rx, ry = _roi_rot_pos()
        roi_rot_src.SetCenter(rx, ry, z)
        etx, ety = mids["eT"]
        roi_rotline_pts.SetPoint(0, etx, ety, z)
        roi_rotline_pts.SetPoint(1, rx, ry, z)
        roi_rotline_pts.Modified()

    def _roi_set_visible(show):
        for actor in (
            roi_outline_actor, roi_handle_actor, roi_rot_actor, roi_rotline_actor
        ):
            actor.SetVisibility(1 if show else 0)

    def _world_to_display(px, py):
        c = vtkCoordinate()
        c.SetCoordinateSystemToWorld()
        c.SetValue(px, py, roi_state["z"])
        return c.GetComputedDoubleDisplayValue(renderer)

    def _display_to_world(dx, dy):
        c = vtkCoordinate()
        c.SetCoordinateSystemToDisplay()
        c.SetValue(float(dx), float(dy), 0.0)
        wx, wy, _wz = c.GetComputedWorldValue(renderer)
        return wx, wy

    def _disp_dist(world_pt, dx, dy):
        sx, sy = _world_to_display(world_pt[0], world_pt[1])
        return math.hypot(sx - dx, sy - dy)

    def _point_in_roi(px, py):
        (ux, uy), (vx, vy) = _roi_axes()
        ddx, ddy = px - roi_state["cx"], py - roi_state["cy"]
        return (
            abs(ddx * ux + ddy * uy) <= roi_state["hw"]
            and abs(ddx * vx + ddy * vy) <= roi_state["hh"]
        )

    def _roi_hit_test(dx, dy):
        thr = 11.0  # px pick tolerance
        if _disp_dist(_roi_rot_pos(), dx, dy) <= thr:
            return "rot"
        for i, corner in enumerate(_roi_corners()):
            if _disp_dist(corner, dx, dy) <= thr:
                return f"c{i}"
        for key, mid in _roi_edge_mids().items():
            if _disp_dist(mid, dx, dy) <= thr:
                return key
        wx, wy = _display_to_world(dx, dy)
        if _point_in_roi(wx, wy):
            return "body"
        return None

    def _roi_update_crop_state():
        if intensity_nx is None or intensity_ny is None:
            return
        corners = _roi_corners()
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        col_lo = max(0, min(int(round(min(xs))), intensity_nx - 1))
        col_hi = max(0, min(int(round(max(xs))), intensity_nx - 1))
        # The displayed frame is flipped vertically (see _show_intensity_frame)
        # to match napari's top-left origin, so the ROI's world-space rows are
        # measured from the bottom. Convert them back to original top-down frame
        # rows so the crop extracts the region the user actually selected.
        disp_row_lo = max(0, min(int(round(min(ys))), intensity_ny - 1))
        disp_row_hi = max(0, min(int(round(max(ys))), intensity_ny - 1))
        row_lo = (intensity_ny - 1) - disp_row_hi
        row_hi = (intensity_ny - 1) - disp_row_lo
        if col_hi <= col_lo:
            col_hi = min(intensity_nx - 1, col_lo + 1)
        if row_hi <= row_lo:
            row_hi = min(intensity_ny - 1, row_lo + 1)
        state.crop_col_min = col_lo
        state.crop_col_max = col_hi
        state.crop_row_min = row_lo
        state.crop_row_max = row_hi
        state.flush()

    def _roi_render():
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    def _roi_init_geometry():
        """Center the ROI box over the frame at ~60% of each dimension."""
        nx = intensity_nx or 1
        ny = intensity_ny or 1
        roi_state["cx"] = (nx - 1) / 2.0
        roi_state["cy"] = (ny - 1) / 2.0
        roi_state["hw"] = max(MIN_HALF, nx * 0.3)
        roi_state["hh"] = max(MIN_HALF, ny * 0.3)
        roi_state["angle"] = 0.0
        # Size the handle disks from the current camera zoom (parallel scale =
        # half the world-height filling the viewport) so they keep a constant
        # on-screen size across successive crops, matching the fixed-pixel line
        # width. Falls back to the frame size if projection isn't parallel yet.
        cam = renderer.GetActiveCamera()
        if cam.GetParallelProjection():
            ps = cam.GetParallelScale()
        else:
            ps = max(nx, ny) * 0.5
        # Pure fraction of the zoom (parallel scale) => constant on-screen size.
        # Do NOT clamp to a world-unit floor: on a small (heavily cropped) frame
        # the zoom is large, so a fixed world floor would render as huge disks
        # that swamp the box. Guard only against a non-positive scale.
        roi_state["handle_r"] = max(ps, 1e-6) * 0.02
        roi_state["z"] = max(1.0, 0.01 * max(nx, ny))
        roi_state["grab"] = None
        roi_handle_src.SetRadius(roi_state["handle_r"])
        roi_rot_src.SetRadius(roi_state["handle_r"])

    def _roi_set_active(active):
        roi_state["active"] = bool(active)
        if active:
            interactor.SetInteractorStyle(roi_lock_style)
            _roi_set_visible(True)
            _roi_refresh_actors()
        else:
            roi_state["grab"] = None
            _roi_set_visible(False)
            interactor.SetInteractorStyle(roi_trackball_style)

    def _roi_on_press(obj, event):
        if not roi_state["active"]:
            return
        dx, dy = interactor.GetEventPosition()
        grab = _roi_hit_test(dx, dy)
        roi_state["grab"] = grab
        if grab is None:
            return
        wx, wy = _display_to_world(dx, dy)
        if grab == "body":
            roi_state["grab_offset"] = (wx - roi_state["cx"], wy - roi_state["cy"])
        elif grab in ("c0", "c1", "c2", "c3"):
            opp = {"c0": 2, "c1": 3, "c2": 0, "c3": 1}[grab]
            roi_state["fixed"] = _roi_corners()[opp]

    def _roi_on_move(obj, event):
        if not roi_state["active"] or roi_state["grab"] is None:
            return
        dx, dy = interactor.GetEventPosition()
        wx, wy = _display_to_world(dx, dy)
        g = roi_state["grab"]
        (ux, uy), (vx, vy) = _roi_axes()
        if g == "body":
            ox, oy = roi_state["grab_offset"]
            roi_state["cx"] = wx - ox
            roi_state["cy"] = wy - oy
        elif g == "rot":
            roi_state["angle"] = (
                math.atan2(wy - roi_state["cy"], wx - roi_state["cx"]) - math.pi / 2.0
            )
        elif g in ("c0", "c1", "c2", "c3"):
            fx, fy = roi_state["fixed"]
            ncx, ncy = (wx + fx) / 2.0, (wy + fy) / 2.0
            ddx, ddy = wx - ncx, wy - ncy
            roi_state["cx"], roi_state["cy"] = ncx, ncy
            roi_state["hw"] = max(MIN_HALF, abs(ddx * ux + ddy * uy))
            roi_state["hh"] = max(MIN_HALF, abs(ddx * vx + ddy * vy))
        elif g in ("eR", "eL"):
            proj = (wx - roi_state["cx"]) * ux + (wy - roi_state["cy"]) * uy
            fixed_s = -roi_state["hw"] if g == "eR" else roi_state["hw"]
            shift = (fixed_s + proj) / 2.0
            roi_state["hw"] = max(MIN_HALF, abs(proj - fixed_s) / 2.0)
            roi_state["cx"] += ux * shift
            roi_state["cy"] += uy * shift
        elif g in ("eT", "eB"):
            proj = (wx - roi_state["cx"]) * vx + (wy - roi_state["cy"]) * vy
            fixed_s = -roi_state["hh"] if g == "eT" else roi_state["hh"]
            shift = (fixed_s + proj) / 2.0
            roi_state["hh"] = max(MIN_HALF, abs(proj - fixed_s) / 2.0)
            roi_state["cx"] += vx * shift
            roi_state["cy"] += vy * shift
        _roi_refresh_actors()
        _roi_update_crop_state()
        _roi_render()

    def _roi_on_release(obj, event):
        if roi_state["grab"] is not None:
            roi_state["grab"] = None
            _roi_update_crop_state()
            _roi_render()

    # Observe the forwarded mouse events (priority above the camera style so a
    # grab on a handle/body edits the ROI rather than moving the camera).
    interactor.AddObserver("LeftButtonPressEvent", _roi_on_press, 10.0)
    interactor.AddObserver("MouseMoveEvent", _roi_on_move, 10.0)
    interactor.AddObserver("LeftButtonReleaseEvent", _roi_on_release, 10.0)

    @state.change("roi_show")
    def _on_roi_show_change(roi_show=True, **kwargs):
        # Only meaningful while the intensity frame viewer is active.
        if not bool(getattr(state, "intensity_slider_show", False)):
            return
        if roi_show and intensity_nx is not None:
            _roi_init_geometry()
            _roi_set_active(True)
            _roi_update_crop_state()
        else:
            _roi_set_active(False)
        _roi_render()

    @ctrl.set("view_intensity")
    def view_intensity(**kwargs):
        if not current_frames:
            _set_status("No intensity frames. Load data first.")
            return
        n = len(current_frames)
        # Frame-by-frame mode: hide the volume / analysis props and show a
        # single 2D image that the bottom slider scrubs through.
        volume_actor.VisibilityOff()
        for actor in slice_actors.values():
            actor.VisibilityOff()
        cyl_actor.VisibilityOff()
        sph_actor.VisibilityOff()
        state.intensity_frame_max = n - 1
        state.intensity_slider_show = True
        if int(_float(state.intensity_frame_index, 0)) != 0:
            # Reset to the first frame; the change handler will render it.
            state.intensity_frame_index = 0
        try:
            _show_intensity_frame(0, reset_camera=True)
        except Exception as exc:
            _set_status(f"Intensity view error: {exc}")
            return
        # Drop an adjustable ROI box on the frame and seed the Crop inputs from
        # it so the user can move/resize/rotate it to refine the crop.
        if bool(getattr(state, "roi_show", True)):
            _roi_init_geometry()
            _roi_set_active(True)
            _roi_update_crop_state()
        else:
            _roi_set_active(False)
        _roi_render()
        _set_status(f"Viewing intensity frames (0–{n - 1}). Use the slider to scrub.")

    def _show_intensity_frame(index, reset_camera=False):
        """Display a single raw intensity frame as a 2D image slice."""
        nonlocal intensity_nx, intensity_ny
        if not current_frames:
            return
        n = len(current_frames)
        index = max(0, min(int(index), n - 1))
        frame = np.asarray(current_frames[index], dtype=np.float32)
        if frame.ndim != 2:
            frame = np.squeeze(frame)
        if frame.ndim != 2:
            _set_status(f"Frame {index} is not 2D; cannot display.")
            return

        # Mirror the volume display path: log-compress and frame the colormap
        # over robust contrast percentiles.
        if bool(getattr(state, "log_view", True)):
            disp = _log1p_clip(frame)
        else:
            disp = np.maximum(frame, 0.0)
        lo = _float(getattr(state, "contrast_lo", 1.0), 1.0)
        hi = _float(getattr(state, "contrast_hi", 99.8), 99.8)
        if not (0.0 <= lo < hi <= 100.0):
            lo, hi = 1.0, 99.8
        frange = _robust_percentiles(disp, (lo, hi))

        ny, nx = disp.shape
        intensity_nx, intensity_ny = nx, ny
        # napari places row 0 at the TOP (origin at the top-left corner), but
        # VTK's image space puts +y upward, so row 0 would otherwise render at
        # the bottom. Flip the frame vertically before handing it to VTK so the
        # web view matches napari's orientation. The ROI->crop mapping in
        # _roi_update_crop_state inverts this flip to keep crop indices in the
        # original (top-down) frame coordinates.
        disp = disp[::-1, :]
        image = vtkImageData()
        image.SetDimensions(nx, ny, 1)
        image.SetSpacing(1.0, 1.0, 1.0)
        image.SetOrigin(0.0, 0.0, 0.0)
        vtk_array = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(disp, dtype=np.float32).ravel(order="C"),
            deep=True,
            array_type=numpy_support.get_vtk_array_type(np.float32),
        )
        vtk_array.SetName("intensity")
        image.GetPointData().SetScalars(vtk_array)

        intensity_mapper.SetInputData(image)
        plane = intensity_mapper.GetSlicePlane()
        plane.SetOrigin(0.0, 0.0, 0.0)
        plane.SetNormal(0.0, 0.0, 1.0)

        lut = _make_lookup_table(_ensure_path(state.colormap), frange)
        prop = intensity_actor.GetProperty()
        prop.SetLookupTable(lut)
        prop.UseLookupTableScalarRangeOn()
        intensity_actor.VisibilityOn()

        if reset_camera:
            # Place the camera deterministically straight-on over the frame
            # using parallel projection. We size the view directly from the
            # frame dimensions and the viewport aspect instead of ResetCamera()
            # so the zoom depends ONLY on the current frame -- not on any other
            # visible prop (e.g. a stale ROI box). This keeps the screen<->image
            # mapping exact and gives a stable parallel scale, so the ROI handle
            # disks keep a constant on-screen size across successive crops.
            cam = renderer.GetActiveCamera()
            cam.ParallelProjectionOn()
            cx = (nx - 1) / 2.0
            cy = (ny - 1) / 2.0
            cam.SetFocalPoint(cx, cy, 0.0)
            cam.SetPosition(cx, cy, float(max(nx, ny) + 10))
            cam.SetViewUp(0.0, 1.0, 0.0)
            win_w, win_h = render_window.GetSize()
            aspect = (win_w / win_h) if win_h else 1.0
            half_h = ny / 2.0
            half_w = (nx / 2.0) / aspect if aspect else nx / 2.0
            cam.SetParallelScale(max(half_h, half_w) * 1.05)
            renderer.ResetCameraClippingRange()
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    @state.change("intensity_frame_index")
    def _on_intensity_frame_change(intensity_frame_index=None, **kwargs):
        if not bool(getattr(state, "intensity_slider_show", False)):
            return
        if not current_frames:
            return
        _show_intensity_frame(int(_float(intensity_frame_index, 0)))

    async def _play_loop():
        """Advance the frame slider at 10 fps, looping back to 0 at the end."""
        nonlocal play_task
        period = 0.1  # seconds between frames -> 10 frames per second
        try:
            while bool(getattr(state, "intensity_playing", False)):
                frame_max = int(_float(getattr(state, "intensity_frame_max", 0), 0))
                if frame_max <= 0 or not current_frames:
                    break
                cur = int(_float(getattr(state, "intensity_frame_index", 0), 0))
                nxt = cur + 1 if cur < frame_max else 0
                # Mutating + flushing inside `with state:` triggers the
                # intensity_frame_index change handler, which renders the frame.
                with state:
                    state.intensity_frame_index = nxt
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass
        finally:
            play_task = None
            with state:
                state.intensity_playing = False

    @ctrl.set("toggle_play")
    def toggle_play(**kwargs):
        """Toggle automatic frame playback (Play <-> Stop)."""
        nonlocal play_task
        if bool(getattr(state, "intensity_playing", False)):
            state.intensity_playing = False
            if play_task is not None:
                play_task.cancel()
                play_task = None
            return
        if not current_frames or not bool(getattr(state, "intensity_slider_show", False)):
            _set_status("Start the intensity viewer (View Intensity) before playing.")
            return
        state.intensity_playing = True
        loop = asyncio.get_event_loop()
        play_task = loop.create_task(_play_loop())

    def _step_intensity_frame(delta):
        """Step one frame (delta = +1 / -1), wrapping at the range ends."""
        nonlocal play_task
        if not current_frames or not bool(getattr(state, "intensity_slider_show", False)):
            return
        # A manual step stops playback so the two don't fight over the index.
        if bool(getattr(state, "intensity_playing", False)):
            state.intensity_playing = False
            if play_task is not None:
                play_task.cancel()
                play_task = None
        frame_max = int(_float(getattr(state, "intensity_frame_max", 0), 0))
        if frame_max <= 0:
            return
        cur = int(_float(getattr(state, "intensity_frame_index", 0), 0))
        state.intensity_frame_index = (cur + delta) % (frame_max + 1)

    @ctrl.set("prev_frame")
    def prev_frame(**kwargs):
        _step_intensity_frame(-1)

    @ctrl.set("next_frame")
    def next_frame(**kwargs):
        _step_intensity_frame(1)

    @ctrl.set("crop_from_roi")
    def crop_from_roi(**kwargs):
        nonlocal current_df, current_frames, current_builder, regrid_volume, regrid_axes
        if current_df is None or not current_frames:
            _set_status("Load data before cropping.")
            return
        try:
            r0 = int(state.crop_row_min)
            r1 = int(state.crop_row_max)
            c0 = int(state.crop_col_min)
            c1 = int(state.crop_col_max)
        except (TypeError, ValueError):
            _set_status("Invalid crop bounds.")
            return
        if r1 <= r0 or c1 <= c0:
            _set_status("Crop bounds must have positive area (max > min).")
            return
        crop_window = ((r0, r1), (c0, c1))
        try:
            current_df = _crop_dataframe_intensity(current_df, crop_window)
            current_frames = list(current_df["intensity"])
            if current_setup is not None:
                _adjust_setup_for_crop(current_setup, crop_window)
                _populate_setup_fields(current_setup, current_frames)
            # Invalidate downstream products built from the un-cropped data.
            current_builder = None
            regrid_volume = None
            regrid_axes = None
            # Refresh the intensity viewer in place on the freshly cropped
            # frames so the user sees the crop immediately, without having to
            # click "View intensity" again.
            if bool(getattr(state, "intensity_slider_show", False)) and current_frames:
                n = len(current_frames)
                state.intensity_frame_max = n - 1
                idx = int(_float(getattr(state, "intensity_frame_index", 0), 0))
                idx = max(0, min(idx, n - 1))
                if int(_float(getattr(state, "intensity_frame_index", 0), 0)) != idx:
                    state.intensity_frame_index = idx
                # Hide the ROI before re-rendering so the stale box (still sized
                # for the previous frame) is never drawn in the new view; it is
                # re-created at the correct place/size right after.
                _roi_set_active(False)
                _show_intensity_frame(idx, reset_camera=True)
                if bool(getattr(state, "roi_show", True)):
                    _roi_init_geometry()
                    _roi_set_active(True)
                    _roi_update_crop_state()
                _roi_render()
            _set_status(
                f"Crop applied: y=[{r0}, {r1}), x=[{c0}, {c1}), detector={c1-c0}x{r1-r0}"
            )
        except Exception as exc:
            _set_status(f"Crop error: {exc}")

    @ctrl.set("stop_task")
    def stop_task(**kwargs):
        nonlocal current_task
        if current_task is not None and not current_task.done():
            current_task.cancel()
            _set_status("Stop requested.")
        else:
            _set_status("Nothing running.")

    @ctrl.set("export_vtr")
    async def export_vtr(**kwargs):
        if current_volume is None or current_axes is None:
            _set_status("No built volume available for export.")
            return

        output_path = Path(_ensure_path(state.export_path))
        if output_path.suffix.lower() != ".vtr":
            output_path = output_path.with_suffix(".vtr")
        loop = asyncio.get_event_loop()
        try:
            _set_status(f"Exporting VTR to {output_path}...")
            await loop.run_in_executor(
                None,
                lambda: write_rsm_volume_to_vtr(
                    current_volume, current_axes, str(output_path), binary=True, compress=True
                ),
            )
            state.export_path = str(output_path)
            _set_status(f"Exported VTR to {output_path}")
        except Exception as exc:
            _set_status(f"Export failed: {exc}")

    @ctrl.set("refresh_rendering")
    def refresh_rendering(**kwargs):
        if regrid_volume is not None and regrid_axes is not None:
            # Re-apply view settings (log/contrast/colormap) by rebuilding the
            # display volume, then refresh slices and push to the client.
            _set_volume_data(regrid_volume, regrid_axes)
        else:
            _update_rendering()
        _update_all_slices()
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    def _fb_target_label(target: str) -> str:
        labels = {
            "setup_path": "YAML setup file",
            "tiff_dir": "TIFF directory",
            "spec_path": "SPEC file",
            "export_path": "Export path",
        }
        return labels.get(_ensure_path(target), "File")

    def _fb_refresh(path: str):
        target_dir = Path(_ensure_path(path)).expanduser()
        if not target_dir.is_dir():
            target_dir = target_dir.parent if target_dir.parent.is_dir() else Path.home()
        mode = _ensure_path(state.fb_mode) or "file"
        entries = []
        try:
            items = sorted(
                target_dir.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except (PermissionError, OSError):
            items = []
        for item in items:
            try:
                is_dir = item.is_dir()
            except OSError:
                continue
            if mode == "dir" and not is_dir:
                continue
            entries.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "is_dir": is_dir,
                    "label": ("\U0001F4C1 " if is_dir else "\U0001F4C4 ") + item.name,
                }
            )
        state.fb_cwd = str(target_dir)
        state.fb_items = entries

    def _fb_open(target, mode="file"):
        target = _ensure_path(target)
        mode = _ensure_path(mode) or "file"
        titles = {
            "setup_path": "Select a YAML setup file",
            "tiff_dir": "Select a TIFF directory",
            "spec_path": "Select a SPEC file",
        }
        state.fb_target = target
        state.fb_mode = mode
        state.fb_title = titles.get(target, "Select a file")
        current = _ensure_path(getattr(state, target, ""))
        start = Path(current).expanduser()
        if current and start.is_dir():
            start_dir = start
        elif current and start.parent.is_dir():
            start_dir = start.parent
        else:
            start_dir = Path.home()
        _fb_refresh(str(start_dir))
        state.fb_show = True

    def _fb_up():
        _fb_refresh(str(Path(_ensure_path(state.fb_cwd)).parent))

    def _fb_click(path, is_dir):
        if is_dir:
            _fb_refresh(_ensure_path(path))
        else:
            target = _ensure_path(state.fb_target)
            if target:
                value = _ensure_path(path)
                setattr(state, target, value)
                _set_status(f"{_fb_target_label(target)} set: {value}")
            state.fb_show = False

    def _fb_select_dir():
        target = _ensure_path(state.fb_target)
        if target:
            value = _ensure_path(state.fb_cwd)
            setattr(state, target, value)
            _set_status(f"{_fb_target_label(target)} set: {value}")
        state.fb_show = False

    def _fb_cancel():
        state.fb_show = False

    # Keep energy (keV) and wavelength (Å) in sync: λ[Å] = 12.398 / E[keV].
    # review_widget.py line 1215
    _ew_sync = {"busy": False}

    @state.change("exp_energy")
    def _on_energy_change(exp_energy=None, **kwargs):
        if _ew_sync["busy"]:
            return
        energy = _float(exp_energy, 0.0)
        if energy > 0:
            _ew_sync["busy"] = True
            state.exp_wavelength = 12.398419843320026 / energy
            _ew_sync["busy"] = False

    @state.change("exp_wavelength")
    def _on_wavelength_change(exp_wavelength=None, **kwargs):
        if _ew_sync["busy"]:
            return
        wavelength = _float(exp_wavelength, 0.0)
        if wavelength > 0:
            _ew_sync["busy"] = True
            state.exp_energy = 12.398419843320026 / wavelength
            _ew_sync["busy"] = False

    # Clamp the CMS angle step to the valid [0, 360] range. Corrects a typed out-of-range value.
    @state.change("cms_angle_step")
    def _on_cms_angle_step_change(cms_angle_step=None, **kwargs):
        clamped = max(0.0, min(360.0, _float(cms_angle_step, 0.0)))
        if clamped != _float(cms_angle_step, None):
            state.cms_angle_step = clamped

    # When a TIFF directory is chosen, seed the scan-number range from the
    # filenames so the first DEFAULT_FRAME_COUNT frames are pre-selected.
    @state.change("tiff_dir")
    def _on_tiff_dir_change(tiff_dir=None, **kwargs):
        scans = _scan_numbers_in_dir(tiff_dir)
        if not scans:
            return
        first = scans[:DEFAULT_FRAME_COUNT]
        state.scan_range = f"{first[0]}-{first[-1]}"
        _set_status(
            f"Found {len(scans)} scan(s); defaulting to the first {len(first)} "
            f"(scans {first[0]}\u2013{first[-1]})."
        )

    with DivLayout(server) as layout:
        # NOTE: VtkRemoteView is instantiated later, inside the right-hand 3D
        # view panel (which has a defined non-zero size). Do NOT create a
        # VtkRemoteView here at the top of the layout: a stray view with no
        # sized container reports a 0x0 client geometry, which makes trame
        # resize the underlying X render window to 0x0 and triggers a fatal
        # "X_ConfigureWindow BadValue (0x0)" error that kills the server.
        html.Style(
            "* { box-sizing: border-box; }"
            "html, body { margin: 0; height: 100%; font-family: sans-serif; }"
            "input, select, button, textarea { font-family: inherit; }"
            # Long rectangular frame slider: flat rectangular track with a
            # rectangular thumb sized to sit inside the bar.
            "input.frame-slider { -webkit-appearance: none; appearance: none; "
            "width: 100%; height: 22px; background: transparent; cursor: pointer; margin: 0; }"
            "input.frame-slider::-webkit-slider-runnable-track { height: 22px; "
            "background: #2a2a2e; border: 1px solid #44444a; border-radius: 2px; }"
            "input.frame-slider::-webkit-slider-thumb { -webkit-appearance: none; "
            "appearance: none; width: 16px; height: 20px; margin-top: 1px; "
            "background: #6aa9ff; border: 1px solid #cfe2ff; border-radius: 2px; }"
            "input.frame-slider::-moz-range-track { height: 22px; background: #2a2a2e; "
            "border: 1px solid #44444a; border-radius: 2px; }"
            "input.frame-slider::-moz-range-thumb { width: 16px; height: 20px; border: none; "
            "background: #6aa9ff; border: 1px solid #cfe2ff; border-radius: 2px; }"
        )
        with html.Div(  # top header bar with the sidebar toggle and title
            style=(
                "display:flex; align-items:center; gap:12px; height:64px; flex:0 0 auto; "
                "padding:8px 16px; border-bottom:1px solid #e0e0e0; "
                "background:#ffffff; color:rgba(0,0,0,0.87); font-family:sans-serif;"
            )
        ):
            html.Button( 
                "\u2630",  # hamburger (3-bar) icon to hide/expand the sidebar
                click="sidebar_open = !sidebar_open",
                title="Toggle sidebar",
                style=(
                    "font-size:1.9rem; line-height:1; background:none; border:none; "
                    "cursor:pointer; padding:4px 10px; color:rgba(0,0,0,0.87);"
                ),
            )
            with html.Div():
                html.H2("Napari ResView Web", style="margin:0; font-size:1.3rem;")
                html.P(
                    "Load experiment profiles, build 3D RSM volumes, and inspect results in the browser.",
                    style="margin:0; font-size:0.9rem; color:#666;",
                )
        with html.Div(  # main container row holding the left control panel and the right 3D view
            style=( 
                "display:flex; flex-direction:row; align-items:stretch; "
                "height:calc(100vh - 64px); overflow:hidden; margin:0; padding:0; "
                "background:#ffffff; color:rgba(0,0,0,0.87); font-family:sans-serif;"
            )
        ):
            with html.Div(
                id="control_panel",
                v_show="sidebar_open",
                style=( # left control panel: width is adjusted via the drag grip on its right edge
                    "width:310px; min-width:200px; max-width:50vw; flex:none; height:100%; padding:16px; "
                    "overflow:auto; background:#ffffff;"
                )
            ):
                # ---- Accordion helpers ------------------------------------
                # Each tab is a horizontal bar (name on the left, arrow on the
                # right). Clicking a bar expands that tab and collapses any
                # other, so only 0 or 1 tab is open at a time.
                _bar = (
                    "display:flex; align-items:center; justify-content:space-between; "
                    "cursor:pointer; padding:12px 14px; margin-bottom:6px; "
                    "background:#f0f0f3; border:1px solid #dcdce0; border-radius:6px; "
                    "font-weight:600; user-select:none;"
                )
                _panel = (
                    "border:1px solid #e0e0e0; border-top:none; border-radius:0 0 6px 6px; "
                    "padding:14px; margin:-6px 0 10px 0; background:#fbfbfc;"
                )
                _lbl = "display:block; margin-top:10px; font-size:0.85rem; color:#444;"
                _inp = "width:100%; margin-top:4px;"
                _btn = "flex:1; padding:10px 8px; cursor:pointer;"

                # ===================== DATA TAB =====================
                with html.Div(
                    style=_bar, click="open_tab = open_tab === 'data' ? '' : 'data'"
                ):
                    html.Span("Data")
                    html.Span("{{ open_tab === 'data' ? '\u25BC' : '\u25B6' }}")
                with html.Div(v_show="open_tab === 'data'", style=_panel):
                    html.Label("Beamline / loader mode", style=_lbl)
                    with html.Select(v_model=("loader_mode", ""), style=_inp):
                        html.Option("CMS", value="CMS")
                        html.Option("ISR", value="ISR")

                    html.Label("TIFF directory", style=_lbl)
                    html.Input(
                        v_model=("tiff_dir", ""),
                        placeholder="Select a TIFF directory",
                        readonly=True,
                        click=(_fb_open, "['tiff_dir', 'dir']"),
                        style=_inp + " cursor:pointer;",
                    )

                    html.Label("Scans", style=_lbl)
                    html.Input(
                        v_model=("scan_range", ""),
                        type="text",
                        placeholder="e.g. 17-20, 30 (blank = all)",
                        style=_inp,
                    )

                    with html.Div(v_show="loader_mode === 'ISR'"):
                        html.Label("SPEC file (ISR only)", style=_lbl)
                        html.Input(
                            v_model=("spec_path", ""),
                            placeholder="Select a SPEC file",
                            readonly=True,
                            click=(_fb_open, "['spec_path', 'file']"),
                            style=_inp + " cursor:pointer;",
                        )

                    # CMS metadata
                    with html.Div(v_show="loader_mode === 'CMS'"):
                        html.Strong("CMS metadata", style="display:block; margin-top:14px;")
                        html.Label("Angle step (\u00b0)", style=_lbl)
                        html.Input(
                            v_model=("cms_angle_step", ""),
                            type="number", step="0.01", min="0", max="360",
                            style=_inp,
                        )

                    # Experimental setup
                    html.Strong("Experimental Setup", style="display:block; margin-top:14px;")
                    html.Label("Distance (m)", style=_lbl)
                    html.Input(v_model=("exp_distance", ""), type="number", step="1e-6", style=_inp)
                    html.Label("Pitch (m)", style=_lbl)
                    html.Input(v_model=("exp_pitch", ""), type="number", step="1e-9", style=_inp)
                    html.Label("Detector height (px)", style=_lbl)
                    html.Input(v_model=("exp_det_h", ""), type="number", step="1", style=_inp)
                    html.Label("Detector width (px)", style=_lbl)
                    html.Input(v_model=("exp_det_w", ""), type="number", step="1", style=_inp)
                    html.Label("Beam center height (px)", style=_lbl)
                    html.Input(v_model=("exp_bc_h", ""), type="number", step="1", style=_inp)
                    html.Label("Beam center width (px)", style=_lbl)
                    html.Input(v_model=("exp_bc_w", ""), type="number", step="1", style=_inp)
                    html.Label("Energy (keV)", style=_lbl)
                    html.Input(v_model=("exp_energy", ""), type="number", step="1e-3", style=_inp)
                    html.Label("Wavelength (\u00c5)", style=_lbl)
                    html.Input(v_model=("exp_wavelength", ""), type="number", step="1e-3", style=_inp)

                    with html.Div(style="display:flex; gap:8px; margin-top:14px;"):
                        html.Button("\U0001F4C2 Load Data", click=ctrl.load_data, style=_btn)
                        html.Button("\U0001F4C8 View Intensity", click=ctrl.view_intensity, style=_btn)

                    # Crop
                    html.Strong("Crop", style="display:block; margin-top:16px;")
                    html.Label("Crop rows (top / bottom)", style=_lbl)
                    with html.Div(style="display:flex; gap:8px;"):
                        html.Input(v_model=("crop_row_min", ""), type="number", placeholder="top", style="flex:1; min-width:0;")
                        html.Input(v_model=("crop_row_max", ""), type="number", placeholder="bottom", style="flex:1; min-width:0;")
                    html.Label("Crop cols (left / right)", style=_lbl)
                    with html.Div(style="display:flex; gap:8px;"):
                        html.Input(v_model=("crop_col_min", ""), type="number", placeholder="left", style="flex:1; min-width:0;")
                        html.Input(v_model=("crop_col_max", ""), type="number", placeholder="right", style="flex:1; min-width:0;")
                    with html.Label(style=_lbl + " display:flex; align-items:center; gap:6px; cursor:pointer;"):
                        html.Input(type="checkbox", v_model=("roi_show", True), style="margin:0; width:auto;")
                        html.Span("ROI")
                    html.Button("\U0001F532 Crop from ROI", click=ctrl.crop_from_roi, style="width:100%; margin-top:12px; padding:10px 8px; cursor:pointer;")

                # ===================== BUILD TAB =====================
                with html.Div(
                    style=_bar, click="open_tab = open_tab === 'build' ? '' : 'build'"
                ):
                    html.Span("Build")
                    html.Span("{{ open_tab === 'build' ? '\u25BC' : '\u25B6' }}")
                with html.Div(v_show="open_tab === 'build'", style=_panel):
                    html.Label("Space", style=_lbl)
                    with html.Select(v_model=("space", ""), style=_inp):
                        html.Option("Q-space", value="q")
                        html.Option("HKL", value="hkl")
                    html.Label("Grid size", style=_lbl)
                    html.Input(v_model=("grid_size", ""), type="number", min="16", step="8", style=_inp)
                    html.Label("Normalize", style=_lbl)
                    with html.Select(v_model=("normalize", ""), style=_inp):
                        html.Option("mean", value="mean")
                        html.Option("sum", value="sum")
                    with html.Div(style="display:flex; gap:8px; margin-top:14px;"):
                        html.Button("\U0001F527 Build RSM", click=ctrl.build_rsm, style=_btn)
                        html.Button("\U0001F9EE Regrid", click=ctrl.regrid, style=_btn)

                # ===================== VIEW TAB =====================
                with html.Div(
                    style=_bar, click="open_tab = open_tab === 'view' ? '' : 'view'"
                ):
                    html.Span("View")
                    html.Span("{{ open_tab === 'view' ? '\u25BC' : '\u25B6' }}")
                with html.Div(v_show="open_tab === 'view'", style=_panel):
                    with html.Div(style="display:flex; gap:8px;"):
                        html.Button("\U0001F52D View RSM", click=ctrl.view_rsm, style=_btn)
                        html.Button("\u21BB Refresh", click=ctrl.refresh_rendering, style=_btn)
                        html.Button("\u23F9 Stop", click=ctrl.stop_task, style=_btn)

                    with html.Div(style="display:flex; align-items:center; margin-top:14px;"):
                        html.Input(v_model=("log_view", ""), type="checkbox", style="margin-right:8px;")
                        html.Span("Log view")

                    html.Label("Colormap", style=_lbl)
                    with html.Select(v_model=("colormap", ""), style=_inp):
                        for name in COLORMAP_NAMES:
                            html.Option(name, value=name)
                    html.Label("Rendering", style=_lbl)
                    with html.Select(v_model=("rendering", ""), style=_inp):
                        html.Option("attenuated_mip", value="attenuated_mip")
                        html.Option("mip", value="mip")
                        html.Option("translucent", value="translucent")
                    html.Label("Contrast low (%)", style=_lbl)
                    html.Input(v_model=("contrast_lo", ""), type="number", min="0", max="100", step="0.1", style=_inp)
                    html.Label("Contrast high (%)", style=_lbl)
                    html.Input(v_model=("contrast_hi", ""), type="number", min="0", max="100", step="0.1", style=_inp)

                    html.Strong("Export", style="display:block; margin-top:16px;")
                    html.Label("Export path", style=_lbl)
                    html.Input(v_model=("export_path", ""), placeholder="/path/to/output.vtr", style=_inp)
                    html.Button("\U0001F4BE Export VTR", click=ctrl.export_vtr, style="width:100%; margin-top:12px; padding:10px 8px; cursor:pointer;")

                # ===================== ANALYSIS TAB =====================
                with html.Div(
                    style=_bar, click="open_tab = open_tab === 'analysis' ? '' : 'analysis'"
                ):
                    html.Span("Analysis")
                    html.Span("{{ open_tab === 'analysis' ? '\u25BC' : '\u25B6' }}")
                with html.Div(v_show="open_tab === 'analysis'", style=_panel):
                    # --- Orthogonal slicing ---
                    html.Strong("Orthogonal Slicing", style="display:block;")
                    for ax, lbl in (("x", "X"), ("y", "Y"), ("z", "Z")):
                        with html.Div(style="display:flex; align-items:center; gap:8px; margin-top:8px;"):
                            html.Input(v_model=(f"slice_{ax}_show", ""), type="checkbox", change=ctrl.update_slices)
                            html.Span(f"{lbl}", style="width:14px;")
                            html.Input(
                                v_model=(f"slice_{ax}_pos", ""), type="range", min="0", max="100", step="1",
                                change=ctrl.update_slices, style="flex:1;",
                            )
                            html.Span("{{ " + f"slice_{ax}_pos" + " }}%", style="width:38px; font-size:0.8rem;")
                    html.Label("Slice opacity", style=_lbl)
                    html.Input(v_model=("slice_opacity", ""), type="number", min="0", max="1", step="0.1", change=ctrl.update_slices, style=_inp)
                    html.Label("Slice colormap", style=_lbl)
                    with html.Select(v_model=("slice_cmap", ""), change=ctrl.update_slices, style=_inp):
                        for name in ["turbo", "viridis", "inferno", "plasma", "gray", "hsv"]:
                            html.Option(name, value=name)
                    with html.Div(style="display:flex; align-items:center; margin-top:10px;"):
                        html.Input(v_model=("slice_show_border", ""), type="checkbox", change=ctrl.update_slices, style="margin-right:8px;")
                        html.Span("Show border")

                    # --- Cylindrical slicing ---
                    html.Strong("Cylindrical Slicing (Q space)", style="display:block; margin-top:18px;")
                    with html.Div(style="display:flex; align-items:center; margin-top:8px;"):
                        html.Input(v_model=("cyl_show", ""), type="checkbox", change=ctrl.update_slices, style="margin-right:8px;")
                        html.Span("Show cylinder")
                    html.Label("Cylinder radius (\u00c5\u207b\u00b9)", style=_lbl)
                    html.Input(v_model=("cyl_radius", ""), type="number", min="0", step="0.01", change=ctrl.update_slices, style=_inp)
                    html.Label("Angular samples", style=_lbl)
                    html.Input(v_model=("cyl_samples", ""), type="number", min="16", max="360", step="8", change=ctrl.update_slices, style=_inp)
                    html.Label("Opacity", style=_lbl)
                    html.Input(v_model=("cyl_opacity", ""), type="number", min="0", max="1", step="0.1", change=ctrl.update_slices, style=_inp)
                    html.Label("Colormap", style=_lbl)
                    with html.Select(v_model=("cyl_cmap", ""), change=ctrl.update_slices, style=_inp):
                        for name in ["turbo", "viridis", "inferno", "plasma", "gray", "hsv"]:
                            html.Option(name, value=name)

                    # --- Spherical slicing ---
                    html.Strong("Spherical Slicing (Q space)", style="display:block; margin-top:18px;")
                    with html.Div(style="display:flex; align-items:center; margin-top:8px;"):
                        html.Input(v_model=("sph_show", ""), type="checkbox", change=ctrl.update_slices, style="margin-right:8px;")
                        html.Span("Show sphere")
                    html.Label("Sphere radius (\u00c5\u207b\u00b9)", style=_lbl)
                    html.Input(v_model=("sph_radius", ""), type="number", min="0", step="0.01", change=ctrl.update_slices, style=_inp)
                    html.Label("Angular samples", style=_lbl)
                    html.Input(v_model=("sph_samples", ""), type="number", min="16", max="180", step="8", change=ctrl.update_slices, style=_inp)
                    html.Label("Opacity", style=_lbl)
                    html.Input(v_model=("sph_opacity", ""), type="number", min="0", max="1", step="0.1", change=ctrl.update_slices, style=_inp)
                    html.Label("Colormap", style=_lbl)
                    with html.Select(v_model=("sph_cmap", ""), change=ctrl.update_slices, style=_inp):
                        for name in ["turbo", "viridis", "inferno", "plasma", "gray", "hsv"]:
                            html.Option(name, value=name)

                # ---- Status (always visible, below the accordion) ----------
                html.Hr(style="border-color:#e0e0e0; margin:16px 0;")
                html.Strong("Status")
                # Scrollable status history. `flex-direction:column-reverse`
                # keeps the view pinned to the newest message (rendered last in
                # the reversed list) while still letting the user scroll up to
                # read past messages.
                with html.Div(
                    style=(
                        "display:flex; flex-direction:column-reverse; "
                        "max-height:180px; overflow-y:auto; background:#f5f5f5; "
                        "padding:12px; border-radius:6px; margin-top:8px; "
                        "color:rgba(0,0,0,0.87); border:1px solid #e0e0e0; "
                        "min-height:90px;"
                    ),
                ):
                    html.Div(
                        "{{ line }}",
                        v_for="(line, i) in [...status_log].reverse()",
                        key="i",
                        style=(
                            "white-space:pre-wrap; font-family:monospace; "
                            "font-size:0.85rem; padding:2px 0;"
                        ),
                    )
                with html.P(style="font-size:0.90rem; margin-top:12px; color:#666;"):
                    html.Span("Scalar range: ")
                    html.Strong(v_text="scalar_range")
                    html.Br()
                    html.Span("Dimensions: ")
                    html.Strong(v_text="volume_dims")
            # Drag grip: click-and-drag the vertical 3-dot handle to resize the
            # left panel. Resizing is handled entirely client-side (no server
            # round-trip) by adjusting the panel element's width on pointermove,
            # clamped between 200px and half the viewport width.
            html.Div(
                "\u22EE",
                v_show="sidebar_open",
                title="Drag to resize panel",
                mousedown=(
                    "$event.preventDefault();"
                    "var p=document.getElementById('control_panel');"
                    "if(!p){return;}"
                    "var sx=$event.clientX;"
                    "var sw=p.getBoundingClientRect().width;"
                    "function mv(e){"
                    "var w=Math.max(200, Math.min(window.innerWidth/2, sw+e.clientX-sx));"
                    "p.style.width=w+'px';"
                    "}"
                    "function up(){"
                    "document.removeEventListener('pointermove',mv);"
                    "document.removeEventListener('pointerup',up);"
                    "document.body.style.userSelect='';"
                    "document.body.style.cursor='';"
                    "}"
                    "document.body.style.userSelect='none';"
                    "document.body.style.cursor='col-resize';"
                    "document.addEventListener('pointermove',mv);"
                    "document.addEventListener('pointerup',up);"
                ),
                style=(
                    "flex:0 0 16px; align-self:stretch; cursor:col-resize; "
                    "display:flex; align-items:center; justify-content:center; "
                    "background:#f0f0f0; border-right:1px solid #e0e0e0; "
                    "color:#9e9e9e; font-size:1.2rem; line-height:1; "
                    "user-select:none; touch-action:none;"
                ),
            )
            with html.Div(style="flex:1; min-width:0; height:100%; background:#0f0f12; display:flex; flex-direction:column;"):
                # Instantiate the remote view that streams the off-screen VTK
                # render window to the browser. This reassigns the `remote_view`
                # closure variable that _update_rendering()/_set_volume_data()
                # read, so live renders now have a surface to push to.
                with html.Div(style="flex:1; min-height:0; position:relative;"):
                    remote_view = VtkRemoteView(render_window, interactive_ratio=1)
                # Frame slider: visible after "View Intensity" so the user can
                # scrub through the loaded frame range one frame at a time.
                with html.Div(
                    v_show="intensity_slider_show",
                    style=(
                        "flex:0 0 auto; display:flex; align-items:center; gap:12px; "
                        "padding:10px 16px; background:#16161a; color:#dddddd; "
                        "border-top:1px solid #2a2a2e; font-family:sans-serif;"
                    ),
                ):
                    _pbtn = (
                        "flex:0 0 auto; width:32px; height:28px; padding:0; "
                        "display:flex; align-items:center; justify-content:center; "
                        "font-size:1rem; line-height:1; cursor:pointer; "
                        "background:#2a2a2e; color:#dddddd; "
                        "border:1px solid #44444a; border-radius:4px;"
                    )
                    html.Button(
                        "\u23EE",
                        click=ctrl.prev_frame,
                        title="Previous frame",
                        style=_pbtn,
                    )
                    html.Button(
                        "{{ intensity_playing ? '\u23F9' : '\u25B6' }}",
                        click=ctrl.toggle_play,
                        title="Play / Stop (10 fps, loops)",
                        style=_pbtn,
                    )
                    html.Button(
                        "\u23ED",
                        click=ctrl.next_frame,
                        title="Next frame",
                        style=_pbtn,
                    )
                    html.Input(
                        type="range",
                        classes="frame-slider",
                        v_model=("intensity_frame_index", 0),
                        min=0,
                        max=("intensity_frame_max", 0),
                        step=1,
                        style="flex:1; min-width:0;",
                    )
                    html.Span(
                        "{{ intensity_frame_index }} / {{ intensity_frame_max }}",
                        style=(
                            "min-width:80px; text-align:right; font-size:0.85rem; "
                            "font-variant-numeric:tabular-nums;"
                        ),
                    )

        # File browser modal dialog
        with html.Div(  
            v_if="fb_show",
            style=(
                "position:fixed; inset:0; background:rgba(0,0,0,0.6); display:flex; "
                "align-items:center; justify-content:center; z-index:1000; font-family:sans-serif;"
            ),
        ):
            with html.Div(
                style=(
                    "width:600px; max-width:90vw; max-height:70vh; background:#ffffff; "
                    "border:1px solid #e0e0e0; border-radius:8px; padding:16px; box-sizing:border-box; "
                    "display:flex; flex-direction:column; color:rgba(0,0,0,0.87);"
                ),
            ):
                html.H3(v_text="fb_title", style="margin:0 0 8px 0;")
                html.Div(
                    v_text="fb_cwd",
                    style="font-size:0.85rem; color:#666; margin-bottom:8px; word-break:break-all;",
                )
                with html.Div(style="display:flex; gap:8px; margin-bottom:8px;"): 
                    html.Button(
                        "\u2B06 Back",
                        click=_fb_up,
                        style="padding:6px 12px;",
                    )
                    html.Button(
                        "Select",
                        v_if="fb_mode === 'dir'",
                        click=_fb_select_dir,
                        style="padding:6px 12px;",
                    )
                with html.Div(
                    style=(
                        "flex:1; overflow:auto; background:#fafafa; border:1px solid #e0e0e0; "
                        "border-radius:6px; min-height:200px;"
                    ),
                ):
                    html.Div(
                        "{{ item.label }}",
                        v_for="(item, index) in fb_items",
                        key="index",
                        click=(_fb_click, "[item.path, item.is_dir]"),
                        style=(
                            "padding:6px 10px; cursor:pointer; border-bottom:1px solid #eeeeee; "
                            "white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
                        ),
                    )
                with html.Div(style="display:flex; justify-content:flex-end; gap:8px; margin-top:12px;"):
                    html.Button(
                        "Cancel",
                        click=_fb_cancel,
                        style="padding:6px 12px;",
                    )

    return server


def run_server(port: int = 0, host: str = "localhost", open_browser: bool = True):
    server = create_server()
    server.start(port=port, host=host, open_browser=open_browser)


if __name__ == "__main__":
    run_server()