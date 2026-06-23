import os
import asyncio
import time
from pathlib import Path
import pathlib
from typing import Optional, Tuple
import importlib
import sys
import types

import numpy as np
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkCommonCore import vtkLookupTable # maps raw scalar data to colors (RGBA)
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
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
    state.setdefault("cms_angle_step", 0.30) # range from 0 to 360
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

    state.setdefault("blend_mode", 0)
    state.setdefault("shade", True)
    state.setdefault("opacity_scale", 1.0)
    state.setdefault("colormap", "viridis")

    state.setdefault("status", "Ready")
    state.setdefault("status_log", ["Ready"])
    state.setdefault("scalar_range", "—")
    state.setdefault("volume_dims", "—")
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

    current_volume = None
    current_axes = None
    current_builder = None
    # Robust display range (log-scaled) used to build the transfer functions.
    render_range = None
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

        if int(_float(state.blend_mode, 0)) == 1:
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
        nonlocal current_volume, current_axes, render_range
        current_volume = np.asarray(volume, dtype=np.float32)
        current_axes = axes

        # Log-compress for display so the high-dynamic-range RSM is visible,
        # and derive the transfer-function range from robust percentiles of the
        # log-scaled data (mirrors the napari viewer). The raw current_volume is
        # kept untouched for VTR export.
        display_volume = _log1p_clip(current_volume)
        render_range = _robust_percentiles(display_volume, (1.0, 99.8))

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
        renderer.ResetCamera()
        _update_rendering()
        return current_volume, current_axes

    @ctrl.set("build_rsm")
    async def build_rsm(**kwargs):
        nonlocal current_builder
        _set_status("Loading data...")
        setup_path = Path(_ensure_path(state.setup_path)).expanduser()
        tiff_dir = Path(_ensure_path(state.tiff_dir)).expanduser()
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

        crop_window = _crop_window_from_state(state)
        loop = asyncio.get_event_loop()

        try:
            # 1) Load experiment + TIFF frames (blocking I/O off the event loop)
            _set_status("Loading experiment and TIFF frames...")
            setup, ub, df = await loop.run_in_executor(
                None, _load_experiment, loader_mode, crop_window
            )

            # grid_size = max(16, int(_float(state.grid_size, 90)))
            # if bool(state.stream_build):
            #     # Memory-frugal streaming path. Constructing the builder is
            #     # cheap (it only configures the xrayutilities geometry); the
            #     # expensive per-pixel mapping happens inside the two streaming
            #     # passes below, one frame at a time, so the full Q cube is
            #     # never held in RAM.
            #     _set_status("Preparing reciprocal-space mapping...")
            #     current_builder = await loop.run_in_executor(
            #         None, _build_builder, setup, ub, df
            #     )

                # # 2a) Pass 1: scan the data extent to fix the grid bins.
                # _set_status("Scanning data extent (pass 1/2)...")
                # ranges = await loop.run_in_executor(
                #     None, _compute_ranges, current_builder
                # )

                # 2b) Pass 2: bin every frame into the 3D grid.
            #     _set_status("Binning frames into 3D grid (pass 2/2)...")
            #     volume, axes = await loop.run_in_executor(
            #         None, _regrid_stream, current_builder, grid_size, ranges
            #     )
            # else:
            #     # Original path: materializes the full Q/HKL cube. Faster for
            #     # small datasets but can exhaust RAM on large ones.
            #     _set_status("Computing Q/HKL mapping...")
            #     current_builder = await loop.run_in_executor(
            #         None, _compute_builder, setup, ub, df
            #     )

            #     _set_status("Regridding to 3D volume...")
            #     volume, axes = await loop.run_in_executor(
            #         None, _regrid_volume, current_builder, grid_size
            #     )

            # 2) Compute Q/HKL mapping
            _set_status("Computing Q/HKL mapping...")
            current_builder = await loop.run_in_executor(
                None, _compute_builder, setup, ub, df
            )

            # 3) Regrid to a 3D volume
            grid_size = max(16, int(_float(state.grid_size, 90)))
            _set_status("Regridding to 3D volume...")
            volume, axes = await loop.run_in_executor(
                None, _regrid_volume, current_builder, grid_size
            )

            # 4) Push into the renderer
            _set_status("Updating 3D view...")
            _set_volume_data(volume, axes)
            state.scalar_range = (
                f"{float(np.nanmin(volume)):.4g} … {float(np.nanmax(volume)):.4g}"
            )
            state.volume_dims = (
                f"{volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]}"
            )
            state.export_path = str(
                Path(_ensure_path(state.export_path) or Path.cwd() / "rsm_output.vtr")
            )
            _set_status("RSM volume built.")
        except Exception as exc:
            _set_status(f"Error: {exc}")

    def _load_experiment(loader_mode, crop_window):
        if loader_mode == "ISR":
            spec_path = Path(_ensure_path(state.spec_path)).expanduser()
            setup_path = Path(_ensure_path(state.setup_path)).expanduser()
            tiff_dir = Path(_ensure_path(state.tiff_dir)).expanduser()
            loader = RSMDataLoader_ISR(
                str(spec_path),
                str(setup_path),
                str(tiff_dir),
                use_dask=False,
            )
            setup, ub, df = loader.load()
            if crop_window is not None:
                df = _crop_dataframe_intensity(df, crop_window)
                _adjust_setup_for_crop(setup, crop_window)
        else:
            setup_path = Path(_ensure_path(state.setup_path)).expanduser()
            tiff_dir = Path(_ensure_path(state.tiff_dir)).expanduser()
            loader = RSMDataloader_CMS(
                str(setup_path),
                str(tiff_dir),
                crop_window=crop_window,
            )
            setup, ub, df = loader.load()
            if crop_window is not None:
                _adjust_setup_for_crop(setup, crop_window)
        return setup, ub, df

    def _compute_builder(setup, ub, df):
        builder = RSMBuilder(setup, ub, df, ub_includes_2pi=True)
        builder.compute_full(verbose=False)
        return builder

    def _regrid_volume(builder, grid_size):
        return builder.regrid_xu(
            space=_ensure_path(state.space) or "q",
            grid_shape=(grid_size, grid_size, grid_size),
            normalize="mean",
        )

    # --- Streaming (low-memory) build helpers -------------------------------
    # def _build_builder(setup, ub, df):
    #     # Cheap: only sets up the xrayutilities QConversion geometry; does NOT
    #     # compute or allocate the per-pixel Q/HKL arrays.
    #     return RSMBuilder(setup, ub, df, ub_includes_2pi=True)

    # def _compute_ranges(builder):
    #     return builder.compute_ranges(_ensure_path(state.space) or "q")

    # def _regrid_stream(builder, grid_size, ranges):
    #     return builder.regrid_stream(
    #         space=_ensure_path(state.space) or "q",
    #         grid_shape=(grid_size, grid_size, grid_size),
    #         ranges=ranges,
    #         normalize="mean",
    #     )

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
        _update_rendering()

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
                html.Label("Loader mode")
                with html.Select(
                    v_model=("loader_mode", ""),
                    style="width:100%; margin-bottom:12px;",
                ):
                    html.Option("CMS", value="CMS")
                    html.Option("ISR", value="ISR")
                # html.Label("Experiment YAML setup file")
                # html.Input(
                #     v_model=("setup_path", ""), # the second arg sets the initial value in the input field
                #     placeholder="Select a YAML setup file",
                #     readonly=True,
                #     click=(_fb_open, "['setup_path', 'file']"),
                #     style="width:100%; margin-bottom:12px; cursor:pointer;",
                # )
                html.Label("TIFF directory")
                html.Input(
                    v_model=("tiff_dir", ""),
                    placeholder="Select a TIFF directory",
                    readonly=True,
                    click=(_fb_open, "['tiff_dir', 'dir']"),
                    style="width:100%; margin-bottom:12px; cursor:pointer;",
                )
                html.Label("SPEC file (ISR only)")
                html.Input(
                    v_model=("spec_path", ""),
                    placeholder="Select a SPEC file",
                    readonly=True,
                    click=(_fb_open, "['spec_path', 'file']"),
                    style="width:100%; margin-bottom:12px; cursor:pointer;",
                )
                html.Label("Space")
                with html.Select(
                    v_model=("space", ""),
                    style="width:100%; margin-bottom:12px;",
                ):
                    html.Option("Q-space", value="q")
                    html.Option("HKL", value="hkl")
                html.Label("Grid size")
                html.Input(
                    v_model=("grid_size", ""),
                    type="number",
                    min="16",
                    step="8",
                    style="width:100%; margin-bottom:12px;",
                )
                # html.Label("Low-memory build")
                # with html.Div(style="display:flex; align-items:center; margin-bottom:12px;"):
                #     html.Input(v_model=("stream_build", ""), type="checkbox", style="margin-right:8px;")
                #     html.Span("Stream frames (avoids large RAM use)")
                html.Label("Colormap")
                with html.Select(
                    v_model=("colormap", ""),
                    style="width:100%; margin-bottom:12px;",
                ):
                    for name in COLORMAP_NAMES:
                        html.Option(name, value=name)
                html.Label("Opacity scale")
                html.Input(
                    v_model=("opacity_scale", ""),
                    type="number",
                    min="0.0",
                    max="1.0",
                    step="0.1",
                    style="width:100%; margin-bottom:12px;",
                )
                html.Label("Blend mode")
                with html.Select(
                    v_model=("blend_mode", ""),
                    style="width:100%; margin-bottom:12px;",
                ):
                    html.Option("Composite", value="0")
                    html.Option("Maximum intensity", value="1")
                html.Label("Crop enabled")
                with html.Div(style="display:flex; align-items:center; margin-bottom:12px;"):
                    html.Input(v_model=("crop_enabled", ""), type="checkbox", style="margin-right:8px;")
                    html.Span("Enable crop window")
                html.Label("Crop rows")
                with html.Div(style="display:flex; gap:8px; margin-bottom:12px;"):
                    html.Input(
                        v_model=("crop_row_min", ""),
                        type="number",
                        placeholder="top",
                        style="flex:1; min-width:0;",
                    )
                    html.Input(
                        v_model=("crop_row_max", ""),
                        type="number",
                        placeholder="bottom",
                        style="flex:1; min-width:0;",
                    )
                html.Label("Crop cols")
                with html.Div(style="display:flex; gap:8px; margin-bottom:12px;"):
                    html.Input(
                        v_model=("crop_col_min", ""),
                        type="number",
                        placeholder="left",
                        style="flex:1; min-width:0;",
                    )
                    html.Input(
                        v_model=("crop_col_max", ""),
                        type="number",
                        placeholder="right",
                        style="flex:1; min-width:0;",
                    )
                with html.Div():
                    html.Button("Load and build RSM", click=ctrl.build_rsm, style="width:100%; margin-bottom:12px; padding:12px 8px;")
                    html.Button("Export VTR", click=ctrl.export_vtr, style="width:100%; margin-bottom:12px; padding:12px 8px;")
                html.Label("Export path")
                html.Input(
                    v_model=("export_path", ""),
                    placeholder="/path/to/output.vtr",
                    style="width:100%; margin-top:12px;",
                )
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
                remote_view = VtkRemoteView(render_window, interactive_ratio=1)

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