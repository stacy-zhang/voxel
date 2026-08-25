import os
import asyncio
import base64
import gc
import io
import math
import time
from pathlib import Path
import pathlib
from typing import Optional, Tuple
import importlib
import sys
import types
import argparse

import matplotlib
matplotlib.use("Agg") # use non-interactive backend for colormap rendering (Agg means "Anti-Grain Geometry")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.cm as mpl_cm            # noqa: E402
import numpy as np

from trame.app import get_server # Trame server framework
from trame.ui.vuetify3 import SinglePageWithDrawerLayout 
from trame.ui.vuetify3 import SinglePageLayout 
from trame.widgets import vuetify3 as v3, vtk as vtk_widgets, html

from vtkmodules.vtkCommonDataModel import (
    vtkPlane, # for slicing the volume with axis-aligned planes
    vtkPiecewiseFunction, # for defining the opacity transfer function (mapping scalar values to opacity)
    vtkImageData, # in-memory 3D image the pipeline's NumPy result is wrapped in for display
)

from vtkmodules.vtkFiltersModeling import vtkOutlineFilter # to create a wireframe box around the volume
from vtkmodules.vtkIOImage import vtkTIFFReader # to read 3D TIFF files as vtkImageData
from vtkmodules.vtkRenderingCore import (
    vtkActor, # for rendering the outline box
    vtkColorTransferFunction, # for defining the color transfer function (mapping scalar values to colors)
    vtkPolyDataMapper, # to map the outline geometry to graphics primitives
    vtkRenderer, # the main rendering engine that manages the scene
    vtkRenderWindow, # the window that displays the rendered scene
    vtkRenderWindowInteractor, # handles user interaction (mouse, keyboard) with the render window
    vtkVolume, # the actor type for volume rendering
    vtkVolumeProperty, # holds the properties of the volume rendering (color, opacity, shading, etc.)
    vtkImageSlice,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper # the GPU-accelerated volume mapper that does the actual rendering of the 3D data
from vtkmodules.vtkRenderingImage import vtkImageResliceMapper # maps a 3D volume to a 2D slice plane
from vtkmodules.util import numpy_support

from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera # allows the user to rotate/zoom/pan the view with mouse interactions (trackball style)
import vtkmodules.vtkInteractionStyle  # noqa – required
import vtkmodules.vtkRenderingOpenGL2  # noqa – required
# the above two imports are needed to ensure the appropriate VTK rendering and interaction styles are registered, even if we don't directly reference them in the code

from voxel.features.base import FeatureContext
from voxel.features.tomo import ui as tomo_ui
from voxel.features.tomo.feature import TomographyFeature
from voxel.features.tomo import pipeline as tomo_pipeline


def create_tomo_server():
    ################
    # VTK pipeline #
    ################
    renderer = vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)  # dark gray background

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1024, 768) # initial window size (can be resized by user)
    render_window.SetOffScreenRendering(1) # enable off-screen rendering for web-based applications (no native window needed)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    interactor.EnableRenderOff()

    volume_mapper = vtkSmartVolumeMapper()
    volume_mapper.SetBlendModeToComposite() # default blend mode (user can change to max/min/avg intensity)

    color_tf = vtkColorTransferFunction()
    opacity_tf = vtkPiecewiseFunction()

    vol_property = vtkVolumeProperty()
    vol_property.SetIndependentComponents(True)
    vol_property.SetInterpolationTypeToLinear() # smooth interpolation between voxels for better quality (can be set to nearest for sharper but blockier look)
    vol_property.SetColor(color_tf)
    vol_property.SetScalarOpacity(opacity_tf)
    vol_property.ShadeOn() # enable lighting by default for better depth perception, user can toggle off if desired
    vol_property.SetAmbient(0.2)
    vol_property.SetDiffuse(0.7)
    vol_property.SetSpecular(0.3)
    vol_property.SetSpecularPower(10.0) # shininess of the specular highlight (higher = smaller, sharper highlight)

    volume_actor = vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(vol_property)
    volume_actor.VisibilityOff() # start with volume hidden until a file is loaded
    renderer.AddVolume(volume_actor)

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

    tilt_slice_plane = vtkPlane()
    tilt_slice_plane.SetNormal(0, 0, 1) # z
    tilt_slice_plane.SetOrigin(0, 0, 0)
    slice_mappers["z"].SetSlicePlane(tilt_slice_plane)

    # Clipping planes for axis-aligned slicing (6 planes: +X, -X, +Y, -Y, +Z, -Z)
    clip_planes = {}
    for axis_name, normal in [("x_min", (1,0,0)), ("x_max", (-1,0,0)),
                            ("y_min", (0,1,0)), ("y_max", (0,-1,0)),
                            ("z_min", (0,0,1)), ("z_max", (0,0,-1))]:
        plane = vtkPlane()
        plane.SetNormal(*normal)
        plane.SetOrigin(0, 0, 0)
        clip_planes[axis_name] = plane
        volume_mapper.AddClippingPlane(plane)

    # Outline (bounding box)
    outline_filter = vtkOutlineFilter()
    outline_mapper = vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
    outline_actor = vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0) # white outline
    outline_actor.GetProperty().SetLineWidth(1)
    outline_actor.VisibilityOff() # start with outline hidden until a file is loaded
    renderer.AddActor(outline_actor)

    renderer.ResetCamera() # reset camera to fit the scene

    # Colormap presets sampled from matplotlib
    _MPL_CMAP_NAMES = [
        # Perceptually uniform
        "viridis", "plasma", "inferno", "magma", "cividis",
        # Sequential
        "gray", "hot", "bone", "copper", "cool", "spring", "summer",
        "autumn", "winter", "YlOrRd", "YlGnBu", "RdPu",
        # Diverging (ParaView favourites)
        "coolwarm", "bwr", "seismic", "RdBu", "RdYlBu", "RdYlGn",
        # Qualitative / misc
        "jet", "rainbow", "turbo", "gnuplot", "gnuplot2", "nipy_spectral",
    ]

    # Keep only the ones actually available in this matplotlib version
    def _cmap_available(name: str) -> bool:
        try:
            matplotlib.colormaps[name]
            return True
        except (ValueError, KeyError):
            return False

    MPL_CMAP_NAMES = [n for n in _MPL_CMAP_NAMES if _cmap_available(n)]
    if not MPL_CMAP_NAMES:
        MPL_CMAP_NAMES = ["viridis"] # fallback to viridis if none of the preferred colormaps are available

    _N_CMAP_SAMPLES = 64  # number of RGB samples per colormap
    # an RGB sample is a tuple of (fraction, r, g, b) where fraction is in [0,1] and r,g,b are in [0,1]
    # fraction is the position along the colormap gradient (0 = start, 1 = end), and r,g,b are the corresponding color values at that position

    def _sample_mpl_colormap(name: str, n: int = _N_CMAP_SAMPLES) -> list[tuple[float, float, float, float]]:
        """Return [(frac, r, g, b), ...] sampled from a matplotlib colormap."""
        cmap = matplotlib.colormaps[name]
        return [(i / (n - 1), *cmap(i / (n - 1))[:3]) for i in range(n)]


    # Pre-sample all colormaps
    COLORMAPS: dict[str, list[tuple[float, float, float, float]]] = {
        name: _sample_mpl_colormap(name) for name in MPL_CMAP_NAMES
    }

    COLORMAP_ITEMS = [{"title": name, "value": name} for name in MPL_CMAP_NAMES]


    def _generate_colormap_preview(name: str, width: int = 300, height: int = 20) -> str:
        """Render a colormap bar as a base64 PNG data-URI."""
        cmap = matplotlib.colormaps[name]
        gradient = np.linspace(0, 1, width).reshape(1, -1)
        gradient = np.vstack([gradient] * height)

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"


    # Pre-render preview images
    COLORMAP_PREVIEWS: dict[str, str] = {
        name: _generate_colormap_preview(name) for name in MPL_CMAP_NAMES
    }


    ##########################
    # Trame server and state #
    ##########################
    server = get_server(name="tomo_app")
    state, ctrl = server.state, server.controller

    state.drawer_split = 0.5  # fraction of drawer height given to the Pipeline section

    state.setdefault("tiff_path", "")
    state.setdefault("opacity_scale", 1.0)
    state.setdefault("shade", True)
    state.setdefault("scalar_range", "—")
    state.setdefault("dimensions", "—")
    state.setdefault("loaded", False)

    # Colormap / contrast / blend / lighting
    state.setdefault("colormap", "coolwarm")
    state.setdefault("colormap_preview", COLORMAP_PREVIEWS.get("coolwarm", ""))
    state.setdefault("contrast_low", 0.0)   # fraction of data range [0..1]
    state.setdefault("contrast_high", 1.0)
    state.setdefault("blend_mode", 0)        # 0=Composite, 1=MaxIP, 2=MinIP, 3=Average
    state.setdefault("ambient", 0.2)
    state.setdefault("diffuse", 0.7)
    state.setdefault("specular", 0.3)
    state.setdefault("show_outline", True)
    # Opacity control points: list of {x: frac, y: opacity} in [0,1]×[0,1]
    state.setdefault("opacity_points", [
        {"x": 0.0,  "y": 0.0},
        {"x": 0.10, "y": 0.02},
        {"x": 0.40, "y": 0.10},
        {"x": 0.70, "y": 0.30},
        {"x": 1.0,  "y": 0.75},
    ])

    # Slicing (clip fractions 0..1 along each axis)
    state.setdefault("slice_x_min", 0.0)
    state.setdefault("slice_x_max", 1.0)
    state.setdefault("slice_y_min", 0.0)
    state.setdefault("slice_y_max", 1.0)
    state.setdefault("slice_z_min", 0.0)
    state.setdefault("slice_z_max", 1.0)

    # File-browser dialog state
    state.setdefault("browser_open", False)
    state.setdefault("browser_path", str(Path.home()))
    state.setdefault("browser_items", [])  # list[dict] shown in the dialog list
    state.setdefault("browser_selected", [])  # currently highlighted item(s)

    # "Select data type" dialog shown after a file is chosen in the browser
    state.setdefault("data_type_open", False)
    state.setdefault("data_type_loading", "")

    # Tilt series projection slider bar
    state.setdefault("tomo_projection_index", 0)
    state.setdefault("tomo_projection_max", 0)
    state.setdefault("tomo_is_open_data", False)
    state.setdefault("tomo_is_tilt_series", False)

    _baseline_opacity: list[tuple[float, float, float, float]] = []
    _active_reader = None  # prevent reader from being garbage-collected
    _tilt_image = None  # keep the tilt-series vtkImageData alive for the z-slice
    _data_range = (0.0, 1.0)  # raw scalar range of the loaded data
    _volume_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)  # xmin,xmax,ymin,ymax,zmin,zmax
    # Bridge between the workflow and the scene: "base" holds the loaded dataset
    # (a pipeline.TomoData) that tomo_run_pipeline feeds through run_pipeline;
    # "result" caches the last pipeline output; "browse_target" routes a
    # file-browser pick into a specific pipeline block param (see _fb_open).
    _pipeline_io: dict = {}

    _DATA_TYPE_KIND = {"Volume": "volume", "Tilt Series": "tilt_series"}


    def _update_clip_planes():
        """Position clipping planes based on current slice fractions and data bounds."""
        xmin, xmax, ymin, ymax, zmin, zmax = _volume_bounds

        fx0, fx1 = float(state.slice_x_min), float(state.slice_x_max)
        fy0, fy1 = float(state.slice_y_min), float(state.slice_y_max)
        fz0, fz1 = float(state.slice_z_min), float(state.slice_z_max)

        # x_min plane: normal (+1,0,0), clips everything below origin.x
        clip_planes["x_min"].SetOrigin(xmin + fx0 * (xmax - xmin), 0, 0)
        # x_max plane: normal (-1,0,0), clips everything above origin.x
        clip_planes["x_max"].SetOrigin(xmin + fx1 * (xmax - xmin), 0, 0)
        # y_min
        clip_planes["y_min"].SetOrigin(0, ymin + fy0 * (ymax - ymin), 0)
        # y_max
        clip_planes["y_max"].SetOrigin(0, ymin + fy1 * (ymax - ymin), 0)
        # z_min
        clip_planes["z_min"].SetOrigin(0, 0, zmin + fz0 * (zmax - zmin))
        # z_max
        clip_planes["z_max"].SetOrigin(0, 0, zmin + fz1 * (zmax - zmin))

    BLEND_MODE_ITEMS = [
        {"title": "Composite", "value": 0},
        {"title": "Max Intensity", "value": 1},
        {"title": "Min Intensity", "value": 2},
        {"title": "Average Intensity", "value": 3},
    ]


    ###########
    # helpers #
    ###########
    def _view_update():
        fn = getattr(ctrl, "view_update", None)
        if callable(fn):
            fn()

    def _snapshot_opacity():
        _baseline_opacity.clear()
        buf = [0.0, 0.0, 0.0, 0.0]
        for i in range(opacity_tf.GetSize()):
            opacity_tf.GetNodeValue(i, buf)
            _baseline_opacity.append(tuple(buf))

    def _apply_opacity_scale(scale: float):
        if not _baseline_opacity:
            return
        new_tf = vtkPiecewiseFunction()
        for x, y, m, s in _baseline_opacity:
            new_tf.AddPoint(x, max(0.0, min(y * scale, 1.0)), m, s)
        vol_property.SetScalarOpacity(new_tf)

    def _apply_colormap():
        """Rebuild the colour transfer function from current state."""
        lo_frac = float(state.contrast_low)
        hi_frac = float(state.contrast_high)
        if hi_frac <= lo_frac:
            hi_frac = lo_frac + 0.001

        raw_lo, raw_hi = _data_range
        raw_span = raw_hi - raw_lo if raw_hi > raw_lo else 1.0
        # Map contrast fractions to actual scalar values
        c_lo = raw_lo + lo_frac * raw_span
        c_hi = raw_lo + hi_frac * raw_span
        c_span = c_hi - c_lo

        cmap_name = state.colormap or "coolwarm"
        nodes = COLORMAPS.get(cmap_name, COLORMAPS["coolwarm"])

        color_tf.RemoveAllPoints()
        for frac, r, g, b in nodes:
            color_tf.AddRGBPoint(c_lo + frac * c_span, r, g, b)

    def _apply_opacity_from_points():
        """Rebuild opacity TF from the control points stored in state."""
        lo, hi = _data_range
        span = hi - lo if hi > lo else 1.0

        opacity_tf.RemoveAllPoints()
        points = state.opacity_points or [{"x": 0, "y": 0}, {"x": 1, "y": 0.75}]
        for pt in sorted(points, key=lambda p: p["x"]):
            opacity_tf.AddPoint(lo + pt["x"] * span, pt["y"], 0.5, 0.0)

        _snapshot_opacity()
        _apply_opacity_scale(float(state.opacity_scale))

    def _setup_transfer_functions(data_range):
        nonlocal _data_range
        lo, hi = data_range
        if hi <= lo:
            hi = lo + 1.0
        _data_range = (lo, hi)

        # Reset contrast to full range
        state.contrast_low = 0.0
        state.contrast_high = 1.0

        # Reset opacity control points to default ramp
        state.opacity_points = [
            {"x": 0.0,  "y": 0.0},
            {"x": 0.10, "y": 0.02},
            {"x": 0.40, "y": 0.10},
            {"x": 0.70, "y": 0.30},
            {"x": 1.0,  "y": 0.75},
        ]

        _apply_colormap()
        _apply_opacity_from_points()

    def load_tiff(path_str: str):
        """Read a 3-D TIFF and feed it into the volume pipeline."""
        resolved = Path(path_str).expanduser().resolve() if path_str else None

        if not resolved or not resolved.is_file():
            volume_actor.VisibilityOff()
            state.loaded = False
            state.scalar_range = "—"
            state.dimensions = "—"
            _view_update()
            return

        nonlocal _active_reader
        try:
            reader = vtkTIFFReader()
            reader.SetFileName(str(resolved))
            reader.Update()
            _active_reader = reader  # prevent garbage collection

            image = reader.GetOutput()
            dims = image.GetDimensions()
            srange = image.GetScalarRange()

            volume_mapper.SetInputConnection(reader.GetOutputPort())
            outline_filter.SetInputConnection(reader.GetOutputPort())
            _setup_transfer_functions(srange)

            spacing = image.GetSpacing()
            unit_dist = max(float(max(spacing)), 1e-3) if spacing else 1.0
            vol_property.SetScalarOpacityUnitDistance(unit_dist)
            vol_property.SetShade(bool(state.shade))

            volume_actor.VisibilityOn()
            outline_actor.SetVisibility(bool(state.show_outline))

            # Store bounds and reset slice fractions
            nonlocal _volume_bounds
            _volume_bounds = image.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
            state.slice_x_min = 0.0
            state.slice_x_max = 1.0
            state.slice_y_min = 0.0
            state.slice_y_max = 1.0
            state.slice_z_min = 0.0
            state.slice_z_max = 1.0
            _update_clip_planes()

            renderer.ResetCamera()

            state.scalar_range = f"{srange[0]:.3f} – {srange[1]:.3f}"
            state.dimensions = f"{dims[0]} × {dims[1]} × {dims[2]}"
            state.loaded = True
            state.tiff_path = str(resolved)
        except Exception as exc:  # noqa: BLE001
            volume_actor.VisibilityOff()
            state.loaded = False

        _view_update()

    # connect the tomography pipeline (works in in NumPy via pipeline.TomoData) 
    # to the VTK scene. load_data() reads a file into a TomoData; 
    # _show_numpy_volume() renders any NumPy volume (a freshly loaded
    # dataset OR a pipeline result) through the same volume_mapper/outline that
    # load_tiff uses.
    def _show_numpy_volume(vol):
        """Wrap a 3-D NumPy array as vtkImageData and display it as the volume."""
        nonlocal _active_reader, _volume_bounds
        arr = np.ascontiguousarray(np.asarray(vol, dtype=np.float32))
        if arr.ndim == 2:  # a single image -> a 1-slice volume
            arr = arr[np.newaxis, ...]
        nz, ny, nx = arr.shape
        img = vtkImageData()
        img.SetDimensions(nx, ny, nz)  # VTK is x-fastest; C-order (z,y,x) matches
        vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="C"), deep=True)
        img.GetPointData().SetScalars(vtk_arr)
        _active_reader = img  # keep the image alive; the mapper only holds a ref

        volume_mapper.SetInputData(img)
        outline_filter.SetInputData(img)
        srange = img.GetScalarRange()
        _setup_transfer_functions(srange)
        vol_property.SetScalarOpacityUnitDistance(1.0)
        vol_property.SetShade(bool(state.shade))

        volume_actor.VisibilityOn()
        slice_actors["z"].VisibilityOff()  # hide any tilt-series projection view
        outline_actor.SetVisibility(bool(state.show_outline))

        _volume_bounds = img.GetBounds()
        state.slice_x_min = 0.0
        state.slice_x_max = 1.0
        state.slice_y_min = 0.0
        state.slice_y_max = 1.0
        state.slice_z_min = 0.0
        state.slice_z_max = 1.0
        _update_clip_planes()
        renderer.ResetCamera()

        state.scalar_range = f"{srange[0]:.3f} – {srange[1]:.3f}"
        state.dimensions = f"{nx} × {ny} × {nz}"
        state.loaded = True
        _view_update()

    def _position_tilt_plane(index):
        """Move the projection slice to z=index (clamped) inside the outline box."""
        idx = max(0, min(int(index), int(state.tomo_projection_max or 0)))
        tilt_slice_plane.SetOrigin(0.0, 0.0, float(idx))
        slice_mappers["z"].Modified()

    def _show_tilt_series(prj, reset_index=False):
        """Display one projection of a tilt series as a 2D image in the box.

        The full ``(theta, y, x)`` stack determines the outline box framing, the
        z-slice actor shows a single projection whose depth follows the slider.
        """
        nonlocal _tilt_image, _volume_bounds
        arr = np.ascontiguousarray(np.asarray(prj, dtype=np.float32))
        if arr.ndim == 2:  # a single projection
            arr = arr[np.newaxis, ...]
        ntheta, ny, nx = arr.shape
        img = vtkImageData()
        img.SetDimensions(nx, ny, ntheta)
        vtk_arr = numpy_support.numpy_to_vtk(arr.ravel(order="C"), deep=True)
        img.GetPointData().SetScalars(vtk_arr)
        _tilt_image = img  # keep alive; mappers only hold a ref

        srange = img.GetScalarRange()
        _setup_transfer_functions(srange)

        outline_filter.SetInputData(img)  # box spans the whole projection stack
        slice_mappers["z"].SetInputData(img)
        img_prop = slice_actors["z"].GetProperty()
        img_prop.SetLookupTable(color_tf)
        img_prop.UseLookupTableScalarRangeOn()

        state.tomo_projection_max = int(ntheta - 1)
        cur = int(state.tomo_projection_index or 0)
        if reset_index or cur > ntheta - 1:
            cur = ntheta // 2
            state.tomo_projection_index = cur
        _position_tilt_plane(cur)

        volume_actor.VisibilityOff()
        slice_actors["z"].VisibilityOn()
        outline_actor.SetVisibility(bool(state.show_outline))

        _volume_bounds = img.GetBounds()
        renderer.ResetCamera()

        state.scalar_range = f"{srange[0]:.3f} – {srange[1]:.3f}"
        state.dimensions = f"{nx} × {ny} × {ntheta}"
        state.loaded = True
        _view_update()

    def _read_zarr_stack(root):
        """Stack a Zarr group of per-projection 2-D arrays into (theta, y, x).

        Targets ``sim_data/scan_64`` (subgroups 0000..0019, each a 2-D
        projection). Falls back to reading a single Zarr array store.
        """
        import zarr  # optional dep; only needed for Zarr datasets

        group = zarr.open_group(str(root), mode="r")
        keys = sorted(group.array_keys())
        if not keys:
            return np.asarray(zarr.open_array(str(root), mode="r"))
        return np.stack([np.asarray(group[k]) for k in keys], axis=0)

    def _read_array(path):
        """Read a dataset path into a NumPy array (TIFF stack, .npy, or Zarr)."""
        p = Path(path)
        if p.is_dir() or (p / "zarr.json").exists():
            return _read_zarr_stack(p)
        suffix = p.suffix.lower()
        if suffix in (".tif", ".tiff"):
            import tifffile
            return np.asarray(tifffile.imread(str(p)))
        if suffix == ".npy":
            return np.load(str(p))
        if suffix == ".zarr":
            return _read_zarr_stack(p)
        raise ValueError(f"Unsupported dataset type: {p.name}")

    def load_data(path_str, kind="tilt_series"):
        """Load a dataset into the *base* TomoData and display it.

        File -> Open Data routes here. The loaded array becomes
        ``_pipeline_io['base']`` -- the TomoData that ``tomo_run_pipeline`` feeds
        through ``pipeline.run_pipeline``. It is also shown immediately so the
        user sees the input before running the pipeline.
        """
        resolved = Path(path_str).expanduser()
        if not resolved.exists():
            return
        try:
            arr = _read_array(resolved)
        except Exception as exc:  # noqa: BLE001
            return
        data = tomo_pipeline.TomoData(kind=kind)
        data = data.with_(recon=arr) if kind == "volume" else data.with_(prj=arr)
        _pipeline_io["base"] = data
        # Record the base extents (x=cols, y=rows, z=depth/theta) so the Crop
        # editor can bound its x/y/z max inputs. Kept off the displayed-result
        # path (which the auto-run mutates) so crop bounds stay at full size.
        base_arr = np.asarray(arr)
        if base_arr.ndim >= 3:
            bnz, bny, bnx = base_arr.shape[-3], base_arr.shape[-2], base_arr.shape[-1]
        elif base_arr.ndim == 2:
            bnz, bny, bnx = 1, base_arr.shape[0], base_arr.shape[1]
        else:
            bnz = bny = bnx = 0
        state.tomo_dim_x = int(bnx)
        state.tomo_dim_y = int(bny)
        state.tomo_dim_z = int(bnz)
        try:
            if kind == "tilt_series":
                _show_tilt_series(arr, reset_index=True)
            else:
                _show_numpy_volume(arr)
        except Exception as exc:  # noqa: BLE001
            return
        state.tiff_path = str(resolved)

    # file browser helpers
    def _list_directory(directory: str) -> list[dict]:
        """Return a sorted list of entries (sub-dirs + tif/tiff files) for *directory*."""
        p = Path(directory).expanduser().resolve()
        if not p.is_dir():
            return []

        items: list[dict] = []

        # Directories first (sorted)
        dirs = sorted(
            (d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")),
            key=lambda d: d.name.lower(),
        )
        for d in dirs:
            items.append(
                {
                    "title": d.name,
                    "value": str(d),
                    "props": {"prependIcon": "mdi-folder"},
                    "is_dir": True,
                }
            )

        # Dataset files: TIFF stacks + .npy arrays (sorted)
        tiff_files = sorted(
            (
                f
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff", ".npy")
            ),
            key=lambda f: f.name.lower(),
        )
        for f in tiff_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            items.append(
                {
                    "title": f.name,
                    "value": str(f),
                    "props": {"prependIcon": "mdi-file-image", "subtitle": f"{size_mb:.1f} MB"},
                    "is_dir": False,
                }
            )

        return items

    def open_browser():
        """Open the file-browser dialog starting at the current browser_path."""
        path = state.browser_path or str(Path.home())
        state.browser_path = path
        state.browser_items = _list_directory(path)
        state.browser_selected = []
        state.browser_open = True

    def browser_navigate(entry_value):
        """Called when the user clicks a row in the file list."""
        if not entry_value:
            return

        target = Path(entry_value)
        if target.is_dir():
            # Navigate into the directory
            state.browser_path = str(target)
            state.browser_items = _list_directory(str(target))
            state.browser_selected = []
        else:
            # It's a file – select it
            state.browser_selected = [entry_value]

    def browser_go_up():
        """Navigate to the parent directory."""
        current = Path(state.browser_path).expanduser().resolve()
        parent = current.parent
        if parent != current:
            state.browser_path = str(parent)
            state.browser_items = _list_directory(str(parent))
            state.browser_selected = []

    def browser_confirm():
        """Load the highlighted file, or route it into a pipeline param.

        Normally the pick becomes the base dataset (load_data). When the browser
        was opened for a specific block parameter (via _fb_open, which sets
        ``browse_target``), the pick is written into that param instead.
        """
        selected = state.browser_selected
        target = _pipeline_io.pop("browse_target", None)
        state.browser_open = False
        if not selected or len(selected) == 0:
            return
        chosen = selected[0]
        if target:
            try:
                _, op_id, name = target.split("::")
                ctrl.tomo_set_param(op_id, name, chosen)
            except ValueError:
                pass
        else:
            # Don't load until user selects data type, Open Data block is added afterward.
            _pipeline_io["pending_open_path"] = chosen
            state.data_type_loading = ""
            state.data_type_open = True

    @server.trigger("tomo_choose_data_type")
    async def choose_data_type(data_type):
        """Load the pending file once a data-type button is clicked.

        Clicking the data type button activates a loading circle until the file is loaded.
        An Open Data pipeline block is added with the chosen ``data_type``, 
        shown in the Properties panel.
        """
        path = _pipeline_io.pop("pending_open_path", None)
        if not path:
            state.data_type_loading = ""
            state.data_type_open = False
            return
        # Yield once so the spinner paints before the blocking load begins.
        await asyncio.sleep(0)
        try:
            ctrl.tomo_add_op(
                "open_data",
                label="Data",
                params={"path": path, "data_type": data_type},
            )
        finally:
            state.data_type_loading = ""
            state.data_type_open = False

    def browser_cancel():
        _pipeline_io.pop("browse_target", None)
        state.browser_open = False

    ctrl.open_browser = open_browser
    ctrl.browser_navigate = browser_navigate
    ctrl.browser_go_up = browser_go_up
    ctrl.browser_confirm = browser_confirm
    ctrl.browser_cancel = browser_cancel

    # Opacity preset curves
    _OPACITY_PRESETS = {
        "linear": [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
        ],
        "ramp_up": [
            {"x": 0.0, "y": 0.0},
            {"x": 0.10, "y": 0.0},
            {"x": 0.40, "y": 0.05},
            {"x": 0.70, "y": 0.25},
            {"x": 1.0,  "y": 0.80},
        ],
        "ramp_down": [
            {"x": 0.0, "y": 0.80},
            {"x": 0.30, "y": 0.25},
            {"x": 0.60, "y": 0.05},
            {"x": 0.90, "y": 0.0},
            {"x": 1.0,  "y": 0.0},
        ],
        "tent": [
            {"x": 0.0, "y": 0.0},
            {"x": 0.50, "y": 1.0},
            {"x": 1.0,  "y": 0.0},
        ],
        "s_curve": [
            {"x": 0.0,  "y": 0.0},
            {"x": 0.15, "y": 0.01},
            {"x": 0.30, "y": 0.05},
            {"x": 0.50, "y": 0.50},
            {"x": 0.70, "y": 0.95},
            {"x": 0.85, "y": 0.99},
            {"x": 1.0,  "y": 1.0},
        ],
        "flat": [
            {"x": 0.0, "y": 0.5},
            {"x": 1.0, "y": 0.5},
        ],
    }

    def set_opacity_preset(preset_name: str):
        """Apply an opacity-curve preset by name."""
        pts = _OPACITY_PRESETS.get(preset_name)
        if pts is not None:
            state.opacity_points = [dict(p) for p in pts]

    ctrl.set_opacity_preset = set_opacity_preset


    ##########################
    # @state.change handlers #
    ##########################
    @state.change("opacity_scale")
    def _on_opacity(opacity_scale, **_kw):
        try:
            _apply_opacity_scale(float(opacity_scale))
        except (TypeError, ValueError):
            pass
        _view_update()

    @state.change("shade")
    def _on_shade(shade, **_kw):
        vol_property.SetShade(bool(shade))
        _view_update()

    @state.change("show_outline")
    def _on_outline(show_outline, **_kw):
        # Only show the outline once data is loaded; otherwise outline_filter has
        # no input and VTK's demand-driven pipeline errors on every render.
        outline_actor.SetVisibility(bool(show_outline) and bool(state.loaded))
        _view_update()

    @state.change("slice_x_min", "slice_x_max",
                "slice_y_min", "slice_y_max",
                "slice_z_min", "slice_z_max")
    def _on_slice(**_kw):
        _update_clip_planes()
        _view_update()

    @state.change("colormap")
    def _on_colormap(colormap, **_kw):
        state.colormap_preview = COLORMAP_PREVIEWS.get(colormap, "")
        _apply_colormap()
        _view_update()

    @state.change("contrast_low", "contrast_high")
    def _on_contrast(contrast_low, contrast_high, **_kw):
        _apply_colormap()
        _view_update()

    @state.change("opacity_points")
    def _on_opacity_points(opacity_points, **_kw):
        _apply_opacity_from_points()
        _view_update()

    @state.change("blend_mode")
    def _on_blend_mode(blend_mode, **_kw):
        mode = int(blend_mode)
        if mode == 0:
            volume_mapper.SetBlendModeToComposite()
        elif mode == 1:
            volume_mapper.SetBlendModeToMaximumIntensity()
        elif mode == 2:
            volume_mapper.SetBlendModeToMinimumIntensity()
        elif mode == 3:
            volume_mapper.SetBlendModeToAverageIntensity()
        _view_update()

    @state.change("ambient")
    def _on_ambient(ambient, **_kw):
        vol_property.SetAmbient(float(ambient))
        _view_update()

    @state.change("diffuse")
    def _on_diffuse(diffuse, **_kw):
        vol_property.SetDiffuse(float(diffuse))
        _view_update()

    @state.change("specular")
    def _on_specular(specular, **_kw):
        vol_property.SetSpecular(float(specular))
        _view_update()

    @state.change("tomo_projection_index")
    def _on_projection_index(tomo_projection_index, **_kw):
        if not slice_actors["z"].GetVisibility():
            return
        _position_tilt_plane(tomo_projection_index)
        _view_update()


    def _fb_open(target, mode="file"):
        # Remember which block param this browse is for, then open the browser;
        # browser_confirm writes the pick back into that param.
        _pipeline_io["browse_target"] = target
        open_browser()

    ctx = FeatureContext(server=server, state=state, ctrl=ctrl, fb_open=_fb_open)
    TomographyFeature().register_controllers(ctx)

    # The shared feature controllers report progress through ctrl.set_status;
    # provide a sink so those calls never raise (this app has no status widget).
    ctrl.set_status = lambda msg="": setattr(state, "status", msg)

    @ctrl.set("tomo_save_data")
    def tomo_save_data():
        return

    @ctrl.set("tomo_run_pipeline")
    def tomo_run_pipeline():
        """Run the pipeline on the loaded data and display the result."""
        base = _pipeline_io.get("base")
        if base is None:
            return
        try:
            result = tomo_pipeline.run_pipeline(
                base,
                list(state.pipeline),
            )
        except ModuleNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001
            return
        _pipeline_io["result"] = result
        # A reconstructed volume is shown as a 3D volume;
        # a tilt series stays a projection stack scrubbed via the slider
        is_tilt = _pipeline_io.get("loaded_kind") == "tilt_series"
        try:
            if result.recon is not None:
                _show_numpy_volume(np.asarray(result.recon))
            elif result.prj is not None and is_tilt:
                _show_tilt_series(np.asarray(result.prj))
            elif result.prj is not None:
                _show_numpy_volume(np.asarray(result.prj))
        except Exception as exc:  # noqa: BLE001
            return

    def _maybe_reload_base():
        """Reload the base dataset when the Open Data block's path/type changes.

        Driving the load off pipeline state means changing the Data Type in the
        Properties dropdown (or the initial "Select data type" dialog) re-reads
        the file with the matching kind.
        """
        block = next((b for b in state.pipeline if b.get("op") == "open_data"), None)
        if block is None:
            return
        path = block["params"].get("path")
        if not path:
            return
        kind = _DATA_TYPE_KIND.get(block["params"].get("data_type"), "tilt_series")
        if (_pipeline_io.get("loaded_path") == path
                and _pipeline_io.get("loaded_kind") == kind):
            return
        _pipeline_io["loaded_path"] = path
        _pipeline_io["loaded_kind"] = kind
        load_data(path, kind=kind)

    @state.change("pipeline")
    def _auto_run_pipeline(**_kw):
        _maybe_reload_base()
        if _pipeline_io.get("base") is not None:
            tomo_run_pipeline()


    ######
    # UI #
    ######
    # File menu items call app-level handlers (open/save), every other op appends a
    # pipeline block via ctrl.tomo_add_op 
    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Tomography Viewer")

        with layout.toolbar:
            v3.VSpacer()
            for menu_title, ops in tomo_ui.TOMO_MENUS:
                subgroups = tomo_ui.MENU_SUBGROUPS.get(menu_title)
                with v3.VMenu(open_on_hover=True, location="bottom end", viewport_margin=0):
                    with v3.Template(v_slot_activator="{ props }"):
                        v3.VBtn(
                            menu_title,
                            v_bind="props",
                            variant="text",
                            append_icon="mdi-chevron-down",
                            rounded="lg",
                        )
                    with v3.VList(density="compact"):
                        if subgroups:
                            for cat_label, cat_ops in subgroups:
                                with v3.VMenu(open_on_hover=True, open_on_click=False, location="start top", viewport_margin=0):
                                    with v3.Template(v_slot_activator="{ props }"):
                                        v3.VListItem(
                                            v_bind="props",
                                            title=cat_label,
                                            append_icon="mdi-chevron-left",
                                        )
                                    with v3.VList(density="compact"):
                                        for op in cat_ops:
                                            v3.VListItem(
                                                title=op["label"],
                                                click=(ctrl.tomo_add_op, f"['{op['id']}']"),
                                            )
                            continue
                        for op in ops:
                            if op["id"] == "open_data":
                                click = ctrl.open_browser
                            elif op["id"] == "save_data":
                                click = ctrl.tomo_save_data
                            else:
                                # Add this operation to the pipeline as a block.
                                click = (ctrl.tomo_add_op, f"['{op['id']}']")
                            v3.VListItem(title=op["label"], click=click)

        with layout.drawer as drawer:
            drawer.width = 360
            with html.Div(classes="d-flex flex-column fill-height", style="height:100%;"):
                # Pipeline
                with v3.VCard(
                    flat=True,
                    classes="d-flex flex-column",
                    style=("`flex: ${drawer_split} 1 0px; overflow:hidden;`",),
                ):
                    v3.VCardTitle("Pipeline", classes="text-subtitle-1 font-weight-medium py-2 mt-2")
                    with html.Div(style="overflow:auto; padding:0 12px 12px;"):
                        tomo_ui.build_pipeline_panel(ctx)
                # draggable divider for resizing sections
                html.Div(
                    v3.VIcon(icon="mdi-drag-horizontal", size=18, classes="mx-auto"),
                    classes="flex-shrink-0",
                    style="height:6px; cursor:row-resize; touch-action:none; "
                    "background-color:rgba(0,0,0,0.12);",
                    __events=["pointerdown", "pointermove", "pointerup"],
                    pointerdown="$event.target.setPointerCapture($event.pointerId)",
                    pointermove=(
                        "$event.buttons===1 && (drawer_split = Math.min(0.9, Math.max(0.1, "
                        "($event.clientY - $event.currentTarget.parentNode.getBoundingClientRect().top) "
                        "/ $event.currentTarget.parentNode.getBoundingClientRect().height)))"
                    ),
                )
                # Properties
                with v3.VCard(
                    flat=True,
                    classes="d-flex flex-column",
                    style=("`flex: ${1 - drawer_split} 1 0px; overflow:hidden;`",),
                ):
                    v3.VCardTitle("Properties", classes="text-subtitle-1 font-weight-medium py-2")
                    with html.Div(style="overflow:auto; padding:0 12px 12px;"):
                        tomo_ui.build_properties_panel(ctx)

        # main 3D view area
        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = vtk_widgets.VtkRemoteView(
                    render_window,
                    ref="view",
                    interactive_ratio=0.5,
                    still_ratio=1,
                )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

        # file browser dialog
        with v3.VDialog(
            v_model=("browser_open",),
            max_width=700,
            scrollable=True,
        ):
            with v3.VCard(rounded="lg"):
                with html.Div(classes="d-flex align-center mt-2"):
                    v3.VIcon("mdi-folder-multiple", classes="ms-10 me-2", size=36)
                    with html.Div():
                        v3.VCardTitle("Upload a dataset", classes="mt-2")
                        v3.VCardSubtitle("Supported formats: TIFF, HDF5, and NPY", classes="mb-4")

                v3.VDivider()
    
                # Back button + current path 
                with v3.VCardText(classes="py-1 mx-1"):
                    with html.Div(classes="d-flex align-center"):
                        v3.VBtn(
                            "Back",
                            variant="text",
                            prepend_icon="mdi-arrow-left",
                            click=ctrl.browser_go_up,
                            rounded="lg",
                            classes="flex-shrink-0 mt-2",
                        )
                        # truncate the middle of the path if too long
                        with html.Div(
                            classes="d-flex justify-end text-grey ms-auto",
                            style="flex:0 1 50%; max-width:50%; min-width:0; overflow:hidden;",
                        ):
                            html.Span(
                                "{{ browser_path.slice(0, -12) }}",
                                style="overflow:hidden; text-overflow:ellipsis; "
                                "white-space:nowrap; min-width:0;",
                            )
                            html.Span(
                                "{{ browser_path.slice(-12) }}",
                                style="white-space:nowrap; flex-shrink:0;",
                            )

                # Scrollable file / directory list
                with v3.VCardText(classes="pt-0 mx-1", style="height: 400px; overflow-y: auto;"):
                    with v3.VList(
                        density="compact",
                        nav=True,
                    ):
                        with v3.VListItem(
                            v_for="(item, idx) in browser_items",
                            key="idx",
                            title=("item.title",),
                            value=("item.value",),
                            v_bind=("item.props",),
                            click=(ctrl.browser_navigate, "[item.value]"),
                            active=("browser_selected.includes(item.value)",),
                            color="primary",
                            rounded="lg",
                        ):
                            pass
    
                    # Empty state
                    html.Div(
                        "No folders or TIFF files here.",
                        v_show="browser_items.length === 0",
                        classes="text-caption text-center mt-4 text-grey",
                    )
    
                v3.VDivider()
    
                with v3.VCardActions():
                    v3.VSpacer() # push the buttons to the right
                    v3.VBtn(
                        "Cancel",
                        variant="text",
                        click=ctrl.browser_cancel,
                        rounded="lg",
                    )
                    v3.VBtn(
                        "Open",
                        color="primary",
                        variant="flat",
                        disabled=("browser_selected.length === 0",),
                        click=ctrl.browser_confirm,
                        rounded="lg",
                    )

        with v3.VDialog(
            v_model=("data_type_open",),
            max_width=320,
            persistent=True,
        ):
            with v3.VCard(rounded="lg"):
                v3.VCardTitle("Select data type", classes="text-center")
                with v3.VCardActions(classes="justify-center pb-4 ga-2"):
                    with v3.VBtn(
                        color="primary",
                        variant="tonal",
                        disabled=("data_type_loading !== ''",),
                        click="data_type_loading = 'Volume'; trigger('tomo_choose_data_type', ['Volume'])",
                        rounded="lg",
                    ):
                        v3.VProgressCircular(
                            v_if="data_type_loading === 'Volume'",
                            indeterminate=True,
                            size=20,
                            width=2,
                        )
                        html.Span("Volume", v_else=True)
                    with v3.VBtn(
                        color="primary",
                        variant="tonal",
                        disabled=("data_type_loading !== ''",),
                        click="data_type_loading = 'Tilt Series'; trigger('tomo_choose_data_type', ['Tilt Series'])",
                        rounded="lg",
                    ):
                        v3.VProgressCircular(
                            v_if="data_type_loading === 'Tilt Series'",
                            indeterminate=True,
                            size=20,
                            width=2,
                        )
                        html.Span("Tilt Series", v_else=True)

    return server

def run_server(port=0, host="localhost", open_browser=True):
    create_tomo_server().start(port=port, host=host, open_browser=open_browser)

def main(argv=None):
    # CLI
    parser = argparse.ArgumentParser(description="Tomography Web Application using Trame")
    parser.add_argument("--port", type=int, default=0, help="Port to bind the Trame server (0 = auto)")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on (default: localhost)")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    run_server(port=args.port, host=args.host, open_browser=not args.no_browser)