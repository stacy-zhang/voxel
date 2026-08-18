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
    outline_actor.GetProperty().SetLineWidth(1.5)
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
    state.setdefault("status", "")
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

    _baseline_opacity: list[tuple[float, float, float, float]] = []
    _active_reader = None  # prevent reader from being garbage-collected
    _data_range = (0.0, 1.0)  # raw scalar range of the loaded data
    _volume_bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)  # xmin,xmax,ymin,ymax,zmin,zmax


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
        state.status = ""
        resolved = Path(path_str).expanduser().resolve() if path_str else None

        if not resolved or not resolved.is_file():
            state.status = f"File not found: {path_str}" if path_str else ""
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
            state.status = str(exc)
            volume_actor.VisibilityOff()
            state.loaded = False

        _view_update()

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

        # TIFF files (sorted)
        tiff_files = sorted(
            (
                f
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff")
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
        """Load the selected file and close the dialog."""
        selected = state.browser_selected
        if selected and len(selected) > 0:
            chosen = selected[0]
            state.browser_open = False
            load_tiff(chosen)
        # If nothing selected, just close
        else:
            state.browser_open = False

    def browser_cancel():
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
        outline_actor.SetVisibility(bool(show_outline))
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


    ######
    # UI #
    ######
    workflow_menus = [
        ("File", ["Open Data", "Save Data", "Export"]),
        ("Data Transforms", ["Data Management", "Volume Manipulation", "Math Operations", "Filters"]),
        ("Tomography", ["Mark Data as Volume", "Mark Data as Tilt Series", "Set Tilt Angles", "Pre-processing", "Alignment", "Reconstruction", "Simulation & Demonstrations"]),
        ("Visualization", ["Volume", "Outline", "Slice", "Contour", "Threshold", "Clip", "Ruler", "Scale Cube"]),
    ]

    option_funcs = {"Open Data": ctrl.open_browser}

    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Tomography Viewer")

        with layout.toolbar:
            v3.VSpacer()
            for menu_title, options in workflow_menus:
                with v3.VMenu(open_on_hover=True, location="bottom end", viewport_margin=0):
                    with v3.Template(v_slot_activator="{ props }"):
                        v3.VBtn(
                            menu_title,
                            v_bind="props",
                            variant="text",
                            append_icon="mdi-chevron-down",
                        )
                    with v3.VList(density="compact"):
                        for option in options:
                            v3.VListItem(
                                title=option,
                                click=option_funcs.get(option),
                            )

        with layout.drawer as drawer:
            drawer.width = 360

            with html.Div(classes="d-flex flex-column fill-height"):
                with v3.VCard(flat=True, classes="d-flex flex-column mt-2", style=("`flex: ${drawer_split} 1 0px`",)):
                    v3.VCardTitle("Pipeline", classes="text-h6 font-weight-regular")
                # draggable divider for resizing drawer sections
                html.Div(
                    v3.VIcon(icon="mdi-drag-horizontal", size=18, classes="mx-auto"),
                    classes="flex-shrink-0",
                    style="height: 5px; cursor: row-resize; touch-action: none; "
                    "background-color: rgba(0,0,0,0.12);",
                    __events=["pointerdown", "pointermove", "pointerup"],
                    pointerdown="$event.target.setPointerCapture($event.pointerId)",
                    pointermove=(
                        "$event.buttons===1 && (drawer_split = Math.min(0.9, Math.max(0.1, "
                        "($event.clientY - $event.currentTarget.parentNode.getBoundingClientRect().top) "
                        "/ $event.currentTarget.parentNode.getBoundingClientRect().height)))"
                    ),
                )
                with v3.VCard(flat=True, classes="d-flex flex-column", style=("`flex: ${1 - drawer_split} 1 0px`",)):
                    v3.VCardTitle("Properties", classes="text-h6 font-weight-regular")

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
            with v3.VCard():
                v3.VCardTitle("Select a TIFF file")
                with v3.VCardSubtitle():
                    html.Span("{{ browser_path }}")
    
                with v3.VCardText(style="height: 400px; overflow-y: auto;"):
                    # Up-one-level button
                    v3.VBtn(
                        "Parent directory",
                        variant="text",
                        prepend_icon="mdi-arrow-up",
                        click=ctrl.browser_go_up,
                        classes="mb-2",
                        block=True,
                    )
    
                    v3.VDivider()
    
                    # File / directory list
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
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        variant="text",
                        click=ctrl.browser_cancel,
                    )
                    v3.VBtn(
                        "Open",
                        color="primary",
                        variant="flat",
                        disabled=("browser_selected.length === 0",),
                        click=ctrl.browser_confirm,
                    )

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