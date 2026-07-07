"""Trame app for visualizing 3D TIFF tomography data with GPU volume rendering.

Features
--------
* Server-side file browser dialog – navigate directories and pick .tif/.tiff files.
* GPU volume rendering via vtkSmartVolumeMapper.
* Opacity multiplier slider and shading toggle.
* Optional CLI: ``python tomo_viewer.py --data /path/to/file.tif``
"""

import argparse
import base64 # for encoding colormap preview images as data-URIs
# data-URI is a way to embed small images directly in HTML/CSS as text, without needing separate files
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # use non-interactive backend for colormap rendering (Agg means "Anti-Grain Geometry")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.cm as mpl_cm            # noqa: E402
import numpy as np                        # noqa: E402

from trame.app import get_server # Trame server framework
from trame.ui.vuetify3 import SinglePageWithDrawerLayout 
from trame.widgets import vuetify3 as vuetify, vtk as vtk_widgets, html
# Vuetify provides pre-built UI components (buttons, sliders, dialogs, etc.)

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
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper # the GPU-accelerated volume mapper that does the actual rendering of the 3D data

from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera # allows the user to rotate/zoom/pan the view with mouse interactions (trackball style)
import vtkmodules.vtkInteractionStyle  # noqa – required
import vtkmodules.vtkRenderingOpenGL2  # noqa – required
# the above two imports are needed to ensure the appropriate VTK rendering and interaction styles are registered, even if we don't directly reference them in the code

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# this text appears when you run `pixi run python my_tests/tomo_viewer.py --help`
parser = argparse.ArgumentParser(description="Trame TIFF Volume Viewer")
parser.add_argument("--data", type=str, default="", help="Path to a 3-D TIFF file")
parser.add_argument("--port", type=int, default=0, help="Server port (0 = auto)")
args, _unknown = parser.parse_known_args()
# Trame TIFF Volume Viewer
#
# options:
#   -h, --help   show this help message and exit
#   --data DATA  Path to a 3-D TIFF file
#   --port PORT  Server port (0 = auto)

# ---------------------------------------------------------------------------
# VTK pipeline
# ---------------------------------------------------------------------------
renderer = vtkRenderer()
renderer.SetBackground(0.15, 0.15, 0.15) # dark gray background

render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1024, 768) # initial window size (can be resized by user)
render_window.SetOffScreenRendering(1)

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

renderer.ResetCamera() # automatically position the camera to fit the scene (will be updated when a file is loaded)

# ---------------------------------------------------------------------------
# Colormap presets – sampled from matplotlib (ParaView-like set)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Trame server & state
# ---------------------------------------------------------------------------
server = get_server(name="TomoViewer")
state, ctrl = server.state, server.controller

state.setdefault("tiff_path", args.data)
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


# ---------------------------------------------------------------------------
# Helpers – rendering
# ---------------------------------------------------------------------------
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
    global _data_range
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

    global _active_reader
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
        global _volume_bounds
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


# ---------------------------------------------------------------------------
# Helpers – server-side file browser
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# State-change handlers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Opacity preset curves
# ---------------------------------------------------------------------------
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


def on_load_click():
    load_tiff(state.tiff_path or "")


ctrl.on_load = on_load_click

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("Tomography Volume Viewer")

    # ---- toolbar ----
    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VBtn(icon="mdi-refresh", click=ctrl.on_load, variant="text")
        vuetify.VBtn(
            icon="mdi-crop-free",
            click="$refs.view.resetCamera()",
            variant="text",
        )

    # ---- drawer ----
    with layout.drawer as drawer:
        drawer.width = 360

        # --- Data Source section ---
        with vuetify.VCard(flat=True, classes="mb-4"):
            vuetify.VCardTitle("Data Source", classes="text-subtitle-1")
            with vuetify.VCardText():
                # Manual path + load
                vuetify.VTextField(
                    v_model=("tiff_path",),
                    label="TIFF file path",
                    placeholder="/path/to/reconstruction.tif",
                    clearable=True,
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                )
                with vuetify.VRow(classes="mt-2", no_gutters=True):
                    with vuetify.VCol(cols=6, classes="pr-1"):
                        vuetify.VBtn(
                            "Browse…",
                            color="secondary",
                            block=True,
                            click=ctrl.open_browser,
                            prepend_icon="mdi-folder-open",
                        )
                    with vuetify.VCol(cols=6, classes="pl-1"):
                        vuetify.VBtn(
                            "Load",
                            color="primary",
                            block=True,
                            click=ctrl.on_load,
                            prepend_icon="mdi-cube-scan",
                        )

                # Error banner
                with vuetify.VAlert(
                    v_show="status",
                    type="error",
                    density="compact",
                    variant="outlined",
                    classes="mt-3",
                ):
                    html.Span("{{ status }}")

                # Info
                html.Div(
                    "Scalar range: {{ scalar_range }}",
                    classes="text-caption mt-4",
                )
                html.Div(
                    "Dimensions: {{ dimensions }}",
                    classes="text-caption",
                )

        vuetify.VDivider()

        # --- Color Map Editor ---
        with vuetify.VCard(flat=True, classes="mt-4"):
            vuetify.VCardTitle("Color Map Editor", classes="text-subtitle-1")
            with vuetify.VCardText():
                # Colormap selector
                vuetify.VSelect(
                    v_model=("colormap", "coolwarm"),
                    items=("colormap_items", COLORMAP_ITEMS),
                    label="Colormap",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                )
                # Colormap preview bar
                html.Img(
                    src=("colormap_preview",),
                    style="width: 100%; height: 20px; margin-top: 8px; border-radius: 4px; border: 1px solid #555;",
                )

                # Contrast window
                html.Div("Contrast (scalar window)", classes="text-caption mt-4 mb-1 font-weight-bold")
                vuetify.VSlider(
                    v_model=("contrast_low", 0.0),
                    min=0.0, max=1.0, step=0.01,
                    label="Low",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify.VSlider(
                    v_model=("contrast_high", 1.0),
                    min=0.0, max=1.0, step=0.01,
                    label="High",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                    classes="mt-1",
                )

                # Opacity presets
                html.Div("Opacity presets", classes="text-caption mt-4 mb-1 font-weight-bold")
                with vuetify.VBtnGroup(density="compact", variant="outlined", divided=True, classes="mb-2"):
                    vuetify.VBtn(
                        "Linear",
                        click=(ctrl.set_opacity_preset, "['linear']"),
                        size="small",
                    )
                    vuetify.VBtn(
                        "Ramp up",
                        click=(ctrl.set_opacity_preset, "['ramp_up']"),
                        size="small",
                    )
                    vuetify.VBtn(
                        "Ramp down",
                        click=(ctrl.set_opacity_preset, "['ramp_down']"),
                        size="small",
                    )
                with vuetify.VBtnGroup(density="compact", variant="outlined", divided=True):
                    vuetify.VBtn(
                        "Tent",
                        click=(ctrl.set_opacity_preset, "['tent']"),
                        size="small",
                    )
                    vuetify.VBtn(
                        "S-curve",
                        click=(ctrl.set_opacity_preset, "['s_curve']"),
                        size="small",
                    )
                    vuetify.VBtn(
                        "Flat",
                        click=(ctrl.set_opacity_preset, "['flat']"),
                        size="small",
                    )

        vuetify.VDivider()

        # --- Rendering ---
        with vuetify.VCard(flat=True, classes="mt-4"):
            vuetify.VCardTitle("Rendering", classes="text-subtitle-1")
            with vuetify.VCardText():
                vuetify.VSelect(
                    v_model=("blend_mode", 0),
                    items=("blend_mode_items", BLEND_MODE_ITEMS),
                    label="Blend Mode",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                )
                vuetify.VSlider(
                    v_model=("opacity_scale", 1.0),
                    min=0.05,
                    max=5.0,
                    step=0.05,
                    label="Opacity",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                    classes="mt-4",
                )
                vuetify.VSwitch(
                    v_model=("shade", True),
                    label="Shading",
                    density="compact",
                    hide_details=True,
                    classes="mt-2",
                )
                vuetify.VSwitch(
                    v_model=("show_outline", True),
                    label="Outline",
                    density="compact",
                    hide_details=True,
                    classes="mt-2",
                )

        vuetify.VDivider()

        # --- Slicing ---
        with vuetify.VCard(flat=True, classes="mt-4"):
            vuetify.VCardTitle("Slicing", classes="text-subtitle-1")
            with vuetify.VCardText():
                html.Div("X axis", classes="text-caption font-weight-bold")
                vuetify.VSlider(
                    v_model=("slice_x_min", 0.0),
                    min=0.0, max=1.0, step=0.01,
                    label="Min",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify.VSlider(
                    v_model=("slice_x_max", 1.0),
                    min=0.0, max=1.0, step=0.01,
                    label="Max",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                    classes="mt-1",
                )
                html.Div("Y axis", classes="text-caption font-weight-bold mt-3")
                vuetify.VSlider(
                    v_model=("slice_y_min", 0.0),
                    min=0.0, max=1.0, step=0.01,
                    label="Min",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify.VSlider(
                    v_model=("slice_y_max", 1.0),
                    min=0.0, max=1.0, step=0.01,
                    label="Max",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                    classes="mt-1",
                )
                html.Div("Z axis", classes="text-caption font-weight-bold mt-3")
                vuetify.VSlider(
                    v_model=("slice_z_min", 0.0),
                    min=0.0, max=1.0, step=0.01,
                    label="Min",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify.VSlider(
                    v_model=("slice_z_max", 1.0),
                    min=0.0, max=1.0, step=0.01,
                    label="Max",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                    classes="mt-1",
                )

        vuetify.VDivider()

        # --- Lighting ---
        with vuetify.VCard(flat=True, classes="mt-4"):
            vuetify.VCardTitle("Lighting", classes="text-subtitle-1")
            with vuetify.VCardText():
                vuetify.VSlider(
                    v_model=("ambient", 0.2),
                    min=0.0, max=1.0, step=0.05,
                    label="Ambient",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                )
                vuetify.VSlider(
                    v_model=("diffuse", 0.7),
                    min=0.0, max=1.0, step=0.05,
                    label="Diffuse",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                    classes="mt-4",
                )
                vuetify.VSlider(
                    v_model=("specular", 0.3),
                    min=0.0, max=1.0, step=0.05,
                    label="Specular",
                    density="compact",
                    hide_details=True,
                    thumb_label=True,
                    classes="mt-4",
                )

    # ---- main content (3-D view) ----
    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
            view = vtk_widgets.VtkRemoteView(
                render_window,
                ref="view",
                interactive_ratio=0.5,
                still_ratio=1,
            )
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

    # ---- file-browser dialog (rendered once, toggled via state) ----
    with vuetify.VDialog(
        v_model=("browser_open",),
        max_width=700,
        scrollable=True,
    ):
        with vuetify.VCard():
            vuetify.VCardTitle("Select a TIFF file")
            with vuetify.VCardSubtitle():
                html.Span("{{ browser_path }}")

            with vuetify.VCardText(style="height: 400px; overflow-y: auto;"):
                # Up-one-level button
                vuetify.VBtn(
                    "Parent directory",
                    variant="text",
                    prepend_icon="mdi-arrow-up",
                    click=ctrl.browser_go_up,
                    classes="mb-2",
                    block=True,
                )

                vuetify.VDivider()

                # File / directory list
                with vuetify.VList(
                    density="compact",
                    nav=True,
                ):
                    with vuetify.VListItem(
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

            vuetify.VDivider()

            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn(
                    "Cancel",
                    variant="text",
                    click=ctrl.browser_cancel,
                )
                vuetify.VBtn(
                    "Open",
                    color="primary",
                    variant="flat",
                    disabled=("browser_selected.length === 0",),
                    click=ctrl.browser_confirm,
                )

    # Auto-load if --data was supplied
    if args.data:
        load_tiff(args.data)

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    server.start(port=args.port if args.port else 0)
