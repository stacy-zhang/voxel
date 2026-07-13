"""Trame + VTK web application: server construction and orchestration.

This is the STAGE-3 orchestration layer (the "app" tier). It wires the browser
UI to the pipeline: it owns the trame reactive ``state``/``controller``, builds
the VTK scene once, and defines the per-step controllers (load / view intensity
/ crop / build / regrid / view / refresh / slices / export). The heavy STAGE-1
(load) and STAGE-2 (build/regrid) work is delegated to ``voxel.services`` /
``voxel.rsm3d``; the color/opacity math is delegated to
``voxel.visualization``; the static UI assets come from ``voxel.ui``.

The pipeline closures below share mutable VTK actors and ``current_*`` scene
state, so they are intentionally kept together inside ``create_server``.
"""

import os
import asyncio
import gc
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
import yaml
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints  # maps raw scalar data to colors (RGBA)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkImageData,
    vtkLine,
    vtkPiecewiseFunction,
    vtkPolyData,
    vtkPolyLine,
)
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkProbeFilter
from vtkmodules.vtkFiltersSources import vtkConeSource, vtkLineSource, vtkRegularPolygonSource, vtkSphereSource  # sphere mesh for slicing; line + cone build the world-axes arrows
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkBillboardTextActor3D,  # camera-facing text placed at a fixed 3D point
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
from vtkmodules.vtkRenderingImage import vtkImageResliceMapper  # maps a 3D volume to a 2D slice plane
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.util import numpy_support

from trame.app import get_server
from trame.ui.html import DivLayout
from trame.widgets import html
from trame.widgets.vtk import VtkRemoteView

# --- pipeline backend (STAGE 1 & 2) ------------------------------------
from voxel.services.backend import (
    RSMDataLoader_ISR,
    RSMDataloader_CMS,
    write_rsm_volume_to_vtr,
    RSMBuilder,
    DEFAULTS_ENV,
    yaml_path,
)

# --- data-access / state coercion (services) ---------------------------
from voxel.services.parsing import (
    _float,
    _ensure_path,
    _scan_numbers_in_dir,
    _parse_scan_list,
    _parse_ub_matrix,
    _format_ub_matrix,
    _parse_axes_list,
    _parse_grid_shape,
    _crop_window_from_state,
    _adjust_setup_for_crop,
    _crop_dataframe_intensity,
)

# --- render helpers (visualization) ------------------------------------
from voxel.visualization.colormaps import (
    _cmap_rgb,
    _apply_color_transfer_function,
    _apply_opacity_function,
    _log1p_clip,
    _robust_percentiles,
    _make_lookup_table,
)

# --- static UI assets (ui) ---------------------------------------------
from voxel.ui.assets import (
    COLORMAP_NAMES,
    DEFAULT_FRAME_COUNT,
    _EYE_ON_SVG,
    _EYE_OFF_SVG,
)


def create_server():
    server = get_server(name="voxel_web", client_type="vue3")
    state, ctrl = server.state, server.controller

    # Data tab
    state.setdefault("loader_mode", "CMS")
    state.setdefault("setup_path", yaml_path())
    state.setdefault("tiff_dir", "")
    state.setdefault("spec_path", "")
    # ISR-only: restrict the load to HKL scans (maps to the loader's
    # process_hklscan_only flag). Ignored in CMS mode.
    state.setdefault("only_hkl", False)
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
    # Grid shape as "nx,ny,nz"; ny/nz may be "*" to auto-scale from the data
    # extents. A single value ("200") expands to "200,*,*".
    state.setdefault("grid_shape", "200,*,*")
    state.setdefault("normalize", "mean")
    # UB orientation matrix (rows separated by newlines). Seeded with identity
    # and overwritten with the loaded UB after Load Data; the user can edit it.
    state.setdefault("ub_matrix", "1 0 0\n0 1 0\n0 0 1")
    state.setdefault("ub_includes_2pi", True)
    # Vue v-model identifiers cannot start with a digit, so the 1-based-center
    # flag uses the key ``one_based_center`` (the legacy ``1_based_center`` key
    # was never wired to the UI).
    state.setdefault("one_based_center", False)
    # Sample/detector goniometer axis strings (mirror the napari widget
    # defaults so the web build reproduces the same Q/HKL mapping).
    state.setdefault("sample_axes", "x+, y+, z-")
    state.setdefault("detector_axes", "x+")
    state.setdefault("fuzzy_gridder", False)
    state.setdefault("width_fuzzy", 0.00) # range from 0 to 999999999

    # View tab
    state.setdefault("log_view", True)
    state.setdefault("rendering", "attenuated_mip")
    state.setdefault("contrast_lo", 1.0)
    state.setdefault("contrast_hi", 99.8)
    # Absolute (display-value) contrast window that the right-panel "Contrast
    # Limits" slider drives. napari seeds contrast_limits once from the
    # percentile boxes at view time, then lets the user adjust the *absolute*
    # limits with a linear mapping -- these mirror that live window, while
    # clim_min/clim_max are the slider bounds (the display data min/max).
    state.setdefault("clim_lo", 0.0)
    state.setdefault("clim_hi", 1.0)
    state.setdefault("clim_min", 0.0)
    state.setdefault("clim_max", 1.0)
    state.setdefault("clim_step", 0.01)
    state.setdefault("export_path", str(Path.cwd() / "rsm_output.vtr"))
    # Grid (TIFF) and grid+edges (NPZ) export paths, mirroring the napari
    # widget's Export section.
    state.setdefault("export_tiff_path", str(Path.cwd() / "rsm_grid.tiff"))
    state.setdefault("export_npz_path", str(Path.cwd() / "rsm_grid.npz"))

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

    ## bounding box outline (rectangular prism drawn around the volume)
    state.setdefault("outline_show", True)

    ## per-corner (Qx, Qy, Qz) coordinate labels -- a separate toggleable layer
    ## from the outline box so the coordinates can be shown/hidden on their own
    state.setdefault("coords_show", True)

    ## world axes overlay (colored +Qx/+Qy/+Qz direction arrows rooted at the
    ## volume's origin corner -- a visual orientation guide, mirrors napari)
    state.setdefault("world_axes_show", True)

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
    # napari renders image volumes unshaded (no lighting model). Default shade
    # off so the web app matches napari's flat colormap appearance; shading only
    # affects translucent/composite modes (VTK ignores it for MIP).
    state.setdefault("shade", False)
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

    # Right-side layer control panel: the list of scene layers available for the
    # active view. Each entry is {"key", "name", "visible"} and drives one row
    # (name + eye toggle) in the panel. Rebuilt whenever the active view changes.
    state.setdefault("layers", [])
    # Key of the currently selected layer (napari-style). Clicking a layer row
    # selects it, highlighting the row and showing that layer's own controls
    # (colormap / contrast / opacity ...) at the top of the panel, so edits
    # target the selected layer instead of always the volume.
    state.setdefault("selected_layer", "")
    # Inline SVG markup for the layer visibility toggle, injected via v-html.
    state.setdefault("eye_on_svg", _EYE_ON_SVG)
    state.setdefault("eye_off_svg", _EYE_OFF_SVG)

    # Progress bar shown at the top of the viewer during long-running pipeline
    # steps (load / build / regrid / export). ``progress_active`` toggles the
    # bar's visibility, ``progress_value`` is the 0-100 percentage, and
    # ``progress_label`` names the running step.
    state.setdefault("progress_active", False)
    state.setdefault("progress_value", 0.0)
    state.setdefault("progress_label", "")
    # When True the bar shows an indeterminate sliding animation with no number
    # (used for steps that can't report a real percentage, e.g. load/export).
    # When False the fill width tracks ``progress_value`` from real per-frame
    # callbacks (build/regrid).
    state.setdefault("progress_indeterminate", False)

    renderer = vtkRenderer()
    renderer.SetBackground(0.10, 0.10, 0.12)
    renderer.SetLayer(0)

    # Overlay renderer for the world-axes gizmo. It sits on a higher layer and
    # shares the main camera, so it rotates/zooms in lockstep with the volume
    # but renders with its own (freshly cleared) depth buffer. That makes the
    # axes always draw on top of the volume instead of being occluded by it.
    overlay_renderer = vtkRenderer()
    overlay_renderer.SetLayer(1)
    overlay_renderer.SetActiveCamera(renderer.GetActiveCamera())
    overlay_renderer.InteractiveOff()

    render_window = vtkRenderWindow()
    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(renderer)
    render_window.AddRenderer(overlay_renderer)
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

    # --- Bounding box outline ---------------------------------------------
    # A rectangular-prism wireframe around the RSM volume. Because it is a 3D
    # actor drawn in the same (Q) world coordinates as the volume, it rotates
    # together with the map when the camera is orbited, yet remains an
    # independent layer that can be toggled on/off. The 8 corners are stored in
    # a fixed index order (bit 4 -> x-axis, bit 2 -> y-axis, bit 1 -> z-axis)
    # so both the 12 edges and the per-corner coordinate labels reference the
    # same points.
    outline_pts = vtkPoints()
    outline_pts.SetNumberOfPoints(8)
    outline_poly = vtkPolyData()
    outline_poly.SetPoints(outline_pts)
    _outline_edges = []
    for _corner in range(8):
        for _bit in (4, 2, 1):
            _other = _corner ^ _bit
            if _corner < _other:
                _outline_edges.append((_corner, _other))
    _outline_cells = vtkCellArray()
    for _a, _b in _outline_edges:
        _edge = vtkLine()
        _edge.GetPointIds().SetId(0, _a)
        _edge.GetPointIds().SetId(1, _b)
        _outline_cells.InsertNextCell(_edge)
    outline_poly.SetLines(_outline_cells)
    outline_mapper = vtkPolyDataMapper()
    outline_mapper.SetInputData(outline_poly)
    outline_actor = vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetColor(0.85, 0.85, 0.9)
    outline_actor.GetProperty().SetLineWidth(1.5)
    outline_actor.GetProperty().LightingOff()
    outline_actor.PickableOff()
    outline_actor.VisibilityOff()
    renderer.AddActor(outline_actor)

    # One camera-facing text label per corner, showing that corner's
    # (Qx, Qy, Qz) coordinate. They live on their own "Coordinates" layer
    # (toggled via ``coords_show``, independent of the outline box) and are
    # colored violet to stand apart from the outline box wireframe.
    _coord_label_color = (0.57, 1.00, 0.85)
    outline_label_actors = []
    for _c in range(8):
        _label = vtkBillboardTextActor3D()
        _label.SetInput("")
        _label.GetTextProperty().SetFontSize(14)
        _label.GetTextProperty().SetColor(*_coord_label_color)
        _label.GetTextProperty().SetJustificationToCentered()
        _label.PickableOff()
        _label.VisibilityOff()
        renderer.AddActor(_label)
        outline_label_actors.append(_label)

    # --- World axes -------------------------------------------------------
    # Three colored direction arrows rooted at the reciprocal-space origin,
    # pointing along +Qx (cyan), +Qy (magenta) and +Qz (yellow) -- the same
    # colors napari uses. This mirrors napari's "World axes" overlay: a purely
    # visual orientation reference that rotates together with the map. It does
    # NOT change the coordinate values reported at the outline-box corners -- it
    # only shows which way the +Qx/+Qy/+Qz (or +H/+K/+L) axes point. Each axis
    # is a thin line shaft plus a cone arrowhead and a tip label; all nine
    # actors share a single "World Axes" layer toggle.
    _world_axis_colors = ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0))
    world_axes_line_sources = []
    world_axes_cone_sources = []
    world_axes_actors = []
    world_axes_labels = []
    # Root position + visibility of the world-axes gizmo, shared between
    # _update_world_axes (sets it when the volume changes) and
    # _rescale_world_axes (re-sizes it every render so it stays a constant
    # on-screen size regardless of zoom). Target shaft length is in pixels.
    # ``signs`` mirrors an axis direction to match napari's handedness (the z
    # world axis is flipped so +Qz points up).
    world_axes_geom = {
        "origin": None,
        "visible": False,
        "target_px": 90.0,
        "signs": (1.0, 1.0, -1.0),
    }
    for _i in range(3):
        _col = _world_axis_colors[_i]
        _ls = vtkLineSource()
        _lm = vtkPolyDataMapper()
        _lm.SetInputConnection(_ls.GetOutputPort())
        _la = vtkActor()
        _la.SetMapper(_lm)
        _la.GetProperty().SetColor(*_col)
        _la.GetProperty().SetLineWidth(3.0)
        _la.GetProperty().LightingOff()
        _la.PickableOff()
        _la.VisibilityOff()
        overlay_renderer.AddActor(_la)

        _cs = vtkConeSource()
        _cs.SetResolution(16)
        _cm = vtkPolyDataMapper()
        _cm.SetInputConnection(_cs.GetOutputPort())
        _ca = vtkActor()
        _ca.SetMapper(_cm)
        _ca.GetProperty().SetColor(*_col)
        _ca.GetProperty().LightingOff()
        _ca.PickableOff()
        _ca.VisibilityOff()
        overlay_renderer.AddActor(_ca)

        _wlbl = vtkBillboardTextActor3D()
        _wlbl.SetInput("")
        _wlbl.GetTextProperty().SetFontSize(14)
        _wlbl.GetTextProperty().SetColor(*_col)
        _wlbl.GetTextProperty().SetJustificationToCentered()
        _wlbl.PickableOff()
        _wlbl.VisibilityOff()
        overlay_renderer.AddActor(_wlbl)

        world_axes_line_sources.append(_ls)
        world_axes_cone_sources.append(_cs)
        world_axes_actors.append(_la)
        world_axes_actors.append(_ca)
        world_axes_labels.append(_wlbl)

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

    # --- Intensity beam-center cross --------------------------------------
    # Two perpendicular line segments marking the beam center over the
    # intensity frame (mirrors the napari "Beam Center" shapes layer).
    # Dragging the cross updates the Data-tab beam-center inputs, and editing
    # those inputs moves the cross.
    cross_state = {
        "active": False,
        "cx": 0.0, "cy": 0.0,   # display-space image coords (x=col, y=flipped row)
        "size": 20.0,           # half-length of each arm (world units)
        "z": 1.0,               # small +z offset so it draws on top of the frame
        "grab": False,          # True while the user is dragging the cross
    }
    cross_pts = vtkPoints()
    cross_pts.SetNumberOfPoints(4)
    cross_poly = vtkPolyData()
    cross_poly.SetPoints(cross_pts)
    _cross_cells = vtkCellArray()
    for _a, _b in ((0, 1), (2, 3)):
        _cross_seg = vtkLine()
        _cross_seg.GetPointIds().SetId(0, _a)
        _cross_seg.GetPointIds().SetId(1, _b)
        _cross_cells.InsertNextCell(_cross_seg)
    cross_poly.SetLines(_cross_cells)
    cross_mapper = vtkPolyDataMapper()
    cross_mapper.SetInputData(cross_poly)
    cross_actor = vtkActor()
    cross_actor.SetMapper(cross_mapper)
    cross_actor.GetProperty().SetColor(1.0, 0.25, 0.25)
    cross_actor.GetProperty().SetLineWidth(2.0)
    cross_actor.GetProperty().LightingOff()
    cross_actor.PickableOff()
    cross_actor.VisibilityOff()
    renderer.AddActor(cross_actor)

    # Interactor styles: the default trackball drives the 3D volume; a no-op
    # style locks the camera straight-on while the 2D ROI selector is active so
    # the screen<->image coordinate mapping stays exact.
    roi_trackball_style = vtkInteractorStyleTrackballCamera()
    # VTK's default mouse-wheel dolly (MouseWheelMotionFactor = 1.0) zooms ~21%
    # per wheel notch, which felt far too sensitive to control. Dial it down so
    # each notch nudges the zoom by a few percent for finer control.
    roi_trackball_style.SetMouseWheelMotionFactor(0.2)
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
    # Background tasks that drive the top-of-viewer progress bar. Determinate
    # steps (build/regrid) advance the percentage directly from per-frame
    # callbacks, but data loading has no granular progress, so ``progress_task``
    # runs an easing ticker that creeps toward 92% until the job finishes. The
    # hide task clears the bar a moment after it reaches 100%.
    progress_task = None
    progress_hide_task = None
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

    def _seed_clim(display: np.ndarray) -> Tuple[float, float]:
        """Seed the absolute contrast window from the percentile boxes.

        Mirrors napari's launch behaviour: the contrast_lo/contrast_hi
        percentiles pick an initial (lo, hi) *display-value* window, and the
        slider bounds (clim_min/clim_max) are set to the full display min/max.
        After this, the right-panel slider adjusts the absolute limits directly
        (linear mapping), so the contrast response matches the napari desktop
        app instead of re-running percentiles on every drag.
        """
        finite = display[np.isfinite(display)]
        if finite.size == 0:
            dmin, dmax = 0.0, 1.0
        else:
            dmin, dmax = float(finite.min()), float(finite.max())
        if dmax <= dmin:
            dmax = dmin + 1.0
        lo = _float(getattr(state, "contrast_lo", 1.0), 1.0)
        hi = _float(getattr(state, "contrast_hi", 99.8), 99.8)
        if not (0.0 <= lo < hi <= 100.0):
            lo, hi = 1.0, 99.8
        clo, chi = _robust_percentiles(display, (lo, hi))
        # Clamp the seed inside the slider bounds and keep a positive span.
        clo = max(dmin, min(clo, dmax))
        chi = max(dmin, min(chi, dmax))
        if chi <= clo:
            chi = min(dmax, clo + (dmax - dmin) * 1e-3)
            if chi <= clo:
                chi = clo + 1.0
        state.clim_min = dmin
        state.clim_max = dmax
        state.clim_step = max((dmax - dmin) / 1000.0, 1e-6)
        state.clim_lo = clo
        state.clim_hi = chi
        return (clo, chi)

    def _hide_volume_overlays():
        """Hide the 3D-only overlays when switching to the 2D intensity viewer.

        The outline box, its per-corner coordinate labels and the world-axes
        gizmo live in world/overlay coordinates tied to the RSM volume. When we
        drop into the 2D intensity frame viewer (a parallel camera framed over a
        small image) they would otherwise stay visible and project to a corner
        of the screen -- this is the "world axes clumped in the bottom-left"
        artifact seen after viewing the RSM and going back to the intensity
        view. Hiding the actors here (and clearing ``world_axes_geom["visible"]``
        so the per-render ``_rescale_world_axes`` observer stops repositioning
        them) removes the artifact. The ``outline_show`` / ``world_axes_show``
        state flags are left untouched, so returning to the RSM view via
        ``_set_volume_data`` -> ``_update_all_slices`` restores whatever the user
        had enabled.
        """
        outline_actor.VisibilityOff()
        for _lbl in outline_label_actors:
            _lbl.VisibilityOff()
        world_axes_geom["visible"] = False
        for _act in world_axes_actors:
            _act.VisibilityOff()
        for _lbl in world_axes_labels:
            _lbl.VisibilityOff()

    def _free_gpu_resources(*mappers):
        """Release cached GPU resources for the given mappers and run a GC pass.

        ``vtkSmartVolumeMapper`` and the 2D image reslice mappers cache their
        uploaded volume/texture on the GPU. Repeatedly switching between the RSM
        volume and the intensity viewer (or re-running load/build/regrid) leaves
        the superseded GPU uploads -- and the large numpy / ``vtkImageData``
        copies they were built from -- lingering, which is what made the server
        progressively slower and eventually crash after many round-trips.
        Dropping the resources for the view we are leaving lets VTK re-upload
        only what the active view needs, and the explicit ``gc.collect`` promptly
        reclaims the arrays that were just replaced.
        """
        for _m in mappers:
            try:
                _m.ReleaseGraphicsResources(render_window)
            except Exception:
                pass
        gc.collect()

    def _update_rendering():
        if current_volume is None:
            return
        state.scalar_range = state.scalar_range or "—"
        volume_property.SetShade(bool(state.shade))
        volume_property.SetAmbient(_float(state.ambient if hasattr(state, "ambient") else 0.2, 0.2))
        volume_property.SetDiffuse(_float(state.diffuse if hasattr(state, "diffuse") else 0.7, 0.7))
        volume_property.SetSpecular(_float(state.specular if hasattr(state, "specular") else 0.3, 0.3))
        volume_property.SetSpecularPower(_float(state.specular_power if hasattr(state, "specular_power") else 10.0, 10.0))
        
        # Map the napari-style rendering choice to a VTK blend mode. napari's
        # default is "attenuated_mip", a maximum-intensity projection -- VTK has
        # no attenuated-MIP mode, so MaximumIntensity is the faithful analogue.
        # Both "mip" and "attenuated_mip" therefore use MIP; only "translucent"
        # (or an explicit composite blend_mode) falls back to composite.
        # Previously "attenuated_mip" fell through to composite, so the default
        # web view used alpha-blended ray casting instead of napari's MIP, which
        # washed the peaks out and changed their apparent shape.
        rendering = _ensure_path(getattr(state, "rendering", "")) or "attenuated_mip"
        if rendering in ("mip", "attenuated_mip") or int(_float(state.blend_mode, 0)) == 1:
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
        # Free the GPU upload + vtkImageData behind the 2D intensity viewer we
        # are leaving, so repeated intensity<->RSM round-trips don't accumulate
        # graphics memory (see _free_gpu_resources).
        _free_gpu_resources(intensity_mapper)
        _roi_set_active(False)  # the ROI selector only applies to the intensity view
        _cross_set_active(False)  # the beam-center cross only applies to the intensity view
        renderer.GetActiveCamera().ParallelProjectionOff()  # restore 3D perspective
        state.intensity_slider_show = False
        state.intensity_playing = False  # halt any in-progress frame playback

        # Mirror napari's _ensure_ascending: flip any descending axis so the
        # volume is stored with monotonically increasing coordinates. VTK's
        # vtkImageData places voxel i at origin + i*spacing and assumes a
        # positive, uniform spacing; a descending regrid axis would otherwise
        # produce a negative spacing that mirrors the volume and corrupts the
        # gradient normals VTK uses for shading/opacity, so parts of the RSM
        # render faint or in the wrong place even though the overall shape is
        # still recognizable.
        vol = np.asarray(volume, dtype=np.float32)
        ax_list = [np.asarray(a, dtype=float) for a in axes]
        for _i in range(min(3, len(ax_list))):
            if ax_list[_i].size > 1 and ax_list[_i][1] < ax_list[_i][0]:
                ax_list[_i] = ax_list[_i][::-1].copy()
                vol = np.flip(vol, axis=_i)
        current_volume = np.ascontiguousarray(vol, dtype=np.float32)
        current_axes = tuple(ax_list)

        # Log-compress for display so the high-dynamic-range RSM is visible,
        # and derive the transfer-function range from user-inputted contrast percentiles.
        # The raw current_volume is kept untouched for VTR export.
        if bool(getattr(state, "log_view", True)):
            display_volume = _log1p_clip(current_volume)
        else:
            display_volume = np.maximum(current_volume, 0.0)
        render_range = _seed_clim(display_volume)

        image = vtkImageData()
        nx, ny, nz = display_volume.shape
        image.SetDimensions(nx, ny, nz)

        spacings = []
        origin = []
        for _ai, axis_values in enumerate(current_axes):
            axis_arr = np.asarray(axis_values, dtype=float)
            if axis_arr.size > 1:
                spacing = float(axis_arr[1] - axis_arr[0])
            else:
                spacing = 1.0
            if _ai == 2:
                # Display the reciprocal-space frame with napari's handedness so
                # +Qz points up (VTK is right-handed and would otherwise render
                # +Qz downward when +Qx is right and +Qy is out of the page).
                # Mirror the z world axis: true Qz value v renders at world
                # z = -v (spacing stays positive; origin = -zax[-1]). The raw
                # ``current_volume`` and ``current_axes`` keep the true Qz, so
                # export and the corner-coordinate labels are unaffected.
                spacings.append(spacing)
                origin.append(-float(axis_arr[-1]))
            else:
                spacings.append(spacing)
                origin.append(float(axis_arr[0]))

        image.SetSpacing(*spacings)
        image.SetOrigin(*origin)

        # Regridded RSM volumes contain non-finite voxels (NaN/inf) wherever a
        # grid cell received no detector samples -- mean-normalization divides
        # by a zero weight there. _log1p_clip propagates those NaNs, and VTK's
        # vtkColorTransferFunction renders NaN through its NanColor (a dark red
        # by default) instead of the colormap, tinting the whole RSM render with
        # the wrong color even though the colormap itself is correct. (The 2D
        # intensity view comes from raw detector frames with no gaps, so it
        # looked fine.) Replace non-finite voxels with the display floor so they
        # map to the fully transparent background instead of the NaN color.
        fill = float(render_range[0]) if render_range is not None else 0.0
        display_volume = np.where(np.isfinite(display_volume), display_volume, fill)

        # Reverse the volume along z so the mirrored origin above still places
        # each voxel at world z = -(its true Qz).
        display_for_vtk = display_volume[:, :, ::-1]
        vtk_array = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(display_for_vtk, dtype=np.float32).ravel(order="F"),
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
        _rebuild_volume_layers()
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
        if current_image is None or render_range is None:
            cyl_actor.VisibilityOff()
            return
        show = bool(state.cyl_show)
        if not show:
            cyl_actor.VisibilityOff()
            return

        # A cylindrical slice in reciprocal space is |Q_xy| = radius around the
        # Qz axis at the Q-space ORIGIN (Qx = Qy = 0), spanning the full Qz
        # range -- matching napari's _extract_cylindrical_surface_mesh
        # (qx = r*cos θ, qy = r*sin θ, sampled with np.argmin nearest-neighbor).
        #
        # We build and sample the mesh manually instead of using vtkProbeFilter.
        # vtkProbeFilter MASKS any surface point that falls outside the volume
        # bounds (setting it to 0), and a radius-~1 tube around the origin lies
        # almost entirely outside the small Qx/Qy extent of a typical RSM -- so
        # the probe returned no real data and the cylinder rendered as a flat,
        # uniform color unrelated to the map. Clamping each vertex to the
        # nearest edge voxel (like napari) guarantees every vertex carries a
        # meaningful intensity so the surface colors track the RSM.
        radius = _float(state.cyl_radius, 1.0)
        samples = max(8, int(_float(state.cyl_samples, 64)))

        dims = current_image.GetDimensions()
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
        if nx < 1 or ny < 1 or nz < 2:
            cyl_actor.VisibilityOff()
            return
        ox, oy, oz = (float(v) for v in current_image.GetOrigin())
        sx, sy, sz = (float(v) for v in current_image.GetSpacing())

        # Pull the display volume back out of current_image. It was uploaded as
        # an x-fastest F-ravel of a (nx, ny, nz) array (see _set_volume_data),
        # so a C-order reshape to (nz, ny, nx) indexes as vol[k, j, i].
        flat = numpy_support.vtk_to_numpy(current_image.GetPointData().GetScalars())
        vol = np.asarray(flat, dtype=np.float32).reshape(nz, ny, nx)

        # Circle of world (x, y) = (Qx, Qy) samples at |Q_xy| = radius; close
        # the loop by repeating the first angle at the end.
        theta = np.linspace(0.0, 2.0 * np.pi, samples + 1)
        n_theta = int(theta.size)
        xs = radius * np.cos(theta)
        ys = radius * np.sin(theta)

        # Nearest voxel index for each angle (clamped into range like np.argmin).
        i_idx = np.clip(np.round((xs - ox) / sx).astype(int), 0, nx - 1) if sx else np.zeros(n_theta, dtype=int)
        j_idx = np.clip(np.round((ys - oy) / sy).astype(int), 0, ny - 1) if sy else np.zeros(n_theta, dtype=int)

        # Sample every z-slab at the clamped (i, j) ring: shape (nz, n_theta).
        sampled = vol[:, j_idx, i_idx].astype(np.float32)

        # Vertex coordinates, ordered k (outer) then theta (inner) so vertex
        # index = k * n_theta + t matches ``sampled`` raveled in C order.
        zc = oz + np.arange(nz, dtype=np.float64) * sz
        coords = np.empty((nz * n_theta, 3), dtype=np.float64)
        coords[:, 0] = np.tile(xs, nz)
        coords[:, 1] = np.tile(ys, nz)
        coords[:, 2] = np.repeat(zc, n_theta)

        points = vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(coords, deep=True))

        poly = vtkPolyData()
        poly.SetPoints(points)
        cells = vtkCellArray()
        for k in range(nz - 1):
            base = k * n_theta
            nbase = base + n_theta
            for t in range(n_theta - 1):
                v00 = base + t
                v01 = base + t + 1
                v10 = nbase + t
                v11 = nbase + t + 1
                cells.InsertNextCell(3)
                cells.InsertCellPoint(v00)
                cells.InsertCellPoint(v01)
                cells.InsertCellPoint(v11)
                cells.InsertNextCell(3)
                cells.InsertCellPoint(v00)
                cells.InsertCellPoint(v11)
                cells.InsertCellPoint(v10)
        poly.SetPolys(cells)

        vtk_scalars = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(sampled.ravel(order="C")),
            deep=True,
            array_type=numpy_support.get_vtk_array_type(np.float32),
        )
        vtk_scalars.SetName("intensity")
        poly.GetPointData().SetScalars(vtk_scalars)

        lut = _make_lookup_table(_ensure_path(state.cyl_cmap), render_range)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(render_range[0], render_range[1])
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("intensity")
        mapper.SetColorModeToMapScalars()
        cyl_actor.SetMapper(mapper)
        cyl_actor.GetProperty().SetOpacity(_float(state.cyl_opacity, 0.7))
        cyl_actor.VisibilityOn()

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

    def _update_outline_box():
        """Draw a bounding-box wireframe around the volume with corner labels.

        The 8 corners are the min/max coordinate of each axis, read directly
        from ``current_axes`` (the regrid bin-center arrays). Using the axis
        endpoints -- rather than deriving the max from VTK image bounds
        (origin + spacing*(dim-1)) -- makes the labels match the napari desktop
        app exactly, which reports ``xax[0]``/``xax[-1]`` and friends. Corner
        ``c`` uses bit 4 for the x extreme, bit 2 for y and bit 1 for z
        (0 = min, 1 = max), matching the edge topology built at actor-creation
        time. Labels are formatted like napari (``Qx=..`` / ``H=..`` with 3
        decimals) so the two front-ends read identically.
        """
        show = bool(getattr(state, "outline_show", True))
        show_coords = bool(getattr(state, "coords_show", True))
        if current_image is None or current_axes is None:
            outline_actor.VisibilityOff()
            for label in outline_label_actors:
                label.VisibilityOff()
            return

        # Axis endpoints (min, max) straight from the regrid axes, matching the
        # values napari displays at the box corners.
        ax_x = np.asarray(current_axes[0], dtype=float)
        ax_y = np.asarray(current_axes[1], dtype=float)
        ax_z = np.asarray(current_axes[2], dtype=float)
        xs = (float(ax_x[0]), float(ax_x[-1]))
        ys = (float(ax_y[0]), float(ax_y[-1]))
        zs = (float(ax_z[0]), float(ax_z[-1]))

        # Match napari's per-space corner label (Qx/Qy/Qz for reciprocal space,
        # H/K/L for crystallographic space) so the coordinate readout is
        # identical across the desktop and web viewers.
        is_q = (_ensure_path(getattr(state, "space", "q")) or "q").lower() == "q"
        n0, n1, n2 = ("Qx", "Qy", "Qz") if is_q else ("H", "K", "L")

        for c in range(8):
            x = xs[1 if c & 4 else 0]
            y = ys[1 if c & 2 else 0]
            z = zs[1 if c & 1 else 0]
            # The z world axis is mirrored (see _set_volume_data) so +Qz points
            # up like napari; place the corner at world z = -z but keep the true
            # Qz value in the label text.
            wz = -z
            outline_pts.SetPoint(c, x, y, wz)
            label = outline_label_actors[c]
            label.SetInput(f"{n0}={x:.3f}, {n1}={y:.3f}, {n2}={z:.3f}")
            label.SetPosition(x, y, wz)
            label.SetVisibility(1 if show_coords else 0)
        outline_pts.Modified()
        outline_actor.SetVisibility(1 if show else 0)

    def _update_world_axes():
        """Draw +Qx/+Qy/+Qz direction arrows from the reciprocal-space origin.

        Mirrors napari's "World axes" overlay. The arrows are rooted at the
        origin ``(0, 0, 0)`` -- clamped into the data bounds so the gizmo stays
        on-screen if the origin lies outside the regridded volume -- and point
        toward increasing Qx (magenta), Qy (green) and Qz (cyan). This is a
        visual orientation cue only; it does not alter the coordinate values
        shown elsewhere. Labels switch to +H/+K/+L in crystallographic space to
        match the outline-box corner labels.

        This routine only fixes the arrows' *origin*, colors, labels and
        visibility. The actual on-screen length is set by
        ``_rescale_world_axes`` (called here and again on every render) so the
        arrows keep a constant pixel size no matter how far the volume is
        zoomed in or out.
        """
        show = bool(getattr(state, "world_axes_show", True))
        if current_image is None or current_axes is None or not show:
            world_axes_geom["visible"] = False
            for actor in world_axes_actors:
                actor.VisibilityOff()
            for label in world_axes_labels:
                label.VisibilityOff()
            return

        ax_x = np.asarray(current_axes[0], dtype=float)
        ax_y = np.asarray(current_axes[1], dtype=float)
        ax_z = np.asarray(current_axes[2], dtype=float)

        # Root the arrows at the reciprocal-space origin (0, 0, 0). If the
        # origin falls outside the regridded extents, clamp each component into
        # the data bounds so the gizmo remains visible next to the volume.
        def _clamp(value, axis):
            lo, hi = float(axis[0]), float(axis[-1])
            if lo > hi:
                lo, hi = hi, lo
            return min(max(value, lo), hi)

        ox = _clamp(0.0, ax_x)
        oy = _clamp(0.0, ax_y)
        oz_true = _clamp(0.0, ax_z)
        # The z world axis is mirrored (see _set_volume_data) so +Qz points up
        # like napari: root the gizmo at world z = -oz_true and make the +Qz
        # arrow point along world -z (via world_axes_geom["signs"]).
        world_axes_geom["origin"] = (ox, oy, -oz_true)
        world_axes_geom["signs"] = (1.0, 1.0, -1.0)
        world_axes_geom["visible"] = True

        is_q = (_ensure_path(getattr(state, "space", "q")) or "q").lower() == "q"
        names = ("+Qx", "+Qy", "+Qz") if is_q else ("+H", "+K", "+L")
        for i in range(3):
            world_axes_labels[i].SetInput(names[i])
            world_axes_labels[i].VisibilityOn()
        for actor in world_axes_actors:
            actor.VisibilityOn()

        # Size the arrows for the current camera so they appear immediately at
        # the right on-screen scale (the render observer keeps them there).
        _rescale_world_axes()

    def _rescale_world_axes(*_):
        """Resize the world-axes arrows to a constant on-screen (pixel) length.

        Registered as a renderer ``StartEvent`` observer so it runs before
        every render with the current camera. It converts a target pixel length
        into world units at the gizmo's origin -- using the parallel scale for
        an orthographic camera, or the perspective view angle and camera
        distance otherwise -- and applies it to the line shafts, cone
        arrowheads and tip labels. Because the world length is recomputed each
        frame, the arrows stay the same size on screen at any zoom level.
        """
        if not world_axes_geom.get("visible"):
            return
        origin = world_axes_geom.get("origin")
        if origin is None:
            return
        ox, oy, oz = origin

        size = render_window.GetSize()
        win_h = int(size[1]) if size and len(size) > 1 else 0
        if win_h <= 0:
            return
        cam = renderer.GetActiveCamera()
        if cam is None:
            return

        if cam.GetParallelProjection():
            # Parallel scale is half the viewport height in world units.
            world_per_px = (2.0 * float(cam.GetParallelScale())) / win_h
        else:
            cpos = np.asarray(cam.GetPosition(), dtype=float)
            fpos = np.asarray(cam.GetFocalPoint(), dtype=float)
            view_dir = fpos - cpos
            norm = float(np.linalg.norm(view_dir))
            if norm == 0.0:
                return
            view_dir /= norm
            # Distance from the camera to the gizmo origin along the view axis.
            dist = abs(float(np.dot(np.array([ox, oy, oz]) - cpos, view_dir)))
            if dist <= 0.0:
                dist = 1e-6
            view_angle = np.deg2rad(float(cam.GetViewAngle()))
            world_per_px = (2.0 * dist * np.tan(view_angle / 2.0)) / win_h

        length = float(world_axes_geom.get("target_px", 90.0)) * world_per_px
        if not np.isfinite(length) or length <= 0.0:
            return

        # Per-axis direction signs mirror an axis to match napari's handedness
        # (the z world axis is flipped so +Qz points up).
        signs = world_axes_geom.get("signs", (1.0, 1.0, 1.0))
        dirs = (
            (signs[0], 0.0, 0.0),
            (0.0, signs[1], 0.0),
            (0.0, 0.0, signs[2]),
        )
        tips = (
            (ox + signs[0] * length, oy, oz),
            (ox, oy + signs[1] * length, oz),
            (ox, oy, oz + signs[2] * length),
        )
        for i in range(3):
            src = world_axes_line_sources[i]
            src.SetPoint1(ox, oy, oz)
            src.SetPoint2(*tips[i])
            src.Modified()
            cone = world_axes_cone_sources[i]
            cone.SetCenter(*tips[i])
            cone.SetDirection(*dirs[i])
            cone.SetHeight(0.28 * length)
            cone.SetRadius(0.09 * length)
            cone.Modified()
            world_axes_labels[i].SetPosition(*tips[i])

    # Keep the arrows a constant on-screen size: recompute their world length
    # from the camera at the start of every render (covers interactive zoom).
    renderer.AddObserver("StartEvent", _rescale_world_axes)

    def _update_all_slices():
        try:
            _update_outline_box()
            _update_world_axes()
            for axis in ("x", "y", "z"):
                _update_ortho_slice(axis)
            _update_cylinder()
            _update_sphere()
        except Exception as exc:  # never let slicing break the main render
            _set_status(f"Slice update skipped: {exc}")

    @ctrl.set("update_slices")
    def update_slices(**kwargs):
        _update_all_slices()
        _rebuild_volume_layers()
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
                process_hklscan_only=bool(getattr(state, "only_hkl", False)),
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

    def _read_profile_section(loader_mode):
        """Return the YAML profile dict matching the chosen ``loader_mode``.

        The bundled loaders always read ExperimentSetup from the YAML's
        ``active_profile`` section, so the Data-tab ISR/CMS choice would
        otherwise be ignored. We read the requested profile directly here
        (web-app only) so the selected beamline's parameters drive the UI and
        the build instead of whatever happens to be the active profile on disk.
        """
        path = Path(_ensure_path(state.setup_path)).expanduser()
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                doc = yaml.safe_load(handle) or {}
        except Exception:
            return None
        profiles = doc.get("profiles")
        if not isinstance(profiles, dict):
            return None
        target = _ensure_path(loader_mode).upper()
        for name, section in profiles.items():
            if str(name).upper() == target and isinstance(section, dict):
                return section
        return None

    def _populate_from_profile(loader_mode):
        """Fill the Data/Build/View fields from the chosen beamline profile.

        Runs immediately when the loader mode changes (and at startup) so the
        experimental setup values and related parameters reflect the ISR/CMS
        choice before any TIFF directory is picked or data is loaded.
        """
        section = _read_profile_section(loader_mode)
        if section is None:
            return

        setup = section.get("ExperimentSetup")
        if isinstance(setup, dict):
            state.exp_distance = _float(setup.get("distance"), state.exp_distance)
            state.exp_pitch = _float(setup.get("pitch"), state.exp_pitch)
            state.exp_det_h = int(_float(setup.get("ypixels"), state.exp_det_h))
            state.exp_det_w = int(_float(setup.get("xpixels"), state.exp_det_w))
            state.exp_bc_h = int(_float(setup.get("ycenter"), state.exp_bc_h))
            state.exp_bc_w = int(_float(setup.get("xcenter"), state.exp_bc_w))
            energy = _float(setup.get("energy"), 0.0)
            if energy > 0:
                # Setting energy keeps wavelength in sync via _on_energy_change.
                state.exp_energy = energy
            else:
                wavelength = _float(setup.get("wavelength"), 0.0)
                if wavelength > 0:
                    state.exp_wavelength = wavelength

        crystal = section.get("Crystal")
        if isinstance(crystal, dict) and crystal.get("ub") is not None:
            try:
                ub_arr = _parse_ub_matrix(str(crystal["ub"]))
            except ValueError:
                ub_arr = None
            if ub_arr is not None:
                state.ub_matrix = _format_ub_matrix(ub_arr)

        build = section.get("build")
        if isinstance(build, dict):
            if build.get("sample_axes") is not None:
                state.sample_axes = str(build["sample_axes"])
            if build.get("detector_axes") is not None:
                state.detector_axes = str(build["detector_axes"])
            if build.get("ub_includes_2pi") is not None:
                state.ub_includes_2pi = bool(build["ub_includes_2pi"])
            if build.get("center_is_one_based") is not None:
                state.one_based_center = bool(build["center_is_one_based"])

        regrid = section.get("regrid")
        if isinstance(regrid, dict):
            if regrid.get("space") is not None:
                state.space = str(regrid["space"])
            if regrid.get("grid_shape") is not None:
                state.grid_shape = str(regrid["grid_shape"])
            if regrid.get("normalize") is not None:
                state.normalize = str(regrid["normalize"])
            if regrid.get("fuzzy") is not None:
                state.fuzzy_gridder = bool(regrid["fuzzy"])
            if regrid.get("fuzzy_width") is not None:
                state.width_fuzzy = _float(regrid.get("fuzzy_width"), state.width_fuzzy)

        data = section.get("data")
        if isinstance(data, dict) and data.get("cms_angle_step") is not None:
            state.cms_angle_step = _float(data.get("cms_angle_step"), state.cms_angle_step)

        view = section.get("view")
        if isinstance(view, dict):
            if view.get("log_view") is not None:
                state.log_view = bool(view["log_view"])
            if view.get("cmap") is not None:
                state.colormap = str(view["cmap"])
            if view.get("rendering") is not None:
                state.rendering = str(view["rendering"])
            if view.get("contrast_lo") is not None:
                state.contrast_lo = _float(view.get("contrast_lo"), state.contrast_lo)
            if view.get("contrast_hi") is not None:
                state.contrast_hi = _float(view.get("contrast_hi"), state.contrast_hi)

    def _override_setup_with_profile(setup, loader_mode):
        """Force a loaded setup's geometry to match the chosen profile.

        The bundled loaders read ExperimentSetup from the YAML ``active_profile``
        section, which need not match the Data-tab ISR/CMS choice. We overwrite
        the geometry here so the loaded ``setup`` (and everything built from it:
        crop adjustments, the displayed fields, and the regrid) reflects the
        selected beamline rather than the active profile on disk.
        """
        if setup is None:
            return
        section = _read_profile_section(loader_mode)
        if section is None:
            return
        exp = section.get("ExperimentSetup")
        if not isinstance(exp, dict):
            return
        try:
            distance = _float(exp.get("distance"), 0.0)
            if distance > 0:
                setup.distance = distance
            pitch = _float(exp.get("pitch"), 0.0)
            if pitch > 0:
                setup.pitch = pitch
            ypixels = int(_float(exp.get("ypixels"), 0))
            if ypixels > 0:
                setup.ypixels = ypixels
            xpixels = int(_float(exp.get("xpixels"), 0))
            if xpixels > 0:
                setup.xpixels = xpixels
            setup.ycenter = int(_float(exp.get("ycenter"), setup.ycenter))
            setup.xcenter = int(_float(exp.get("xcenter"), setup.xcenter))
            energy = _float(exp.get("energy"), 0.0)
            if energy > 0:
                setup.energy = energy
                setup.energy_keV = energy
                setup.wavelength = 12.398419843320026 / energy
            else:
                wavelength = _float(exp.get("wavelength"), 0.0)
                if wavelength > 0:
                    setup.wavelength = wavelength
                    setup.energy_keV = 12.398419843320026 / wavelength
                    setup.energy = setup.energy_keV
        except (TypeError, ValueError):
            pass

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

    def _compute_builder(setup, ub, df, sample_axes, detector_axes,
                         ub_includes_2pi, center_is_one_based,
                         progress_callback=None):
        builder = RSMBuilder(
            setup,
            ub,
            df,
            sample_axes=sample_axes or None,
            detector_axes=detector_axes or None,
            ub_includes_2pi=ub_includes_2pi,
            center_is_one_based=center_is_one_based,
        )
        builder.compute_full(verbose=False, progress_callback=progress_callback)
        return builder

    def _regrid_volume(builder, grid_shape, progress_callback=None):
        fuzzy = bool(getattr(state, "fuzzy_gridder", False))
        kwargs = dict(
            space=_ensure_path(state.space) or "q",
            grid_shape=grid_shape,
            normalize=_ensure_path(state.normalize) or "mean",
            fuzzy=fuzzy,
            # Accumulate the scattered points frame-by-frame (mirrors the napari
            # widget). Without streaming, regrid_xu ravels every frame's points
            # into one giant array, which can need gigabytes of RAM and fail to
            # allocate for large scans.
            stream=True,
        )
        width = _float(getattr(state, "width_fuzzy", 0.0), 0.0)
        if fuzzy and width > 0:
            kwargs["width"] = width
        return builder.regrid_xu(progress_callback=progress_callback, **kwargs)

    def _track(coro):
        """Run a coroutine as a cancellable task (for the Stop button)."""
        nonlocal current_task
        loop = asyncio.get_event_loop()
        current_task = loop.create_task(coro)
        return current_task

    # ---------------------------------------------------------------------
    # Progress bar helpers
    # ---------------------------------------------------------------------
    def _apply_progress_value(pct: float):
        """Set the determinate progress percentage (runs on the event loop)."""
        with state:
            # Cap below 100 while work is ongoing; _stop_progress sets the
            # final 100 so the bar never shows 100% before the job returns.
            state.progress_value = round(min(max(float(pct), 0.0), 99.0), 1)

    def _make_thread_progress_cb():
        """Return a thread-safe ``cb(done, total)`` for executor jobs.

        The loaders/builders run in a thread pool, so their per-frame progress
        must be marshaled back onto the server event loop. Updates are throttled
        to whole-percent steps to avoid flooding the client with state pushes
        (a build can iterate hundreds of frames).
        """
        loop = asyncio.get_event_loop()
        last = {"pct": -1}

        def cb(done, total):
            pct = 100.0 * float(done) / float(max(1, total))
            ip = int(pct)
            if ip <= last["pct"]:
                return
            last["pct"] = ip
            loop.call_soon_threadsafe(_apply_progress_value, pct)

        return cb

    async def _hide_progress_later():
        """Clear the finished progress bar after a brief 100% hold."""
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return
        with state:
            state.progress_active = False
            state.progress_value = 0.0

    async def _progress_ticker():
        """Ease the progress bar toward ~92% while an untracked job runs.

        Data loading can't report granular progress, so the bar creeps up with
        a decelerating step (never exceeding 92%) to convey ongoing work.
        ``_stop_progress`` cancels this and snaps the bar to 100% when the job
        actually finishes.
        """
        try:
            while True:
                await asyncio.sleep(0.15)
                v = float(getattr(state, "progress_value", 0.0) or 0.0)
                v += max(0.4, (92.0 - v) * 0.06)
                with state:
                    state.progress_value = round(min(v, 92.0), 1)
        except asyncio.CancelledError:
            pass

    def _start_progress(label: str, indeterminate: bool = False,
                        ticker: bool = False):
        """Show the progress bar for a new job.

        ``indeterminate=True`` shows a sliding animation with no number for
        steps that can't report progress (export). ``ticker=True`` runs the
        easing ticker toward 92% (data loading). Otherwise the fill tracks
        ``progress_value``, which a per-frame callback advances (build/regrid).
        """
        nonlocal progress_task, progress_hide_task
        loop = asyncio.get_event_loop()
        if progress_hide_task is not None:
            progress_hide_task.cancel()
            progress_hide_task = None
        if progress_task is not None:
            progress_task.cancel()
            progress_task = None
        state.progress_active = True
        state.progress_indeterminate = bool(indeterminate)
        state.progress_value = 0.0
        state.progress_label = label
        state.flush()
        if ticker:
            progress_task = loop.create_task(_progress_ticker())

    def _stop_progress(success: bool = True):
        """Finish the current job's progress bar.

        On success the bar snaps to 100% and auto-hides shortly after; on
        failure/cancellation it is hidden immediately.
        """
        nonlocal progress_task, progress_hide_task
        loop = asyncio.get_event_loop()
        if progress_task is not None:
            progress_task.cancel()
            progress_task = None
        if success:
            state.progress_indeterminate = False
            state.progress_value = 100.0
            state.flush()
            progress_hide_task = loop.create_task(_hide_progress_later())
        else:
            state.progress_active = False
            state.progress_value = 0.0
            state.flush()

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
            _start_progress("Loading data", ticker=True)
            setup, ub, df, frames = await loop.run_in_executor(
                None, _load_experiment, loader_mode
            )

            current_setup, current_ub, current_df, current_frames = setup, ub, df, frames
            current_builder = None
            regrid_volume = None
            regrid_axes = None
            # A fresh load supersedes the previous experiment/dataframe/frames
            # and any built map or regridded volume; reclaim them now so
            # repeated loads don't pile up in RAM.
            gc.collect()
            state.intensity_slider_show = False
            # The bundled loaders read ExperimentSetup from the YAML's
            # active_profile section, which may differ from the Data-tab ISR/CMS
            # choice. Force the loaded geometry to the selected profile so the
            # displayed fields, crop adjustments, and build all stay consistent.
            _override_setup_with_profile(current_setup, loader_mode)
            _populate_setup_fields(current_setup, frames)
            # Reflect the loaded UB into the editable Build-tab field.
            if ub is not None:
                state.ub_matrix = _format_ub_matrix(ub)
            n = len(frames) if frames else 0
            _set_status(f"Data loaded ({n} frame(s)). Ready to build.")
            _stop_progress(success=True)
        except asyncio.CancelledError:
            _set_status("Load cancelled.")
            _stop_progress(success=False)
        except Exception as exc:
            _set_status(f"Load error: {exc}")
            _stop_progress(success=False)

    @ctrl.set("load_data")
    def load_data(**kwargs):
        _track(_do_load_data())
        
    async def _do_build_rsm():
        nonlocal current_builder
        if current_setup is None or current_df is None:
            _set_status("Load data first.")
            return
        _apply_setup_overrides(current_setup)

        # Parse the editable Build-tab parameters on the main thread (so a bad
        # UB is reported before the worker runs). Fall back to the loaded UB
        # when the field is blank.
        try:
            ub_arr = _parse_ub_matrix(getattr(state, "ub_matrix", ""))
        except ValueError as exc:
            _set_status(f"Invalid UB: {exc}")
            return
        if ub_arr is None:
            ub_arr = current_ub
        sample_axes = _parse_axes_list(getattr(state, "sample_axes", ""))
        detector_axes = _parse_axes_list(getattr(state, "detector_axes", ""))
        ub_includes_2pi = bool(getattr(state, "ub_includes_2pi", True))
        center_is_one_based = bool(getattr(state, "one_based_center", False))

        loop = asyncio.get_event_loop()
        try:
            _set_status("Computing Q/HKL mapping...")
            _start_progress("Building RSM")
            build_cb = _make_thread_progress_cb()
            current_builder = await loop.run_in_executor(
                None,
                _compute_builder,
                current_setup,
                ub_arr,
                current_df,
                sample_axes,
                detector_axes,
                ub_includes_2pi,
                center_is_one_based,
                build_cb,
            )
            _set_status("RSM map built. Ready to regrid.")
            _stop_progress(success=True)
        except asyncio.CancelledError:
            _set_status("Build cancelled.")
            _stop_progress(success=False)
        except Exception as exc:
            _set_status(f"Build error: {exc}")
            _stop_progress(success=False)

    @ctrl.set("build_rsm")
    def build_rsm(**kwargs):
        _track(_do_build_rsm())

    async def _do_regrid():
        nonlocal regrid_volume, regrid_axes
        if current_builder is None:
            _set_status("Build the RSM map first.")
            return
        # Parse the grid-shape string on the main thread so a malformed value
        # is reported before the worker runs. nx is required; ny/nz may be
        # auto-scaled ("*").
        try:
            grid_shape = _parse_grid_shape(getattr(state, "grid_shape", ""))
        except ValueError as exc:
            _set_status(f"Invalid grid: {exc}")
            return
        if grid_shape[0] is None:
            _set_status("Grid X (first value) is required, e.g. 90,*,*")
            return
        loop = asyncio.get_event_loop()
        try:
            shape_label = ",".join("*" if d is None else str(d) for d in grid_shape)
            _set_status(f"Regridding to ({shape_label}) volume...")
            _start_progress("Regridding")
            regrid_cb = _make_thread_progress_cb()
            volume, axes = await loop.run_in_executor(
                None, _regrid_volume, current_builder, grid_shape, regrid_cb
            )
            regrid_volume, regrid_axes = volume, axes
            # Drop the previous regridded volume (a full 3D array) that this run
            # just replaced, so repeated regrids don't accumulate.
            gc.collect()
            state.scalar_range = (
                f"{float(np.nanmin(volume)):.4g} … {float(np.nanmax(volume)):.4g}"
            )
            state.volume_dims = (
                f"{volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]}"
            )
            _set_status("Regrid complete. Use View RSM to display.")
            _stop_progress(success=True)
        except asyncio.CancelledError:
            _set_status("Regrid cancelled.")
            _stop_progress(success=False)
        except Exception as exc:
            _set_status(f"Regrid error: {exc}")
            _stop_progress(success=False)

    @ctrl.set("regrid")
    def regrid(**kwargs):
        _track(_do_regrid())

    @ctrl.set("view_rsm")
    def view_rsm(**kwargs):
        if regrid_volume is None or regrid_axes is None:
            _set_status("Regrid first.")
            return
        _set_status("Updating 3D view...")
        # Reset the contrast limits to their defaults each time the RSM is
        # (re)viewed, so the user starts from the standard 1-99.8% window and
        # can then fine-tune it with the right-panel slider. Set before
        # _set_volume_data so it frames the transfer functions over the defaults.
        state.contrast_lo = 1.0
        state.contrast_hi = 99.8
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

    def _update_interactor_style():
        # Lock the camera straight-on while either the ROI selector or the
        # beam-center cross is active so the screen<->image mapping used for
        # dragging stays exact; otherwise restore the 3D trackball.
        if roi_state["active"] or cross_state["active"]:
            interactor.SetInteractorStyle(roi_lock_style)
        else:
            interactor.SetInteractorStyle(roi_trackball_style)

    def _roi_set_active(active):
        roi_state["active"] = bool(active)
        if active:
            _roi_set_visible(True)
            _roi_refresh_actors()
        else:
            roi_state["grab"] = None
            _roi_set_visible(False)
        _update_interactor_style()

    # --- Beam-center cross helpers -----------------------------------------
    # World space vs. display space: the cross is drawn in world space, but the user interacts with it in display space. 
    # The cross's world-space position is stored in the state (exp_bc_w / exp_bc_h) and is used to update the cross actor's points. 
    # When the user drags the cross, the display-space position is converted back to world space and pushed to the state.
    def _cross_position_from_state():
        """Place the cross at the beam center stored in exp_bc_w / exp_bc_h."""
        if intensity_nx is None or intensity_ny is None:
            return
        nx, ny = intensity_nx, intensity_ny
        col = max(0, min(int(_float(getattr(state, "exp_bc_w", 0), 0)), nx - 1))
        row = max(0, min(int(_float(getattr(state, "exp_bc_h", 0), 0)), ny - 1))
        cross_state["cx"] = float(col)
        # The frame is displayed vertically flipped, so the world row is
        # measured from the bottom (see _show_intensity_frame).
        cross_state["cy"] = float((ny - 1) - row)

    def _cross_refresh_actors(): 
        """ Update the cross actor's points from the current state. """
        cx, cy = cross_state["cx"], cross_state["cy"]
        s = cross_state["size"]
        z = cross_state["z"]
        cross_pts.SetPoint(0, cx, cy - s, z)
        cross_pts.SetPoint(1, cx, cy + s, z)
        cross_pts.SetPoint(2, cx - s, cy, z)
        cross_pts.SetPoint(3, cx + s, cy, z)
        cross_pts.Modified()

    def _cross_init_geometry(): 
        """ Size the cross from the current frame dimensions and camera zoom. """
        nx = intensity_nx or 1
        ny = intensity_ny or 1
        cross_state["size"] = max(5.0, min(nx, ny) * 0.08)
        cross_state["z"] = max(1.0, 0.01 * max(nx, ny)) + 0.5
        cross_state["grab"] = False
        _cross_position_from_state()
        _cross_refresh_actors()

    def _cross_set_active(active): 
        """ Show/hide cross and update interacor style. """
        cross_state["active"] = bool(active)
        cross_actor.SetVisibility(1 if active else 0)
        if not active:
            cross_state["grab"] = False
        _update_interactor_style()

    def _seg_dist(px, py, a, b):
        """Distance from point (px, py) to the segment a->b (display space)."""
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        denom = vx * vx + vy * vy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        qx, qy = ax + t * vx, ay + t * vy
        return math.hypot(px - qx, py - qy)

    def _cross_hit_test(dx, dy): 
        """
        Check if the display-space point (dx, dy) is within 9px of the cross. 
        Need to do this because the cross is drawn with a fixed line width, so the actual world-space lines are too thin to reliably pick.
        """
        cx, cy, s = cross_state["cx"], cross_state["cy"], cross_state["size"]
        top = _world_to_display(cx, cy + s)
        bot = _world_to_display(cx, cy - s)
        left = _world_to_display(cx - s, cy)
        right = _world_to_display(cx + s, cy)
        d = min(_seg_dist(dx, dy, top, bot), _seg_dist(dx, dy, left, right))
        return d <= 9.0

    def _cross_set_world(wx, wy): 
        """ Update the cross position from a world-space point (wx, wy) and push it to the Data-tab beam-center inputs. """
        if intensity_nx is None or intensity_ny is None:
            return
        nx, ny = intensity_nx, intensity_ny
        col = int(round(max(0.0, min(wx, nx - 1))))
        disp_row = int(round(max(0.0, min(wy, ny - 1))))
        cross_state["cx"] = float(col)
        cross_state["cy"] = float(disp_row)
        _cross_refresh_actors()
        # Convert the displayed (flipped) row back to the original frame row
        # before pushing it to the Data-tab beam-center inputs.
        state.exp_bc_w = col
        state.exp_bc_h = (ny - 1) - disp_row
        state.flush()
        _roi_render()

    def _roi_on_press(obj, event):
        dx, dy = interactor.GetEventPosition()
        # The beam-center cross takes priority over the ROI body.
        if cross_state["active"] and _cross_hit_test(dx, dy):
            cross_state["grab"] = True
            return
        if not roi_state["active"]:
            return
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
        dx, dy = interactor.GetEventPosition()
        if cross_state["active"] and cross_state["grab"]:
            wx, wy = _display_to_world(dx, dy)
            _cross_set_world(wx, wy)
            return
        if not roi_state["active"] or roi_state["grab"] is None:
            return
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
        if cross_state["active"] and cross_state["grab"]:
            cross_state["grab"] = False
            return
        if roi_state["grab"] is not None:
            roi_state["grab"] = None
            _roi_update_crop_state()
            _roi_render()

    def _roi_rescale_handles():
        """Resize the ROI handle disks to the current camera zoom.

        The handle radius is a fixed fraction of the parallel scale so the
        disks keep a constant on-screen size, matching _roi_init_geometry. This
        is called after a wheel zoom (which changes the parallel scale) without
        resetting the ROI box position/size.
        """
        cam = renderer.GetActiveCamera()
        if cam.GetParallelProjection():
            ps = cam.GetParallelScale()
        else:
            ps = max(intensity_nx or 1, intensity_ny or 1) * 0.5
        roi_state["handle_r"] = max(ps, 1e-6) * 0.02
        roi_handle_src.SetRadius(roi_state["handle_r"])
        roi_rot_src.SetRadius(roi_state["handle_r"])

    def _intensity_wheel(obj, event):
        """Zoom the intensity frame while the camera is locked.

        The ROI selector / beam-center cross swap in ``roi_lock_style``
        (vtkInteractorStyleUser), which has no wheel binding, so scrolling did
        nothing over the intensity map. Handle the wheel here by adjusting the
        parallel scale directly (the frame uses parallel projection). When the
        3D trackball is active we return early and let it handle the wheel.
        """
        if not (roi_state["active"] or cross_state["active"]):
            return
        cam = renderer.GetActiveCamera()
        # Forward = zoom in (smaller parallel scale). ~10% per notch keeps it
        # gentle and roughly matches the RSM viewer's reduced sensitivity.
        factor = 0.9 if event == "MouseWheelForwardEvent" else 1.0 / 0.9
        if cam.GetParallelProjection():
            cam.SetParallelScale(max(cam.GetParallelScale() * factor, 1e-6))
        else:
            cam.Dolly(1.0 / factor)
        renderer.ResetCameraClippingRange()
        if roi_state["active"]:
            _roi_rescale_handles()
            _roi_refresh_actors()
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    # Observe the forwarded mouse events (priority above the camera style so a
    # grab on a handle/body edits the ROI rather than moving the camera).
    interactor.AddObserver("LeftButtonPressEvent", _roi_on_press, 10.0)
    interactor.AddObserver("MouseMoveEvent", _roi_on_move, 10.0)
    interactor.AddObserver("LeftButtonReleaseEvent", _roi_on_release, 10.0)
    interactor.AddObserver("MouseWheelForwardEvent", _intensity_wheel, 10.0)
    interactor.AddObserver("MouseWheelBackwardEvent", _intensity_wheel, 10.0)

    @state.change("roi_show")
    def _on_roi_show_change(roi_show=True, **kwargs):
        # Only meaningful while the intensity frame viewer is active.
        if not bool(getattr(state, "intensity_slider_show", False)):
            return
        if roi_show and intensity_nx is not None:
            # Only seed a fresh box when the ROI isn't already on screen, so
            # re-showing a hidden ROI keeps the user's previous placement.
            if not roi_state["active"]:
                _roi_init_geometry()
                _roi_set_active(True)
            _roi_update_crop_state()
        else:
            _roi_set_active(False)
        _sync_layer_visible("roi", bool(roi_show))
        _roi_render()

    @state.change("exp_bc_w", "exp_bc_h")
    def _on_beam_center_change(**kwargs):
        # Reposition the cross when the user edits the beam-center inputs.
        # Skip while the cross is being dragged (the drag already set these).
        if not cross_state["active"] or cross_state["grab"]:
            return
        _cross_position_from_state()
        _cross_refresh_actors()
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
        # Hide the outline box / corner labels / world-axes gizmo that belong to
        # the 3D RSM view; otherwise they linger and project into a corner of
        # the 2D frame viewer.
        _hide_volume_overlays()
        # Release the RSM volume's GPU upload (and reclaim the superseded arrays)
        # now that the 3D view is no longer visible, so repeated switches don't
        # accumulate graphics memory.
        _free_gpu_resources(volume_mapper, *slice_mappers.values())
        state.intensity_frame_max = n - 1
        state.intensity_slider_show = True
        if int(_float(state.intensity_frame_index, 0)) != 0:
            # Reset to the first frame; the change handler will render it.
            state.intensity_frame_index = 0
        try:
            _show_intensity_frame(0, reset_camera=True, reseed=True)
        except Exception as exc:
            _set_status(f"Intensity view error: {exc}")
            return
        # Beam-center cross marker, always shown over the 2D frame so the user
        # can drag it to set the beam center (updates the Data-tab inputs).
        _cross_init_geometry()
        _cross_set_active(True)
        # Drop an adjustable ROI box on the frame and seed the Crop inputs from
        # it so the user can move/resize/rotate it to refine the crop.
        if bool(getattr(state, "roi_show", True)):
            _roi_init_geometry()
            _roi_set_active(True)
            _roi_update_crop_state()
        else:
            _roi_set_active(False)
        _roi_render()
        _set_layers([
            ("intensity_map", "Intensity"),
            ("roi", "ROI"),
            ("cross", "Beam Center"),
        ])
        _set_status(f"Viewing intensity frames (0–{n - 1}). Use the slider to scrub.")

    def _show_intensity_frame(index, reset_camera=False, reseed=False):
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
        # over the absolute contrast window. On the initial view we seed the
        # window from the percentile boxes; subsequent renders (frame scrub or
        # slider drag) reuse the current clim_lo/clim_hi so the contrast the
        # user set persists across frames, matching napari.
        if bool(getattr(state, "log_view", True)):
            disp = _log1p_clip(frame)
        else:
            disp = np.maximum(frame, 0.0)
        if reseed:
            frange = _seed_clim(disp)
        else:
            frange = (
                _float(getattr(state, "clim_lo", 0.0), 0.0),
                _float(getattr(state, "clim_hi", 1.0), 1.0),
            )
            if frange[1] <= frange[0]:
                frange = _seed_clim(disp)

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
                # Push the live Data-tab values (including any beam center the
                # user dragged/typed on the intensity view) onto the setup
                # FIRST, so the crop shift is applied to the current beam
                # center -- not the stale loaded one. This mirrors napari's
                # on_crop_from_roi, which shifts the live widget values by the
                # crop origin so the cross stays put relative to the pattern.
                _apply_setup_overrides(current_setup)
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
                # re-placed at the correct spot right after.
                _roi_set_active(False)
                # Capture the OLD frame height before the redisplay overwrites
                # intensity_ny; it is needed to remap the ROI through the
                # vertical flip into the cropped frame's coordinates.
                h_old = intensity_ny
                _show_intensity_frame(idx, reset_camera=True)
                # Re-place the beam-center cross for the cropped frame (its
                # dimensions and the beam center both changed).
                _cross_init_geometry()
                _cross_set_active(True)
                if bool(getattr(state, "roi_show", True)):
                    # Keep the ROI box over the same physical region of the
                    # (now cropped) intensity map instead of snapping back to a
                    # default centered box (mirrors the beam-center cross).
                    # Columns shift by the crop origin c0; because the frame is
                    # displayed vertically flipped, world-y shifts by
                    # (r1 - h_old). Box size and rotation are unchanged.
                    old_cx, old_cy = roi_state["cx"], roi_state["cy"]
                    old_hw, old_hh = roi_state["hw"], roi_state["hh"]
                    old_angle = roi_state["angle"]
                    _roi_init_geometry()  # recompute handle_r / z for new frame+zoom
                    if h_old is not None:
                        roi_state["cx"] = old_cx - c0
                        roi_state["cy"] = old_cy - (h_old - r1)
                        roi_state["hw"] = old_hw
                        roi_state["hh"] = old_hh
                        roi_state["angle"] = old_angle
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
            _start_progress("Exporting VTR", indeterminate=True)
            await loop.run_in_executor(
                None,
                lambda: write_rsm_volume_to_vtr(
                    current_volume, current_axes, str(output_path), binary=True, compress=True
                ),
            )
            state.export_path = str(output_path)
            _set_status(f"Exported VTR to {output_path}")
            _stop_progress(success=True)
        except Exception as exc:
            _set_status(f"Export failed: {exc}")
            _stop_progress(success=False)

    @ctrl.set("export_tiff")
    async def export_tiff(**kwargs):
        # Export the raw regridded 3D grid as a compressed TIFF (mirrors the
        # napari widget's on_export_grid).
        if regrid_volume is None:
            _set_status("Regrid first, then export.")
            return
        try:
            import tifffile
        except ImportError:
            _set_status("Export failed: tifffile is not installed.")
            return
        output_path = Path(_ensure_path(state.export_tiff_path))
        if output_path.suffix.lower() not in (".tif", ".tiff"):
            output_path = output_path.with_suffix(".tiff")
        grid_data = np.asarray(regrid_volume)
        loop = asyncio.get_event_loop()
        try:
            _set_status(f"Exporting grid TIFF to {output_path}...")
            _start_progress("Exporting TIFF", indeterminate=True)
            await loop.run_in_executor(
                None,
                lambda: tifffile.imwrite(str(output_path), grid_data, compression="zlib"),
            )
            state.export_tiff_path = str(output_path)
            _set_status(f"Exported grid TIFF to {output_path}")
            _stop_progress(success=True)
        except Exception as exc:
            _set_status(f"Export grid failed: {exc}")
            _stop_progress(success=False)

    @ctrl.set("export_npz")
    async def export_npz(**kwargs):
        # Export the grid together with its coordinate axes as a compressed
        # .npz (mirrors the napari widget's on_export_edges). Axis names follow
        # the chosen space: Qx/Qy/Qz for q-space, H/K/L for hkl.
        if regrid_volume is None or regrid_axes is None:
            _set_status("Regrid first, then export.")
            return
        output_path = Path(_ensure_path(state.export_npz_path))
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        grid_data = np.asarray(regrid_volume)
        xaxis, yaxis, zaxis = regrid_axes
        space = _ensure_path(state.space).lower()
        loop = asyncio.get_event_loop()
        try:
            _set_status(f"Exporting grid+edges NPZ to {output_path}...")
            _start_progress("Exporting NPZ", indeterminate=True)
            if space == "q":
                await loop.run_in_executor(
                    None,
                    lambda: np.savez_compressed(
                        str(output_path), grid=grid_data, Qx=xaxis, Qy=yaxis, Qz=zaxis
                    ),
                )
                label = "Qx, Qy, Qz"
            else:
                await loop.run_in_executor(
                    None,
                    lambda: np.savez_compressed(
                        str(output_path), grid=grid_data, H=xaxis, K=yaxis, L=zaxis
                    ),
                )
                label = "H, K, L"
            state.export_npz_path = str(output_path)
            _set_status(f"Exported grid+edges ({label}) to {output_path}")
            _stop_progress(success=True)
        except Exception as exc:
            _set_status(f"Export failed: {exc}")
            _stop_progress(success=False)

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

    # ---------------------------------------------------------------------
    # Layer control panel (right side)
    # ---------------------------------------------------------------------
    # Maps each layer key to the VTK prop that backs it, so we can read the
    # current visibility when (re)building the panel.
    def _layer_prop(key):
        return {
            "volume": volume_actor,
            "outline": outline_actor,
            "coords": outline_label_actors[0] if outline_label_actors else None,
            "world_axes": world_axes_actors[0] if world_axes_actors else None,
            "intensity_map": intensity_actor,
            "roi": roi_outline_actor,
            "cross": cross_actor,
            "slice_x": slice_actors["x"],
            "slice_y": slice_actors["y"],
            "slice_z": slice_actors["z"],
            "cylinder": cyl_actor,
            "sphere": sph_actor,
        }.get(key)

    def _layer_is_visible(key):
        prop = _layer_prop(key)
        return bool(prop.GetVisibility()) if prop is not None else True

    def _set_layers(items):
        """Rebuild the right-side layer list from (key, name) pairs."""
        state.layers = [
            {"key": key, "name": name, "visible": _layer_is_visible(key)}
            for key, name in items
        ]
        # Keep the selection valid: if the previously selected layer is gone
        # (e.g. a slice was hidden and dropped from the list), fall back to the
        # first layer so the top-of-panel controls always target a real layer.
        keys = [key for key, _ in items]
        if state.selected_layer not in keys:
            state.selected_layer = keys[0] if keys else ""

    def _rebuild_volume_layers():
        """List the layers present in the 3D volume view (volume + active slices)."""
        if bool(getattr(state, "intensity_slider_show", False)):
            return  # the intensity view manages its own layer list
        items = [("volume", "RSM Volume"), ("outline", "Outline Box"), ("coords", "Coordinates"), ("world_axes", "World Axes")]
        for ax, lbl in (("x", "Slice X"), ("y", "Slice Y"), ("z", "Slice Z")):
            if slice_actors[ax].GetVisibility():
                items.append((f"slice_{ax}", lbl))
        if cyl_actor.GetVisibility():
            items.append(("cylinder", "Cylinder"))
        if sph_actor.GetVisibility():
            items.append(("sphere", "Sphere"))
        _set_layers(items)

    def _sync_layer_visible(key, visible):
        """Reflect an external visibility change (e.g. a tab checkbox) in the panel."""
        layers = state.layers or []
        changed = False
        updated = []
        for item in layers:
            if item["key"] == key and item["visible"] != bool(visible):
                item = {**item, "visible": bool(visible)}
                changed = True
            updated.append(item)
        if changed:
            state.layers = updated

    def _apply_layer_visibility(key, visible):
        """Show/hide the prop(s) backing a layer, keeping related controls in sync."""
        visible = bool(visible)
        if key == "volume":
            volume_actor.SetVisibility(1 if visible else 0)
        elif key == "outline":
            state.outline_show = visible
            _update_outline_box()
        elif key == "coords":
            state.coords_show = visible
            _update_outline_box()
        elif key == "world_axes":
            state.world_axes_show = visible
            _update_world_axes()
        elif key == "intensity_map":
            intensity_actor.SetVisibility(1 if visible else 0)
        elif key == "roi":
            if visible and intensity_nx is not None:
                _roi_set_active(True)
                _roi_update_crop_state()
            else:
                _roi_set_active(False)
            if bool(getattr(state, "roi_show", True)) != visible:
                state.roi_show = visible
        elif key == "cross":
            _cross_set_active(visible)
        elif key in ("slice_x", "slice_y", "slice_z"):
            axis = key[-1]
            setattr(state, f"slice_{axis}_show", visible)
            _update_ortho_slice(axis)
        elif key == "cylinder":
            state.cyl_show = visible
            _update_cylinder()
        elif key == "sphere":
            state.sph_show = visible
            _update_sphere()

    @ctrl.set("toggle_layer")
    def toggle_layer(key=None, **kwargs):
        if not key:
            return
        layers = [dict(item) for item in (state.layers or [])]
        target = None
        for item in layers:
            if item["key"] == key:
                item["visible"] = not item["visible"]
                target = item
                break
        if target is None:
            return
        _apply_layer_visibility(key, target["visible"])
        state.layers = layers
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

    # The View-tab percentile boxes seed the *absolute* contrast window.
    # Editing them recomputes clim_lo/clim_hi (display-value limits) from the
    # requested percentiles of the currently displayed data; the clim change
    # handler below then applies that window. This mirrors napari, where the
    # percentile spinboxes set the initial contrast_limits and the live
    # contrast control operates on absolute data values afterwards.
    @state.change("contrast_lo", "contrast_hi")
    def _on_contrast_change(**kwargs):
        lo = _float(getattr(state, "contrast_lo", 1.0), 1.0)
        hi = _float(getattr(state, "contrast_hi", 99.8), 99.8)
        if not (0.0 <= lo < hi <= 100.0):
            return
        if bool(getattr(state, "intensity_slider_show", False)) and current_frames:
            n = len(current_frames)
            idx = max(0, min(int(_float(getattr(state, "intensity_frame_index", 0), 0)), n - 1))
            frame = np.asarray(current_frames[idx], dtype=np.float32)
            if frame.ndim != 2:
                frame = np.squeeze(frame)
            if frame.ndim != 2:
                return
            disp = _log1p_clip(frame) if bool(getattr(state, "log_view", True)) else np.maximum(frame, 0.0)
            _seed_clim(disp)
            return
        if current_volume is None:
            return
        if bool(getattr(state, "log_view", True)):
            display_volume = _log1p_clip(current_volume)
        else:
            display_volume = np.maximum(current_volume, 0.0)
        _seed_clim(display_volume)

    # Live-update the rendering when the absolute contrast window (driven by the
    # right-panel "Contrast Limits" slider, or by _seed_clim above) changes.
    # Only the transfer-function range depends on it -- the scalar data fed to
    # VTK is unchanged -- so we re-apply the color/opacity functions over the
    # new [clim_lo, clim_hi] window. An inverted window (hi <= lo) is ignored.
    @state.change("clim_lo", "clim_hi")
    def _on_clim_change(**kwargs):
        nonlocal render_range
        lo = _float(getattr(state, "clim_lo", 0.0), 0.0)
        hi = _float(getattr(state, "clim_hi", 1.0), 1.0)
        if hi <= lo:
            return
        render_range = (lo, hi)
        if bool(getattr(state, "intensity_slider_show", False)) and current_frames:
            _show_intensity_frame(
                int(_float(getattr(state, "intensity_frame_index", 0), 0))
            )
            return
        if current_volume is None:
            return
        _update_rendering()
        _update_all_slices()
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    # Live-update the display when the colormap dropdown (right panel) changes,
    # so the new colors appear immediately without re-clicking View Intensity /
    # View RSM. Mirrors the clim handler: the raw scalar data is unchanged, so
    # we just re-apply the color transfer function (volume) or rebuild the
    # intensity frame's lookup table over the current data.
    @state.change("colormap")
    def _on_colormap_change(**kwargs):
        if bool(getattr(state, "intensity_slider_show", False)) and current_frames:
            _show_intensity_frame(
                int(_float(getattr(state, "intensity_frame_index", 0), 0))
            )
            return
        if current_volume is None:
            return
        _update_rendering()
        _update_all_slices()
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    # Live-update the Analysis-tab slicing colormaps (orthogonal / cylindrical /
    # spherical) the moment the user picks a new one from the dropdown, without
    # needing to toggle the slice off and on again. The select's own
    # ``change=ctrl.update_slices`` handler can miss the fresh value because the
    # v-model hasn't flushed yet; a state.change observer always sees the new
    # value, so it re-slices reliably.
    @state.change("slice_cmap", "cyl_cmap", "sph_cmap")
    def _on_slice_cmap_change(**kwargs):
        if current_volume is None:
            return
        _update_all_slices()
        render_window.Render()
        if remote_view is not None:
            remote_view.update()

    # Clamp the CMS angle step to the valid [0, 360] range. Corrects a typed out-of-range value.
    @state.change("cms_angle_step")
    def _on_cms_angle_step_change(cms_angle_step=None, **kwargs):
        clamped = max(0.0, min(360.0, _float(cms_angle_step, 0.0)))
        if clamped != _float(cms_angle_step, None):
            state.cms_angle_step = clamped

    # Populate the experimental setup (and related build/view) fields from the
    # chosen beamline profile as soon as the loader mode changes (this also
    # fires once at startup), so the ISR/CMS selection is reflected immediately
    # rather than waiting for a TIFF directory to be picked.
    @state.change("loader_mode")
    def _on_loader_mode_change(loader_mode=None, **kwargs):
        _populate_from_profile(loader_mode)

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
            # Dual-range \"Contrast Limits\" slider (right panel): two range
            # inputs stacked over a shared track, each carrying one thumb. The
            # inputs are transparent and pointer-events:none except on the
            # thumbs, so both handles stay independently draggable; a colored
            # fill div marks the selected [low, high] span between them.
            ".dual-slider { position: relative; height: 28px; margin: 4px 0 2px; }"
            ".dual-slider .track { position: absolute; top: 12px; left: 0; right: 0; "
            "height: 4px; background: #44444a; border-radius: 2px; }"
            ".dual-slider .fill { position: absolute; top: 12px; height: 4px; "
            "background: #6aa9ff; border-radius: 2px; }"
            ".dual-slider input[type=range] { -webkit-appearance: none; appearance: none; "
            "position: absolute; top: 0; left: 0; width: 100%; height: 28px; margin: 0; "
            "background: none; pointer-events: none; }"
            ".dual-slider input[type=range]::-webkit-slider-runnable-track { height: 4px; "
            "background: transparent; }"
            ".dual-slider input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; "
            "appearance: none; width: 16px; height: 16px; margin-top: -6px; border-radius: 50%; "
            "background: #6aa9ff; border: 1px solid #cfe2ff; cursor: pointer; pointer-events: auto; }"
            ".dual-slider input[type=range]::-moz-range-track { height: 4px; background: transparent; }"
            ".dual-slider input[type=range]::-moz-range-thumb { width: 16px; height: 16px; "
            "border-radius: 50%; background: #6aa9ff; border: 1px solid #cfe2ff; "
            "cursor: pointer; pointer-events: auto; }"
            # Animated barber-pole stripes for the top-of-viewer progress bar so
            # the fill visibly \"moves\" while a job runs.
            "@keyframes progress-stripes { from { background-position: 0 0; } "
            "to { background-position: 40px 0; } }"
            ".progress-stripes { background-image: linear-gradient(45deg, "
            "rgba(255,255,255,0.20) 25%, transparent 25%, transparent 50%, "
            "rgba(255,255,255,0.20) 50%, rgba(255,255,255,0.20) 75%, "
            "transparent 75%, transparent); background-size: 40px 40px; "
            "animation: progress-stripes 0.8s linear infinite; }"
            # Indeterminate mode: a fixed-width fill that slides across the
            # track, used for steps that can't report a real percentage
            # (load/export). The !important width overrides the bound
            # progress_value width, and transition:none stops it snapping.
            "@keyframes progress-indeterminate { 0% { left: -40%; } "
            "100% { left: 100%; } }"
            ".progress-indeterminate { width: 40% !important; "
            "animation: progress-indeterminate 1.1s ease-in-out infinite, "
            "progress-stripes 0.8s linear infinite; "
            "transition: none !important; }"
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
                html.H2("Voxel", style="margin:0; font-size:1.3rem;")
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
                        with html.Div(style="margin-top:8px;"):
                            html.Input(
                                v_model=("only_hkl", ""),
                                type="checkbox",
                                style="margin-right:8px;",
                            )
                            html.Span("Only HKL scans")

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
                    html.Label("UB matrix", style=_lbl)
                    html.Textarea(
                        v_model=("ub_matrix", ""),
                        rows="3",
                        style=_inp + " font-family:monospace; resize:vertical;",
                    )
                    with html.Label(style=_lbl + " display:flex; align-items:center; gap:6px; cursor:pointer;"):
                        html.Input(type="checkbox", v_model=("ub_includes_2pi", True), style="margin:0; width:auto;")
                        html.Span("UB includes 2\u03c0")
                    with html.Label(style=_lbl + " display:flex; align-items:center; gap:6px; cursor:pointer;"):
                        html.Input(type="checkbox", v_model=("one_based_center", False), style="margin:0; width:auto;")
                        html.Span("1-based center")
                    html.Label("Sample axes", style=_lbl)
                    html.Input(v_model=("sample_axes", ""), type="text", placeholder="x+, y+, z-", style=_inp)
                    html.Label("Detector axes", style=_lbl)
                    html.Input(v_model=("detector_axes", ""), type="text", placeholder="x+", style=_inp)
                    html.Label("Space", style=_lbl)
                    with html.Select(v_model=("space", ""), style=_inp):
                        html.Option("Q-space", value="q")
                        html.Option("HKL", value="hkl")
                    html.Label("Grid (x,y,z), '*' allowed", style=_lbl)
                    html.Input(
                        v_model=("grid_shape", ""),
                        type="text",
                        placeholder="100,*,*",
                        style=_inp,
                    )
                    html.Label("Normalize", style=_lbl)
                    with html.Select(v_model=("normalize", ""), style=_inp):
                        html.Option("mean", value="mean")
                        html.Option("sum", value="sum")
                    with html.Label(style=_lbl + " display:flex; align-items:center; gap:6px; cursor:pointer;"):
                        html.Input(type="checkbox", v_model=("fuzzy_gridder", False), style="margin:0; width:auto;")
                        html.Span("Fuzzy gridder")
                    html.Label("Fuzzy width", style=_lbl)
                    html.Input(v_model=("width_fuzzy", ""), type="number", min="0", step="0.01", style=_inp)
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
                    html.Label("Grid (.tiff)", style=_lbl)
                    html.Input(v_model=("export_tiff_path", ""), placeholder="/path/to/grid.tiff", style=_inp)
                    html.Button("\U0001F4BE Export TIFF", click=ctrl.export_tiff, style="width:100%; margin-top:12px; padding:10px 8px; cursor:pointer;")
                    html.Label("Grid+Edges (.npz)", style=_lbl)
                    html.Input(v_model=("export_npz_path", ""), placeholder="/path/to/grid.npz", style=_inp)
                    html.Button("\U0001F4BE Export NPZ", click=ctrl.export_npz, style="width:100%; margin-top:12px; padding:10px 8px; cursor:pointer;")

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
                    html.Input(v_model=("cyl_radius", ""), type="number", min="0", max="10", step="0.01", change=ctrl.update_slices, style=_inp)
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
                    html.Input(v_model=("sph_radius", ""), type="number", min="0", max="10", step="0.01", change=ctrl.update_slices, style=_inp)
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
                # Progress bar pinned to the top of the viewer. Shown only while
                # a long-running step runs (load / build / regrid / export). For
                # build/regrid the fill width is a real per-frame percentage;
                # for load/export it slides as an indeterminate animation (no
                # number, since those steps can't report accurate progress).
                with html.Div(
                    v_show="progress_active",
                    style=(
                        "flex:0 0 auto; position:relative; height:24px; "
                        "background:#16161a; border-bottom:1px solid #2a2a2e; "
                        "overflow:hidden;"
                    ),
                ):
                    html.Div(
                        classes=(
                            "progress_indeterminate ? "
                            "'progress-stripes progress-indeterminate' "
                            ": 'progress-stripes'",
                        ),
                        style=(
                            "`position:absolute; top:0; left:0; bottom:0; "
                            "width:${progress_value}%; "
                            "background-color:#3a7bff; "
                            "transition:width 0.2s linear;`",
                        ),
                    )
                    html.Div(
                        "{{ progress_label }}{{ progress_indeterminate ? "
                        "'\u2026' : ' ' + Math.round(progress_value) + '%' }}",
                        style=(
                            "position:absolute; inset:0; display:flex; "
                            "align-items:center; justify-content:center; "
                            "font-size:0.78rem; color:#f0f0f0; pointer-events:none; "
                            "font-family:sans-serif; "
                            "text-shadow:0 1px 2px rgba(0,0,0,0.6);"
                        ),
                    )
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

            # Right-side layer control panel. Lists the scene layers for the
            # active view; each row shows the layer name centered with an eye
            # icon on the left that toggles that layer's visibility.
            with html.Div(
                v_show="layers && layers.length",
                style=(
                    "flex:0 0 210px; height:100%; padding:12px; overflow:auto; "
                    "background:#16161a; color:#dddddd; border-left:1px solid #2a2a2e; "
                    "font-family:sans-serif;"
                ),
            ):
                # ---- Selected-layer controls (napari-style) -------------
                # napari shows the controls for the *selected* layer at the top
                # of the layer panel; which set appears depends on
                # ``selected_layer``. Each control is bound to that layer's own
                # state (e.g. slice_cmap / cyl_cmap vs. the volume's colormap),
                # so editing the colormap here affects only the selected layer.
                # The colors themselves come from vtkplotlib's colormapping (see
                # _cmap_rgb / vpl.colors.as_vtk_cmap).
                _pl_hdr = "display:block; margin-bottom:6px; font-size:0.95rem;"
                _pl_lbl = "display:block; margin:10px 0 4px; font-size:0.85rem; color:#bbbbbb;"
                _pl_inp = "width:100%; margin-bottom:6px;"
                _slice_cmaps = ["turbo", "viridis", "inferno", "plasma", "gray", "hsv"]

                # Selected-layer name header.
                html.Div(
                    "{{ (layers.find(l => l.key === selected_layer) || {}).name }}",
                    v_show="selected_layer",
                    style=(
                        "margin-bottom:10px; font-size:0.9rem; font-weight:600; "
                        "color:#6aa9ff; white-space:nowrap; overflow:hidden; "
                        "text-overflow:ellipsis;"
                    ),
                )

                # Volume / intensity image: colormap + contrast limits.
                with html.Div(
                    v_if="selected_layer === 'volume' || selected_layer === 'intensity_map'"
                ):
                    html.Strong("Colormap", style=_pl_hdr)
                    with html.Select(v_model=("colormap", ""), style=_pl_inp):
                        for name in COLORMAP_NAMES:
                            html.Option(name, value=name)
                    # Contrast Limits dual-range slider. Bound to the absolute
                    # clim_lo / clim_hi display-value window (seeded once from
                    # the View-tab percentile boxes at view time), ranging over
                    # the display data min/max (clim_min / clim_max). Dragging a
                    # handle adjusts the absolute contrast limits with a linear
                    # mapping, matching napari's live contrast control.
                    html.Strong("Contrast Limits", style=_pl_hdr + " margin-top:12px;")
                    with html.Div(classes="dual-slider"):
                        html.Div(classes="track")
                        html.Div(
                            classes="fill",
                            style=(
                                "`left:${clim_max>clim_min?"
                                "(Math.min(clim_lo,clim_hi)-clim_min)/(clim_max-clim_min)*100:0}%;"
                                "right:${clim_max>clim_min?"
                                "(1-(Math.max(clim_lo,clim_hi)-clim_min)/(clim_max-clim_min))*100:0}%`",
                            ),
                        )
                        html.Input(
                            type="range",
                            v_model=("clim_lo", 0.0),
                            min=("clim_min",),
                            max=("clim_max",),
                            step=("clim_step",),
                        )
                        html.Input(
                            type="range",
                            v_model=("clim_hi", 1.0),
                            min=("clim_min",),
                            max=("clim_max",),
                            step=("clim_step",),
                        )
                    html.Div(
                        "{{ Number(clim_lo).toPrecision(3) }} \u2013 "
                        "{{ Number(clim_hi).toPrecision(3) }}",
                        style=(
                            "margin-bottom:14px; font-size:0.8rem; color:#aaaaaa; "
                            "text-align:center; font-variant-numeric:tabular-nums;"
                        ),
                    )

                # Orthogonal slice (x/y/z share one colormap + opacity).
                with html.Div(
                    v_if="selected_layer === 'slice_x' || selected_layer === 'slice_y' || selected_layer === 'slice_z'"
                ):
                    html.Strong("Colormap", style=_pl_hdr)
                    with html.Select(
                        v_model=("slice_cmap", ""), change=ctrl.update_slices, style=_pl_inp
                    ):
                        for name in _slice_cmaps:
                            html.Option(name, value=name)
                    html.Label("Opacity", style=_pl_lbl)
                    html.Input(
                        v_model=("slice_opacity", ""), type="number", min="0", max="1",
                        step="0.1", change=ctrl.update_slices, style=_pl_inp,
                    )

                # Cylindrical probe surface.
                with html.Div(v_if="selected_layer === 'cylinder'"):
                    html.Strong("Colormap", style=_pl_hdr)
                    with html.Select(
                        v_model=("cyl_cmap", ""), change=ctrl.update_slices, style=_pl_inp
                    ):
                        for name in _slice_cmaps:
                            html.Option(name, value=name)
                    html.Label("Opacity", style=_pl_lbl)
                    html.Input(
                        v_model=("cyl_opacity", ""), type="number", min="0", max="1",
                        step="0.1", change=ctrl.update_slices, style=_pl_inp,
                    )
                    html.Label("Radius (\u00c5\u207b\u00b9)", style=_pl_lbl)
                    html.Input(
                        v_model=("cyl_radius", ""), type="number", min="0", max="10",
                        step="0.01", change=ctrl.update_slices, style=_pl_inp,
                    )
                    html.Label("Angular samples", style=_pl_lbl)
                    html.Input(
                        v_model=("cyl_samples", ""), type="number", min="16", max="360",
                        step="8", change=ctrl.update_slices, style=_pl_inp,
                    )

                # Spherical probe surface.
                with html.Div(v_if="selected_layer === 'sphere'"):
                    html.Strong("Colormap", style=_pl_hdr)
                    with html.Select(
                        v_model=("sph_cmap", ""), change=ctrl.update_slices, style=_pl_inp
                    ):
                        for name in _slice_cmaps:
                            html.Option(name, value=name)
                    html.Label("Opacity", style=_pl_lbl)
                    html.Input(
                        v_model=("sph_opacity", ""), type="number", min="0", max="1",
                        step="0.1", change=ctrl.update_slices, style=_pl_inp,
                    )
                    html.Label("Radius (\u00c5\u207b\u00b9)", style=_pl_lbl)
                    html.Input(
                        v_model=("sph_radius", ""), type="number", min="0", max="10",
                        step="0.01", change=ctrl.update_slices, style=_pl_inp,
                    )
                    html.Label("Angular samples", style=_pl_lbl)
                    html.Input(
                        v_model=("sph_samples", ""), type="number", min="16", max="180",
                        step="8", change=ctrl.update_slices, style=_pl_inp,
                    )

                # Layers with no adjustable image properties (overlays/markers).
                with html.Div(
                    v_if="['outline','world_axes','roi','cross'].indexOf(selected_layer) !== -1"
                ):
                    html.Div(
                        "No adjustable properties for this layer.",
                        style="margin-bottom:14px; font-size:0.82rem; color:#888888;",
                    )

                html.Hr(style="border-color:#2a2a2e; margin:6px 0 12px;")
                html.Strong(
                    "Layers",
                    style="display:block; margin-bottom:10px; font-size:0.95rem;",
                )
                with html.Div(
                    v_for="(layer, li) in layers",
                    key="layer.key",
                    # Clicking anywhere on the row selects the layer (napari
                    # style), swapping the controls above to that layer's own.
                    click="selected_layer = layer.key",
                    # Highlight the selected row blue (background + border).
                    style=(
                        "`position:relative; display:flex; align-items:center; "
                        "height:36px; margin-bottom:8px; padding:0 10px; "
                        "background:${layer.key === selected_layer ? '#1e3a5f' : '#2a2a2e'}; "
                        "border:1px solid ${layer.key === selected_layer ? '#6aa9ff' : '#44444a'}; "
                        "border-radius:6px; user-select:none; cursor:pointer;`",
                    ),
                ):
                    # Eye toggle: an inline SVG eye / eye-off glyph (injected
                    # via v-html) whose shape tracks the layer's visibility.
                    # Clicking it flips the layer via the toggle_layer
                    # controller (which updates the VTK actor and pushes a fresh
                    # frame to the remote view).
                    html.Div(
                        v_html="layer.visible ? eye_on_svg : eye_off_svg",
                        click=(ctrl.toggle_layer, "[layer.key]"),
                        title="Show / hide this layer",
                        style=(
                            "position:relative; z-index:2; display:flex; align-items:center; "
                            "justify-content:center; width:24px; height:24px; cursor:pointer;"
                        ),
                    )
                    html.Span(
                        "{{ layer.name }}",
                        style=(
                            "position:absolute; left:0; right:0; text-align:center; "
                            "pointer-events:none; padding:0 34px; font-size:0.85rem; "
                            "white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
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