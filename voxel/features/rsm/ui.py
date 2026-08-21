"""RSM feature: left-panel tabs and layer-property controls.

These are the *feature-specific* slices of the shared app shell. The shell
(``voxel/app/shell.py`` / ``voxel/app/server.py``) owns the left control panel
container, the accordion styling, the status area, the 3D view and the layer
list; this module fills in the parts that are specific to the 3D-RSM workflow:

  * ``build_tabs``            -> the Data / Build / View / Analyze accordion tabs
  * ``build_layer_controls`` -> the per-layer property editors shown at the top
                                of the right-hand layer panel

Both are *slot* functions: they emit ``trame.widgets.html`` elements into
whatever layout context is active when they are called, so the shell simply
calls them at the right spot. A future tomography feature will provide its own
``build_tabs`` / ``build_layer_controls`` with different variables and inputs
while reusing the exact same shell.

``ctx`` is a small namespace supplied by the shell carrying the handles these
slots need: ``ctx.ctrl`` (the trame controller) and ``ctx.fb_open`` (the file
browser opener).
"""

from trame.widgets import html
from trame_client.widgets.html import HtmlElement  # base class for raw SVG tags

from voxel.ui.assets import COLORMAP_NAMES
from voxel.ui.styles import bar_style as _bar_style, PANEL as _panel, LBL as _lbl, INP as _inp, BTN as _btn


def build_tabs(ctx):
    """Emit the RSM Data / Build / View / Analyze accordion tabs.

    Called inside the shell's left-control-panel context.
    """
    ctrl = ctx.ctrl
    _fb_open = ctx.fb_open

    # ===================== DATA TAB =====================
    with html.Div(
        style=_bar_style("data"),
        click="open_tab = open_tab === 'data' ? '' : 'data'",
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

        html.Button("\U0001F4C2 Load Data", click=ctrl.load_data, style="width:100%; margin-top:12px; padding:10px 8px; cursor:pointer;")

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

        html.Button("\U0001F4C8 View Intensity", click=ctrl.view_intensity, style="width:100%; margin-top:12px; padding:10px 8px; cursor:pointer;")

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
        style=_bar_style("build"), click="open_tab = open_tab === 'build' ? '' : 'build'"
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
        style=_bar_style("view"), click="open_tab = open_tab === 'view' ? '' : 'view'"
    ):
        html.Span("View")
        html.Span("{{ open_tab === 'view' ? '\u25BC' : '\u25B6' }}")
    with html.Div(v_show="open_tab === 'view'", style=_panel):
        with html.Div(style="display:flex; gap:8px;"):
            html.Button("\U0001F52D View RSM", click=ctrl.view_rsm, style=_btn)
            html.Button("\u21BB Refresh", click=ctrl.refresh_rendering, style=_btn)
            html.Button("\u23F9 Stop", click=ctrl.stop_task, style=_btn)

        # Faithful napari-style attenuated-MIP snapshot of the
        # current view (CPU-rendered; takes a few seconds). The
        # snapshot always produces an *attenuated* MIP, which only
        # matches what the live view represents when the rendering
        # mode is ``attenuated_mip`` -- so the button is disabled
        # (and dimmed) in any other mode, and while one is already
        # rendering.
        html.Button(
            "{{ hq_snapshot_busy ? '\u23F3 Rendering\u2026' : "
            "'\u2728 HQ Snapshot (attenuated MIP)' }}",
            click=ctrl.hq_snapshot,
            disabled=("hq_snapshot_busy || rendering !== 'attenuated_mip'",),
            title=(
                "rendering === 'attenuated_mip' ? 'Render a high-quality "
                "attenuated-MIP snapshot of the current view' : 'Set Rendering "
                "to attenuated_mip to enable the HQ snapshot'",
            ),
            style=(
                "`width:100%; margin-top:8px; padding:10px 8px; "
                "cursor:${(hq_snapshot_busy || rendering !== 'attenuated_mip')"
                " ? 'not-allowed' : 'pointer'}; "
                "opacity:${rendering === 'attenuated_mip' ? 1 : 0.5};`",
            ),
        )

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

    # ===================== ANALYZE TAB =====================
    with html.Div(
        style=_bar_style("analyze"), click="open_tab = open_tab === 'analyze' ? '' : 'analyze'"
    ):
        html.Span("Analyze")
        html.Span("{{ open_tab === 'analyze' ? '\u25BC' : '\u25B6' }}")
    with html.Div(v_show="open_tab === 'analyze'", style=_panel):
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


def build_layer_controls(ctx):
    """Emit the RSM per-layer property editors.

    Called inside the shell's right-hand layer panel, above the shared layer
    list. Which control set shows depends on ``selected_layer``.
    """
    ctrl = ctx.ctrl

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

        # ---- Opacity transfer-function editor (ParaView-style) ----
        # A graph whose x-axis spans the color range (painted with
        # the active colormap) and whose y-axis is opacity. Users
        # drag the round handles to reshape the piecewise-linear
        # opacity ramp, double-click on the graph to add a point, and
        # right-click a handle to remove it. Points are stored in the
        # ``opacity_points`` state ([x,y], x normalized over the
        # contrast window); the server rebuilds the VTK opacity
        # transfer function from them (see _apply_opacity_points).
        # During a drag we only ``set`` the client state (instant
        # visual feedback) and ``flushState`` once on release, so the
        # volume re-renders a single time per edit instead of on
        # every pointer move.
        html.Strong("Opacity Transfer Function", style=_pl_hdr + " margin-top:12px;")
        with html.Div(
            style=(
                "position:relative; width:100%; height:130px; "
                "border:1px solid #3a3a40; border-radius:4px; "
                "background:#0b0b0e; overflow:visible; "
                "touch-action:none; user-select:none; margin-bottom:6px;"
            ),
            dblclick=(
                "var r=$event.currentTarget.getBoundingClientRect();"
                "var x=Math.max(0.01,Math.min(0.99,($event.clientX-r.left)/r.width));"
                "var y=Math.max(0,Math.min(1,1-($event.clientY-r.top)/r.height));"
                "var pts=opacity_points.map(function(q){return [q[0],q[1]];});"
                "pts.push([x,y]);pts.sort(function(a,b){return a[0]-b[0];});"
                "opacity_points=pts;"
            ),
        ):
            # Full-area colormap gradient painted behind the curve.
            # ParaView colors the region *under* the opacity line
            # with the colormap; the SVG "mask" polygon below paints
            # the area ABOVE the line with the panel background, so
            # only the region beneath the curve shows the gradient.
            # A vertical fade (dark at the bottom -> clear at the
            # top) is layered over the colormap so the fill also
            # reads as opacity: low-opacity columns sit near the
            # dimmed bottom, high-opacity columns reach the bright
            # top near the line.
            html.Div(
                style=(
                    "{position:'absolute',inset:'0',pointerEvents:'none',"
                    "background:'linear-gradient(to top, rgba(11,11,14,0.92) 0%,"
                    " rgba(11,11,14,0.08) 100%), '+cmap_gradient}",
                ),
            )
            # SVG overlay: filled area + polyline through the points.
            # pointer-events:none so double-clicks fall through to the
            # container's add-point handler. NOTE: trame filters
            # element attributes by an allow-list, and SVG attributes
            # (viewBox / points / stroke ...) are NOT in it, so they
            # must be declared via ``__properties`` or they are
            # silently dropped (leaving an empty <polyline/> that
            # draws nothing).
            with html.Svg(
                viewBox="0 0 100 100",
                preserveAspectRatio="none",
                __properties=["viewBox", "preserveAspectRatio"],
                style=(
                    "position:absolute; inset:0; width:100%; height:100%; "
                    "pointer-events:none;"
                ),
            ):
                # Mask polygon: covers the area ABOVE the curve with
                # the panel background color so the colormap gradient
                # only shows *under* the piecewise line. Its points
                # are the curve vertices followed by the two TOP
                # corners (100,0) and (0,0).
                HtmlElement(
                    "polygon",
                    points=(
                        "opacity_points.map(function(p){return (p[0]*100)+','"
                        "+((1-p[1])*100);}).join(' ')+' 100,0 0,0'",
                    ),
                    fill="#0b0b0e",
                    stroke="none",
                    __properties=["points", "fill", "stroke"],
                )
                HtmlElement(
                    "polyline",
                    points=(
                        "opacity_points.map(function(p){return (p[0]*100)+','"
                        "+((1-p[1])*100);}).join(' ')",
                    ),
                    fill="none",
                    stroke="#6aa9ff",
                    stroke_width="2",
                    vector_effect="non-scaling-stroke",
                    __properties=[
                        "points",
                        "fill",
                        "stroke",
                        ("stroke_width", "stroke-width"),
                        ("vector_effect", "vector-effect"),
                    ],
                )
            # Draggable handles (HTML divs so they stay round
            # regardless of the non-uniform SVG scaling). Dragging
            # uses pointer capture + a Vue-bound pointermove (gated
            # on the button being held) so the point follows the
            # cursor even outside the handle, and writes straight to
            # the ``opacity_points`` state (which re-renders the
            # volume and moves the handle live).
            html.Div(
                v_for="(pt, pi) in opacity_points",
                key=("pi",),
                # trame's html.Div only registers a fixed set of DOM
                # events (click / mousedown / dblclick / contextmenu
                # ...); pointer events are NOT among them, so they
                # must be declared explicitly via ``__events`` or the
                # ``@pointer*`` bindings are silently dropped (which
                # is why dragging did nothing while dblclick/
                # contextmenu worked).
                __events=["pointerdown", "pointermove", "pointerup"],
                style=(
                    "`position:absolute; left:${pt[0]*100}%; top:${(1-pt[1])*100}%;"
                    " width:10px; height:10px; margin:-5px 0 0 -5px; border-radius:50%;"
                    " background:#ffffff; border:2px solid #6aa9ff; box-shadow:0 0 3px"
                    " rgba(0,0,0,.7); touch-action:none; cursor:${pi===0||"
                    "pi===opacity_points.length-1?'ns-resize':'move'}`",
                ),
                pointerdown=(
                    "$event.preventDefault();$event.stopPropagation();"
                    "$event.currentTarget.setPointerCapture($event.pointerId);"
                ),
                pointermove=(
                    "if($event.buttons!==1)return;"
                    "$event.preventDefault();"
                    "var r=$event.currentTarget.parentElement.getBoundingClientRect();"
                    "var x=Math.max(0,Math.min(1,($event.clientX-r.left)/r.width));"
                    "var y=Math.max(0,Math.min(1,1-($event.clientY-r.top)/r.height));"
                    "var pts=opacity_points.map(function(q){return [q[0],q[1]];});"
                    "if(pi===0){x=0;}else if(pi===pts.length-1){x=1;}"
                    "else{x=Math.max(pts[pi-1][0]+0.005,Math.min(pts[pi+1][0]-0.005,x));}"
                    "pts[pi]=[x,y];opacity_points=pts;"
                ),
                pointerup=(
                    "try{$event.currentTarget.releasePointerCapture($event.pointerId);}"
                    "catch(e){}"
                ),
                contextmenu=(
                    "$event.preventDefault();$event.stopPropagation();"
                    "if(pi<=0||pi>=opacity_points.length-1)return;"
                    "var pts=opacity_points.map(function(q){return [q[0],q[1]];});"
                    "pts.splice(pi,1);opacity_points=pts;"
                ),
            )
        # X-axis colormap bar below the graph (ParaView's horizontal
        # color-range strip), reading as the data value across x.
        html.Div(
            style=(
                "{width:'100%',height:'14px',borderRadius:'3px',"
                "marginBottom:'8px',background:cmap_gradient}",
            ),
        )
        with html.Div(
            style="display:flex; justify-content:space-between; "
            "align-items:center; margin-bottom:14px;"
        ):
            html.Span(
                "double-click: add \u00b7 drag: move \u00b7 right-click: remove",
                style="font-size:0.68rem; color:#777;",
            )
            html.Button(
                "Reset",
                click="opacity_points=[[0,0],[1,1]]",
                style=(
                    "font-size:0.72rem; padding:2px 8px; cursor:pointer; "
                    "background:#26262c; color:#ccc; border:1px solid #3a3a40; "
                    "border-radius:3px;"
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
        # Tilt of the slice plane. Each plane can tilt about the two
        # world axes other than its own, so it can be oblique in two
        # directions instead of perpendicular to the pure x/y/z axis:
        # X tilts about Qz and Qy, Y about Qz and Qx, Z about Qx and
        # Qy (see _update_ortho_slice). Each rotation binds its own
        # ``slice_{axis}_tilt{rot}`` slider.
        html.Label("Tilt (\u00b0)", style=_pl_lbl)
        for _ax, _rots in (
            ("x", ("z", "y")),
            ("y", ("z", "x")),
            ("z", ("x", "y")),
        ):
            with html.Div(v_if=f"selected_layer === 'slice_{_ax}'"):
                for _rot in _rots:
                    with html.Div(
                        style="display:flex; align-items:center; gap:8px; margin-top:4px;",
                    ):
                        html.Span(
                            f"Q{_rot}",
                            style="width:22px; font-size:0.8rem; color:#bbbbbb;",
                        )
                        html.Input(
                            v_model=(f"slice_{_ax}_tilt{_rot}", ""), type="range",
                            min="-90", max="90", step="1",
                            change=ctrl.update_slices, style="flex:1;",
                        )
                        html.Span(
                            "{{ " + f"slice_{_ax}_tilt{_rot}" + " }}\u00b0",
                            style="width:42px; font-size:0.8rem; text-align:right;",
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
        # Tilt of the cylinder axis away from Qz (see
        # _update_cylinder), so the tube need not be centered on the
        # z axis. Qx leans the axis into the Qy-Qz plane, Qy into the
        # Qx-Qz plane; each binds its own slider.
        html.Label("Tilt (\u00b0)", style=_pl_lbl)
        for _rot in ("x", "y"):
            with html.Div(
                style="display:flex; align-items:center; gap:8px; margin-top:4px;",
            ):
                html.Span(
                    f"Q{_rot}",
                    style="width:22px; font-size:0.8rem; color:#bbbbbb;",
                )
                html.Input(
                    v_model=(f"cyl_tilt{_rot}", ""), type="range",
                    min="-90", max="90", step="1",
                    change=ctrl.update_slices, style="flex:1;",
                )
                html.Span(
                    "{{ " + f"cyl_tilt{_rot}" + " }}\u00b0",
                    style="width:42px; font-size:0.8rem; text-align:right; margin-right:4px;",
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
