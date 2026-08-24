"""Tomography feature UI: header dropdown menus, Pipeline blocks, Properties.

The feature-specific slice of the shared app shell for the tomography workflow.
The shell (``voxel/app/server.py`` / ``create_server``) owns the header bar, the
left control-panel container, the 3D view and the shared Layers
list; this module fills in the tomography-specific left-panel content through the
:class:`~voxel.features.base.VoxelFeature` contract.

Unlike RSM's fixed Data/Build/View/Analyze accordion, tomography follows the
*tomviz* model:

  * header dropdown menus (File / Data Transforms / Tomography / Visualization)
    list the available operations;
  * clicking an operation appends a *block* to the ``pipeline`` list;
  * the ``pipeline`` renders as an ordered list of blocks (selectable, movable,
    deletable);
  * selecting a block shows its editable parameters in the Properties section.

The single source of truth is :data:`TOMO_OPS`: it drives BOTH the dropdown menu
entries and the Properties parameter editors, so adding an operator is a
one-place change. The op ids here MUST match the numeric operator ids in
:data:`voxel.features.tomo.pipeline.OP_FUNCS` (the ones that are numeric
transforms); File/Visualization ops have no numeric counterpart and are handled
by the feature's controllers / the shell.

This module is UI + reactive state only. The numerical operators live in
``voxel/features/tomo/pipeline.py`` and are invoked by the controllers in
``voxel/features/tomo/feature.py``.
"""

from trame.widgets import html, vuetify3 as v3


# ---------------------------------------------------------------------------
# Operation catalog
# ---------------------------------------------------------------------------
# Each entry describes one operation the user can add to the pipeline:
#   id      stable key stored on the block dict ("op"); numeric ops must match
#           voxel.features.tomo.pipeline.OP_FUNCS
#   label   text shown in the menu and on the pipeline block
#   params  ordered list of parameter descriptors, each:
#             {name, label, type, default, [choices]}
#           type in {"number", "int", "text", "bool", "choice", "path"}.
_ALIGN_PARAMS = [
    {"name": "iters", "label": "Iterations", "type": "int", "default": 10},
    {
        "name": "algorithm",
        "label": "Recon algorithm",
        "type": "choice",
        "default": "sirt",
        "choices": ["art", "gridrec", "mlem", "sirt", "tv", "grad"],
    },
    {"name": "upsample_factor", "label": "Upsample factor", "type": "int", "default": 10},
    {"name": "pad_x", "label": "Pad X", "type": "int", "default": 0},
    {"name": "pad_y", "label": "Pad Y", "type": "int", "default": 0},
    {"name": "blur", "label": "Blur edges", "type": "bool", "default": True},
    {"name": "rin", "label": "Blur inner radius", "type": "number", "default": 0.5},
    {"name": "rout", "label": "Blur outer radius", "type": "number", "default": 0.8},
]

FILE_OPS = [
    {
        "id": "open_data",
        "label": "Open Data",
        "params": [
            {"name": "path", "label": "File / directory", "type": "path", "default": ""},
            {
                "name": "data_type",
                "label": "Data Type",
                "type": "choice",
                "default": "Volume",
                "choices": ["Tilt Series", "Volume"],
            },
        ],
    },
    {
        "id": "save_data",
        "label": "Save Data",
        "params": [
            {"name": "path", "label": "Output path", "type": "path", "default": ""},
        ],
    },
]

DATA_TRANSFORM_OPS = [
    {
        "id": "crop",
        "label": "Crop",
        "params": [
            {"name": "x_min", "label": "X min", "type": "int", "default": 0},
            {"name": "x_max", "label": "X max", "type": "int", "default": 0},
            {"name": "y_min", "label": "Y min", "type": "int", "default": 0},
            {"name": "y_max", "label": "Y max", "type": "int", "default": 0},
            {"name": "z_min", "label": "Z min", "type": "int", "default": 0},
            {"name": "z_max", "label": "Z max", "type": "int", "default": 0},
        ],
    },
    {
        "id": "downsample",
        "label": "Downsample x2 (Volume)",
        "params": [
            #{"name": "level", "label": "Bin level (2^n)", "type": "int", "default": 1, "min": 0},
            {"name": "axis", "label": "Axis", "type": "choice", "default": "All Axes (Uniform)", "choices": ["All Axes (Uniform)", "Projection Angles", "Vertical Detector Height", "Horizontal Detector Width"]},
        ],
    },
    {
            "id": "resample",
            "label": "Resample x2 (Tilt Series)",
            "params": [
                #{"name": "level", "label": "Bin level (2^n)", "type": "int", "default": 1, "min": 0},
            ],
        },
    {
        "id": "median_filter",
        "label": "Median Filter",
        "params": [
            {"name": "size", "label": "Size", "type": "int", "default": 2, "min": 1},
        ],
    },
    {
        "id": "gaussian_filter",
        "label": "Gaussian Filter",
        "params": [
            {"name": "sigma", "label": "Sigma", "type": "number", "default": 2.000, "step": 0.500, "min": 0.000},
        ],
    },
    {
        "id": "wiener_filter",
        "label": "Wiener Filter",
        "params": [
                {"name": "sigma_x", "label": "Sigma-X", "type": "number", "default": 0.500, "step": 0.500, "min": 0.001, "max": 1.000},
                {"name": "sigma_y", "label": "Sigma-Y", "type": "number", "default": 0.500, "step": 0.500, "min": 0.001, "max": 1.000},
                {"name": "sigma_z", "label": "Sigma-Z", "type": "number", "default": 0.500, "step": 0.500, "min": 0.001, "max": 1.000},
                {"name": "snr", "label": "SNR", "type": "number", "default": 15.000, "step": 0.500, "min": 0.000},
        ],
    },
]

# Tomography ops are grouped into submenu categories. Each entry is
# (category label, [ops]); the Tomography header menu renders these as
# hover-expandable submenus (see build_menu_bar). The flat TOMOGRAPHY_OPS list
# below is derived from these groups for the TOMO_OPS lookup.
TOMOGRAPHY_GROUPS = [
    (
        "Pre-processing",
        [
            {
                "id": "normalize",
                "label": "Normalize (Dark/Flat)",
                "params": [
                    {"name": "cutoff", "label": "Cutoff (blank = none)", "type": "number", "default": ""},
                ],
            },
            {
                "id": "normalize_bg",
                "label": "Background Normalize",
                "params": [
                    {"name": "air", "label": "Air pixels", "type": "int", "default": 1},
                ],
            },
            {"id": "minus_log", "label": "Minus log (\u2212log)", "params": []},
            {
                "id": "remove_stripe",
                "label": "Ring Removal (Stripe)",
                "params": [
                    {"name": "level", "label": "Level (blank = auto)", "type": "int", "default": ""},
                    {"name": "wname", "label": "Wavelet", "type": "text", "default": "db5"},
                    {"name": "sigma", "label": "Sigma", "type": "number", "default": 2.0},
                ],
            },
            {
                "id": "retrieve_phase",
                "label": "Phase Retrieval (Paganin)",
                "params": [
                    {"name": "pixel_size", "label": "Pixel size (cm)", "type": "number", "default": 1e-4},
                    {"name": "dist", "label": "Propagation dist (cm)", "type": "number", "default": 50.0},
                    {"name": "energy", "label": "Energy (keV)", "type": "number", "default": 20.0},
                    {"name": "alpha", "label": "Regularization \u03b1", "type": "number", "default": 1e-3},
                ],
            },
            {
                "id": "set_angles",
                "label": "Set Tilt Angles",
                "params": [
                    {"name": "img1", "label": "Start Image #", "type": "int", "default": 0},
                    {"name": "img2", "label": "End Image #", "type": "int", "default": 0},
                    {"name": "ang1", "label": "Start Angle (°)", "type": "number", "default": 0.0},
                    {"name": "ang2", "label": "End Angle (°)", "type": "number", "default": 0.0},
                    {"name": "ang_inc", "label": "Angle Increment (°)", "type": "number", "default": 0.0, "readonly": True},
                ], 
            },
        ],
    ),
    (
        "Alignment",
        [
            {"id": "align_seq", "label": "Auto-align (sequential)", "params": _ALIGN_PARAMS},
            {"id": "align_joint", "label": "Auto-align (joint)", "params": _ALIGN_PARAMS},
            {
                "id": "shift_images",
                "label": "Manual Shift",
                "params": [
                    {"name": "sx", "label": "Shift X", "type": "number", "default": 0.0},
                    {"name": "sy", "label": "Shift Y", "type": "number", "default": 0.0},
                ],
            },
            {"id": "scale", "label": "Scale to [\u22121, 1]", "params": []},
            {
                "id": "blur_edges",
                "label": "Blur Edges",
                "params": [
                    {"name": "low", "label": "Low ratio", "type": "number", "default": 0.0},
                    {"name": "high", "label": "High ratio", "type": "number", "default": 0.8},
                ],
            },
        ],
    ),
    (
        "Reconstruction",
        [
            {
                "id": "find_center",
                "label": "Find Center (Manual)",
                "params": [
                    {"name": "init", "label": "Initial guess (blank = mid)", "type": "number", "default": ""},
                    {"name": "tol", "label": "Tolerance", "type": "number", "default": 0.5},
                ],
            },
            {"id": "find_center_vo", "label": "Find Center (Auto, Vo)", "params": []},
            {
                "id": "recon",
                "label": "Reconstruct",
                "params": [
                    {
                        "name": "algorithm",
                        "label": "Algorithm",
                        "type": "choice",
                        "default": "Fourier Grid",
                        "choices": ["Fourier Grid", "Filtered Back-Projection", "Algebraic (ART)", "Simultaneous Iterative (SIRT)", "Maximum-Likelihood Expectation Maximization (MLEM)", "Total Variation (TV)"],
                    },
                    {"name": "center", "label": "Rotation center (blank = auto)", "type": "number", "default": ""},
                    {"name": "num_iter", "label": "Iterations (iterative only)", "type": "int", "default": 1},
                    {
                        "name": "filter_name",
                        "label": "Filter (Fourier Grid/Filtered Back-Projection)",
                        "type": "choice",
                        "default": "shepp",
                        "choices": ["none", "shepp", "cosine", "hann", "hamming", "ramlak", "parzen", "butterworth"],
                    },
                ],
            },
            {
                "id": "circ_mask",
                "label": "Circular Mask",
                "params": [
                    {"name": "axis", "label": "Axis", "type": "int", "default": 0},
                    {"name": "ratio", "label": "Radius ratio", "type": "number", "default": 1.0},
                ],
            },
        ],
    ),
    (
        "Simulation & Demonstrations",
        [
            {
                "id": "add_noise",
                "label": "Add Noise (Sim)",
                "params": [
                    {"name": "ratio", "label": "Std / max ratio", "type": "number", "default": 0.05},
                ],
            },
            {
                "id": "add_jitter",
                "label": "Add Jitter (Sim)",
                "params": [
                    {"name": "low", "label": "Low", "type": "number", "default": 0.0},
                    {"name": "high", "label": "High", "type": "number", "default": 1.0},
                ],
            },
        ],
    ),
]

# Flat list of all tomography ops, derived from the grouped structure above.
TOMOGRAPHY_OPS = [op for _cat, ops in TOMOGRAPHY_GROUPS for op in ops]

VISUALIZATION_OPS = [
    {
        "id": "volume",
        "label": "Volume Render",
        "params": [
            {
                "name": "mode",
                "label": "Mode",
                "type": "choice",
                "default": "composite",
                "choices": ["composite", "mip", "minip", "average"],
            },
        ],
    },
    {   "id": "outline_box", 
        "label": "Outline Box",
        "params": []
    },
    {
        "id": "scale_cube", 
        "label": "Scale Cube", 
        "params": [
            {"name": "side_length", "label": "Side Length", "type": "number", "default": 1.0, "min": 0.0}
        ]
    },
    {
        "id": "ortho_slices",
        "label": "Orthogonal slices",
        "params": [
            {"name": "show_x", "label": "Show X slice", "type": "bool", "default": True},
            {"name": "show_y", "label": "Show Y slice", "type": "bool", "default": True},
            {"name": "show_z", "label": "Show Z slice", "type": "bool", "default": True},
        ],
    },
    {
        "id": "clip",
        "label": "Clip",
        "params": [
            {"name": "direction", "label": "Direction", "type": "choice", "default": "XY Plane", "choices": ["XY Plane", "YZ Plane", "XZ Plane"]},
            {"name": "plane", "label": "Plane", "type": "int", "default": 0, "min": 0},
        ],
    },
    {
        "id": "background",
        "label": "Background Color",
        "params": [
            {"name": "color", "label": "Color", "type": "color", "default": "#000000"},
        ]   
    },
]

# Menu grouping used to build the header dropdowns (label -> list of ops).
TOMO_MENUS = [
    ("File", FILE_OPS),
    ("Data Transforms", DATA_TRANSFORM_OPS),
    ("Tomography", TOMOGRAPHY_OPS),
    ("Visualization", VISUALIZATION_OPS),
]

# Header menus whose ops are further split into hover-expandable submenu
# categories (menu label -> list of (category label, [ops])). Menus not listed
# here render their ops as a single flat list.
MENU_SUBGROUPS = {
    "Tomography": TOMOGRAPHY_GROUPS,
}

# Flat lookup: op id -> op descriptor (single source of truth for Properties).
TOMO_OPS = {op["id"]: op for _label, ops in TOMO_MENUS for op in ops}


def default_params(op_id):
    """Return a fresh ``{name: default}`` dict for the given op id."""
    return {p["name"]: p.get("default", "") for p in TOMO_OPS[op_id]["params"]}


# ---------------------------------------------------------------------------
# Header dropdown menus
# ---------------------------------------------------------------------------
def build_menu_bar(ctx):
    """Emit the tomviz-style header dropdown menu bar (vuetify3).

    Each entry calls ``ctrl.tomo_add_op(op_id)`` to append a block to the pipeline.
    Menus listed in :data:`MENU_SUBGROUPS` render their ops as hover-expandable
    submenu categories instead of a single flat list.
    """
    with html.Div(classes="d-flex flex-wrap ga-1 mb-2"):
        for menu_label, ops in TOMO_MENUS:
            subgroups = MENU_SUBGROUPS.get(menu_label)
            with v3.VMenu(location="bottom start", open_on_hover=bool(subgroups)):
                with v3.Template(v_slot_activator="{ props }"):
                    v3.VBtn(
                        menu_label,
                        v_bind="props",
                        variant="text",
                        size="small",
                        append_icon="mdi-chevron-down",
                    )
                with v3.VList(density="compact"):
                    if subgroups:
                        for cat_label, cat_ops in subgroups:
                            _build_submenu(ctx, cat_label, cat_ops)
                    else:
                        for op in ops:
                            v3.VListItem(
                                title=op["label"],
                                click=(ctx.ctrl.tomo_add_op, f"['{op['id']}']"),
                            )


def _build_submenu(ctx, cat_label, ops):
    """Emit one hover-expandable category as a nested submenu list item."""
    with v3.VMenu(location="end top", open_on_hover=True, open_on_click=False):
        with v3.Template(v_slot_activator="{ props }"):
            v3.VListItem(
                v_bind="props",
                title=cat_label,
                append_icon="mdi-chevron-right",
            )
        with v3.VList(density="compact"):
            for op in ops:
                v3.VListItem(
                    title=op["label"],
                    click=(ctx.ctrl.tomo_add_op, f"['{op['id']}']"),
                )


# ---------------------------------------------------------------------------
# Pipeline + Properties (left control panel)
# ---------------------------------------------------------------------------
# The Pipeline and Properties *bodies* are factored out so they can be rendered
# either as collapsible accordion tabs (``build_tabs``) or as two always-open
# sections (``build_pipeline_panel`` / ``build_properties_panel``) that a host
# can stack with its own resizable divider. All bind to the same reactive state
# (``pipeline`` / ``selected_op_*``) managed by TomographyFeature's controllers.
def _pipeline_body(ctx):
    """Emit the Pipeline block list + Clear button (vuetify3, no section chrome)."""
    ctrl = ctx.ctrl
    html.Div(
        "No operations yet \u2014 add one from the menus.",
        v_show="pipeline.length === 0",
        classes="text-caption text-medium-emphasis pa-2",
    )
    with v3.VList(density="compact", nav=True, classes="pa-0", v_show="pipeline.length > 0"):
        with v3.VListItem(
            v_for="(block, bi) in pipeline",
            key="block.id",
            click="selected_op_id = block.id",
            active=("selected_op_id === block.id",),
            color="primary",
            rounded="lg",
            classes="mb-1",
        ):
            v3.VListItemTitle(
                "{{ bi + 1 }}. {{ block.label }}",
                classes=("block.enabled ? '' : 'text-decoration-line-through text-disabled'",),
            )
            # Row actions live in the append slot; click_stop keeps them from
            # also selecting the row.
            with v3.Template(v_slot_append=True):
                v3.VBtn(
                    icon=("block.enabled ? 'mdi-eye' : 'mdi-eye-off'",),
                    size="x-small",
                    variant="text",
                    v_on_click_stop=(ctrl.tomo_toggle_op, "[block.id]"),
                    title="Enable / disable",
                )
                v3.VBtn(
                    icon="mdi-arrow-up",
                    size="x-small",
                    variant="text",
                    v_on_click_stop=(ctrl.tomo_move_op, "[block.id, -1]"),
                    title="Move up",
                )
                v3.VBtn(
                    icon="mdi-arrow-down",
                    size="x-small",
                    variant="text",
                    v_on_click_stop=(ctrl.tomo_move_op, "[block.id, 1]"),
                    title="Move down",
                )
                v3.VBtn(
                    icon="mdi-delete",
                    size="x-small",
                    variant="text",
                    color="error",
                    v_on_click_stop=(ctrl.tomo_remove_op, "[block.id]"),
                    title="Delete",
                )
    v3.VBtn(
        "Clear",
        v_show="pipeline.length > 0",
        variant="tonal",
        size="small",
        block=True,
        classes="mt-2",
        prepend_icon="mdi-close",
        click=ctrl.tomo_clear_pipeline,
    )


def _properties_body(ctx):
    """Emit the Properties editor for the selected block (vuetify3, no chrome)."""
    ctrl = ctx.ctrl
    html.Div(
        "Select a pipeline block to edit its parameters.",
        v_show="!selected_op_id",
        classes="text-caption text-medium-emphasis pa-2",
    )
    with html.Div(v_show="selected_op_id", classes="pa-1"):
        html.Div("{{ selected_op_label }}", classes="text-subtitle-2 font-weight-medium mb-2")
        html.Div(
            "This operation has no parameters.",
            v_show="selected_op_params.length === 0",
            classes="text-caption text-medium-emphasis",
        )
        with html.Div(v_for="(p, pi) in selected_op_params", key="p.name"):
            v3.VTextField(
                v_show="(p.type === 'number' || p.type === 'int') && !p.readonly",
                label=("p.label",),
                v_model=("p.value",),
                type="number",
                min=("p.min",),
                max=("p.max",),
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
                input=(ctrl.tomo_set_param, "[selected_op_id, p.name, p.value]"), 
                # blur component events commits the values when the user clicks outside the text input field
                # input commits it immediately as the user types
            )
            # read-only / computed number 
            v3.VTextField(
                v_show="(p.type === 'number' || p.type === 'int') && p.readonly",
                label=("p.label",),
                model_value=("p.value",),
                readonly=True,
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
            )
            # text
            v3.VTextField(
                v_show="p.type === 'text'",
                label=("p.label",),
                v_model=("p.value",),
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
                input=(ctrl.tomo_set_param, "[selected_op_id, p.name, p.value]"),
            )
            # path -- read-only; clicking opens the shared file browser
            v3.VTextField(
                v_show="p.type === 'path'",
                label=("p.label",),
                model_value=("p.value",),
                readonly=True,
                placeholder="Select\u2026",
                append_inner_icon="mdi-folder-open",
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
                click=(ctrl.tomo_browse_param, "[selected_op_id, p.name]"),
            )
            # bool
            v3.VCheckbox(
                v_show="p.type === 'bool'",
                label=("p.label",),
                model_value=("p.value",),
                density="compact",
                hide_details=True,
                update_modelValue=(ctrl.tomo_set_param, "[selected_op_id, p.name, $event]"),
            )
            # choice
            v3.VSelect(
                v_show="p.type === 'choice'",
                label=("p.label",),
                model_value=("p.value",),
                items=("p.choices",),
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
                update_modelValue=(ctrl.tomo_set_param, "[selected_op_id, p.name, $event]"),
            )

        # Tilt series projection slider bar
        with html.Div(v_show="tomo_is_tilt_series", classes="mt-3"):
            html.Div(
                "Projection {{ tomo_projection_index }} / {{ tomo_projection_max }}",
                classes="text-caption text-medium-emphasis mb-1",
            )
            v3.VSlider(
                v_model=("tomo_projection_index",),
                min=0,
                max=("tomo_projection_max",),
                step=1,
                density="compact",
                hide_details=True,
                thumb_label=True,
            )


def build_pipeline_panel(ctx):
    """Public: the Pipeline section body, for an always-open two-section layout."""
    _pipeline_body(ctx)


def build_properties_panel(ctx):
    """Public: the Properties section body, for an always-open two-section layout."""
    _properties_body(ctx)


def build_tabs(ctx, include_menu=True):
    """Emit the tomography left-panel content: menu bar + Pipeline + Properties.

    This is the *accordion* layout (vuetify3 ``VExpansionPanels``): Pipeline and
    Properties are collapsible panels, one open at a time, bound to ``open_tab``.
    Hosts that prefer two always-open sections should call
    :func:`build_pipeline_panel` / :func:`build_properties_panel` instead.

    ``include_menu`` controls whether the header dropdown menu bar is emitted
    here. The RSM-shell path leaves it True (the menus live in the left panel);
    the standalone ``tomo_app`` sets it False because its Vuetify toolbar already
    hosts the op menus (both call the same ``ctrl.tomo_add_op``).
    """
    if include_menu:
        build_menu_bar(ctx)

    with v3.VExpansionPanels(v_model=("open_tab",), variant="accordion"):
        with v3.VExpansionPanel(value="pipeline"):
            v3.VExpansionPanelTitle("Pipeline")
            with v3.VExpansionPanelText():
                _pipeline_body(ctx)
        with v3.VExpansionPanel(value="properties"):
            v3.VExpansionPanelTitle("Properties")
            with v3.VExpansionPanelText():
                _properties_body(ctx)


def build_layer_controls(ctx):
    """Per-layer property editors for tomography Visualization ops.

    Placeholder for Step D: the Visualization ops reuse the shell's existing
    slice/volume/colormap machinery, so this stays a no-op for now.
    """
    return None
