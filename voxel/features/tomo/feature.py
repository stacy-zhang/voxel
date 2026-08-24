"""Tomography feature: binds the tomviz-style workflow to the shell contract.

The single object the shell talks to for the tomography front-end (mirrors
RSM Feature). The UI hooks delegate to the tomo ui.py file; 
:meth:`register_controllers` installs the pipeline-management step handlers 
the UI binds to.

*UI + reactive pipeline state* layer:

  * ``pipeline``        -- ordered list of block dicts
                           ``{id, op, label, params, enabled}``
  * ``selected_op_id``  -- id of the block shown in Properties
  * ``selected_op_*``   -- flattened view of the selected block for the editor
  * ``menu_open``       -- open header dropdown ('' = none)

The controllers here mutate that state. ``tomo_run_pipeline`` is a stub
that reports the plan. Keeping the numeric work out of this module
(and importing ``pipeline`` lazily) preserves import isolation: importing this
feature pulls neither TomoPy nor the RSM engine.
"""

from __future__ import annotations

import itertools

from voxel.features.base import FeatureContext, VoxelFeature
from voxel.features.tomo import ui as tomo_ui


class TomographyFeature(VoxelFeature):
    """The tomography front-end, exposed through the shared feature contract."""

    key = "Tomography"
    title = "Tomography"

    def __init__(self) -> None:
        # ID numbering for pipeline blocks (string ids stay stable across
        # reordering / deletion, unlike list indices).
        self._id_counter = itertools.count(1)

    # ui
    def build_tabs(self, ctx: FeatureContext) -> None:
        tomo_ui.build_tabs(ctx)

    def build_layer_controls(self, ctx: FeatureContext) -> None:
        tomo_ui.build_layer_controls(ctx)

    # controllers
    def register_controllers(self, ctx: FeatureContext) -> None:
        """Install the pipeline-management step handlers (reactive state only)."""
        state = ctx.state
        ctrl = ctx.ctrl

        # Seed state defaults (idempotent).
        state.setdefault("pipeline", [])
        state.setdefault("selected_op_id", None)
        state.setdefault("selected_op_label", "")
        state.setdefault("selected_op_params", [])
        state.setdefault("menu_open", "")
        state.setdefault("open_tab", "pipeline")

        # Full extents of the loaded base dataset (x=cols, y=rows, z=depth/theta),
        # 0 until data is loaded. Used to bound the Crop editor's max inputs.
        state.setdefault("tomo_dim_x", 0)
        state.setdefault("tomo_dim_y", 0)
        state.setdefault("tomo_dim_z", 0)

        # Crop param name -> the state key holding that axis's full extent.
        _CROP_AXIS = {
            "x_min": "tomo_dim_x", "x_max": "tomo_dim_x",
            "y_min": "tomo_dim_y", "y_max": "tomo_dim_y",
            "z_min": "tomo_dim_z", "z_max": "tomo_dim_z",
        }

        def _crop_dim(name) -> int:
            """Full extent of the axis a crop param bounds (0 if unknown)."""
            return int(getattr(state, _CROP_AXIS[name], 0) or 0)

        def _clamp_crop(name, value, dim) -> int:
            """Clamp a crop value into ``[0, dim]``; max params snap 0/overflow to dim."""
            try:
                v = int(float(value)) # value is the user input
            except (TypeError, ValueError):
                v = 0
            if dim <= 0:
                return v  # extents unknown: keep the raw value (0 = end fallback)
            if name.endswith("_max"): # an input for the max crop bound 
                return dim if v <= 0 or v > dim else v # if input is 0 or too big, clamp to dim
            return max(0, min(v, dim))

        def _find(blocks, op_id):
            for i, b in enumerate(blocks):
                if b["id"] == op_id:
                    return i, b
            return -1, None

        def _recompute_ang_inc(block) -> None:
            """Derive angle increment from the image/angle range."""
            p = block["params"]
            try:
                span = float(p.get("img2", 0)) - float(p.get("img1", 0))
                p["ang_inc"] = (float(p.get("ang2", 0)) - float(p.get("ang1", 0))) / span if span else 0.0
            except (TypeError, ValueError):
                p["ang_inc"] = 0.0

        def _sync_selected_params() -> None:
            """Rebuild ``selected_op_*`` from the currently selected block.

            The Properties editor renders ``selected_op_params`` (descriptor +
            live value) rather than reaching into ``pipeline`` directly, so this
            keeps that flattened view in sync on selection / value change.
            """
            op_id = state.selected_op_id
            _, block = _find(state.pipeline, op_id)
            if block is None:
                state.selected_op_label = ""
                state.selected_op_params = []
                state.tomo_is_open_data = False
                state.tomo_is_tilt_series = False
                return
            spec = tomo_ui.TOMO_OPS[block["op"]]
            is_crop = block["op"] == "crop"
            is_open = block["op"] == "open_data"
            state.tomo_is_open_data = is_open
            state.tomo_is_tilt_series = (
                is_open and block["params"].get("data_type") == "Tilt Series"
            )
            state.selected_op_label = block["label"]
            params = []
            for p in spec["params"]:
                entry = {
                    "name": p["name"],
                    "label": p["label"],
                    "type": p["type"],
                    "choices": p.get("choices", []),
                    "value": block["params"].get(p["name"], p.get("default", "")),
                    "min": p.get("min"),
                    "max": p.get("max"),
                    "readonly": p.get("readonly", False),
                }
                if is_crop and _crop_dim(p["name"]):
                    entry["min"] = 0
                    entry["max"] = _crop_dim(p["name"])
                params.append(entry)
            state.selected_op_params = params

        # Sync properties panel to the selected block
        @state.change("selected_op_id")
        def _on_selected_op_change(**_kw):
            _sync_selected_params()

        @ctrl.set("tomo_add_op")
        def tomo_add_op(op_id, label=None, params=None):
            spec = tomo_ui.TOMO_OPS.get(op_id)
            if spec is None:
                return
            block = {
                "id": f"op{next(self._id_counter)}",
                "op": op_id,
                "label": label or spec["label"],
                "params": tomo_ui.default_params(op_id),
                "enabled": True,
            }
            # Crop maxes default to the full extent (last slice) of each axis
            # rather than the placeholder 0, so the user sees the real bounds.
            if op_id == "crop":
                for name in ("x_max", "y_max", "z_max"):
                    dim = _crop_dim(name)
                    if dim:
                        block["params"][name] = dim

            if op_id == "set_angles":
                block["params"]["img2"] = int(state.tomo_projection_max or 0)
                ang_span = (block["params"]["img2"] - block["params"]["img1"]) * 2
                block["params"]["ang1"] = -1 * ang_span / 2
                block["params"]["ang2"] = ang_span / 2
                _recompute_ang_inc(block)
            if params:
                block["params"] = {**block["params"], **params} # Merge user-provided params with default ones
            state.pipeline = state.pipeline + [block]
            state.selected_op_id = block["id"]
            state.menu_open = ""
            state.open_tab = "properties"
            _sync_selected_params()

        @ctrl.set("tomo_remove_op")
        def tomo_remove_op(op_id):
            state.pipeline = [b for b in state.pipeline if b["id"] != op_id]
            if state.selected_op_id == op_id:
                state.selected_op_id = state.pipeline[-1]["id"] if state.pipeline else None
            _sync_selected_params()

        @ctrl.set("tomo_toggle_op")
        def tomo_toggle_op(op_id):
            blocks = [dict(b) for b in state.pipeline]
            _, block = _find(blocks, op_id)
            if block is not None:
                block["enabled"] = not block["enabled"]
                state.pipeline = blocks

        @ctrl.set("tomo_move_op")
        def tomo_move_op(op_id, delta):
            blocks = list(state.pipeline)
            i, _ = _find(blocks, op_id)
            j = i + int(delta)
            if i < 0 or j < 0 or j >= len(blocks):
                return
            blocks[i], blocks[j] = blocks[j], blocks[i]
            state.pipeline = blocks

        @ctrl.set("tomo_set_param")
        def tomo_set_param(op_id, name, value):
            blocks = [dict(b) for b in state.pipeline]
            _, block = _find(blocks, op_id)
            if block is None:
                return
            if block["op"] == "crop" and name in _CROP_AXIS:
                value = _clamp_crop(name, value, _crop_dim(name))
            block["params"] = {**block["params"], name: value}
            if block["op"] == "set_angles":
                _recompute_ang_inc(block)
            state.pipeline = blocks
            if state.selected_op_id == op_id:
                _sync_selected_params()

        @ctrl.set("tomo_browse_param")
        def tomo_browse_param(op_id, name):
            # Route the shared file browser back into the selected block's param.
            ctx.fb_open(f"tomo_browse::{op_id}::{name}", "file")

        @ctrl.set("tomo_clear_pipeline")
        def tomo_clear_pipeline():
            state.pipeline = []
            state.selected_op_id = None
            _sync_selected_params()

        @ctrl.set("tomo_run_pipeline")
        def tomo_run_pipeline():
            enabled = [b for b in state.pipeline if b.get("enabled", True)]
            if not enabled:
                return
            names = " \u2192 ".join(b["label"] for b in enabled)