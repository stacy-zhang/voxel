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

        def _find(blocks, op_id):
            for i, b in enumerate(blocks):
                if b["id"] == op_id:
                    return i, b
            return -1, None

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
                return
            spec = tomo_ui.TOMO_OPS[block["op"]]
            state.selected_op_label = block["label"]
            state.selected_op_params = [
                {
                    "name": p["name"],
                    "label": p["label"],
                    "type": p["type"],
                    "choices": p.get("choices", []),
                    "value": block["params"].get(p["name"], p.get("default", "")),
                }
                for p in spec["params"]
            ]

        @ctrl.set("tomo_add_op")
        def tomo_add_op(op_id):
            spec = tomo_ui.TOMO_OPS.get(op_id)
            if spec is None:
                return
            block = {
                "id": f"op{next(self._id_counter)}",
                "op": op_id,
                "label": spec["label"],
                "params": tomo_ui.default_params(op_id),
                "enabled": True,
            }
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
            block["params"] = {**block["params"], name: value}
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