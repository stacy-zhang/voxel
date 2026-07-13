"""Backward-compatible shim for the web app entry points.

The web application has been repackaged under ``voxel`` with data access,
reconstruction, visualization, and UI concerns split into separate packages:

    voxel/services       - data access + state coercion (STAGE 1 loaders,
                           parsing/crop helpers, backend bridge)
    voxel/rsm3d          - reconstruction engine (STAGE 2, unchanged)
    voxel/visualization  - render helpers (colormaps / transfer functions)
    voxel/ui             - static UI assets (colormap names, icons)
    voxel/app            - trame server construction + orchestration (STAGE 3)

``create_server`` / ``run_server`` now live in ``voxel.app.server``. This module
re-exports them so existing callers (``main.py``, ``import web_app``) keep
working.
"""

from voxel.app.server import create_server, run_server

__all__ = ("create_server", "run_server")


if __name__ == "__main__":
    run_server()
