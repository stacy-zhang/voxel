"""Bridge to the RSM reconstruction backend (``voxel/rsm3d``).

STAGE 1 (LOAD) and STAGE 2 (BUILD/REGRID) of the pipeline live in the
``rsm3d`` package (the original napari ResView engine). Those two stages are
front-end agnostic and are reused verbatim by the web app.

The backend's own ``rsm3d/__init__.py`` eagerly imports ``resview_widget`` and
``data_viz``, which depend on napari/Qt. The web app runs headless and must
avoid that GUI dependency, so instead of ``import voxel.rsm3d`` we register a
lightweight ``rsm3d`` namespace package pointed at the source directory and
import only the pure ``data_io`` / ``rsm3d`` submodules from it. This mirrors
the loader logic that used to live at the top of ``web_app.py``.
"""

import importlib
import sys
import types
from pathlib import Path

PACKAGE_NAME = "rsm3d"
# voxel/services/backend.py -> voxel/rsm3d
PACKAGE_PATH = Path(__file__).resolve().parent.parent / PACKAGE_NAME

if PACKAGE_NAME not in sys.modules:
    _package_module = types.ModuleType(PACKAGE_NAME)
    _package_module.__path__ = [str(PACKAGE_PATH)]
    sys.modules[PACKAGE_NAME] = _package_module

_data_io = importlib.import_module("rsm3d.data_io")
_rsm3d = importlib.import_module("rsm3d.rsm3d")

RSMDataLoader_ISR = _data_io.RSMDataLoader_ISR
RSMDataloader_CMS = _data_io.RSMDataloader_CMS
write_rsm_volume_to_vtr = _data_io.write_rsm_volume_to_vtr
RSMBuilder = _rsm3d.RSMBuilder

# --- defaults YAML path -------------------------------------------------
# partly copied from resview_widget.py
DEFAULTS_ENV = "RSM3D_DEFAULTS_YAML"
# The bundled defaults YAML lives inside the rsm3d package. Use PACKAGE_PATH so
# the auto-filled setup path points at the file that actually exists.
import os  # noqa: E402  (kept after the package bootstrap above)

os.environ.setdefault(
    DEFAULTS_ENV,
    str(PACKAGE_PATH / "rsm3d_defaults.yaml"),
)


def yaml_path() -> str:
    """Resolve the defaults YAML path.

    Allows override via the ``RSM3D_DEFAULTS_YAML`` environment variable, else
    falls back to ``~/.rsm3d_defaults.yaml``.
    """
    p = os.environ.get(DEFAULTS_ENV, "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.join(os.path.expanduser("~"), ".rsm3d_defaults.yaml")


__all__ = (
    "PACKAGE_NAME",
    "PACKAGE_PATH",
    "RSMDataLoader_ISR",
    "RSMDataloader_CMS",
    "write_rsm_volume_to_vtr",
    "RSMBuilder",
    "DEFAULTS_ENV",
    "yaml_path",
)
