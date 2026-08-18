"""Import-isolation guard.

The separate-server design (``pixi run rsm`` vs a future ``pixi run tomo``)
depends on the *shared* surface — parsing/coercion utilities, colormap math,
UI assets, the feature UI, and the app shell module — staying free of the heavy
RSM reconstruction engine (``xrayutilities`` pulled in via
``voxel.services.backend`` -> ``rsm3d``). If any of these imports xrayutilities
at *import time*, a tomography app that reuses them would needlessly load RSM's
dependencies.

Each module is checked in a FRESH interpreter (subprocess) because
``sys.modules`` is process-global and would otherwise leak state between checks.

Run directly (``python tests/test_import_isolation.py``) or under pytest.
"""

import subprocess
import sys

# Modules that MUST NOT import xrayutilities at import time.
CLEAN_MODULES = [
    "voxel.services.parsing",
    "voxel.visualization.colormaps",
    "voxel.ui.assets",
    "voxel.features.rsm.feature",
    "voxel.features.rsm.ui",
    "voxel.app.server",
]

# Modules that MUST NOT import tomopy at import time. The tomography numeric
# pipeline wraps TomoPy but lazy-imports it *inside* each operator, so importing
# the module (as the shared surface may) never pulls TomoPy's heavy stack -- and
# it stays importable even where TomoPy is not installed (the dev/pixi env).
TOMOPY_CLEAN_MODULES = [
    "voxel.features.tomo.pipeline",
]


def _imports_dep(module: str, dep: str) -> bool:
    """True if importing ``module`` in a fresh interpreter pulls in ``dep``."""
    code = (
        "import importlib, sys;"
        f"importlib.import_module({module!r});"
        f"print({dep!r} in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"importing {module} failed:\n{proc.stderr}"
    # The last stdout line is the boolean; earlier lines may be env warnings.
    return proc.stdout.strip().splitlines()[-1] == "True"


def _imports_xrayutilities(module: str) -> bool:
    """True if importing ``module`` in a fresh interpreter pulls in xrayutilities."""
    return _imports_dep(module, "xrayutilities")


def test_shared_surface_is_xrayutilities_free():
    offenders = [m for m in CLEAN_MODULES if _imports_xrayutilities(m)]
    assert not offenders, (
        "these modules pull in xrayutilities at import time (heavy RSM engine "
        "must be lazy-imported instead): " + ", ".join(offenders)
    )


def test_tomo_pipeline_is_tomopy_free():
    offenders = [m for m in TOMOPY_CLEAN_MODULES if _imports_dep(m, "tomopy")]
    assert not offenders, (
        "these modules pull in tomopy at import time (TomoPy must be lazy-imported "
        "inside operators instead): " + ", ".join(offenders)
    )


if __name__ == "__main__":
    test_shared_surface_is_xrayutilities_free()
    print("OK: shared surface is xrayutilities-free")
    test_tomo_pipeline_is_tomopy_free()
    print("OK: tomo pipeline is tomopy-free at import")
