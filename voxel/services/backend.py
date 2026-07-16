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

import contextlib
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


def write_rsm_volume_to_vtr_compat(rsm, coords, filename, compress=True):
    """Write a 3D RSM volume to a portable, widely-readable .vtr file.

    This is a ParaView-compatibility wrapper around the same rectilinear-grid
    export that ``rsm3d.data_io.write_rsm_volume_to_vtr`` performs. The rsm3d
    writer emits the data in *appended* mode (a trailing ``<AppendedData>``
    section), which relies on VTK's ``EncodeAppendedData`` default: when that
    default is OFF the appended block is written as raw binary, so raw bytes
    (including ``<`` / ``&`` / control characters) land in the middle of the
    XML and older readers such as ParaView 5.9.1 abort with

        Error parsing XML in stream ... not well-formed (invalid token)

    Even when the appended block is base64-encoded, the ``<AppendedData>``
    seek path is a frequent source of cross-version incompatibility.

    To avoid both problems this writer uses *inline binary* data mode
    (``format="binary"`` inside each ``DataArray``): the payload is always
    base64-encoded and lives inside the element content, so there is no
    ``<AppendedData>`` section at all and the file is valid, self-contained XML
    that every ParaView/VTK version can parse. The grid geometry, bin-edge
    inference, descending-axis handling and NaN sanitisation mirror the rsm3d
    writer exactly, so the exported volume is identical -- only the on-disk
    encoding differs.

    Parameters
    ----------
    rsm : (nx, ny, nz) ndarray
        Cell-centered intensities (one value per bin).
    coords : [x_coords, y_coords, z_coords]
        For each axis, either bin EDGES (length n+1) or bin CENTERS (length n;
        edges are inferred). Ascending or descending order is accepted.
    filename : str
        Output path; ``.vtr`` is enforced if missing.
    compress : bool
        Enable zlib compression of the inline binary data.
    """
    import numpy as np
    import vtk
    from vtkmodules.util import numpy_support

    x_c, y_c, z_c = [np.asarray(a, dtype=np.float64) for a in coords]
    nx, ny, nz = map(int, rsm.shape)

    def _as_edges(arr, n):
        """Return edges of length n+1, from either edges (n+1) or centers (n)."""
        m = arr.size
        if m == n + 1:
            return arr.copy()
        if m == n:
            edges = np.empty(n + 1, dtype=np.float64)
            edges[1:-1] = 0.5 * (arr[1:] + arr[:-1])
            edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
            edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])
            return edges
        raise ValueError(
            f"Coordinate array must have length {n} or {n+1}; got {m}."
        )

    x_edges = _as_edges(x_c, nx)
    y_edges = _as_edges(y_c, ny)
    z_edges = _as_edges(z_c, nz)

    # Ensure each axis is ascending; if not, flip both coords and data.
    rsm_work = np.asarray(rsm, dtype=np.float32)
    if x_edges[1] < x_edges[0]:
        x_edges = x_edges[::-1].copy()
        rsm_work = np.flip(rsm_work, axis=0)
    if y_edges[1] < y_edges[0]:
        y_edges = y_edges[::-1].copy()
        rsm_work = np.flip(rsm_work, axis=1)
    if z_edges[1] < z_edges[0]:
        z_edges = z_edges[::-1].copy()
        rsm_work = np.flip(rsm_work, axis=2)

    if (
        (np.diff(x_edges) <= 0).any()
        or (np.diff(y_edges) <= 0).any()
        or (np.diff(z_edges) <= 0).any()
    ):
        raise ValueError("Non-positive bin width detected after adjustment.")

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(nx + 1, ny + 1, nz + 1)
    grid.SetExtent(0, nx, 0, ny, 0, nz)
    grid.SetXCoordinates(numpy_support.numpy_to_vtk(x_edges, deep=True))
    grid.SetYCoordinates(numpy_support.numpy_to_vtk(y_edges, deep=True))
    grid.SetZCoordinates(numpy_support.numpy_to_vtk(z_edges, deep=True))

    # Cell data: sanitize + Fortran order so I (x) is fastest (VTK IJK).
    rsm_work = np.nan_to_num(rsm_work, copy=True)
    vtk_int = numpy_support.numpy_to_vtk(rsm_work.ravel(order="F"), deep=True)
    vtk_int.SetName("intensity")
    grid.GetCellData().SetScalars(vtk_int)
    grid.GetCellData().SetActiveScalars("intensity")

    if not filename.lower().endswith(".vtr"):
        base = filename.rsplit(".", 1)[0] if "." in filename else filename
        filename = base + ".vtr"

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetInputData(grid)
    writer.SetFileName(filename)
    # Inline binary (base64) data => no <AppendedData> section, so the file is
    # valid self-contained XML that every ParaView/VTK version can parse.
    writer.SetDataModeToBinary()
    # UInt32 headers keep the file readable by older (ParaView 5.x-era) readers.
    with contextlib.suppress(AttributeError):
        writer.SetHeaderTypeToUInt32()
    if compress:
        with contextlib.suppress(AttributeError):
            writer.SetCompressorTypeToZLib()
    else:
        with contextlib.suppress(AttributeError):
            writer.SetCompressorTypeToNone()

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTR file: {filename}")


__all__ = (
    "PACKAGE_NAME",
    "PACKAGE_PATH",
    "RSMDataLoader_ISR",
    "RSMDataloader_CMS",
    "write_rsm_volume_to_vtr",
    "write_rsm_volume_to_vtr_compat",
    "RSMBuilder",
    "DEFAULTS_ENV",
    "yaml_path",
)
