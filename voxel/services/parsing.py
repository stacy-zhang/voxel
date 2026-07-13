"""State-coercion and text-parsing helpers for the web app.

These are pure functions that translate the browser's string/JSON state into
the typed values the pipeline expects (scan lists, UB matrices, goniometer
axes, grid shapes) and mirror the crop logic from the napari widget. They have
no VTK or trame dependency, so the data-access layer can be tested and extended
independently of the UI.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .backend import RSMDataloader_CMS


def _float(value: Optional[object], default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_path(value: Optional[str]) -> str:
    return str(value).strip() if value is not None else ""


def _scan_numbers_in_dir(tiff_dir: Optional[str]) -> list:
    """Return the scan numbers parsed from TIFF filenames in ``tiff_dir``.

    Mirrors ``RSMDataloader_CMS``'s filename parsing (the original napari
    behavior) so the Data-tab scan-range inputs reference the real scan ids
    embedded in the file names rather than positional frame indices. The list
    is sorted ascending, matching the order the CMS loader presents frames.
    """
    directory = Path(_ensure_path(tiff_dir)).expanduser()
    if not directory.is_dir():
        return []
    pattern = re.compile(RSMDataloader_CMS.SCAN_REGEX)
    scans = []
    for path in sorted(
        list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    ):
        match = pattern.search(path.name)
        if match:
            scans.append(int(match.group(1)))
    return sorted(scans)


def _parse_scan_list(text: Optional[str]) -> list:
    """Parse a scan-range string like "17-20, 30" into a sorted list of ints.

    Accepts comma/whitespace-separated single scans ("30") and inclusive
    ranges ("17-20"). Mirrors the napari widget's parse_scan_list so the web
    app accepts the same syntax. Raises ValueError on malformed input.
    """
    text = _ensure_path(text)
    if not text:
        return []
    out: set = set()
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.isdigit() and b.isdigit():
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                raise ValueError(f"Bad scan range: '{part}'")
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"Bad scan id: '{part}'")
    return sorted(out)


def _parse_ub_matrix(text: Optional[str]) -> Optional[np.ndarray]:
    """Parse a UB matrix string into an ndarray (mirrors the napari widget).

    Rows are separated by newlines; values within a row by commas/whitespace.
    Returns None when the text is blank. Raises ValueError on ragged rows.
    """
    stripped = _ensure_path(text)
    if not stripped:
        return None
    rows = []
    for line in stripped.splitlines():
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        if parts:
            rows.append([float(p) for p in parts])
    if not rows:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("UB rows must have equal length.")
    return np.array(rows, dtype=float)


def _format_ub_matrix(ub: Optional[object]) -> str:
    """Format a UB matrix (loaded from a profile) into editable text."""
    if ub is None:
        return ""
    try:
        arr = np.asarray(ub, dtype=float)
    except (TypeError, ValueError):
        return str(ub)
    if arr.ndim == 1:
        return " ".join(f"{v:.6g}" for v in arr)
    if arr.ndim == 2:
        return "\n".join(" ".join(f"{v:.6g}" for v in row) for row in arr)
    return np.array2string(arr, precision=6, separator=" ")


def _parse_axes_list(text: Optional[str]) -> list:
    """Split a goniometer-axis string like "x+, y+, z-" into a list."""
    text = _ensure_path(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]


def _parse_grid_shape(
    text: Optional[str],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse a grid-shape string like "100,*,*" into a (nx, ny, nz) tuple.

    Mirrors the napari widget's ``parse_grid_shape``. A single value ("100")
    expands to "100,*,*". Any dimension given as ``*`` (or blank) becomes
    ``None``, which tells ``regrid_xu`` to auto-scale that dimension from the
    data extents. Raises ValueError on malformed input.
    """
    text = _ensure_path(text)
    if not text:
        return (None, None, None)
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 1:
        parts += ["*", "*"]
    if len(parts) != 3:
        raise ValueError("Grid must be 'x,y,z' (y/z may be '*').")

    def _one(p: str) -> Optional[int]:
        if p in ("*", ""):
            return None
        if not p.isdigit():
            raise ValueError(f"Bad grid value: '{p}'")
        v = int(p)
        if v <= 0:
            raise ValueError("Grid values must be positive.")
        return v

    return tuple(_one(p) for p in parts)  # type: ignore[return-value]


# --- crop helpers (mirror napari's on_crop_from_roi) --------------------
def _crop_window_from_state(state) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if not getattr(state, "crop_enabled", False):
        return None

    try:
        r0 = int(state.crop_row_min)
        r1 = int(state.crop_row_max)
        c0 = int(state.crop_col_min)
        c1 = int(state.crop_col_max)
    except (TypeError, ValueError):
        return None

    if r0 < 0 or c0 < 0 or r1 <= r0 or c1 <= c0:
        return None
    return ((r0, r1), (c0, c1))


def _adjust_setup_for_crop(setup, crop_window: Tuple[Tuple[int, int], Tuple[int, int]]):
    # Shift the beam center by the crop origin and resize the detector to the
    # crop dimensions, mirroring napari's on_crop_from_roi (new_yc = old_yc -
    # y_min, new detector size = crop height/width). The beam center thus stays
    # over the same physical pixel of the (now cropped) intensity map.
    (r0, r1), (c0, c1) = crop_window
    setup.xcenter = max(0, int(setup.xcenter) - c0)
    setup.ycenter = max(0, int(setup.ycenter) - r0)
    setup.xpixels = max(1, int(c1 - c0))
    setup.ypixels = max(1, int(r1 - r0))


def _crop_dataframe_intensity(df, crop_window):
    if crop_window is None:
        return df
    (r0, r1), (c0, c1) = crop_window
    if r1 <= r0 or c1 <= c0:
        return df

    df = df.copy()
    df["intensity"] = [frame[r0:r1, c0:c1] for frame in df["intensity"]]
    return df
