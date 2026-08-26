"""Tomography numeric pipeline: pure, render-free wrappers over TomoPy.

Tomography counterpart to :mod:`voxel.features.rsm.pipeline`: a set
of small, pure functions that take the in-memory dataset (:class:`TomoData`) plus
a block's parameter dict and return a new :class:`TomoData`. There is **no** trame
state and **no** VTK here, so the whole module is importable and testable headless.

Design rules:

* **Import isolation.** ``tomopy`` is imported *lazily inside each operator*, never
  at module top level, so importing this module (or the shared surface that may
  reach it) does not pull TomoPy's heavy stack. In the dev (pixi) environment
  TomoPy is not installed at all; the module still imports, and only *calling* a
  TomoPy-backed operator raises a clear error.
* **Angle-first, radians.** ``prj`` is a 3-D stack ``(theta, y, x)`` and ``ang`` is
  in radians as expected by TomoPy operators.
* **Single source of truth for the runner.** :data:`OP_FUNCS` maps a pipeline
  block's ``op`` id to its operator function; :func:`run_pipeline` threads a
  ``TomoData`` through the enabled blocks top-to-bottom. Ops that are not numeric
  transforms (Open/Save/Visualization) are simply absent from :data:`OP_FUNCS`
  and skipped.

Operator contract::

    def op(data: TomoData, params: dict) -> TomoData

``params`` values arrive as strings from the trame Properties editor, so each
operator coerces the fields it needs (blank -> sensible default / ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field
from typing import Any, Callable, Optional

import numpy as np
import tomopy
import dxchange
import matplotlib.pyplot as plt
import logging

# set up logging to monitor reconstruction process
logging.basicConfig(level=logging.INFO) 

# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
@dataclass
class TomoData:
    """The single in-memory dataset threaded through the pipeline.

    Attributes mirror TomoPy conventions so operators can pass ``prj``/``ang``
    straight through. ``prj`` is the working projection stack; ``recon`` holds the
    reconstructed volume once :func:`op_recon` has run.
    """

    prj: Optional[np.ndarray] = None          # (theta, y, x) projection stack
    ang: Optional[np.ndarray] = None          # projection angles, radians
    flat: Optional[np.ndarray] = None         # flat / white field(s)
    dark: Optional[np.ndarray] = None         # dark field(s)
    center: Optional[float] = None            # rotation center (px)
    recon: Optional[np.ndarray] = None        # (z, y, x) reconstructed volume
    kind: str = "tilt_series"                 # "tilt_series" | "volume"
    align_err: Optional[np.ndarray] = None    # per-iteration error from align_*

    def with_(self, **changes: Any) -> "TomoData":
        """Return a shallow copy with ``changes`` applied (functional update)."""
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Parameter coercion (Properties editor sends strings)
# ---------------------------------------------------------------------------
def _f(params: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    """Float param, or ``default`` when blank / missing / unparseable."""
    v = params.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(params: dict, key: str, default: Optional[int] = None) -> Optional[int]:
    """Int param, or ``default`` when blank / missing / unparseable."""
    v = params.get(key, "")
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _b(params: dict, key: str, default: bool = False) -> bool:
    """Bool param tolerant of trame's string / native forms."""
    v = params.get(key, default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _s(params: dict, key: str, default: str = "") -> str:
    """String param, trimmed."""
    v = params.get(key, default)
    return str(v).strip() if v is not None else default


def _require_prj(data: TomoData) -> np.ndarray:
    if data.prj is None:
        raise ValueError("No projection data loaded (prj is None).")
    return data.prj


def _working_field(data: TomoData) -> str:
    """Array a data transform edits: ``recon`` for volumes, ``prj`` for tilt series."""
    return "recon" if data.kind == "volume" else "prj"


def _require_working(data: TomoData) -> np.ndarray:
    """The active image array (volume ``recon`` or projection ``prj``)."""
    field = _working_field(data)
    arr = getattr(data, field)
    if arr is None:
        raise ValueError("No image data loaded.")
    return arr


def _require_ang(data: TomoData) -> np.ndarray:
    if data.ang is None:
        raise ValueError(
            "Projection angles are not set. Add 'Set Tilt Angles' (or load angle "
            "metadata) before alignment / center finding / reconstruction."
        )
    return np.asarray(data.ang)


# ===========================================================================
# Data Transforms
# ===========================================================================
def op_crop(data: TomoData, params: dict) -> TomoData:
    """Crop the volume / projection stack to a row/column ROI (pure NumPy, no TomoPy)."""
    arr = _require_working(data)
    nz, ny, nx = arr.shape
    r0 = max(0, _i(params, "y_min", 0) or 0)
    r1 = _i(params, "y_max", 0) or 0
    c0 = max(0, _i(params, "x_min", 0) or 0)
    c1 = _i(params, "x_max", 0) or 0
    d0 = max(0, _i(params, "z_min", 0) or 0)
    d1 = _i(params, "z_max", 0) or 0
    r1 = ny if r1 <= r0 else min(ny, r1)
    c1 = nx if c1 <= c0 else min(nx, c1)
    d1 = nz if d1 <= d0 else min(nz, d1)
    return data.with_(**{_working_field(data): arr[d0:d1, r0:r1, c0:c1]})


def op_downsample(data: TomoData, params: dict) -> TomoData:
    """Bin volumes by x2 along all a    xes (tomopy.misc.morph.downsample)."""

    axes = {"All Axes (Uniform)": [0,1,2], "Projection Angles": [0], "Vertical Detector Height": [1], "Horizontal Detector Width": [2]}

    arr = _require_working(data)
    for axis in axes.get(params.get('axis')):
        arr=tomopy.misc.morph.downsample(arr, level=1, axis=axis)
    return data.with_(**{_working_field(data): arr}) # ** unpacks the dictionary into keyword args


def op_resample(data: TomoData, params: dict) -> TomoData:
    """Bin tilt series x2 along X and Y image dimensions (tomopy.misc.morph.downsample)."""

    arr = _require_working(data)
    for axis in [1,2]:
        arr=tomopy.misc.morph.downsample(arr, level=1, axis=axis)
    return data.with_(**{_working_field(data): arr})


def op_median_filter(data: TomoData, params: dict) -> TomoData:
    """Median-filter each projection (tomopy.misc.corr.median_filter)."""
    #from tomopy.misc.corr import median_filter

    arr = _require_working(data)
    size = _i(params, "size", 3) or 3
    axis = _i(params, "axis", 0)
    return data.with_(**{_working_field(data): tomopy.misc.corr.median_filter(arr, size=size, axis=axis)})

def op_gaussian_filter(data: TomoData, params: dict) -> TomoData:
    """Gaussian-filter each projection (tomopy.misc.corr.gaussian_filter)."""
    #from tomopy.misc.corr import gaussian_filter

    arr = _require_working(data)
    sigma = _f(params, "sigma", 2.0) or 2.0
    axis = _i(params, "axis", 0)
    return data.with_(**{_working_field(data): tomopy.misc.corr.gaussian_filter(arr, sigma=sigma, axis=axis)})

def op_wiener_filter(data: TomoData, params: dict) -> TomoData:
    """Wiener-filter each projection (tomopy.misc.corr.wiener_filter)."""
    #from tomopy.misc.corr import wiener_filter

    arr = _require_working(data)
    sigma_x = _f(params, "sigma_x", 0.5) or 0.5
    sigma_y = _f(params, "sigma_y", 0.5) or 0.5
    sigma_z = _f(params, "sigma_z", 0.5) or 0.5
    snr = _f(params, "snr", 15.0) or 15.0
    axis = _i(params, "axis", 0)
    return data.with_(**{_working_field(data): tomopy.misc.corr.wiener_filter(arr, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z, snr=snr, axis=axis)})


# ===========================================================================
# Pre-processing
# ===========================================================================
def op_normalize(data: TomoData, params: dict) -> TomoData:
    """Flat/dark-field normalize (tomopy.prep.normalize.normalize).

    Requires ``flat`` and ``dark`` on the dataset; raises otherwise so the user
    gets a clear message instead of a silent no-op.
    """
    #from tomopy.prep.normalize import normalize

    prj = _require_prj(data)
    if data.flat is None or data.dark is None:
        raise ValueError(
            "Normalize needs flat and dark fields; none were loaded with this dataset."
        )
    cutoff = _f(params, "cutoff", None)
    out = tomopy.prep.normalize.normalize(prj, np.asarray(data.flat), np.asarray(data.dark), cutoff=cutoff)
    return data.with_(prj=out)


def op_normalize_bg(data: TomoData, params: dict) -> TomoData:
    """Background (air) normalization (tomopy.prep.normalize.normalize_bg)."""
    #from tomopy.prep.normalize import normalize_bg

    prj = _require_prj(data)
    air = _i(params, "air", 1) or 1
    return data.with_(prj=tomopy.prep.normalize.normalize_bg(prj, air=air))


def op_minus_log(data: TomoData, params: dict) -> TomoData:
    """Transmission -> attenuation, i.e. ``-log`` (tomopy.prep.normalize.minus_log)."""
    #from tomopy.prep.normalize import minus_log

    prj = _require_prj(data)
    return data.with_(prj=tomopy.prep.normalize.minus_log(prj))


def op_remove_stripe(data: TomoData, params: dict) -> TomoData:
    """Fourier-wavelet ring/stripe removal (tomopy.prep.stripe.remove_stripe_fw)."""
    #from tomopy.prep.stripe import remove_stripe_fw

    prj = _require_prj(data)
    level = _i(params, "level", None)          # None -> TomoPy auto
    wname = _s(params, "wname", "db5") or "db5"
    sigma = _f(params, "sigma", 2.0)
    out = tomopy.prep.stripe.remove_stripe_fw(prj, level=level, wname=wname, sigma=sigma, pad=True)
    return data.with_(prj=out)


def op_retrieve_phase(data: TomoData, params: dict) -> TomoData:
    """Paganin single-material phase retrieval (tomopy.prep.phase.retrieve_phase)."""
    #from tomopy.prep.phase import retrieve_phase

    prj = _require_prj(data)
    out = tomopy.prep.phase.retrieve_phase(
        prj,
        pixel_size=_f(params, "pixel_size", 1e-4),
        dist=_f(params, "dist", 50.0),
        energy=_f(params, "energy", 20.0),
        alpha=_f(params, "alpha", 1e-3),
    )
    return data.with_(prj=out)


# ===========================================================================
# Angles
# ===========================================================================
def op_set_angles(data: TomoData, params: dict) -> TomoData:
    """Generate an evenly spaced angle sweep in radians (tomopy.angles).

    Number of angles is taken from the projection stack so it always matches.
    """
    prj = _require_prj(data)

    nang = prj.shape[0]
    ang1 = _f(params, "ang1", 0.0)
    ang2 = _f(params, "ang2", 180.0)
    return data.with_(ang=tomopy.angles(nang, ang1, ang2))


# ===========================================================================
# Alignment
# ===========================================================================
def _align(data: TomoData, params: dict, joint: bool) -> TomoData:
    prj = _require_prj(data)
    ang = _require_ang(data)
    #from tomopy.prep.alignment import align_joint, align_seq

    fn = tomopy.prep.alignment.align_joint if joint else tomopy.prep.alignment.align_seq
    prj_out, err = fn(
        prj,
        ang,
        iters=_i(params, "iters", 10) or 10,
        pad=(_i(params, "pad_x", 0) or 0, _i(params, "pad_y", 0) or 0),
        blur=_b(params, "blur", True),
        center=data.center,
        algorithm=_s(params, "algorithm", "sirt") or "sirt",
        upsample_factor=_i(params, "upsample_factor", 10) or 10,
        rin=_f(params, "rin", 0.5),
        rout=_f(params, "rout", 0.8),
    )
    return data.with_(prj=prj_out, align_err=np.asarray(err))


def op_align_seq(data: TomoData, params: dict) -> TomoData:
    """Sequential re-projection alignment (tomopy.prep.alignment.align_seq)."""
    return _align(data, params, joint=False)


def op_align_joint(data: TomoData, params: dict) -> TomoData:
    """Joint re-projection alignment (tomopy.prep.alignment.align_joint)."""
    return _align(data, params, joint=True)


def op_shift_images(data: TomoData, params: dict) -> TomoData:
    """Rigidly shift every projection by ``(sx, sy)`` (tomopy.prep.alignment.shift_images)."""
    #from tomopy.prep.alignment import shift_images

    prj = _require_prj(data)
    sx = _f(params, "sx", 0.0)
    sy = _f(params, "sy", 0.0)
    # shift_images shifts in place and returns the array.
    return data.with_(prj=tomopy.prep.alignment.shift_images(prj.copy(), sx, sy))


def op_scale(data: TomoData, params: dict) -> TomoData:
    """Linearly scale projections into ``[-1, 1]`` (tomopy.prep.alignment.scale)."""
    #from tomopy.prep.alignment import scale

    prj = _require_prj(data)
    out = tomopy.prep.alignment.scale(prj)
    # scale() returns (prj_scaled, scl_factor) in some versions; keep the array.
    if isinstance(out, tuple):
        out = out[0]
    return data.with_(prj=out)


def op_blur_edges(data: TomoData, params: dict) -> TomoData:
    """Blur projection edges before registration (tomopy.prep.alignment.blur_edges)."""
    #from tomopy.prep.alignment import blur_edges

    prj = _require_prj(data)
    low = _f(params, "low", 0.0)
    high = _f(params, "high", 0.8)
    return data.with_(prj=tomopy.prep.alignment.blur_edges(prj, low, high))


# ===========================================================================
# Reconstruction
# ===========================================================================
def op_find_center(data: TomoData, params: dict) -> TomoData:
    """Entropy-based rotation-center search (tomopy.recon.rotation.find_center)."""
    prj = _require_prj(data)
    ang = _require_ang(data)
    #from tomopy.recon.rotation import find_center

    init = _f(params, "init", None)
    tol = _f(params, "tol", 0.5)
    center = float(tomopy.recon.rotation.find_center(prj, ang, init=init, tol=tol))
    return data.with_(center=center)


def op_find_center_vo(data: TomoData, params: dict) -> TomoData:
    """Automatic rotation-center detection, Vo's method (find_center_vo)."""
    prj = _require_prj(data)
    #from tomopy.recon.rotation import find_center_vo

    center = float(tomopy.recon.rotation.find_center_vo(prj))
    return data.with_(center=center)


def op_recon(data: TomoData, params: dict) -> TomoData:
    """Reconstruct a volume from projections (tomopy.recon.algorithm.recon).

    ``filter_name`` applies to gridrec/fbp; ``num_iter`` to iterative algorithms.
    Both are forwarded and TomoPy ignores the irrelevant one. Center defaults to
    the dataset's found center, then the image midpoint.
    """
    prj = _require_prj(data)
    ang = _require_ang(data)

    algs = {
        "Fourier Grid": "gridrec",
        "Filtered Back-Projection": "fbp",
        "Algebraic (ART)": "art",
        "Simultaneous Iterative (SIRT)": "sirt",
        "Maximum-Likelihood Expectation Maximization (MLEM)": "mlem",
        "Total Variation (TV)": "tv",
    }
    algorithm = algs.get(_s(params, "algorithm", "Fourier Grid")) or "gridrec"
    center = _f(params, "center", None)
    if center is None:
        center = data.center if data.center is not None else prj.shape[2] / 2.0

    kwargs: dict = {"algorithm": algorithm, "center": center}
    if algorithm in ("gridrec", "fbp"):
        filt = _s(params, "filter_name", "shepp") or "shepp"
        kwargs["filter_name"] = filt
    else:
        kwargs["num_iter"] = _i(params, "num_iter", 1) or 1

    vol = tomopy.recon(prj, ang, **kwargs)
    return data.with_(recon=vol, center=float(center))


def op_circ_mask(data: TomoData, params: dict) -> TomoData:
    """Zero the volume outside an inscribed cylinder (tomopy.misc.corr.circ_mask)."""
    if data.recon is None:
        raise ValueError("Circular mask needs a reconstructed volume; run Reconstruct first.")
    #from tomopy.misc.corr import circ_mask

    axis = _i(params, "axis", 0) or 0
    ratio = _f(params, "ratio", 1.0)
    return data.with_(recon=tomopy.misc.corr.circ_mask(data.recon, axis=axis, ratio=ratio))


# ===========================================================================
# Simulation & Demonstrations
# ===========================================================================
def op_add_noise(data: TomoData, params: dict) -> TomoData:
    """Add Gaussian noise as a ratio of the max (tomopy.prep.alignment.add_noise)."""
    #from tomopy.prep.alignment import add_noise

    prj = _require_prj(data)
    ratio = _f(params, "ratio", 0.05)
    return data.with_(prj=tomopy.prep.alignment.add_noise(prj, ratio=ratio))


def op_add_jitter(data: TomoData, params: dict) -> TomoData:
    """Add random per-projection jitter (tomopy.prep.alignment.add_jitter)."""
    #from tomopy.prep.alignment import add_jitter

    prj = _require_prj(data)
    low = _f(params, "low", 0.0)
    high = _f(params, "high", 1.0)
    out = tomopy.prep.alignment.add_jitter(prj, low=low, high=high)
    # add_jitter returns (prj, shift arrays) in some versions; keep the stack.
    if isinstance(out, tuple):
        out = out[0]
    return data.with_(prj=out)

# ===========================================================================
# Visualization
# ===========================================================================
# placeholders
def op_volume(data: TomoData, params:dict) -> TomoData:
    return data

def op_outline_box(data: TomoData, params:dict) -> TomoData:
    return data

def op_scale_cube(data: TomoData, params:dict) -> TomoData:
    return data

def op_ortho_slices(data: TomoData, params:dict) -> TomoData:
    return data

def op_clip(data: TomoData, params:dict) -> TomoData:
    return data

def op_background(data: TomoData, params:dict) -> TomoData:
    return data

# ---------------------------------------------------------------------------
# Operator registry + runner
# ---------------------------------------------------------------------------
# Maps a pipeline block's ``op`` id to its operator. Non-numeric ops (open_data,
# save_data, and all Visualization ops) are intentionally absent -> the runner
# skips them. Keep these ids in sync with the UI catalog (Step A ``ui.py``).
OP_FUNCS: dict[str, Callable[[TomoData, dict], TomoData]] = {
    # Data Transforms
    "crop": op_crop,
    "downsample": op_downsample,
    "resample": op_resample,
    "median_filter": op_median_filter,
    "gaussian_filter": op_gaussian_filter,
    "wiener_filter": op_wiener_filter,
    # Pre-processing
    "normalize": op_normalize,
    "normalize_bg": op_normalize_bg,
    "minus_log": op_minus_log,
    "remove_stripe": op_remove_stripe,
    "retrieve_phase": op_retrieve_phase,
    # Angles
    "set_angles": op_set_angles,
    # Alignment
    "align_seq": op_align_seq,
    "align_joint": op_align_joint,
    "shift_images": op_shift_images,
    "scale": op_scale,
    "blur_edges": op_blur_edges,
    # Reconstruction
    "find_center": op_find_center,
    "find_center_vo": op_find_center_vo,
    "recon": op_recon,
    "circ_mask": op_circ_mask,
    # Simulation
    "add_noise": op_add_noise,
    "add_jitter": op_add_jitter,
    # Visualization
    "volume": op_volume,
    "outline_box": op_outline_box,
    "scale_cube": op_scale_cube,
    "ortho_slices": op_ortho_slices,
    "clip": op_clip,
    "background": op_background,
}


def apply_op(data: TomoData, op_id: str, params: dict) -> TomoData:
    """Apply a single operator by id. Unknown / non-numeric ids return ``data`` unchanged."""
    fn = OP_FUNCS.get(op_id)
    if fn is None:
        return data
    return fn(data, params or {})


def run_pipeline(
    data: TomoData,
    blocks: list,
) -> TomoData:
    """Run the enabled numeric blocks top-to-bottom, threading ``data``.

    ``blocks`` are the pipeline block dicts ``{id, op, label, params, enabled}``
    from the UI. Blocks that are disabled, or whose ``op`` is not a numeric
    transform, are skipped.
    """
    def _log(msg: str) -> None:
        pass

    for block in blocks:
        if not block.get("enabled", True):
            continue
        op_id = block.get("op", "")
        if op_id not in OP_FUNCS:
            continue
        label = block.get("label", op_id)
        _log(f"Running: {label}\u2026")
        try:
            data = apply_op(data, op_id, block.get("params", {}))
        except Exception as exc:  # noqa: BLE001 - surface to UI, then re-raise
            _log(f"Error in '{label}': {exc}")
            raise
    _log("Pipeline complete.")
    return data
