import contextlib
import logging
from collections.abc import Iterable, Sequence
from typing import Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import napari  # type: ignore
except Exception as e:
    raise ImportError(
        "napari must be installed to use RSMNapariViewer. `pip install napari`"
    ) from e


__all__ = ("RSMNapariViewer", "IntensityNapariViewer")


class RSMNapariViewer:
    """
    Napari viewer for 3D reciprocal-space maps (HKL or Q). Always shows scale bar in Å⁻¹.
    Layers (top→bottom) after reordering:
        RSM3D, Outline box, Outline corners, Corner Labels, World axes
    """

    def __init__(
        self,
        grid: np.ndarray,
        axes: Tuple[Iterable[float], Iterable[float], Iterable[float]],
        *,
        space: str = "hkl",
        name: str = "RSM",
        log_view: bool = True,
        contrast_percentiles: Tuple[float, float] = (1.0, 99.8),
        cmap: str = "viridis",
        rendering: str = "attenuated_mip",
        viewer_kwargs: dict | None = None,
    ) -> None:
        self._validate_grid_axes(grid, axes)
        self.grid, (self.xax, self.yax, self.zax) = self._ensure_ascending(
            grid, axes
        )
        self.space = space.lower()
        if self.space not in {"hkl", "q"}:
            raise ValueError("space must be 'hkl' or 'q'")
        self.name = name
        self.log_view = bool(log_view)
        self.contrast_percentiles = contrast_percentiles
        self.cmap = cmap
        self.rendering = rendering
        self.viewer_kwargs = viewer_kwargs or {}
        self.volume, self.scale, self.translate, self.is_uniform = (
            self._volume_from_grid_axes(
                self.grid, (self.xax, self.yax, self.zax)
            )
        )
        self.viewer: napari.Viewer | None = None
        self.img_layer = None
        self._hud_enabled = True
        self._corner_labels_layer = None  # restored (even if unused)

    # ------------------------------ public ------------------------------------
    def launch(self, viewer: napari.Viewer | None = None) -> napari.Viewer:
        """Launch into a new Napari viewer or reuse the provided `viewer`.

        If `viewer` is None a new `napari.Viewer` is created; otherwise layers
        are added to the supplied viewer.
        """
        v = viewer or napari.Viewer(
            title=f"{self.name} viewer", **self.viewer_kwargs
        )
        # If reusing an existing viewer, clear its current layers first so the
        # new visualization replaces the previous content.
        if viewer is not None:
            with contextlib.suppress(Exception):
                for _ly in list(v.layers):
                    with contextlib.suppress(Exception):
                        v.layers.remove(_ly)
        v.dims.ndisplay = 3
        data = self._log1p_clip(self.volume) if self.log_view else self.volume
        lo, hi = self._robust_percentiles(data, self.contrast_percentiles)

        # STAGE 3 (napari path): hand the dense RSM volume straight to napari.
        # napari builds the GPU volume rendering internally -- `scale`/`translate`
        # place the voxels at the correct reciprocal-space coordinates (derived
        # from the regrid axes), and `colormap`/`contrast_limits` control the
        # appearance. The trame/VTK web app (web_app.py) reproduces this same
        # step manually instead: it wraps the identical volume in a vtkImageData
        # and builds explicit color + opacity transfer functions, because there
        # is no high-level "add_image" helper on the VTK side.
        layer = v.add_image(
            data,
            name="RSM3D",
            colormap=self.cmap,
            scale=self.scale,
            translate=self.translate,
            contrast_limits=(float(lo), float(hi)),
        )
        self._force_volume(layer)

        # REMOVE old simple camera block and RESTORE robust auto zoom
        self._auto_zoom(v, layer)

        v.axes.visible = True
        v.axes.colored = True
        v.axes.arrows = True
        v.scale_bar.visible = True
        v.scale_bar.unit = "Å⁻¹"
        v.dims.axis_labels = (
            ("Qz", "Qy", "Qx") if self.space == "q" else ("L", "K", "H")
        )

        self._add_outline_and_corners(v)
        self._add_axes_vectors(v)
        self._install_hud(v, layer)

        # Desired panel order top→bottom
        with contextlib.suppress(Exception):
            # original intended order
            self._apply_display_order(
                v,
                [
                    "RSM3D",
                    "Outline box",
                    "Outline corners",
                    "Corner Labels",
                    "World axes",
                ],
            )

        self.viewer = v
        self.img_layer = layer
        return v

    def add_grid_overlay(
        self,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        *,
        thickness_vox: int = 1,
        opacity: float = 0.25,
        color: str = "white",
        name: str = "grid-planes",
        max_planes_per_axis: int = 200,
    ) -> None:
        self._require_viewer()
        nx, ny, nz = self.grid.shape
        xax, yax, zax = self.xax, self.yax, self.zax

        def nearest_indices(ax: np.ndarray, step: float) -> np.ndarray:
            if step <= 0 or ax.size < 2:
                return np.array([], dtype=int)
            start = np.ceil(ax[0] / step) * step
            coords = np.arange(start, ax[-1] + 0.5 * step, step)
            idx = np.searchsorted(ax, coords)
            idx = idx[(idx >= 0) & (idx < ax.size)]
            return np.unique(idx)

        ix = nearest_indices(xax, spacing[0])
        iy = nearest_indices(yax, spacing[1])
        iz = nearest_indices(zax, spacing[2])
        if any(sz > max_planes_per_axis for sz in (ix.size, iy.size, iz.size)):
            raise ValueError(
                "Too many grid planes — adjust spacing or max_planes_per_axis."
            )

        mask = np.zeros((nx, ny, nz), dtype=np.uint8)
        half = max(int(thickness_vox) // 2, 0)
        for k in ix:
            mask[max(0, k - half) : min(nx, k + half + 1), :, :] = 1
        for k in iy:
            mask[:, max(0, k - half) : min(ny, k + half + 1), :] = 1
        for k in iz:
            mask[:, :, max(0, k - half) : min(nz, k + half + 1)] = 1

        mask_vol = np.ascontiguousarray(mask.transpose(2, 1, 0))
        layer = self.viewer.add_image(
            mask_vol,
            name=name,
            opacity=float(opacity),
            blending="additive",
            colormap=color,
            scale=self.scale,
            translate=self.translate,
            contrast_limits=(0, 1),
            rendering="translucent",
        )
        with contextlib.suppress(Exception):
            layer.interpolation = "nearest"

    def flip_axis(self, axis: str) -> None:
        self._require_viewer()
        axis = axis.lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if axis == "x":
            self.grid = self.grid[::-1, :, :]
            self.xax = self.xax[::-1].copy()
        elif axis == "y":
            self.grid = self.grid[:, ::-1, :]
            self.yax = self.yax[::-1].copy()
        else:
            self.grid = self.grid[:, :, ::-1]
            self.zax = self.zax[::-1].copy()
        self.volume, self.scale, self.translate, self.is_uniform = (
            self._volume_from_grid_axes(
                self.grid, (self.xax, self.yax, self.zax)
            )
        )
        data = self._log1p_clip(self.volume) if self.log_view else self.volume
        if self.img_layer is not None:
            self.img_layer.data = data
            self.img_layer.scale = self.scale
            self.img_layer.translate = self.translate
            self._force_volume(self.img_layer)

    def add_slices(
        self,
        axis: str,
        positions: Union[float, Sequence[float]],
        *,
        use_index: bool = False,
        opacity: float = 0.8,
        colormap: str = "turbo",
        name_prefix: str = "slice",
        blending: str = "translucent",
        add_border: bool = False,
    ) -> list:
        """
        Add 2D slice planes through the 3D volume.

        Parameters
        ----------
        axis : str
            Slicing axis: 'x', 'y', or 'z' (or 'h', 'k', 'l' for HKL space).
        positions : float or sequence of float
            Position(s) along the axis to extract slices. By default, these are
            coordinate values (H/K/L or Qx/Qy/Qz). Set `use_index=True` to
            specify voxel indices instead.
        use_index : bool, optional
            If True, `positions` are interpreted as integer voxel indices.
            If False (default), they are coordinate values and the nearest voxel
            is selected.
        opacity : float, optional
            Opacity of the slice layers (0.0=transparent, 1.0=opaque).
        colormap : str, optional
            Napari colormap name for the slices.
        name_prefix : str, optional
            Prefix for slice layer names. Each slice will be named
            "{name_prefix}_{axis}_{value}".
        blending : str, optional
            Blending mode for slice layers ('translucent', 'additive', etc.).
        add_border : bool, optional
            If True, add red border pixels around the edge of each slice.

        Returns
        -------
        list
            List of added napari Image layers.
        """
        self._require_viewer()
        axis = axis.lower()

        # Map HKL names to XYZ
        axis_map = {"h": "x", "k": "y", "l": "z"}
        axis = axis_map.get(axis, axis)

        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', 'z', 'h', 'k', or 'l'")

        # Ensure positions is a sequence
        if isinstance(positions, (int, float)):
            positions = [positions]
        positions = list(positions)

        # Select axis and shape info
        if axis == "x":
            ax_array = self.xax
            ax_dim = 0  # grid dimension
            ax_label = "H" if self.space == "hkl" else "Qx"
        elif axis == "y":
            ax_array = self.yax
            ax_dim = 1
            ax_label = "K" if self.space == "hkl" else "Qy"
        else:  # z
            ax_array = self.zax
            ax_dim = 2
            ax_label = "L" if self.space == "hkl" else "Qz"

        # Convert positions to indices if needed
        if use_index:
            indices = [int(p) for p in positions]
        else:
            indices = [
                int(np.argmin(np.abs(ax_array - pos))) for pos in positions
            ]

        # Validate indices
        max_idx = self.grid.shape[ax_dim] - 1
        indices = [idx for idx in indices if 0 <= idx <= max_idx]
        if not indices:
            logger.warning("No valid slice positions found")
            return []

        # Extract slices and add to viewer
        layers = []
        data = self._log1p_clip(self.volume) if self.log_view else self.volume

        for idx in indices:
            # Extract 2D slice from 3D volume (ZYX ordering) and reshape to 3D
            # with singleton dimension to properly position in 3D space
            coord_value = float(ax_array[idx])

            if axis == "x":
                # X corresponds to last dimension in volume
                slice_2d = data[:, :, idx]  # (Z, Y)
                # Reshape to (Z, Y, 1) for 3D positioning
                slice_3d = slice_2d[:, :, np.newaxis]
                # Full 3D scale and translate, with X position at coord_value
                scale_3d = self.scale  # (dz, dy, dx)
                translate_3d = (
                    self.translate[0],
                    self.translate[1],
                    coord_value,
                )
            elif axis == "y":
                # Y corresponds to middle dimension
                slice_2d = data[:, idx, :]  # (Z, X)
                # Reshape to (Z, 1, X) for 3D positioning
                slice_3d = slice_2d[:, np.newaxis, :]
                # Full 3D scale and translate, with Y position at coord_value
                scale_3d = self.scale  # (dz, dy, dx)
                translate_3d = (
                    self.translate[0],
                    coord_value,
                    self.translate[2],
                )
            else:  # z
                # Z corresponds to first dimension
                slice_2d = data[idx, :, :]  # (Y, X)
                # Reshape to (1, Y, X) for 3D positioning
                slice_3d = slice_2d[np.newaxis, :, :]
                # Full 3D scale and translate, with Z position at coord_value
                scale_3d = self.scale  # (dz, dy, dx)
                translate_3d = (
                    coord_value,
                    self.translate[1],
                    self.translate[2],
                )

            # Add red border to slice if requested
            if add_border:
                slice_3d = self._add_red_border_to_slice(slice_3d)

            layer_name = f"{name_prefix}_{ax_label}_{coord_value:.3f}"

            # Compute percentile-based contrast limits
            lo, hi = self._robust_percentiles(
                slice_2d, self.contrast_percentiles
            )

            layer = self.viewer.add_image(
                slice_3d,
                name=layer_name,
                colormap=colormap,
                scale=scale_3d,
                translate=translate_3d,
                opacity=opacity,
                blending=blending,
                contrast_limits=(float(lo), float(hi)),
            )
            layers.append(layer)
            logger.info(
                "Added slice at %s=%.3f (index %d)", ax_label, coord_value, idx
            )

        return layers

    def _add_red_border_to_slice(
        self, slice_3d: np.ndarray, border_width: int = 1
    ) -> np.ndarray:
        """Add red border pixels to a slice array.

        Parameters
        ----------
        slice_3d : ndarray
            3D slice array (can have singleton dimension)
        border_width : int
            Width of the border in pixels

        Returns
        -------
        ndarray
            Modified slice with red border (uses max value from slice)
        """
        # Get the max value from the slice to use for the border
        max_val = np.nanmax(slice_3d)
        if not np.isfinite(max_val):
            max_val = 1.0

        # Multiply by a factor to make border brighter/more visible
        border_val = max_val * 2.0

        # Create a copy to modify
        result = slice_3d.copy()

        # Find the non-singleton dimensions
        shape = result.shape

        # Apply border to the first two non-singleton dimensions
        # Handle different slice orientations
        if shape[0] > 1 and shape[1] > 1:  # Z-Y plane or Y-X plane
            # Top and bottom edges
            result[:border_width, :, ...] = border_val
            result[-border_width:, :, ...] = border_val
            # Left and right edges
            result[:, :border_width, ...] = border_val
            result[:, -border_width:, ...] = border_val
        elif shape[0] > 1 and shape[2] > 1:  # Z-X plane
            # Top and bottom edges
            result[:border_width, :, :] = border_val
            result[-border_width:, :, :] = border_val
            # Left and right edges
            result[:, :, :border_width] = border_val
            result[:, :, -border_width:] = border_val
        elif shape[1] > 1 and shape[2] > 1:  # Y-X plane
            # Top and bottom edges
            result[:, :border_width, :] = border_val
            result[:, -border_width:, :] = border_val
            # Left and right edges
            result[:, :, :border_width] = border_val
            result[:, :, -border_width:] = border_val

        return result

    def extract_subvolume(
        self,
        x_range: Tuple[float, float] | None = None,
        y_range: Tuple[float, float] | None = None,
        z_range: Tuple[float, float] | None = None,
        *,
        use_index: bool = False,
        add_to_viewer: bool = True,
        name: str = "subvolume",
        **layer_kwargs,
    ) -> dict:
        """
        Extract a 3D sub-volume within specified coordinate ranges.

        Parameters
        ----------
        x_range : tuple of (float, float) or None
            (min, max) range along X axis (H or Qx). None = full range.
        y_range : tuple of (float, float) or None
            (min, max) range along Y axis (K or Qy). None = full range.
        z_range : tuple of (float, float) or None
            (min, max) range along Z axis (L or Qz). None = full range.
        use_index : bool, optional
            If True, ranges are voxel indices. If False (default), coordinate values.
        add_to_viewer : bool, optional
            If True (default), add the sub-volume as a new layer to the viewer.
        name : str, optional
            Name for the new layer if `add_to_viewer=True`.
        **layer_kwargs
            Additional keyword arguments passed to viewer.add_image().

        Returns
        -------
        dict
            Dictionary with keys:
            - 'grid': 3D numpy array (cropped grid data)
            - 'axes': tuple of (xax, yax, zax) arrays (cropped axes)
            - 'layer': napari Image layer (if added to viewer, else None)
        """

        # Determine index ranges for each axis
        def get_indices(ax_array, range_spec):
            if range_spec is None:
                return 0, len(ax_array) - 1
            rmin, rmax = range_spec
            if use_index:
                imin, imax = int(rmin), int(rmax)
            else:
                imin = int(np.argmin(np.abs(ax_array - rmin)))
                imax = int(np.argmin(np.abs(ax_array - rmax)))
            imin = max(0, min(imin, imax))
            imax = min(len(ax_array) - 1, max(imin, imax))
            return imin, imax

        x_min, x_max = get_indices(self.xax, x_range)
        y_min, y_max = get_indices(self.yax, y_range)
        z_min, z_max = get_indices(self.zax, z_range)

        # Extract sub-grid and sub-axes
        sub_grid = self.grid[
            x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1
        ]
        sub_xax = self.xax[x_min : x_max + 1].copy()
        sub_yax = self.yax[y_min : y_max + 1].copy()
        sub_zax = self.zax[z_min : z_max + 1].copy()

        result = {
            "grid": sub_grid,
            "axes": (sub_xax, sub_yax, sub_zax),
            "layer": None,
        }

        if add_to_viewer:
            self._require_viewer()
            # Compute volume representation
            sub_volume, sub_scale, sub_translate, _ = (
                self._volume_from_grid_axes(
                    sub_grid, (sub_xax, sub_yax, sub_zax)
                )
            )
            data = (
                self._log1p_clip(sub_volume) if self.log_view else sub_volume
            )
            lo, hi = self._robust_percentiles(data, self.contrast_percentiles)

            # Merge user kwargs with defaults
            kwargs = {
                "colormap": self.cmap,
                "opacity": 0.9,
                "blending": "translucent",
                "contrast_limits": (float(lo), float(hi)),
            }
            kwargs.update(layer_kwargs)

            layer = self.viewer.add_image(
                data,
                name=name,
                scale=sub_scale,
                translate=sub_translate,
                **kwargs,
            )
            self._force_volume(layer)
            result["layer"] = layer

            x_label = "H" if self.space == "hkl" else "Qx"
            y_label = "K" if self.space == "hkl" else "Qy"
            z_label = "L" if self.space == "hkl" else "Qz"
            logger.info(
                "Extracted subvolume: %s=[%.3f, %.3f], %s=[%.3f, %.3f], %s=[%.3f, %.3f]",
                x_label,
                sub_xax[0],
                sub_xax[-1],
                y_label,
                sub_yax[0],
                sub_yax[-1],
                z_label,
                sub_zax[0],
                sub_zax[-1],
            )

        return result

    # ------------------------------ overlays ----------------------------------
    def _add_outline_and_corners(self, v: napari.Viewer) -> None:
        zmin, zmax = float(self.zax[0]), float(self.zax[-1])
        ymin, ymax = float(self.yax[0]), float(self.yax[-1])
        xmin, xmax = float(self.xax[0]), float(self.xax[-1])
        corners_world = np.array(
            [
                [zmin, ymin, xmin],
                [zmin, ymin, xmax],
                [zmin, ymax, xmin],
                [zmin, ymax, xmax],
                [zmax, ymin, xmin],
                [zmax, ymin, xmax],
                [zmax, ymax, xmin],
                [zmax, ymax, xmax],
            ],
            dtype=float,
        )

        mean_vox = float(np.mean(self.scale))
        corner_size = mean_vox * 0.15

        # Outline box (add first for final order control)
        box_edges = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [1, 3],
                [1, 5],
                [2, 3],
                [2, 6],
                [3, 7],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
            ],
            dtype=int,
        )
        edge_segments = [corners_world[e] for e in box_edges]
        v.add_shapes(
            edge_segments,
            shape_type="line",
            edge_color="yellow",
            edge_width=mean_vox * 0.05,
            opacity=0.9,
            blending="additive",
            name="Outline box",
        )

        v.add_points(
            corners_world,
            name="Outline corners",
            size=np.full(8, corner_size),
            face_color="red",
            opacity=0.9,
            blending="additive",
        )

        labels = []
        if self.space == "q":
            for z, y, x in corners_world:
                labels.append(f"Qx={x:.3f}, Qy={y:.3f}, Qz={z:.3f}")
        else:
            for z, y, x in corners_world:
                labels.append(f"H={x:.3f}, K={y:.3f}, L={z:.3f}")

        try:
            v.add_points(
                corners_world,
                name="Corner Labels",
                text={"string": labels, "size": 10},
                face_color="white",
                size=0.0,
                blending="additive",
            )
        except (TypeError, ValueError, AttributeError):
            v.add_points(
                corners_world,
                name="Corner Labels",
                text=labels,
                face_color="white",
                size=0.0,
                blending="additive",
            )

    def _add_axes_vectors(self, v: napari.Viewer) -> None:
        with contextlib.suppress(Exception):
            Lx = float(self.xax[-1] - self.xax[0])
            Ly = float(self.yax[-1] - self.yax[0])
            Lz = float(self.zax[-1] - self.zax[0])
            max_extent = max(Lx, Ly, Lz) or 1.0
            length = 0.10 * max_extent
            origin = np.array(
                [self.zax[0], self.yax[0], self.xax[0]], dtype=float
            )
            vectors = np.stack(
                [
                    np.vstack(
                        [origin, origin + np.array([length, 0, 0])]
                    ),  # +Z
                    np.vstack(
                        [origin, origin + np.array([0, length, 0])]
                    ),  # +Y
                    np.vstack(
                        [origin, origin + np.array([0, 0, length])]
                    ),  # +X
                ],
                axis=0,
            )
            width = float(np.mean(self.scale)) * 0.05
            kwargs = {
                "name": "World axes",
                "edge_color": ["cyan", "lime", "magenta"],
                "edge_width": width,
                "blending": "translucent_no_depth",
            }
            try:
                v.add_vectors(vectors, **kwargs)
            except TypeError:
                kwargs.pop("edge_width", None)
                kwargs["width"] = max(width, 1.0)
                v.add_vectors(vectors, **kwargs)

    def _install_hud(self, v: napari.Viewer, layer) -> None:
        def on_mouse_move(viewer, event):
            if not self._hud_enabled:
                return
            pos_world = viewer.cursor.position
            if pos_world is None:
                return
            try:
                zi, yi, xi = layer.world_to_data(pos_world)
            except (AttributeError, TypeError, ValueError):
                return
            intensity_val = np.nan
            zi_i, yi_i, xi_i = (int(np.round(q)) for q in (zi, yi, xi))
            if (
                0 <= zi_i < self.volume.shape[0]
                and 0 <= yi_i < self.volume.shape[1]
                and 0 <= xi_i < self.volume.shape[2]
            ):
                intensity_val = float(self.volume[zi_i, yi_i, xi_i])
            H = self._index_to_axis_value(self.xax, xi)
            K = self._index_to_axis_value(self.yax, yi)
            L = self._index_to_axis_value(self.zax, zi)
            if self.space == "q":
                text = f"Qx={H:.4f} Å⁻¹   Qy={K:.4f} Å⁻¹   Qz={L:.4f} Å⁻¹   I={intensity_val:.3g}"
            else:
                text = f"H={H:.4f}   K={K:.4f}   L={L:.4f}   I={intensity_val:.3g}"
            overlay = getattr(viewer, "text_overlay", None)
            if overlay is not None:
                overlay.visible = True
                overlay.position = "top_left"
                overlay.color = "white"
                overlay.font_size = 12
                overlay.text = text

        v.mouse_move_callbacks.append(on_mouse_move)

        @v.bind_key("C", overwrite=True)
        def _toggle_coords(viewer):
            self._hud_enabled = not self._hud_enabled
            overlay = getattr(viewer, "text_overlay", None)
            if overlay is not None:
                overlay.visible = self._hud_enabled

    # ------------------------------ internals ---------------------------------
    def _apply_display_order(self, v, desired_order):
        """
        Previous ordering strategy: iterate desired_order in given order and move each
        existing layer to the end so final list order matches desired_order exactly.
        """
        for name in desired_order:
            try:
                current_idx = [ly.name for ly in v.layers].index(name)
                target_idx = len(v.layers) - 1
                if current_idx != target_idx:
                    v.layers.move(current_idx, target_idx)
            except ValueError:
                continue  # layer not present; ignore

    def _force_volume(self, layer) -> None:
        if self.viewer is not None:
            self.viewer.dims.ndisplay = 3
        with contextlib.suppress(Exception):
            if hasattr(layer, "depiction"):
                layer.depiction = "volume"
        with contextlib.suppress(Exception):
            if hasattr(layer, "rendering"):
                for r in [
                    self.rendering,
                    "attenuated_mip",
                    "mip",
                    "translucent",
                ]:
                    try:
                        layer.rendering = r
                        break
                    except (AttributeError, TypeError):
                        continue

    @staticmethod
    def _validate_grid_axes(grid: np.ndarray, axes) -> None:
        if not isinstance(grid, np.ndarray) or grid.ndim != 3:
            raise ValueError("grid must be a 3D numpy array (nx, ny, nz).")
        xax, yax, zax = axes
        xax = np.asarray(xax)
        yax = np.asarray(yax)
        zax = np.asarray(zax)
        nx, ny, nz = grid.shape
        if any(a.ndim != 1 for a in (xax, yax, zax)):
            raise ValueError("All axes must be 1D.")
        if len(xax) != nx or len(yax) != ny or len(zax) != nz:
            raise ValueError("Axis lengths must match grid shape.")

    @staticmethod
    def _ensure_ascending(grid: np.ndarray, axes):
        xax, yax, zax = [np.asarray(a) for a in axes]
        G = grid
        if xax.size > 1 and xax[1] < xax[0]:
            xax = xax[::-1]
            G = G[::-1, :, :]
        if yax.size > 1 and yax[1] < yax[0]:
            yax = yax[::-1]
            G = G[:, ::-1, :]
        if zax.size > 1 and zax[1] < zax[0]:
            zax = zax[::-1]
            G = G[:, :, ::-1]
        return G, (xax, yax, zax)

    @staticmethod
    def _volume_from_grid_axes(grid, axes):
        xax, yax, zax = axes
        vol = np.ascontiguousarray(grid.transpose(2, 1, 0))

        def avg_step(a):
            return float(np.diff(a).mean()) if a.size > 1 else 1.0

        dx, dy, dz = avg_step(xax), avg_step(yax), avg_step(zax)
        translate = (float(zax[0]), float(yax[0]), float(xax[0]))
        scale = (dz, dy, dx)
        is_uniform = (
            (zax.size < 2 or np.allclose(np.diff(zax), dz))
            and (yax.size < 2 or np.allclose(np.diff(yax), dy))
            and (xax.size < 2 or np.allclose(np.diff(xax), dx))
        )
        return vol, scale, translate, is_uniform

    @staticmethod
    def _robust_percentiles(
        a: np.ndarray, prc: Tuple[float, float]
    ) -> Tuple[float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (0.0, 1.0)
        lo, hi = np.percentile(a, prc)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(a)), float(np.max(a))
            if hi <= lo:
                hi = lo + 1.0
        return float(lo), float(hi)

    @staticmethod
    def _log1p_clip(a: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(np.asarray(a), 0.0))

    @staticmethod
    def _index_to_axis_value(ax: np.ndarray, idx: float) -> float:
        n = ax.size
        if n == 0:
            return np.nan
        if idx <= 0:
            return float(ax[0])
        if idx >= n - 1:
            return float(ax[-1])
        i0 = int(np.floor(idx))
        t = float(idx - i0)
        return float((1 - t) * ax[i0] + t * ax[i0 + 1])

    def _require_viewer(self) -> None:
        if self.viewer is None:
            raise RuntimeError("Call launch() first.")

    def _auto_zoom(self, v: napari.Viewer, layer) -> None:
        """
        Robust auto-zoom to fit the full data extent, restoring earlier behavior.
        Attempts (in order):
          1. reset_view()
          2. camera.set_range(...)
          3. set camera.center manually
          4. set an oblique viewing angle + reasonable zoom
        All steps are guarded; silently continues if not supported.
        """
        # Reset to a known baseline
        with contextlib.suppress(Exception):
            v.reset_view()

        cam = getattr(v, "camera", None)
        if cam is None:
            return

        # Get data extent (Z,Y,X)
        try:
            (z0, z1), (y0, y1), (x0, x1) = layer.extent.data
        except (AttributeError, TypeError, ValueError, IndexError):
            return

        # Try the direct napari convenience if available
        with contextlib.suppress(Exception):
            cam.set_range(range_z=(z0, z1), range_y=(y0, y1), range_x=(x0, x1))

        # Center camera explicitly (older napari sometimes needs this)
        with contextlib.suppress(Exception):
            cam.center = ((z0 + z1) / 2.0, (y0 + y1) / 2.0, (x0 + x1) / 2.0)

        # Provide a stable oblique angle
        with contextlib.suppress(Exception):
            cam.angles = (30, 30, 0)

        # Heuristic zoom: inversely related to max span (kept mild to avoid over-zoom)
        with contextlib.suppress(Exception):
            spans = np.array([z1 - z0, y1 - y0, x1 - x0], dtype=float)
            span_max = float(np.max(spans)) or 1.0
            # Normalize zoom to typical scale; clamp to sensible bounds
            target_zoom = 1.0
            if span_max > 0:
                target_zoom = min(
                    1.5, max(0.2, 1.0 * (100.0 / (50.0 + span_max)))
                )
            cam.zoom = target_zoom


# ------------------------------ Utilities -------------------------------------
def _maybe_series_to_list(obj):
    if (
        hasattr(obj, "to_numpy")
        and hasattr(obj, "values")
        and hasattr(obj, "iloc")
    ):
        try:
            return list(obj.to_numpy())
        except (AttributeError, TypeError, ValueError):
            try:
                return list(obj.values)
            except (AttributeError, TypeError, ValueError):
                return list(obj)
    return obj


def _stack_list_of_2d(
    frames: Sequence[np.ndarray],
    pad_value: float = np.nan,
    dtype=np.float32,
) -> np.ndarray:
    frames = [np.asarray(f) for f in frames if f is not None]
    if not frames:
        raise ValueError("Empty intensity list.")
    H = max(f.shape[0] for f in frames)
    W = max(f.shape[1] for f in frames)
    if all(f.shape == (H, W) for f in frames):
        return np.stack([f.astype(dtype, copy=False) for f in frames], 0)
    out = np.full((len(frames), H, W), pad_value, dtype=dtype)
    for i, f in enumerate(frames):
        h, w = f.shape
        out[i, :h, :w] = f
    return out


def _to_tyx_any(
    intensity: Union[np.ndarray, Sequence[np.ndarray]],
) -> np.ndarray:
    intensity = _maybe_series_to_list(intensity)
    if isinstance(intensity, (list, tuple)):
        return _stack_list_of_2d(intensity)
    a = np.asarray(intensity)
    if a.ndim == 3 and a.dtype != object:
        return a
    if a.ndim == 2 and a.dtype != object:
        return a[None, ...]
    if a.ndim == 1 and a.dtype == object:
        return _stack_list_of_2d(list(a))
    raise ValueError("Unsupported intensity input.")


class IntensityNapariViewer:
    """
    Simple intensity viewer with adjustable rectangular ROI.
    Layers (top→bottom after reorder): Intensity, ROI
    """

    def __init__(
        self,
        intensity,
        *,
        name: str = "Intensity",
        log_view: bool = True,
        contrast_percentiles=(1.0, 99.8),
        cmap: str = "inferno",
        rendering: str = "attenuated_mip",  # kept for API compatibility
        add_timeseries: bool = True,  # backward compat (unused)
        add_volume: bool = False,  # backward compat (unused)
        scale_tzyx=(1.0, 1.0, 1.0),
        pad_value: float = np.nan,
    ):
        self._name = name
        self._log = bool(log_view)
        self._p_lo, self._p_hi = map(float, contrast_percentiles)
        self._cmap = cmap
        self._rendering = rendering
        self._scale = tuple(map(float, scale_tzyx))
        self._raw_tyx = _to_tyx_any(intensity).astype(np.float32, copy=False)
        self._viewer: napari.Viewer | None = None
        self._layer_ts = None
        self._pad_value = float(pad_value)
        self._add_timeseries = bool(add_timeseries)
        self._add_volume = bool(add_volume)
        self._roi_layer = None

    def launch(self, viewer: napari.Viewer | None = None) -> napari.Viewer:
        v = viewer or napari.Viewer(title=self._name)
        self._viewer = v

        # If reusing viewer, clear layers so intensity replaces existing content
        if viewer is not None:
            with contextlib.suppress(Exception):
                for _ly in list(v.layers):
                    with contextlib.suppress(Exception):
                        v.layers.remove(_ly)

        data = self._prepare(self._raw_tyx)
        finite = data[np.isfinite(data)]
        lo, hi = (
            np.percentile(finite, [self._p_lo, self._p_hi])
            if finite.size
            else (0.0, 1.0)
        )
        if lo == hi:
            hi = lo + 1e-6

        self._layer_ts = v.add_image(
            data,
            name="Intensity",
            contrast_limits=(float(lo), float(hi)),
            colormap=self._cmap,
            blending="translucent",
            scale=self._scale,
        )
        v.dims.ndisplay = 2

        # ROI rectangle
        _, H, W = data.shape
        rect = np.array(
            [
                [H / 4, W / 4],
                [H / 4, 3 * W / 4],
                [3 * H / 4, 3 * W / 4],
                [3 * H / 4, W / 4],
            ],
            dtype=float,
        )

        # ROI Shapes layer (rectangle only)
        self._roi_layer = v.add_shapes(
            [rect],
            shape_type="rectangle",
            edge_color="red",
            face_color="transparent",
            name="ROI",
        )
        self._roi_layer.editable = True
        # Select the ROI shape to enable transformation
        self._roi_layer.selected_data = {0}
        with contextlib.suppress(Exception):
            self._roi_layer.mode = "select"

        logger.debug(
            "ROI layer initialized with data: %s", self._roi_layer.data
        )

        return v

    def _prepare(self, a: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(a, 0.0)) if self._log else a

    def get_roi_bounds(self) -> tuple[int, int, int, int] | None:
        """Get current ROI bounds as (y_min, y_max, x_min, x_max).

        Returns None if ROI layer doesn't exist or has no data.
        """
        if self._roi_layer is None or not self._roi_layer.data:
            return None
        try:
            corners = np.round(self._roi_layer.data[0]).astype(int)
            y_coords = corners[:, 0]
            x_coords = corners[:, 1]
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            return (y_min, y_max, x_min, x_max)
        except Exception as e:
            logger.exception("Failed to get ROI bounds: %s", e)
            return None

    def apply_crop(
        self, y_min: int, y_max: int, x_min: int, x_max: int
    ) -> None:
        """Crop the intensity display to the specified region and update ROI position.

        Args:
            y_min: Minimum y coordinate (row)
            y_max: Maximum y coordinate (row, exclusive)
            x_min: Minimum x coordinate (column)
            x_max: Maximum x coordinate (column, exclusive)
        """
        if self._layer_ts is None or self._viewer is None:
            logger.warning(
                "apply_crop: No intensity layer or viewer available"
            )
            return

        try:
            # Get the original raw data
            T, H, W = self._raw_tyx.shape

            # Validate bounds
            y_min = max(0, min(y_min, H))
            y_max = max(0, min(y_max, H))
            x_min = max(0, min(x_min, W))
            x_max = max(0, min(x_max, W))

            if y_max <= y_min or x_max <= x_min:
                logger.warning(
                    "Invalid crop bounds: y=(%s, %s), x=(%s, %s)",
                    y_min,
                    y_max,
                    x_min,
                    x_max,
                )
                return

            # Crop the raw data
            cropped_data = self._raw_tyx[:, y_min:y_max, x_min:x_max]

            # Update internal raw data to cropped version
            self._raw_tyx = cropped_data

            # Prepare (log transform if needed) and update layer
            prepared_data = self._prepare(cropped_data)
            self._layer_ts.data = prepared_data

            # Update contrast limits for the cropped data
            finite = prepared_data[np.isfinite(prepared_data)]
            if finite.size > 0:
                lo, hi = np.percentile(finite, [self._p_lo, self._p_hi])
                if lo == hi:
                    hi = lo + 1e-6
                self._layer_ts.contrast_limits = (float(lo), float(hi))

            # Update ROI layer position to match new coordinate system
            if self._roi_layer is not None and len(self._roi_layer.data) > 0:
                try:
                    # Get current ROI coordinates (in original frame)
                    old_roi = self._roi_layer.data[0].copy()

                    # Shift ROI coordinates by crop offset
                    new_roi = old_roi - np.array([y_min, x_min])

                    # Get new dimensions
                    new_H, new_W = cropped_data.shape[1], cropped_data.shape[2]

                    # Clamp ROI to new bounds
                    new_roi[:, 0] = np.clip(new_roi[:, 0], 0, new_H)
                    new_roi[:, 1] = np.clip(new_roi[:, 1], 0, new_W)

                    # Update ROI data
                    self._roi_layer.data = [new_roi]
                    self._roi_layer.refresh()

                    logger.debug(
                        "Updated ROI position: shifted by (%d, %d)",
                        -y_min,
                        -x_min,
                    )
                except (
                    AttributeError,
                    TypeError,
                    IndexError,
                    ValueError,
                ) as roi_err:
                    logger.warning(
                        "Failed to update ROI position: %s", roi_err
                    )

            # Reset viewer to fit the new cropped data
            with contextlib.suppress(Exception):
                self._viewer.reset_view()

            logger.debug(
                "Applied crop: y=(%s, %s), x=(%s, %s)",
                y_min,
                y_max,
                x_min,
                x_max,
            )

        except Exception as e:
            logger.exception("Failed to apply crop: %s", e)

    def close(self):
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.close()
            self._viewer = None
