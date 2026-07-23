# Voxel Web Application

A browser-based tool for reconstructing and exploring 3D reciprocal-space maps
(RSMs) from diffraction data. Built with [Trame](https://kitware.github.io/trame/)
and VTK, it reproduces the interactive functionality of the Napari ResView plugin
in a single-window web app.

## What it does

You load detector data, build a reciprocal-space map, regrid it into a uniform
volume, and inspect the result with GPU-accelerated 3D rendering and slicing —
all from a browser.

The left panel is a four-tab accordion that follows the pipeline:

- **Data** — load and prepare
  - Two loaders: **CMS** (TIFF directory + angle step, no SPEC metadata) or
    **ISR** (SPEC file + TIFF frames)
  - Server-side file browser for picking paths
  - Editable experimental setup (detector distance, pixel pitch, detector
    size, beam center, energy/wavelength — energy and wavelength stay in sync)
    populated from a YAML setup file
  - **View Intensity** to inspect raw detector frames before building
  - **ROI crop** (row/col min-max) to re-crop the loaded data
- **Build** — compute the Q/HKL mapping (**Build RSM**), then scatter it into a
  uniform 3D grid (**Regrid**, via xrayutilities)
- **View** — render the volume (**View RSM**) in Q-space (Å⁻¹) or HKL, with
  colormap, log-intensity, contrast percentiles, and rendering mode
  (composite / MIP / attenuated MIP); export to VTK `.vtr`
- **Analyze** — orthogonal X/Y/Z plane slices plus cylindrical and spherical
  surface sampling, each with adjustable position, **tilt**, opacity, and
  colormap; optional **φ=0 reference plane** marking the diffractometer origin

Each pipeline step is its own button, so any stage can be run and inspected
independently. A **Stop** button cancels a running load/build/regrid.

## Requirements

- Python 3.11+
- [Pixi](https://pixi.sh) for environment management

## Install

```bash
pixi install
```

This creates a managed environment in `.pixi/` with the correct Python version
and all packages (declared in `pixi.toml`, resolved from conda-forge).

## Run

```bash
pixi run start          # or: pixi run python -m voxel
```

The server picks an available port, prints it, and opens the browser.

Options (`pixi run python -m voxel --help`):

- `--port PORT` — bind a specific port (default: auto)
- `--host HOST` — bind a specific host (default: localhost)
- `--no-browser` — don't open the browser

```bash
pixi run python -m voxel --port 8080 --host 0.0.0.0
```

## Typical workflow

1. **Data**: pick CMS or ISR, provide paths, review/override the setup fields,
   then **Load Data**. Optionally **View Intensity** or **Crop from ROI**.
2. **Build**: **Build RSM**, then **Regrid**.
3. **View**: **View RSM**, adjust colormap / log / contrast / mode, **Refresh**,
   and export to `.vtr`.
4. **Analyze**: toggle slices and surfaces; adjust position, tilt, opacity, and
   colormap.

**3D view controls**: left drag to rotate, wheel or right drag to zoom, middle
drag to pan.

## Architecture

Launched as a package module: `python -m voxel`
([voxel/__main__.py](voxel/__main__.py) → [voxel/cli.py](voxel/cli.py) →
`run_server()` in [voxel/app/server.py](voxel/app/server.py)).

Code is split by concern so new loaders, reconstruction methods, or
visualizations can be added in isolation:

- `voxel/services` — data access (`backend.py` bridges the headless `rsm3d`
  loaders/builder; `parsing.py` coerces browser state into typed values;
  `tiled_io.py` provides a Tiled-based loader)
- `voxel/rsm3d` — the reconstruction engine (loaders, `RSMBuilder`, VTR export)
- `voxel/visualization` — VTK color/opacity transfer functions and percentile
  helpers
- `voxel/ui` — static UI assets (colormap names, icons)
- `voxel/app/server.py` — builds the Trame server, the VTK scene, and the
  per-step controllers (load, view intensity, crop, build, regrid, view,
  refresh, stop, slicing, export)

The backend is imported headlessly (a lightweight `rsm3d` namespace) so no
Napari/Qt dependency is required to run the web app.

## Troubleshooting

- **Won't start**: run `pixi install`; confirm `pixi run python --version` shows
  3.11.x; verify with
  `pixi run python -c "from voxel.app.server import create_server; print('OK')"`.
- **Load fails**: use absolute paths; ensure the YAML has an `ExperimentSetup`
  section; confirm the TIFF directory has valid `.tif`/`.tiff` files; for ISR,
  check the SPEC file is accessible.
- **Rendering issues**: ensure a VTK display is available (SSH `-X` for remote);
  keep grid size reasonable (16–256) — very large grids may time out.

## References

- [Trame](https://trame.readthedocs.io/)
- [VTK Python](https://vtk.org/doc/nightly/html/python/)
- [xrayutilities](https://sourceforge.net/projects/xrayutilities/)
