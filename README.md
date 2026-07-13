# Napari ResView Web Application

A web-based interface for exploring 3D reconstructed reciprocal-space maps (RSMs), built with [Trame](https://kitware.github.io/trame/) and VTK. Reproduces all interactive functionality from the Napari ResView plugin in a browser-accessible single-window application.

## Features

- **Interactive 3D Volume Rendering**: GPU-accelerated volume visualization with real-time rotation, zoom, and panning
- **Tabbed Accordion Control Panel**: The left panel is organized into four collapsible tabs — **Data**, **Build**, **View**, and **Analysis**
- **Step-by-Step Pipeline**: Each stage has its own button — Load Data, View Intensity, Crop from ROI, Build RSM, Regrid, View RSM, Refresh, and Stop — so each step can be run and inspected independently
- **Dual Loader Support**:
  - **CMS mode**: Load 3D diffraction data from a TIFF directory (with a configurable angle step) without SPEC metadata
  - **ISR mode**: Load a SPEC file with TIFF intensity frames for comprehensive experimental tracking
- **Editable Experimental Setup**: Detector distance, pixel pitch, detector height/width (px), beam-center height/width (px), photon energy, and wavelength are populated from the setup file and can be overridden before building (energy and wavelength stay in sync)
- **Raw Intensity Inspection**: View the loaded detector frames directly as a stacked volume before building the RSM
- **Flexible Data Space**: Render in Q-space (reciprocal Ångström) or HKL (Miller indices)
- **Adjustable Visualization Parameters**:
  - Colormap selection (viridis, plasma, coolwarm, etc.)
  - Log-intensity toggle and low/high contrast percentile clipping
  - Rendering mode (composite / maximum-intensity / attenuated MIP)
  - Shading and lighting properties
- **ROI-Based Cropping**: Define rectangular crop windows (row/col min-max) on detector images and re-crop the loaded data
- **Analysis Slicing**: Orthogonal (X/Y/Z) plane slices plus cylindrical and spherical surface sampling (Q-space) with per-slice opacity and colormap
- **3D Regridding**: Scatter-to-grid interpolation using xrayutilities for uniform 3D volumes
- **Export Capabilities**: Save reconstructed volumes as VTK XML RectilinearGrid (.vtr) format

## Requirements

- Python 3.11+ (due to `from __future__ import annotations` in Trame)
- [Pixi](https://pixi.sh) for environment management

## Installation

The environment is defined in `pixi.toml`. Install all dependencies with:

```bash
pixi install
```

This creates a managed environment in `.pixi/` with the correct Python version and all packages.

### Dependencies

- **Core**: trame, trame-vtk, vtk, numpy, scipy
- **Scientific**: xrayutilities, pandas, pyyaml
- **Data I/O**: tifffile, h5py, dask
- **Visualization**: matplotlib, scikit-image, lmfit

All packages are declared in `pixi.toml` and resolved from conda-forge.

## Running the Web App

### Launch with Default Settings

```bash
pixi run start
```

or equivalently:

```bash
pixi run python main.py
```

The server will:
1. Start on `localhost` (by default)
2. Assign an available port (printed to console)
3. Automatically open the browser

### Command-Line Options

```bash
pixi run python main.py --help
```

Options:
- `--port PORT`: Bind to a specific port (default: 0 = auto)
- `--host HOST`: Bind to a specific host address (default: localhost)
- `--no-browser`: Do not open the browser automatically

### Example Launches

```bash
# Run on port 8080, accessible from any interface
pixi run python main.py --port 8080 --host 0.0.0.0

# Run on localhost without opening browser
pixi run python main.py --no-browser
```

## Usage Workflow

### 1. Data tab — load and prepare

1. **Select Loader Mode**: CMS or ISR
2. **Provide Paths** (via the server-side file browser):
   - **TIFF Directory**: folder containing detector images (required)
   - **YAML Setup**: experiment configuration file
   - **SPEC File**: ISR mode only
   - **Angle Step**: CMS mode only
3. (Optional) Review/override the range of scans to load and **Experimental Setup** fields (distance, pitch,
   detector height/width, beam-center height/width, energy, wavelength).
4. Click **Load Data** (loads data only; no rendering yet).
5. (Optional) Click **View Intensity** to inspect the raw detector frames.
6. (Optional) Enter crop bounds (row/col min-max) and click **Crop from ROI** to
   re-crop the loaded data (a rebuild is required afterward).

### 2. Build tab — compute the RSM

1. Click **Build RSM** to compute the Q/HKL mapping.
2. Click **Regrid** to scatter the cloud into a uniform 3D volume.

### 3. View tab — render and export

1. Click **View RSM** to display the regridded volume.
2. Adjust **Log view**, **colormap**, **rendering mode**, and **low/high contrast
   percentiles**, then click **Refresh** to re-apply.
3. Click **Stop** to cancel a running load/build/regrid task.
4. Set the **Export Path** (.vtr) and use the export control to save.

### 4. Analysis tab — slicing

- Toggle **orthogonal** X/Y/Z plane slices and adjust their position, opacity,
  and colormap.
- Toggle **cylindrical** and **spherical** surface sampling (Q-space) and adjust
  radius, sample count, opacity, and colormap.

### Interacting with the 3D view

- **Rotate**: Left mouse drag
- **Zoom**: Mouse wheel or right drag
- **Pan**: Middle mouse drag

## Application Architecture

### `main.py`
Entry point that parses command-line arguments and launches the Trame server.

### `web_app.py`
Thin backward-compatibility shim that re-exports `create_server` / `run_server`
from `voxel.app.server`, so `import web_app` and `main.py` keep working.

### Package layout (`voxel/`)
The application is split by concern so new data types, reconstruction methods,
or visualization capabilities can be added in isolation:

- **`voxel/services`** — data access and state coercion:
  - `backend.py`: bridge that imports the headless `rsm3d` loaders/builder and
    resolves the defaults YAML path (`yaml_path`)
  - `parsing.py`: pure helpers that turn browser state into typed values
    (scan lists, UB matrix, goniometer axes, grid shape) plus the crop helpers
- **`voxel/rsm3d`** — the reconstruction engine (STAGE 2), unchanged
- **`voxel/visualization`** — render helpers:
  - `colormaps.py`: `vtkColorTransferFunction` / `vtkPiecewiseFunction` /
    `vtkLookupTable` builders, log-compression, and robust-percentile utilities
- **`voxel/ui`** — static UI assets:
  - `assets.py`: `COLORMAP_NAMES`, layer-toggle SVG icons, `DEFAULT_FRAME_COUNT`
- **`voxel/app`** — orchestration (STAGE 3):
  - `server.py`: `create_server()` builds the Trame server, the VTK scene, and
    the per-step controllers (`load_data`, `view_intensity`, `crop_from_roi`,
    `build_rsm`, `regrid`, `view_rsm`, `refresh_rendering`, `stop_task`,
    `update_slices`, `export_vtr`); `run_server()` starts it

### Integration with `napari_resview`
- **Loaders**: `RSMDataLoader_ISR`, `RSMDataloader_CMS` – load experimental data
- **Builder**: `RSMBuilder` – compute Q/HKL mappings and 3D regridding
- **Export**: `write_rsm_volume_to_vtr` – save results to VTK format

## Technical Details

### Python Environment
- Uses the Pixi-managed environment in `.pixi/envs/default` (Python 3.11)
- Dynamically creates the `napari_resview` package namespace to import the backend modules without a Napari dependency

### VTK Rendering
- **vtkSmartVolumeMapper**: GPU-accelerated volume rendering (composite / MIP / attenuated MIP)
- **vtkImageData**: Regular grid representation of 3D RSM volumes
- **vtkImageResliceMapper / vtkImageSlice**: Orthogonal plane slicing in the Analysis tab
- **vtkCylinderSource / vtkSphereSource + vtkProbeFilter**: Cylindrical and spherical surface sampling
- **vtkLookupTable**: Maps scalar values to colors for slice surfaces
- **Trame VtkRemoteView**: Streams off-screen rendered frames to the browser client

### Trame UI
- **DivLayout**: Simple HTML container layout
- **Accordion control panel**: Four stacked tabs (Data / Build / View / Analysis), each a header bar plus a `v_show`-gated panel; only one tab is open at a time
- **html widgets**: Lightweight interactive controls (inputs, selects, buttons, checkboxes)
- **State binding**: Two-way synchronization between UI controls and server state
- **Controllers**: Server-side handlers for each pipeline step (load, build, regrid, view, slicing, export)

## Troubleshooting

### App Won't Start
- Ensure the Pixi environment is installed: `pixi install`
- Check Python version: `pixi run python --version` should show 3.11.x
- Verify imports: `pixi run python -c "import web_app; print('OK')"`

### Data Load Fails
- Verify file paths are absolute and correct
- Check YAML file format (must contain `ExperimentSetup` section with required keys)
- Ensure TIFF directory contains valid .tif or .tiff files
- For ISR mode, verify SPEC file is accessible

### Rendering Issues
- Ensure VTK display is available (SSH with -X for remote sessions)
- Check grid size is reasonable (16–256; very large grids may timeout)
- If regridding hangs, try smaller grid size or shorter axis ranges

## Future Improvements (Deferred)

- **Performance**: Async/streamed loading for large datasets, caching strategies
- **Export Formats**: FITS, NetCDF, raw binary alongside VTR

## References

- [Trame Documentation](https://trame.readthedocs.io/)
- [VTK Python Bindings](https://vtk.org/doc/nightly/html/python/)
- [xrayutilities](https://sourceforge.net/projects/xrayutilities/)
