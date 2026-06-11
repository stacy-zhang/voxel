# Napari ResView Web Application

A web-based interface for exploring 3D reconstructed reciprocal-space maps (RSMs), built with [Trame](https://kitware.github.io/trame/) and VTK. Reproduces all interactive functionality from the Napari ResView plugin in a browser-accessible single-window application.

## Features

- **Interactive 3D Volume Rendering**: GPU-accelerated volume visualization with real-time rotation, zoom, and panning
- **Dual Loader Support**:
  - **CMS mode**: Load 3D diffraction data directly from TIFF files without metadata
  - **ISR mode**: Load SPEC metadata with TIFF intensity frames for comprehensive experimental tracking
- **Flexible Data Space**: Render in Q-space (reciprocal Ångström) or HKL (Miller indices)
- **Adjustable Visualization Parameters**:
  - Colormap selection (viridis, plasma, coolwarm, etc.)
  - Opacity transfer function scaling
  - Blend mode (composite or maximum intensity)
  - Shading and lighting properties
- **ROI-Based Cropping**: Define rectangular crop windows on detector images before processing
- **3D Regridding**: Scatter-to-grid interpolation using xrayutilities for uniform 3D volumes
- **Export Capabilities**: Save reconstructed volumes as VTK XML RectilinearGrid (.vtr) format
- **Responsive UI**: Sidebar controls + full-width 3D visualization viewport

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

### 1. Load Data

1. **Select Loader Mode**: CMS or ISR
2. **Provide Paths**:
   - **YAML Setup**: Path to experiment configuration file (required)
   - **TIFF Directory**: Folder containing detector images (required)
   - **SPEC File**: For ISR mode only
3. (Optional) **Configure Crop**: Enable and specify detector ROI bounds (row/col min-max)
4. Click **Load and Build RSM**

### 2. Inspect Volume

- **Rotate**: Left mouse drag
- **Zoom**: Mouse wheel or right drag
- **Pan**: Middle mouse drag
- **Adjust Rendering**:
  - Change colormap
  - Adjust opacity scale
  - Toggle shading
  - Switch blend mode
  - Modify grid size for finer/coarser regridding

### 3. Export Results

1. Specify **Export Path** (.vtr filename)
2. Click **Export VTR**
3. File is saved as compressed VTK XML RectilinearGrid

## Application Architecture

### `main.py`
Entry point that parses command-line arguments and launches the Trame server.

### `web_app.py`
Core application logic:
- **`create_server()`**: Initializes the Trame server with UI and VTK rendering
- **`run_server()`**: Starts the server with optional host/port configuration
- Helper functions for data processing, crop handling, and rendering

### Integration with `napari_resview`
- **Loaders**: `RSMDataLoader_ISR`, `RSMDataloader_CMS` – load experimental data
- **Builder**: `RSMBuilder` – compute Q/HKL mappings and 3D regridding
- **Export**: `write_rsm_volume_to_vtr` – save results to VTK format

## Technical Details

### Python Environment
- Uses Python 3.11 from `.venv3` (custom module shim avoids requiring Napari in the environment)
- Dynamically creates `napari_resview` package namespace to import backend modules without Napari dependency

### VTK Rendering
- **vtkSmartVolumeMapper**: GPU-accelerated composite blending
- **vtkImageData**: Regular grid representation of 3D RSM volumes
- **Trame VtkRemoteView**: Sends rendered images to the browser client

### Trame UI
- **DivLayout**: Simple HTML container layout
- **html widgets**: Lightweight interactive controls (inputs, selects, buttons)
- **State binding**: Two-way synchronization between UI controls and server state
- **Controllers**: Server-side handlers for load, export, and rendering actions

## Workflow Example

```bash
# 1. Launch the app
./.venv3/bin/python main.py

# 2. In the browser that opens:
#    - Set loader mode to "CMS"
#    - Enter paths:
#      setup_path: /data/setup.yaml
#      tiff_dir: /data/tiffs/
#    - Leave spec_path empty (CMS doesn't need it)
#    - Click "Load and Build RSM"

# 3. Wait for regridding to complete (status updates in real-time)

# 4. Interact with the 3D volume:
#    - Drag to rotate
#    - Scroll to zoom
#    - Adjust opacity/colormap as desired

# 5. Export when ready:
#    - Set export path: /output/my_rsm.vtr
#    - Click "Export VTR"
```

## Troubleshooting

### App Won't Start
- Ensure `.venv3` is activated: `source ./.venv3/bin/activate`
- Check Python version: `python --version` should show 3.11.x
- Verify imports: `./.venv3/bin/python -c "import web_app; print('OK')"`

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

- **Modularity**: Refactor into package structure with separate UI and backend modules
- **Advanced Controls**: Per-frame visualization, ROI-based slicing
- **Performance**: Async loading for large datasets, caching strategies
- **Export Formats**: FITS, NetCDF, raw binary alongside VTR

## References

- [Trame Documentation](https://trame.readthedocs.io/)
- [VTK Python Bindings](https://vtk.org/doc/nightly/html/python/)
- [xrayutilities](https://sourceforge.net/projects/xrayutilities/)
