# SpaceExplorer Rewrite - Summary

**Date:** 2026-03-25  
**Status:** ✓ Complete and Tested

## Overview

Successfully rewrote `src/cobrabox/visualization/space_explorer.py` to match its description and new requirements. The visualization now provides flexible, mode-based exploration with a single plot that switches between time_series and heatmap views.

## Key Changes

### 1. Removed Plot Linking
- **Removed:** `_XRangeLinker` class (no longer needed for single plot)
- **Reason:** Plot linking is only useful when multiple plots need to stay synchronized

### 2. Removed Point Positions / Event Markers
- **Removed:** `point_positions` parameter from ExplorerState
- **Removed:** `event_index` parameter from ExplorerState
- **Removed:** Related methods: `_update_event_bounds()`, `_sync_event_position()`
- **Removed:** Event navigation UI from ControlsPanel
- **Reason:** SpaceExplorer is designed for general spatial exploration, not seizure-specific analysis

### 3. Simplified to Single Plot
- **Removed:** `ChannelPlot`, `HeatmapPanel`, `AveragePlot` classes
- **Removed:** `zoom_window` parameter
- **Created:** New `DataPlot` class that renders different visualizations based on `viz_mode`
- **Result:** Same reusable plot component for both modes

### 4. Added Visualization Mode Selector
- **Added:** `viz_mode` parameter to ExplorerState with options: `["time_series", "heatmap"]`
- **Behavior:** 
  - **"time_series" mode:** Shows full-length offset traces with position marker
    - Controls: Position slider, Window size slider
    - Supports click-to-navigate on the plot
  - **"heatmap" mode:** Shows windowed 2D heatmap of trace_dim × sweep_dim
    - Controls: Colormap selector, Color range (clim) slider
    - Automatically recomputes when window/position changes

### 5. Updated ControlsPanel
- **Always shows:** Visualization mode selector, Dimension selectors, Metadata
- **Conditionally shows (based on viz_mode):**
  - **time_series:** Position and window_size sliders
  - **heatmap:** Colormap and clim controls
- **Implementation:** `_update_panel_layout()` method dynamically rebuilds layout on mode change

### 6. Simplified Top-Level Class
- **Renamed:** `SeizureExplorer` → `SpaceExplorer` (matches file purpose)
- **Removed:** `point_positions` parameter from `__init__`
- **Removed:** Linker initialization and management
- **Result:** Cleaner, more straightforward initialization

## Component Structure

```
ExplorerState
├── Parameters: sweep_dim, trace_dim, position, window_size, viz_mode, cmap, clim_*
├── Properties: data, xr
└── Helpers: get_window_slice(), sweep_coords(), windowed_2d(), etc.

DataPlot
├── _plot() → dispatches to _plot_time_series() or _plot_heatmap()
├── _plot_time_series() → HoloViews NdOverlay with offset traces
├── _plot_heatmap() → HoloViews Image with colorbar
└── _make_ts_hook() → Bokeh tap handler for navigation

ControlsPanel
├── Mode selector (always visible)
├── Dimension selectors (always visible)
├── Mode-specific controls (conditionally shown)
└── Metadata display (always visible)

SpaceExplorer
├── __panel__() → Returns Panel layout for Jupyter
└── create_app() → Returns FastListTemplate for served app
```

## Testing

All tests pass successfully:

### Functionality Tests ✓
- ExplorerState creation and parameter validation
- Visualization mode switching
- Window slice calculation
- Dimension selection
- Component creation

### Rendering Tests ✓
- Time series plot rendering (returns NdOverlay)
- Heatmap plot rendering (returns Image)
- Mode switching produces different plot types
- Parameter responsiveness
- Controller widgets creation
- SpaceExplorer layout assembly
- create_app() returns FastListTemplate

### Test Coverage
- Basic instantiation: ✓
- State properties: ✓
- Mode switching: ✓
- Plot rendering: ✓
- Parameter changes: ✓
- Jupyter integration: ✓
- Panel served integration: ✓

## Usage

### Jupyter Notebook
```python
import cobrabox as cb
from cobrabox.visualization import SpaceExplorer

data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
explorer = SpaceExplorer(data)
explorer  # Displays interactively
```

### Panel Served App
```bash
panel serve src/cobrabox/visualization/space_explorer.py --show --dev
```

## Files Modified
- `src/cobrabox/visualization/space_explorer.py` - Complete rewrite

## Files Created (for testing)
- `test_space_explorer.py` - Functionality tests
- `test_space_explorer_rendering.py` - Rendering tests

## Future Enhancements
The `viz_mode` parameter can be easily extended with more options:
- `"spectrogram"` - Frequency-domain visualization
- `"3d"` - 3D spatial visualization
- `"animated"` - Time-animated slice through data
- `"statistics"` - Statistical summary plots

The architecture supports adding new modes by:
1. Adding the mode to `_VIZ_MODES`
2. Creating a `_plot_<mode_name>()` method in `DataPlot`
3. Adding mode-specific controls to `ControlsPanel._update_panel_layout()`
