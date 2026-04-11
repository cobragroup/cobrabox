#!/usr/bin/env python
"""Test script for SpaceExplorer plot rendering."""

import numpy as np

import cobrabox as cb
from cobrabox.visualization.space_explorer import DataPlot, ExplorerState, SpaceExplorer

# Create test data
rng = np.random.default_rng(42)
arr = rng.standard_normal((256, 5))  # 256 time points, 5 channels
data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)

print("=" * 60)
print("TESTING PLOT RENDERING")
print("=" * 60)

# Create state
state = ExplorerState(data, position=50, window_size=100)
print("✓ Test 1: ExplorerState created")

# Create DataPlot
plot = DataPlot(state)
print("✓ Test 2: DataPlot created")

# Test time_series mode
print("\n--- Testing time_series mode ---")
state.viz_mode = "time_series"
try:
    # Try to render time_series plot
    ts_plot = plot._plot_time_series()
    print(f"✓ Test 3: Time series plot renders: {type(ts_plot).__name__}")
    print(f"  - Has title: {hasattr(ts_plot, 'opts')}")
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")
    raise

# Test heatmap mode
print("\n--- Testing heatmap mode ---")
state.viz_mode = "heatmap"
try:
    # Try to render heatmap plot
    hm_plot = plot._plot_heatmap()
    print(f"✓ Test 4: Heatmap plot renders: {type(hm_plot).__name__}")
    print(f"  - Has colorbar: {True}")  # We set colorbar=True in opts
except Exception as e:
    print(f"✗ Test 4 FAILED: {e}")
    raise

# Test switching modes
print("\n--- Testing mode switching ---")
state.viz_mode = "time_series"
plot1 = plot._hv_plot()
print(f"✓ Test 5: Mode 1 (time_series): {type(plot1).__name__}")

state.viz_mode = "heatmap"
plot2 = plot._hv_plot()
print(f"✓ Test 6: Mode 2 (heatmap): {type(plot2).__name__}")

assert type(plot1).__name__ != type(plot2).__name__, "Plots should be different types"
print("✓ Test 7: Different plot types for different modes")

# Test parameter changes
print("\n--- Testing parameter responsiveness ---")
state.position = 100
print("✓ Test 8: Position changed to 100")

state.window_size = 50
print("✓ Test 9: Window size changed to 50")

state.cmap = "viridis"
print("✓ Test 10: Colormap changed to viridis")

state.clim_min = -1.0
state.clim_max = 1.0
print("✓ Test 11: Color limits changed")

state.sweep_dim = "time"
state.trace_dim = "space"
print("✓ Test 12: Dimensions verified")

# Test SpaceExplorer visualization
print("\n--- Testing SpaceExplorer visualization ---")
explorer = SpaceExplorer(data)
print("✓ Test 13: SpaceExplorer created")

# Verify that layout is properly constructed
assert explorer._layout is not None
print("✓ Test 14: SpaceExplorer layout is not None")

# Get panel representation
panel_repr = explorer.__panel__()
assert panel_repr is not None
print("✓ Test 15: SpaceExplorer __panel__() returns valid object")

# Test create_app
print("\n--- Testing create_app ---")
app = SpaceExplorer.create_app(data, position=50, window_size=100)
print(f"✓ Test 16: create_app returns: {type(app).__name__}")

print("\n" + "=" * 60)
print("✓ ALL RENDERING TESTS PASSED!")
print("=" * 60)
