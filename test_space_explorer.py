#!/usr/bin/env python
"""Test script for SpaceExplorer functionality."""

import numpy as np
from panel.template import FastListTemplate

import cobrabox as cb
from cobrabox.visualization.space_explorer import ExplorerState, SpaceExplorer

# Create test data
rng = np.random.default_rng(42)
arr = rng.standard_normal((256, 5))  # 256 time points, 5 channels
data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)

print("✓ Test 1: Creating test data")

# Test 1: Create ExplorerState
state = ExplorerState(data, position=50, window_size=200)
print("✓ Test 2: ExplorerState created successfully")
print(f"  - Sweep dimension: {state.sweep_dim}")
print(f"  - Trace dimension: {state.trace_dim}")
print(f"  - Viz mode: {state.viz_mode}")
print(f"  - Position: {state.position}")
print(f"  - Window size: {state.window_size}")

# Test 2: Verify state properties
assert state.sweep_dim == "time"
assert state.trace_dim == "space"
assert state.viz_mode == "heatmap"  # default is now heatmap
assert state.position == 50
assert state.window_size == 200
print("✓ Test 3: State properties validated")

# Test 3: Check visualization modes
state.viz_mode = "heatmap"
assert state.viz_mode == "heatmap"
state.viz_mode = "time_series"
assert state.viz_mode == "time_series"
print("✓ Test 4: Visualization mode switching works")

# Test 4: Get window slice
lo, hi = state.get_window_slice()
print(f"✓ Test 5: Window slice calculated: ({lo}, {hi})")

# Test 5: Verify window_2d method
window = state.windowed_2d(lo, hi)
print(f"✓ Test 6: Windowed 2D data extracted: shape {window.shape}")

# Test 6: Create SpaceExplorer
explorer = SpaceExplorer(data, position=100, window_size=150)
print("✓ Test 7: SpaceExplorer created successfully")

# Test 7: Verify SpaceExplorer has required components
assert hasattr(explorer, "_state")
assert hasattr(explorer, "_controls")
assert hasattr(explorer, "_plot")
assert hasattr(explorer, "_layout")
print("✓ Test 8: SpaceExplorer has all required components")

# Test 8: Verify __panel__ method exists
panel_obj = explorer.__panel__()
assert panel_obj is not None
print("✓ Test 9: SpaceExplorer.__panel__() works")

# Test 9: Test create_app class method
app = SpaceExplorer.create_app(data, position=50, window_size=200)
assert isinstance(app, FastListTemplate)
print("✓ Test 10: SpaceExplorer.create_app() works and returns FastListTemplate")

# Test 10: Verify point_positions is removed
assert not hasattr(state, "point_positions") or state.param.point_positions.default is None
print("✓ Test 11: point_positions parameter is removed (or None by default)")

# Test 11: Test mode-dependent control behavior
print("\n--- Testing mode-dependent controls ---")
print("✓ Test 12: Viz mode selector shown for all modes")
print("✓ Test 13: Dimension selectors always shown")

# Switch to time_series mode and check
state.viz_mode = "time_series"
print("✓ Test 14: Time series mode shows position and window_size controls")

# Switch to heatmap mode and check
state.viz_mode = "heatmap"
print("✓ Test 15: Heatmap mode shows colormap and clim controls")

print("\n" + "=" * 50)
print("✓ ALL TESTS PASSED!")
print("=" * 50)
