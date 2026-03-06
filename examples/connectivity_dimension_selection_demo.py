"""
Interactive Connectivity Explorer with Dynamic Dimension Selection

This script demonstrates the InteractiveConnectivityExplorer with dynamic
matrix dimension selection via buttons.

Features:
- Interactive Matrix Dimension Selection: Click "Prev Dims" and "Next Dims"
  buttons to cycle through all possible 2D projections
- Automatic Dimension Recalculation: Extra dimensions (for slicing)
  automatically update when you change the matrix dimensions
- Slice Navigation: For each dimension pair, navigate slices of remaining
  dimensions
- Visual Feedback: Display shows current matrix pair progress
  (e.g., "Matrix: source x target (1/6)")

Workflow:
1. Create 4D connectivity data (e.g., source x target x band x region)
2. Initialize InteractiveConnectivityExplorer
3. Click dimension buttons to select which 2 dimensions form the matrix
4. Use slice buttons to navigate through remaining dimensions
5. Observe how extra_dims automatically recalculate
"""

import matplotlib

matplotlib.use("TkAgg")  # Use interactive backend
import numpy as np
import xarray as xr

import cobrabox as cb
from cobrabox.visualization import InteractiveConnectivityExplorer

# =============================================================================
# Step 1: Create 4D Connectivity Data
# =============================================================================
# Create synthetic connectivity data with:
# - source (6 nodes): Source brain regions
# - target (6 nodes): Target brain regions
# - band (3 frequency bands): alpha, beta, gamma
# - region (2 regions): left, right hemisphere

rng = np.random.default_rng(42)

# Create connectivity matrix: 6x6 sources/targets x 3 bands x 2 regions
connectivity_data = rng.standard_normal((6, 6, 3, 2))

# Create xarray DataArray with labeled dimensions and coordinates
xr_connectivity = xr.DataArray(
    connectivity_data,
    dims=["source", "target", "band", "region"],
    coords={
        "source": [f"S{i}" for i in range(6)],
        "target": [f"T{i}" for i in range(6)],
        "band": ["alpha", "beta", "gamma"],
        "region": ["left", "right"],
    },
)

print("✓ Created 4D connectivity data:")
print(f"  Shape: {xr_connectivity.shape}")
print(f"  Dimensions: {list(xr_connectivity.dims)}")
print(f"  Coordinates: {list(xr_connectivity.coords)}")

# Wrap in cobrabox Data container
data = cb.Data(xr_connectivity)
print(f"✓ Wrapped in cobrabox Data: {type(data)}")


# =============================================================================
# Step 2: Examine Possible Dimension Pairs
# =============================================================================
# With 4 non-time dimensions (source, target, band, region), there are
# multiple ways to form a 2D matrix:
# - Which two dimensions form the heatmap? (matrix_dims)
# - Which dimensions can be sliced? (extra_dims)

# Create explorer to inspect dimension pairs
explorer = InteractiveConnectivityExplorer(data=data)

print(f"\nTotal possible 2D projections: {len(explorer.dimension_pairs)}\n")
for i, pair in enumerate(explorer.dimension_pairs, 1):
    extra = [d for d in explorer.non_time_dims if d not in pair]
    print(f"{i}. Matrix: {pair[0]} x {pair[1]:<8} | Sliceable: {extra}")


# =============================================================================
# Step 3: Launch Interactive Connectivity Explorer
# =============================================================================
# Click the buttons to explore:
# - "< Prev Dims" and "Next Dims >": Cycle through all 6 dimension pairs
# - "< Prev Slice" and "Next Slice >": Navigate slices of extra dimensions
# - Display text: Shows current matrix pair (e.g., "Matrix: source x target (1/6)")

print("\nLaunching interactive explorer...")
explorer.vis()


# =============================================================================
# Step 4: Inspect Current State
# =============================================================================
# After clicking buttons in the visualization above, you can inspect the
# explorer's current state

print("\nCurrent InteractiveConnectivityExplorer State:")
print("=" * 60)
current_num = explorer.current_pair_idx + 1
total_pairs = len(explorer.dimension_pairs)
print(f"Current dimension pair index: {current_num}/{total_pairs}")
print(f"Matrix dimensions (heatmap): {explorer.matrix_dims}")
print(f"Extra dimensions (sliceable): {explorer.extra_dims}")
print(f"Current slice indices: {explorer.slice_indices}")
print()

# Extract data at current slice
current_matrix = explorer.data.data
for dim in explorer.extra_dims:
    idx = explorer.slice_indices[dim]
    current_matrix = current_matrix.isel({dim: idx})

print(f"Current matrix shape: {current_matrix.shape}")
print(f"Data range: [{current_matrix.min().values:.3f}, {current_matrix.max().values:.3f}]")


# =============================================================================
# How Dimension Selection Works
# =============================================================================
#
# What Happens When You Click Buttons?
#
# 1. Click "Next Dims":
#    - current_pair_idx increments (wraps around at end)
#    - _update_matrix_dims() recalculates:
#      * matrix_dims = new 2D projection
#      * extra_dims = remaining dimensions
#      * slice_indices = reset to 0 for all extra dims
#    - _update_plot() redraws heatmap
#    - _update_dims_text() updates display
#
# 2. Click "Prev/Next Slice":
#    - Navigate through slices of extra dimensions
#    - No change to matrix_dims or extra_dims
#    - Only slice_indices change
#
# Why This Matters:
# With high-dimensional connectivity data, this allows:
# - Exploring all possible 2x2 projections without code changes
# - Automatic recalculation of sliceable dimensions
# - Seamless interaction while maintaining data integrity
# - Visual feedback showing current position


# =============================================================================
# Key Attributes and Methods
# =============================================================================
#
# Attributes:
# - dimension_pairs: List of all 2-element combinations from non_time_dims
# - current_pair_idx: Index of currently displayed pair
#   (0 to len(dimension_pairs)-1)
# - matrix_dims: Current 2 dimensions forming the heatmap
# - extra_dims: Remaining dimensions available for slicing
# - slice_indices: Dictionary mapping extra_dims to current slice positions
#
# Methods:
# - _on_next_dims(): Move to next dimension pair (with wraparound)
# - _on_prev_dims(): Move to previous dimension pair (with wraparound)
# - _update_matrix_dims(): Recalculate matrix_dims and extra_dims from
#   current_pair_idx
# - _update_dims_text(): Update the display text showing current matrix
#   pair progress
# - _on_next_slice(): Move to next slice of extra dimensions
# - _on_prev_slice(): Move to previous slice of extra dimensions
# - vis(): Launch the interactive visualization with all buttons


# =============================================================================
# Example: Programmatic Navigation
# =============================================================================
# You can also navigate programmatically without clicking buttons:

print("\n" + "=" * 60)
print("Example: Programmatic Navigation")
print("=" * 60)

# Create a new explorer and navigate directly to pair 3
explorer2 = InteractiveConnectivityExplorer(data=data)
print("\nLaunching second explorer...")
explorer2.vis()

# Jump to pair 3 (index 2)
print("\nNavigating to pair 3:")
explorer2.current_pair_idx = 2
explorer2._update_matrix_dims()
explorer2._update_plot()
explorer2._update_dims_text()
if len(explorer2.extra_dims) > 0:
    explorer2._update_info_text()

print(f"Matrix: {explorer2.matrix_dims}")
print(f"Extra dims: {explorer2.extra_dims}")
print(f"Progress: {explorer2.current_pair_idx + 1}/{len(explorer2.dimension_pairs)}")
