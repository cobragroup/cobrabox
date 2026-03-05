"""Example: Connectivity matrix visualization.

This example demonstrates how to use:
1. ConnectivityPlotter - for static visualization of 2D connectivity matrices
2. InteractiveConnectivityExplorer - for interactive exploration of 3D+ connectivity data
"""

import numpy as np
import xarray as xr

import cobrabox as cb
from cobrabox.visualization import (
    ConnectivityPlotter,
    InteractiveConnectivityExplorer,
    plot_connectivity,
)

# =========================================================================
# Example 1: Static 2D Connectivity Matrix
# =========================================================================
print("Example 1: Static 2D Connectivity Matrix")
print("-" * 50)

# Create synthetic connectivity data (10x10 matrix)
rng = np.random.default_rng(42)
conn_matrix = rng.standard_normal((10, 10))

# Add some diagonal structure for realistic connectivity
conn_matrix = (conn_matrix + conn_matrix.T) / 2  # Make symmetric

# Create xarray DataArray with proper dimensions
xr_data = xr.DataArray(
    conn_matrix,
    dims=["node_from", "node_to"],
    coords={
        "node_from": [f"Node_{i}" for i in range(10)],
        "node_to": [f"Node_{i}" for i in range(10)],
    },
)

# Create Data object
data = cb.Data(xr_data, subjectID="example_subject")

# Method 1: Using ConnectivityPlotter class
plotter = ConnectivityPlotter(figsize=(8, 7))
fig, ax = plotter.plot(data, cmap="RdBu_r", title="Example: Connectivity Matrix (10x10 nodes)")
print("✓ Static 2D connectivity matrix created")

# Method 2: Using convenience function
fig2, ax2 = plot_connectivity(
    data, cmap="viridis", title="Alternative: Connectivity Matrix (viridis colormap)"
)
print("✓ Alternative visualization created\n")

# =========================================================================
# Example 2: Interactive 3D Connectivity Data (Connectivity across bands)
# =========================================================================
print("Example 2: Interactive 3D Connectivity Data")
print("-" * 50)

# Create 3D connectivity data: 10x10 connectivity for 5 frequency bands
conn_3d = rng.standard_normal((10, 10, 5))
conn_3d = (conn_3d + np.transpose(conn_3d, (1, 0, 2))) / 2  # Make symmetric

xr_data_3d = xr.DataArray(
    conn_3d,
    dims=["node_from", "node_to", "band"],
    coords={
        "node_from": [f"Node_{i}" for i in range(10)],
        "node_to": [f"Node_{i}" for i in range(10)],
        "band": ["delta", "theta", "alpha", "beta", "gamma"],
    },
)

data_3d = cb.Data(xr_data_3d, subjectID="interactive_example")

# Create interactive explorer
explorer = InteractiveConnectivityExplorer(data=data_3d)
print("✓ Interactive connectivity explorer created")
print("  - Dimensions: 10x10 connectivity x 5 frequency bands")
print("  - Use Prev/Next buttons to navigate through bands")
print("  - Call explorer.vis() to display interactive plot\n")

# =========================================================================
# Example 3: 4D Connectivity Data (Multiple extra dimensions)
# =========================================================================
print("Example 3: 4D Connectivity Data (Multiple extra dimensions)")
print("-" * 50)

# Create 4D data: 8x8 connectivity for 3 bands and 2 time windows
conn_4d = rng.standard_normal((8, 8, 3, 2))

xr_data_4d = xr.DataArray(
    conn_4d,
    dims=["source", "target", "frequency_band", "time_window"],
    coords={
        "source": [f"S{i}" for i in range(8)],
        "target": [f"T{i}" for i in range(8)],
        "frequency_band": ["low", "mid", "high"],
        "time_window": ["early", "late"],
    },
)

data_4d = cb.Data(xr_data_4d, subjectID="4d_example")

# Create interactive explorer for 4D data
explorer_4d = InteractiveConnectivityExplorer(data=data_4d, cmap="coolwarm", figsize=(10, 8))
print("✓ 4D connectivity explorer created")
print("  - Dimensions: 8x8 connectivity x 3 frequency bands x 2 time windows")
print("  - Navigate through (frequency_band, time_window) combinations")
print("  - Call explorer_4d.vis() to display interactive plot\n")

print("=" * 50)
print("All examples created successfully!")
print("=" * 50)
