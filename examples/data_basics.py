"""Basics: loading datasets, accessing data, working with coordinates.

Run from the project root:
    uv run python examples/data_basics.py

This example is aimed at users who are new to CobraBox and xarray.
It shows the most common operations you'll need in a Jupyter notebook.
"""

import numpy as np
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# 1. Load a built-in dataset
# ---------------------------------------------------------------------------

ds = cb.load_dataset("dummy_chain")

print("--- Dataset overview ---")
ds.describe()
print()

# Number of items
print(f"Number of items : {len(ds)}")

# Metadata across items
print(f"Subject IDs     : {[item.subjectID for item in ds]}")
print(f"Group IDs       : {[item.groupID for item in ds]}")
print(f"Conditions      : {[item.condition for item in ds]}")
print()

# ---------------------------------------------------------------------------
# 2. Access a single item
# ---------------------------------------------------------------------------

item = ds[0]
print("--- Single item ---")
print(item)
print()

# Shape and dimension names
print(f"Shape           : {item.data.shape}")
print(f"Dimensions      : {list(item.data.dims)}")
print(f"Sampling rate   : {item.sampling_rate} Hz")
print()

# ---------------------------------------------------------------------------
# 3. Coordinates — the most common question
# ---------------------------------------------------------------------------

# The underlying xarray DataArray lives at item.data
# Each dimension may have coordinates attached to it.

print("--- Coordinates ---")

# Which coordinates exist?
print(f"Available coords: {list(item.data.coords)}")

# Get the space dimension coordinate values as a Python list
if "space" in item.data.coords:
    space_coords = item.data.coords["space"].values.tolist()
    print(f"Space coords    : {space_coords}")

# Get the time coordinate values as a numpy array
if "time" in item.data.coords:
    time_coords = item.data.coords["time"].values
    print(f"Time coords     : first 5 = {time_coords[:5].tolist()}")
    print(f"Time duration   : {time_coords[-1]:.3f} s  ({len(time_coords)} samples)")

print()

# ---------------------------------------------------------------------------
# 4. Get the sizes of each dimension
# ---------------------------------------------------------------------------

print("--- Dimension sizes ---")
# item.data.sizes is a dict-like {dim_name: size}
for dim, size in item.data.sizes.items():
    print(f"  {dim}: {size}")
print()

# ---------------------------------------------------------------------------
# 5. Convert to numpy or pandas
# ---------------------------------------------------------------------------

print("--- Conversion ---")

arr_np = item.to_numpy()
print(f"numpy array     : shape={arr_np.shape}, dtype={arr_np.dtype}")

df = item.to_pandas()
print(f"pandas DataFrame:\n{df.head()}")
print()

# ---------------------------------------------------------------------------
# 6. Create your own Data from a numpy array
# ---------------------------------------------------------------------------

print("--- Create Data from numpy ---")

rng = np.random.default_rng(0)
my_arr = rng.normal(size=(200, 8))  # 200 time steps, 8 channels

my_data = cb.from_numpy(
    arr=my_arr,
    dims=["time", "space"],
    sampling_rate=100.0,
    subjectID="sub-01",
    groupID="control",
    condition="rest",
)

print(my_data)

# Named space coordinates (e.g., electrode labels)
channel_labels = [f"E{i + 1}" for i in range(8)]

xr_arr = xr.DataArray(
    my_arr,
    dims=["time", "space"],
    coords={
        "time": np.arange(200) / 100.0,  # time in seconds
        "space": channel_labels,
    },
)
my_data_labelled = cb.Data.from_xarray(xr_arr, subjectID="sub-01", sampling_rate=100.0)

print(f"\nLabelled space coords: {my_data_labelled.data.coords['space'].values.tolist()}")
print()

# ---------------------------------------------------------------------------
# 7. Select data by coordinate value
# ---------------------------------------------------------------------------

print("--- Selecting by coordinate ---")

# Select a single channel by label
single_channel = my_data_labelled.data.sel(space="E3")
print(f"Single channel (E3) shape: {single_channel.shape}")

# Select a subset of channels
subset = my_data_labelled.data.sel(space=["E1", "E2", "E5"])
print(f"Channel subset shape     : {subset.shape}")

# Select a time window (0.5 s to 1.0 s)
window = my_data_labelled.data.sel(time=slice(0.5, 1.0))
print(f"Time window shape        : {window.shape}")
print()

# ---------------------------------------------------------------------------
# 8. Filter and group a Dataset
# ---------------------------------------------------------------------------

print("--- Filter and group ---")

# Build a small dataset with mixed metadata
items = []
for i in range(6):
    arr_i = rng.normal(size=(100, 4))
    items.append(
        cb.from_numpy(
            arr=arr_i,
            dims=["time", "space"],
            sampling_rate=100.0,
            subjectID=f"S{i + 1:02d}",
            groupID="control" if i < 3 else "patient",
            condition="rest" if i % 2 == 0 else "task",
        )
    )

mixed_ds = cb.Dataset(items)

# Filter by groupID
controls = mixed_ds.filter(groupID="control")
print(f"Controls : {len(controls)} items")

# Group by condition
by_condition = mixed_ds.groupby("condition")
for cond, group in by_condition.items():
    print(f"  condition={cond!r}: {len(group)} items")
