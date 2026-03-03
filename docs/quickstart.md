# Quick Start

Get up and running with CobraBox in 5 minutes.

## 1. Create Data

Start with a numpy array and wrap it in a `Data` container:

```python
import cobrabox as cb
import numpy as np

# Create synthetic data: 100 timepoints, 4 channels
my_array = np.random.default_rng(seed=0).normal(size=(100, 4))

# Wrap in Data container
data = cb.from_numpy(
    arr=my_array,
    dims=["time", "space"],
    sampling_rate=100.0,  # Hz
    subjectID="sub-01",
    condition="baseline"
)

print(data)
```

## 2. Apply Features

Use built-in features or create your own:

```python
# Apply line length feature
line_len = cb.feature.line_length(data)

print(f"Shape: {line_len.data.shape}")
print(f"History: {line_len.history}")
```

## 3. Chain Operations

Features return new `Data` objects, so you can chain them:

```python
# Sliding window → compute min/max per window → line length
wdata = cb.feature.sliding_window(data, window_size=10, step_size=5)
win_min = cb.feature.min(wdata, dim="window_index")
win_max = cb.feature.max(wdata, dim="window_index")
```

## 4. Access Data

Extract data in different formats:

```python
# As numpy array
arr = data.to_numpy()

# As pandas DataFrame
df = data.to_pandas()

# Access underlying xarray.DataArray
xr_data = data.data
```

## 5. Load Built-in Datasets

Try the dummy datasets:

```python
# Load a dataset
datasets = cb.dataset("dummy_chain")
data = datasets[0]

# Apply your pipeline
result = cb.feature.line_length(data)
```

## What's Next?

- [Core Concepts](guide/core-concepts.md) - Understand immutability and metadata
- [Data Containers](guide/data-containers.md) - Deep dive into the Data class
- [Feature Guide](guide/features.md) - Learn all available features
