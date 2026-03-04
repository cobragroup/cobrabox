# Features

Features are the core building blocks of CobraBox pipelines. They are functions that operate on `Data` objects and return new `Data` objects with updated history.

## What is a Feature?

A feature is a function decorated with `@feature`:

```python
from cobrabox.function_wrapper import feature
from cobrabox import Data
import xarray as xr

@feature
def my_feature(data: Data, param: float) -> xr.DataArray:
    """Compute custom feature.
    
    Args:
        data: Input Data with 'time' and 'space' dimensions
        param: Custom parameter
        
    Returns:
        xarray DataArray with computed feature values
    """
    # Your computation here
    result = data.data.mean(dim="time")
    return result
```

The `@feature` decorator automatically:
- Repackages the return value into a new `Data` object
- Preserves metadata (subjectID, groupID, condition, etc.)
- Appends the function name to `history`
- Merges `extra` dicts

## Built-in Features

### Line Length

```python
from cobrabox import feature

# Compute line length (sum of absolute differences)
line_len = feature.line_length(data)
```

### Sliding Window

```python
# Create sliding windows
wdata = feature.sliding_window(
    data,
    window_size=10,  # samples per window
    step_size=5      # samples between windows
)

# wdata now has a 'window_index' dimension
```

### Min/Max/Mean

```python
# Compute statistics across a dimension
win_min = feature.min(wdata, dim="window_index")
win_max = feature.max(wdata, dim="window_index")
mean_val = feature.mean(wdata, dim="time")
```

## Feature Discovery

Features are auto-discovered from the `cobrabox/features/` directory. Any function decorated with `@feature` is automatically registered and accessible via `cb.feature.*`.

To see available features:

```python
import cobrabox as cb

# Access feature module
print(dir(cb.feature))
```

## Creating Custom Features

### Basic Feature

```python
# src/cobrabox/features/my_feature.py
import xarray as xr
from cobrabox import Data
from cobrabox.function_wrapper import feature

@feature
def spectral_power(data: Data, freq_band: tuple) -> xr.DataArray:
    """Compute spectral power in a frequency band.
    
    Args:
        data: Input Data
        freq_band: (low, high) frequency range in Hz
        
    Returns:
        DataArray with power values
    """
    # Your implementation
    xr_data = data.data
    
    # Validate dimensions
    if "time" not in xr_data.dims:
        raise ValueError("data must have 'time' dimension")
    
    # Compute feature
    # ... FFT, bandpass, etc.
    result = xr_data.mean(dim="time")
    
    return result
```

### Feature with Extra Metadata

```python
@feature
def custom_feature(data: Data) -> xr.DataArray:
    """Custom feature with extra metadata."""
    result = data.data.std(dim="time")
    
    # Return Data with custom extra dict
    # (will be merged with original extra)
    return result
```

### Returning Data Objects

Features can return `Data` objects directly:

```python
@feature
def complex_feature(data: Data) -> Data:
    """Feature that returns Data directly."""
    # Intermediate computation
    intermediate = data.data.mean(dim="time")
    
    # Wrap in Data with custom metadata
    return Data(
        data=intermediate,
        subjectID="custom",  # Will override if not None
        extra={"custom": "metadata"}
    )
```

When a feature returns a `Data` object, metadata is merged:
- Non-None values from returned `Data` override originals
- Histories are concatenated
- `extra` dicts are merged (returned values override)

## Feature Validation

Features should validate their inputs:

```python
@feature
def safe_feature(data: Data, dim: str) -> xr.DataArray:
    """Feature with validation."""
    xr_data = data.data
    
    # Check required dimensions
    if dim not in xr_data.dims:
        raise ValueError(f"dim '{dim}' not found in {xr_data.dims}")
    
    # Check data type, range, etc.
    if xr_data.isnull().any():
        raise ValueError("data contains NaN values")
    
    return xr_data.mean(dim=dim)
```

## Best Practices

1. **Keep features focused** - One feature, one computation
2. **Validate inputs** - Check dimensions, data types, ranges
3. **Document thoroughly** - Use Google-style docstrings
4. **Preserve dimensions** - Only remove dimensions when intentional
5. **Test extensively** - Write tests for edge cases

## Feature Naming

Use descriptive, lowercase names with underscores:

```python
@feature
def line_length(data: Data) -> xr.DataArray:
    ...

@feature
def spectral_edge_frequency(data: Data) -> xr.DataArray:
    ...
```

## Accessing Feature History

After applying features, check the history:

```python
data = cb.from_numpy(arr, dims=["time", "space"])
result = cb.feature.line_length(data)

print(result.history)  # ['line_length']

# Chain multiple features
wdata = cb.feature.sliding_window(data, window_size=10)
win_min = cb.feature.min(wdata, dim="window_index")

print(win_min.history)  # ['sliding_window', 'min']
```
