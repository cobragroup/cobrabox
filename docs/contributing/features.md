# Contributing a Feature

This guide shows the recommended workflow for adding a new feature to CobraBox.

## Quick Checklist

1. Make a new branch
2. Create a new file in `src/cobrabox/features/` (e.g., `my_feature.py`)
3. Create a new test file in `tests/` (e.g., `test_feature_my_feature.py`)
4. Implement and test
5. Open a pull request

## 1. Create a Branch

```bash
git checkout main
git pull
git checkout -b feature/add-mean-absolute-value
```

Pick a branch name that describes the feature.

## 2. Implement the Feature

Create `src/cobrabox/features/my_feature.py`:

```python
import xarray as xr
from cobrabox import Data
from cobrabox.function_wrapper import feature

@feature
def my_feature(data: Data, dim: str) -> xr.DataArray:
    """Compute my custom feature.
    
    Args:
        data: Input Data with 'time' and 'space' dimensions
        dim: Dimension to reduce over
        
    Returns:
        xarray DataArray with computed feature values
        
    Example:
        >>> result = cb.feature.my_feature(data, dim="time")
    """
    xr_data = data.data
    
    # Validate dimensions
    if dim not in xr_data.dims:
        raise ValueError(f"dim '{dim}' not found in {xr_data.dims}")
    
    # Your computation here
    result = xr_data.mean(dim=dim)
    
    return result
```

**Key points:**

- Use the `@feature` decorator
- First parameter must be `data: Data`
- Return `xr.DataArray` or `Data`
- Add parameters as needed
- Validate inputs
- Write a clear docstring (Google style)

**Reference implementations:**

- `src/cobrabox/features/line_length.py`
- `src/cobrabox/features/sliding_window.py`
- `src/cobrabox/features/mean.py`

**Note:** Features are auto-discovered. No need to manually register in `features/__init__.py`.

## 3. Add Tests

Create `tests/test_feature_my_feature.py`:

```python
import numpy as np
import cobrabox as cb

def test_my_feature_basic():
    """Test basic functionality."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data = cb.from_numpy(arr, dims=["time", "space"])
    
    result = cb.feature.my_feature(data, dim="time")
    
    # Check shape
    assert result.data.shape == (2,)  # space dimension only
    
    # Check history
    assert "my_feature" in result.history
    
    # Check values (mean of [1,3,5] and [2,4,6])
    expected = np.array([3.0, 4.0])
    np.testing.assert_array_almost_equal(result.data.values, expected)

def test_my_feature_invalid_dim():
    """Test error handling for invalid dimension."""
    arr = np.random.normal(size=(10, 4))
    data = cb.from_numpy(arr, dims=["time", "space"])
    
    import pytest
    with pytest.raises(ValueError, match="dim 'invalid' not found"):
        cb.feature.my_feature(data, dim="invalid")
```

**Test coverage:**

- Basic functionality with known inputs
- Metadata/history preservation
- Edge cases (empty data, single timepoint, etc.)
- Error handling (invalid dimensions, bad parameters)

**Reference tests:**

- `tests/test_feature_dummy.py`
- `tests/test_feature_line_length.py`
- `tests/test_feature_sliding_window.py`

Run tests:

```bash
uv run pytest tests/test_feature_my_feature.py -v
```

## 4. Commit and Push

```bash
# Add feature implementation
git add src/cobrabox/features/my_feature.py
git commit -m "feat: add my_feature implementation"

# Add tests
git add tests/test_feature_my_feature.py
git commit -m "test: add tests for my_feature"

# Push to remote
git push -u origin feature/add-mean-absolute-value
```

Pre-commit hooks will automatically run ruff linting.

## 5. Create Pull Request

Go to GitHub and create a pull request. Include:

- **What** the feature computes
- **Why** it's useful
- **How** to use it (example code)
- **Test coverage** summary

## Feature Naming Conventions

- Use lowercase with underscores: `line_length`, `spectral_power`
- Be descriptive but concise
- Match the function name to the file name

## Best Practices

1. **Keep it focused** - One feature, one computation
2. **Validate inputs** - Check dimensions, ranges, data types
3. **Document thoroughly** - Args, returns, examples in docstring
4. **Test edge cases** - Empty data, NaN values, boundary conditions
5. **Preserve dimensions** - Only remove dimensions intentionally
6. **Use type hints** - Full type annotations for all parameters

## Common Patterns

### Reducing a Dimension

```python
@feature
def mean_over_time(data: Data) -> xr.DataArray:
    """Compute mean over time dimension."""
    return data.data.mean(dim="time")
```

### Adding a New Dimension

```python
@feature
def sliding_window(data: Data, window_size: int) -> xr.DataArray:
    """Create sliding windows."""
    # Your implementation that adds 'window_index' dimension
    ...
```

### Conditional Logic

```python
@feature
def threshold_feature(data: Data, threshold: float) -> xr.DataArray:
    """Apply thresholding."""
    xr_data = data.data
    return (xr_data > threshold).sum(dim="time")
```
