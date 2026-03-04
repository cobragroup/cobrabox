# CobraBox

**Lightweight toolbox for feature extraction from EEG and fMRI time-series data**

CobraBox provides a clean, immutable data model for neuroimaging time-series analysis with automatic history tracking and a flexible feature system.

## Key Features

- **Immutable data containers** - `Data`, `EEG`, and `FMRI` classes wrapping `xarray.DataArray`
- **Automatic history tracking** - Every operation is recorded in the `history` attribute
- **Feature system** - Decorator-based feature functions that preserve metadata
- **Multi-modal support** - Works with EEG, fMRI, and other time-series data
- **Type-safe** - Full type hints for better IDE support and error detection

## Quick Example

```python
import cobrabox as cb
import numpy as np

# Create synthetic data: [time, space]
my_array = np.random.default_rng(seed=0).normal(size=(100, 4))

# Wrap in Data container
data = cb.from_numpy(
    arr=my_array,
    dims=["time", "space"],
    sampling_rate=100.0,
    subjectID="sub-01"
)

# Apply features
feat = cb.feature.LineLength().apply(data)

print(f"Shape: {feat.data.shape}")
print(f"History: {feat.history}")
```

## Why CobraBox?

Traditional neuroimaging workflows often lose track of preprocessing steps and metadata. CobraBox solves this by:

1. **Immutability** - Data is never modified in-place; every operation returns a new instance
2. **Metadata preservation** - Subject IDs, conditions, and sampling rates travel with your data
3. **Transparent pipelines** - The `history` attribute shows exactly what operations were applied
4. **xarray integration** - Leverage the full power of labeled multi-dimensional arrays

## Installation

```bash
uv sync --group docs
```

See [Installation](installation.md) for detailed setup instructions.

## Next Steps

- [Quick Start](quickstart.md) - Get up and running in 5 minutes
- [Core Concepts](guide/core-concepts.md) - Understand the data model
- [Feature Guide](guide/features.md) - Learn to use and create features
