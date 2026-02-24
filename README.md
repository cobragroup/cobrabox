# CobraBox

A lightweight toolbox for extracting time-series features from EEG and fMRI
data. This repository provides a clean, reproducible starting point for feature
engineering across modalities.

## Repository layout

- `src/cobrabox/` - Core library code (feature extractors, pipelines).
- `examples/` - Minimal end-to-end example scripts and notebooks.
- `scripts/` - CLI utilities for batch processing and dataset prep.
- `tests/` - Unit and integration tests.
- `docs/` - User guides and developer notes.
- `data/`
  - `raw/` - Unmodified input data (not tracked).
  - `processed/` - Derived outputs (not tracked).
  - `synthetic/` - Generated datasets for testing (not tracked).

## Getting started

1. Create a virtual environment and install dependencies.
2. Add a small EEG or fMRI example to `data/raw/`.
3. Run an example from `examples/` once available.

## Feature API (sample docs)

### Core ideas

- Data is represented by `cobrabox.Data` (internally wraps an xarray `DataArray`).
- Feature functions are called from `cb.feature.*`.
- Features return a new `Data` object and append to `history`.

### Minimal feature pipeline

```python
import cobrabox as cb

data = cb.from_numpy(
    arr=my_array,                     # shape: (time, space, ...)
    dims=["time", "space"],           # must include "time"
    sampling_rate=100.0,              # optional
)
wdata = cb.feature.sliding_window(data, window_size=20, step_size=10)
feat = cb.feature.line_length(wdata)

print(feat.data.shape)
print(feat.history)  # ['sliding_window', 'line_length']
```

### Built-in feature functions

- `cb.feature.sliding_window(data, window_size=10, step_size=5)`
  - Adds a `window_index` dimension from overlapping windows on `time`.
- `cb.feature.line_length(data)`
  - Computes line-length feature from temporal differences.

### Loading dummy datasets

`cb.dataset(name)` currently supports dummy identifiers and returns a list of
`Data` parts loaded from compressed CSV files:

- `dummy_chain`
- `dummy_random`
- `dummy_star`
- `dummy_noise`

Example:

```python
parts = cb.dataset("dummy_random")
print(len(parts))
print(parts[0].data.dims)
```

## Planned scope

- Time-domain features: basic statistics, complexity, entropy.
- Frequency-domain features: band power, spectral entropy.
- Connectivity features: coherence, correlation, graph metrics.
- Windowing utilities for multichannel time series.
- Modality helpers for EEG and fMRI I/O and preprocessing hooks.

## Contributing

Contributions are welcome. Open an issue to propose new features or datasets.

## License

Add a license before distributing the toolbox publicly.

