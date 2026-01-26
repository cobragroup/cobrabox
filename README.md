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

