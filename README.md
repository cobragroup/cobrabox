# CobraBox

CobraBox is a lightweight toolbox for feature extraction from EEG and fMRI
time-series data.

## Quick Start

For setup, use the onboarding:
[`docs`](docs)

Short version:

1. Install `git-lfs` on your system (`brew install git-lfs` / `apt-get install git-lfs`)
2. Run:

```bash
make setup
```

## Repository Layout

- `src/cobrabox/` - Core package code (data model, features, loaders)
- `tests/` - Unit tests
- `examples/` - Minimal runnable examples
- `docs/` - Setup and contribution guides
- `data/` - Dummy/example data files

## Python Version

- Minimum supported: Python 3.11
- Development target: Python 3.14

## Minimal Runnable Usage

```python
import cobrabox as cb
import numpy as np

# Create synthetic 2D input data: [time, space]
my_array = np.random.default_rng(seed=0).normal(size=(100, 4))

data = cb.from_numpy(arr=my_array, dims=["time", "space"], sampling_rate=100.0)

# Single feature
feat = cb.feature.LineLength().apply(data)

# Pipeline with sliding window (chord: fan-out → map → fan-in)
result = cb.Chord(
    split=cb.feature.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.feature.LineLength(),
    aggregate=cb.feature.MeanAggregate(),
).apply(data)

print(result.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

## Core Concepts

- **Data container**: `cobrabox.Data` (with `EEG` and `FMRI` subclasses) — immutable, xarray-backed
- **Features** (`BaseFeature`): dataclasses under `cb.feature.*`; call `.apply(data)` or chain with `|`
- **Splitters** (`SplitterFeature`): yield a lazy stream of `Data` per window (e.g. `SlidingWindow`)
- **Aggregators** (`AggregatorFeature`): fold a stream back into one `Data` (e.g. `MeanAggregate`)
- **Chord**: combines a splitter + pipeline + aggregator into a single composable feature
- All features append to `history` automatically

## Built-in Features

### Standard Features
- `LineLength` - Sum of absolute differences per channel
- `Min` / `Max` / `Mean` - Reduce over any dimension
- `Bandpower` - Power in frequency bands using Welch's method
- `Coherence` - Magnitude-squared coherence between channel pairs
- `Spectrogram` - Time-frequency power spectrogram
- `SpikesCalc` - Outlier detection using IQR method

### Connectivity Features
- `EnvelopeCorrelation` - Amplitude envelope correlation (AEC)
- `PartialCorrelation` - Partial correlation controlling for other variables

### Specialized Features
- `EpileptogenicityIndex` - Quantify epileptogenicity from SEEG (Bartolomei et al., 2008)

### Windowing & Aggregation
- `SlidingWindow` - Split data into overlapping windows (splitter)
- `MeanAggregate` - Average windowed results (aggregator)
- `Chord` - Combine splitter + feature + aggregator

## Built-in Dummy Datasets

Use `cb.dataset(name)` with:

- `dummy_chain`
- `dummy_random`
- `dummy_star`
- `dummy_noise`

## Coverage

- Test coverage is measured with `pytest-cov`.
- Coverage output is shown by default in test runs (configured in `pyproject.toml`).
- Run tests with:

```bash
uv run pytest -q
```

## Documentation

- Setup repo: [`docs/setup_repo.md`](docs/setup_repo.md)
- Contribute a feature: [`docs/contributing_feature.md`](docs/contributing_feature.md)
- Make a pull request: [`docs/how_to_make_a_pr.md`](docs/how_to_make_a_pr.md)
- Set up GitHub SSH key: [`docs/setup_github_ssh_key.md`](docs/setup_github_ssh_key.md)
- Docs index: [`docs/README.md`](docs/README.md)
