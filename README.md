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
- **Serialization**: save/load any feature, pipeline, or chord to YAML or JSON
- All features append to `history` automatically

## Working with Dimensions and Coordinates

Every `Data` object wraps an `xarray.DataArray` at `data.data`. You don't need to know xarray to use
CobraBox, but these one-liners cover the most common needs:

```python
item = cb.dataset("dummy_chain")[0]

# Dimension names and sizes
list(item.data.dims)                             # ['space', 'time']
dict(item.data.sizes)                            # {'space': 4, 'time': 200}

# Coordinate values as a Python list
item.data.coords["space"].values.tolist()        # [0, 1, 2, 3]
item.data.coords["time"].values.tolist()         # [0.0, 0.005, 0.01, ...]

# Select by label (returns xarray.DataArray)
item.data.sel(space=0)                           # one channel
item.data.sel(time=slice(0.0, 0.5))             # time window

# Convert to numpy or pandas
item.to_numpy()                                  # plain ndarray
item.to_pandas()                                 # DataFrame with MultiIndex
```

To attach named coordinates (e.g., electrode labels), build the DataArray explicitly:

```python
import xarray as xr
import numpy as np

xr_arr = xr.DataArray(
    np.random.normal(size=(200, 8)),
    dims=["time", "space"],
    coords={"time": np.arange(200) / 100.0, "space": [f"E{i+1}" for i in range(8)]},
)
data = cb.Data.from_xarray(xr_arr, sampling_rate=100.0, subjectID="sub-01")
data.data.coords["space"].values.tolist()        # ['E1', 'E2', ..., 'E8']
```

See [`examples/data_basics.py`](examples/data_basics.py) for a full walkthrough, and
[`docs/guide/data-containers.md`](docs/guide/data-containers.md) for the complete reference.

## Built-in Features

### Standard Features

- `LineLength` - Sum of absolute differences per channel
- `Min` / `Max` / `Mean` - Reduce over any dimension
- `AmplitudeVariation` - Amplitude variation (standard deviation) over time
- `Bandpower` - Power in frequency bands using Welch's method
- `BandFilter` - Butterworth bandpass filter into frequency bands
- `Coherence` - Magnitude-squared coherence between channel pairs
- `Spectrogram` - Time-frequency power spectrogram
- `Hilbert` - Analytic signal, envelope, phase, or instantaneous frequency
- `SpikeCount` - Outlier detection using IQR method
- `Autocorr` - Normalized autocorrelation at a single lag
- `LempelZiv` - Lempel-Ziv complexity per channel
- `FractalDimHiguchi` - Higuchi Fractal Dimension (signal roughness/complexity)
- `FractalDimKatz` - Katz Fractal Dimension (fast, parameter-free complexity)
- `SampleEntropy` - Sample Entropy (signal regularity/complexity measure)
- `AmplitudeEntropy` - Amplitude entropy from histogram-based distribution
- `Nonreversibility` - Normalised deviation from causal normality (time-irreversibility)
- `MutualInformation` - Pairwise mutual information matrix between channels

### Connectivity Features

- `Correlation` - Pairwise Pearson or Spearman correlation matrix between channels
- `Covariance` - Pairwise sample covariance matrix between channels
- `PartialCorrelation` / `PartialCorrelationMatrix` - Partial correlation controlling for other variables
- `PartialDirectedCoherence` - Partial Directed Coherence via VAR model (directional frequency-domain connectivity)
- `ReciprocalConnectivity` - Net directional role per channel (source/sink detection from PDC)
- `EnvelopeCorrelation` - Amplitude envelope correlation (AEC)
- `PhaseLockingValue` / `PhaseLockingValueMatrix` - Phase locking value between channels
- `GrangerCausality` / `GrangerCausalityMatrix` - Granger causality testing
- `RecurrenceMatrix` - Pairwise recurrence (self-similarity) matrix across time-points or windows

### Specialized Features

- `EpileptogenicityIndex` - Quantify epileptogenicity from SEEG (Bartolomei et al., 2008)

### Windowing & Aggregation

- `SlidingWindow` - Split data into overlapping windows (splitter)
- `SlidingWindowReduce` - Single-step windowing + aggregation (simpler alternative to Chord)
- `MeanAggregate` - Average windowed results (aggregator)
- `ConcatAggregate` - Stack windowed results along new dimension (aggregator)
- `Chord` - Combine splitter + feature + aggregator

### Surrogate Generation

- `FourierTransformSurrogates` - Generate Fourier transform surrogates preserving power spectrum

### Wavelet Transforms

- `DiscreteWaveletTransform` - Multi-level discrete wavelet decomposition (DWT)
- `ContinuousWaveletTransform` - Continuous wavelet transform for time-frequency analysis

### Signal Decomposition

- `EMD` - Empirical Mode Decomposition into Intrinsic Mode Functions (IMFs)

### Specialized Features

- `EpileptogenicityIndex` - Quantify epileptogenicity from SEEG (Bartolomei et al., 2008)

### qEEG Measures

- `Cordance` - Quantitative EEG cordance combining absolute and relative bandpower

## Serialization

Save any feature, pipeline, or chord to YAML or JSON and reload it later — or share it with collaborators:

```python
# Save to file
cb.save(pipeline, "my_pipeline.yaml")

# Load from file
pipeline = cb.load("my_pipeline.yaml")

# Or work with strings directly
yaml_str = cb.serialize(pipeline)
pipeline  = cb.deserialize(yaml_str)
```

See [`examples/serialization_demo.py`](examples/serialization_demo.py) for a full walkthrough.

## Built-in Dummy Datasets

`cb.dataset(name)` returns a `Dataset[SignalData]` — an immutable, typed collection with helpers:

```python
ds = cb.dataset("dummy_chain")

ds.describe()                        # print summary: shapes, metadata
ds.filter(groupID="control")         # Dataset[SignalData] with matching items
ds.groupby("condition")              # dict[str, Dataset[SignalData]]
ds[0]                                # first item
ds[1:3]                              # slice → Dataset[SignalData]
ds1 + ds2                            # concatenate two Datasets
```

Available identifiers:

- `dummy_chain` - Sequential data with known ground truth
- `dummy_random` - Random Gaussian noise
- `dummy_star` - Star-shaped pattern with one central channel
- `dummy_noise` - High-dimensional noise for stress testing

## Coverage

- Test coverage is measured with `pytest-cov` (target: 95%).
- Coverage output is shown by default in test runs (configured in `pyproject.toml`).
- Run tests with:

```bash
uv run pytest -q                    # run all tests
uv run pytest --cov-fail-under=95   # enforce 95% coverage threshold
uv run pytest --cov-report=html     # generate HTML report in htmlcov/
```

## Feature Alignments (D&D Style)

Every feature in CobraBox has a D&D alignment that captures its "moral character" — how it treats your data:

```bash
# See the full roster
uv run python -m cobrabox.dnd_alignment --roster

# Check a pipeline's aggregate alignment
uv run python -m cobrabox.dnd_alignment SlidingWindow LineLength MeanAggregate

# Check a chord pipeline (splitter + map + aggregator)
uv run python -m cobrabox.dnd_alignment --chord SlidingWindow LineLength MeanAggregate
```

The alignment grid categorizes features by their "moral character":
- **Law axis**: Lawful (+1) imposes structure (windowing, categorization); Neutral (0) passively describes; Chaotic (-1) is disruptive
- **Good axis**: Good (+1) preserves meaning; Neutral (0) is indifferent; Evil (-1) discards/distorts

**Current roster:**

| Alignment | Features |
|-----------|----------|
| Lawful Good | SlidingWindow, BandFilter, Bandpower |
| Lawful Neutral | SlidingWindowReduce, MeanAggregate, Mean, ConcatAggregate, Cordance, FractalDimKatz, FourierTransformSurrogates, EpileptogenicityIndex |
| Lawful Evil | SpikeCount, Max, Min |
| Neutral Good | AmplitudeEntropy, AmplitudeVariation, LineLength, DiscreteWaveletTransform, Nonreversibility, RecurrenceMatrix, ContinuousWaveletTransform, Hilbert, Coherence, Autocorr, Spectrogram, EnvelopeCorrelation, FractalDimHiguchi, GrangerCausality, GrangerCausalityMatrix, PartialCorrelation, PartialCorrelationMatrix, PhaseLockingValue, PhaseLockingValueMatrix, PartialDirectedCoherence, ReciprocalConnectivity |
| True Neutral | LempelZiv, MutualInformation, SampleEntropy |
| Chaotic Neutral | Dummy |

Run the command above to see the full grid!

## Documentation

- Setup repo: [`docs/setup_repo.md`](docs/setup_repo.md)
- Contribute a feature: [`docs/contributing/features.md`](docs/contributing/features.md)
- Make a pull request: [`docs/how_to_make_a_pr.md`](docs/how_to_make_a_pr.md)
- Set up GitHub SSH key: [`docs/setup_github_ssh_key.md`](docs/setup_github_ssh_key.md)
- Docs index: [`docs/README.md`](docs/README.md)
