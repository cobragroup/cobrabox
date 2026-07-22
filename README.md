# CobraBox

CobraBox is a lightweight toolbox for feature extraction from EEG and fMRI
time-series data.

## Documentation

📖 **Full docs: <https://cobragroup.github.io/cobrabox/>** — browse and **filter the
entire feature catalog by tag** right on the home page, plus the feature-by-domain
guides and the auto-generated API reference.

## Quick Start

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

# Single feature — one-shot function, or the class
feat = cb.line_length(data)
feat = cb.LineLength().apply(data)

# Pipeline with sliding window (chord: fan-out → map → fan-in)
result = cb.Chord(
    split=cb.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.LineLength(),
    aggregate=cb.MeanAggregate(),
).apply(data)

print(result.history)  # ['SlidingWindow', 'LineLength', 'MeanAggregate', 'Chord']
```

## Core Concepts

- **Data container**: `cobrabox.Data` (with `EEG` and `FMRI` subclasses) — immutable, xarray-backed
- **Features** (`BaseFeature`): every feature has a one-shot function (`cb.line_length(data)`) and a class (`cb.LineLength()`) for chaining with `|`
- **Splitters** (`SplitterFeature`): yield a lazy stream of `Data` per window (e.g. `SlidingWindow`)
- **Aggregators** (`AggregatorFeature`): fold a stream back into one `Data` (e.g. `MeanAggregate`)
- **Chord**: combines a splitter + pipeline + aggregator into a single composable feature
- **Serialization**: save/load any feature, pipeline, or chord to YAML or JSON
- All features append to `history` automatically

## Working with Dimensions and Coordinates

Every `Data` object wraps an `xarray.DataArray` at `data.data`. You don't need to know xarray to use
CobraBox, but these one-liners cover the most common needs:

```python
item = cb.load_dataset("dummy_chain")[0]

# Shape metadata — straight off the Data object
item.dims                                        # ('space', 'time')
item.shape                                       # (4, 200)                  as numpy has it
item.size                                        # 800                       total elements
item.sizes                                       # {'space': 4, 'time': 200} as xarray has it

# Reach the wrapped object without the .data.data double-take
item.xarr                                        # the xarray.DataArray
item.numpy                                       # the numpy array, no copy

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

The full catalog is browsable and **filterable by tag** in the
[docs](https://cobragroup.github.io/cobrabox/). Highlights by domain:

### Signal statistics & information

- `LineLength` - Sum of absolute differences per channel
- `Min` / `Max` / `Mean` - Reduce over any dimension
- `AmplitudeVariation` - Amplitude variation (standard deviation) over time
- `Autocorrelation` - Normalized autocorrelation at a single lag
- `SpikeCount` - Outlier detection using the IQR method
- `EpileptogenicityIndex` - Quantify epileptogenicity from SEEG (Bartolomei et al., 2008)
- `SampleEntropy` - Sample entropy (signal regularity/complexity)
- `LempelZiv` - Lempel-Ziv complexity per channel
- `FractalDimension` - Higuchi or Katz fractal dimension (select with `method=`)
- `AmplitudeEntropy` - Amplitude entropy from a histogram-based distribution
- `Nonreversibility` - Time-irreversibility (deviation from causal normality)
- `RecurrenceMatrix` - Pairwise recurrence (self-similarity) matrix

### Spectral & transforms

- `BandPower` - Power in frequency bands using Welch's method
- `Spectrogram` - Time-frequency power spectrogram
- `Cordance` - Quantitative EEG cordance (Leuchter et al., 1994)
- `ContinuousWaveletTransform` / `DiscreteWaveletTransform` - Wavelet time-frequency / sub-band analysis
- `AnalyticSignal` - Analytic signal: envelope, phase, or instantaneous frequency
- `BandpassFilter` - Butterworth bandpass filter into frequency bands
- `FourierTransform` / `InverseFourierTransform` - FFT to and from the frequency domain

### Connectivity

- `Correlation` / `Covariance` - Pairwise correlation / covariance matrix
- `PartialCorrelation` - Partial correlation controlling for other channels
- `Coherence` - Magnitude-squared coherence between channel pairs
- `PhaseLockingValue` - Phase locking value between channels
- `EnvelopeCorrelation` - Amplitude envelope correlation (AEC)
- `MutualInformation` - Pairwise mutual information matrix
- `GrangerCausality` - Granger causality (directed)
- `PartialDirectedCoherence` / `DirectedTransferFunction` - Directed frequency-domain connectivity via MVAR
- `ReciprocalConnectivity` - Net directional role per channel (source/sink) from a directed matrix

### Decomposition

- `SVD` - Truncated SVD over one dimension with optional centering/z-scoring
- `EMD` - Empirical Mode Decomposition into Intrinsic Mode Functions (IMFs)

### Windowing & aggregation

- `SlidingWindow` - Split data into overlapping windows (splitter)
- `SlidingWindowReduce` - Single-step windowing + aggregation (simpler alternative to Chord)
- `MeanAggregate` / `ConcatAggregate` - Fold windowed results back into one `Data` (aggregators)
- `Chord` - Combine splitter + feature + aggregator

### Surrogates

- `FourierTransformSurrogates` - Fourier surrogates preserving the power spectrum

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

`cb.load_dataset(name)` returns a `Dataset[SignalData]` — an immutable, typed collection with helpers:

```python
ds = cb.load_dataset("dummy_chain")

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

## Documentation

- Setup repo: [`docs/setup_repo.md`](docs/setup_repo.md)
- Contribute a feature: [`docs/contributing/features.md`](docs/contributing/features.md)
- Make a pull request: [`docs/how_to_make_a_pr.md`](docs/how_to_make_a_pr.md)
- Set up GitHub SSH key: [`docs/setup_github_ssh_key.md`](docs/setup_github_ssh_key.md)
- Docs index: [`docs/README.md`](docs/README.md)
