# Contributing a Feature

This guide shows the recommended workflow for adding a new feature to CobraBox.

## Quick Checklist

1. Make a new branch
2. Create `src/cobrabox/features/<domain>/my_feature.py`
3. Create `tests/test_feature_my_feature.py`
4. Implement, test, lint
5. Open a pull request

### Choose the Right Domain

| Domain | Features |
| ------ | -------- |
| `time_domain/` | Morphological & temporal statistics (LineLength, AmplitudeVariation, SampleEntropy, LempelZiv, FractalDimHiguchi, FractalDimKatz, SpikeCount, Autocorr, EnvelopeCorrelation, RecurrenceMatrix, Nonreversibility) |
| `frequency_domain/` | Spectral analysis (Bandpower, BandFilter, Cordance, Spectrogram, EpileptogenicityIndex) |
| `time_frequency/` | Joint time-frequency methods (Hilbert, ContinuousWaveletTransform, DiscreteWaveletTransform, EMD, AmplitudeEntropy) |
| `connectivity/` | Inter-channel relationships (Correlation, PartialCorrelation, PartialCorrelationMatrix, Covariance, Coherence, GrangerCausality, GrangerCausalityMatrix, PartialDirectedCoherence, PhaseLockingValue, PhaseLockingValueMatrix, MutualInformation, ReciprocalConnectivity) |
| `decomposition/` | Signal decomposition methods (FourierTransformSurrogates) |
| `windowing/` | Windowing & aggregation (SlidingWindow, SlidingWindowReduce, MeanAggregate, ConcatAggregate) |
| `reductions/` | Basic statistical reductions (Mean, Max, Min) |

## 1. Create a Branch

```bash
git checkout main
git pull
git checkout -b feature/add-variance
```

## 2. Choose the Right Base Class and Type

| You want to…                                | Inherit from                  | Type Parameter |
| ------------------------------------------- | ----------------------------- | -------------- |
| Transform any `Data` → `Data`               | `BaseFeature[Data]`           | Generic        |
| Transform time-series `SignalData` → `Data` | `BaseFeature[SignalData]`     | Time-series    |
| Split time-series into windows              | `SplitterFeature[SignalData]` | Time-series    |
| Fold a stream back into one `Data`          | `AggregatorFeature`           | (not generic)  |

## 3. Implement the Feature

### Generic feature (`BaseFeature[Data]`)

Use for features that work with any data container:

```python
# src/cobrabox/features/reductions/variance.py
from __future__ import annotations
from dataclasses import dataclass
import xarray as xr
from ..base_feature import BaseFeature
from ..data import Data

@dataclass
class Variance(BaseFeature[Data]):
    """Compute variance over a dimension.

    Args:
        dim: Dimension to reduce over.

    Example:
        >>> result = cb.feature.Variance(dim="time").apply(data)
    """

    dim: str

    def __call__(self, data: Data) -> xr.DataArray:
        if self.dim not in data.data.dims:
            raise ValueError(f"dim '{self.dim}' not found in {data.data.dims}")
        return data.data.var(dim=self.dim)
```

**Key points:**

- `@dataclass` on the class, not `@feature` on a function
- Store parameters as dataclass fields (with defaults where sensible)
- Use `BaseFeature[Data]` for generic features or `BaseFeature[SignalData]` for time-series
- Implement `__call__(self, data: DataT)` — no `apply()` needed, that's inherited
- Return `xr.DataArray` or `Data`
- Validate inputs and raise `ValueError` with clear messages
- Write a Google-style docstring with an `Example:` block

### Time-series feature (`BaseFeature[SignalData]`)

Use for features that require time-series data:

```python
# src/cobrabox/features/frequency_domain/spectral_power.py
from __future__ import annotations
from dataclasses import dataclass
import xarray as xr
from ..base_feature import BaseFeature
from ..data import SignalData

@dataclass
class SpectralPower(BaseFeature[SignalData]):
    """Compute power in a frequency band.

    Args:
        fmin: Lower frequency bound in Hz.
        fmax: Upper frequency bound in Hz.

    Example:
        >>> result = cb.feature.SpectralPower(fmin=8, fmax=12).apply(data)
    """

    fmin: float
    fmax: float

    def __call__(self, data: SignalData) -> xr.DataArray:
        # SignalData guarantees 'time' dimension exists
        # No need to check: if "time" not in data.data.dims
        xr_data = data.data
        # ... compute spectral power
        return result
```

**Key points for time-series features:**

- Import `SignalData` instead of `Data`
- Use `BaseFeature[SignalData]` (not just `BaseFeature`)
- Type hint `data: SignalData` in `__call__`
- SignalData validates 'time' dimension at construction — no need to check in feature
- SignalData guarantees time is the last dimension

### Splitter feature (`SplitterFeature[SignalData]`)

```python
# src/cobrabox/features/windowing/trial_split.py
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from ..base_feature import SplitterFeature
from ..data import Data, SignalData

@dataclass
class TrialSplit(SplitterFeature[SignalData]):
    """Yield one Data per fixed-length trial block."""

    trial_length: int

    def __call__(self, data: SignalData) -> Iterator[Data]:
        n = data.data.sizes["time"]
        for start in range(0, n - self.trial_length + 1, self.trial_length):
            window = data.data.isel(time=slice(start, start + self.trial_length))
            yield data._copy_with_new_data(new_data=window, operation_name="TrialSplit")
```

### Aggregator feature (`AggregatorFeature`)

```python
# src/cobrabox/features/windowing/max_aggregate.py
from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
import xarray as xr
from ..base_feature import AggregatorFeature
from ..data import Data

@dataclass
class MaxAggregate(AggregatorFeature):
    """Take element-wise max across a stream of per-window Data."""

    def __call__(self, data: Data, stream: Iterator[Data]) -> Data:
        items = list(stream)
        if not items:
            raise ValueError("MaxAggregate received an empty stream")
        stacked = xr.concat([w.data for w in items], dim="window", join="override")
        result = stacked.max(dim="window")
        window_history = [op for op in items[0].history if op not in data.history]
        return Data(
            data=result,
            subjectID=data.subjectID,
            groupID=data.groupID,
            condition=data.condition,
            history=list(data.history) + window_history + ["MaxAggregate"],
            extra=data.extra,
        )
```

Note: `AggregatorFeature` is responsible for building history manually — include the per-window pipeline ops and the aggregator's own name.

## 4. Add Tests

```python
# tests/test_feature_variance.py
from __future__ import annotations
import numpy as np
import pytest
import cobrabox as cb
from cobrabox.features.reductions.mean import Mean


def test_variance_reduces_time_dimension() -> None:
    """Test with SignalData for time-series features."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Variance(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert "time" not in out.data.dims
    np.testing.assert_allclose(out.to_numpy().flatten(), np.var([[1,3,5],[2,4,6]], axis=1))
    assert out.history == ["Variance"]


def test_variance_raises_for_unknown_dimension() -> None:
    """Test error handling with SignalData."""
    data = cb.SignalData.from_numpy(np.ones((5, 3)), dims=["time", "space"])
    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.Variance(dim="band_index").apply(data)
```

**Cover at minimum:**

- Correct values on known input
- `history` contains the class name
- Metadata (`subjectID`, `sampling_rate`) is preserved
- `ValueError` on invalid dimension/parameters
- Use `SignalData` for time-series features, `Data` for generic features

## 5. Auto-discovery

Features are discovered automatically — no registration needed. The discovery looks for classes where:

- `_is_cobrabox_feature` is `True` (inherited from all base classes), **and**
- `__module__` matches the feature's own file

So just drop the file in `src/cobrabox/features/` and run the tests.

## 6. Lint and Format

```bash
uvx ruff check --fix src/ tests/
uvx ruff format src/ tests/
```

## 7. Commit and Push

```bash
git add src/cobrabox/features/variance.py tests/test_feature_variance.py
git commit -m "feat: add Variance feature"
git push -u origin feature/add-variance
```

Pre-commit hooks run ruff automatically on commit.

## Reference Implementations

### Time-series Features (use `SignalData`)

- `src/cobrabox/features/time_domain/line_length.py` — simple `BaseFeature[SignalData]`
- `src/cobrabox/features/frequency_domain/bandpower.py` — `BaseFeature[SignalData]` with parameters
- `src/cobrabox/features/connectivity/coherence.py` — `BaseFeature[SignalData]` with internal helpers
- `src/cobrabox/features/windowing/sliding_window.py` — `SplitterFeature[SignalData]`

### Generic Features (use `Data`)

- `src/cobrabox/features/reductions/mean.py` — `BaseFeature[Data]` with parameter
- `src/cobrabox/features/reductions/max.py` — `BaseFeature[Data]` with parameter
- `src/cobrabox/features/reductions/min.py` — `BaseFeature[Data]` with parameter

### Aggregators

- `src/cobrabox/features/windowing/mean_aggregate.py` — `AggregatorFeature`
