# Test Plan: ampent

**Feature**: `src/cobrabox/features/ampent.py`
**Test file**: `tests/test_feature_ampent.py` (does not exist)
**Date**: 2026-03-06
**Verdict**: MISSING

## Summary

Ampent computes amplitude entropy from time-series data using histogram-based probability estimation. It takes a `Data` object (not SignalData - no time dimension requirement) and two parameters: `data_window` (a Data object) and `band_width` (float for histogram bin width). It returns a scalar value representing the mean entropy across all time points, wrapped in a Data container with `output_type = Data` (removing time dimension).

The algorithm:

1. Iterates through each row (time point) of the input data
2. Computes histogram edges from min to max of each row using band_width as step
3. Calculates probability distribution via normalized histogram counts
4. Computes Shannon entropy: -sum(p * log2(p)) for each row
5. Returns mean entropy as a scalar (shape (1, 1) with dims ["time", "space"])

## Proposed test file

```python
"""Tests for cb.feature.Ampent."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n_time: int = 100,
    n_space: int = 10,
    subjectID: str = "s1",
    groupID: str = "g1",
    condition: str = "rest",
) -> cb.Data:
    """Create a Data object for testing."""
    arr = np.random.randn(n_time, n_space)
    return cb.Data.from_numpy(
        arr,
        dims=["time", "space"],
        subjectID=subjectID,
        groupID=groupID,
        condition=condition,
    )


def _make_data_window(
    n_time: int = 10,
    n_space: int = 5,
) -> cb.Data:
    """Create a small Data object to use as data_window parameter."""
    arr = np.random.randn(n_time, n_space)
    return cb.Data.from_numpy(arr, dims=["time", "space"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ampent_basic() -> None:
    """Ampent returns a Data object with scalar entropy value."""
    data = _make_data(n_time=50, n_space=10)
    data_window = _make_data_window(n_time=10, n_space=5)
    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.shape == (1, 1)
    assert result.data.dims == ("time", "space")
    assert not np.any(np.isnan(result.to_numpy()))
    assert not np.any(np.isinf(result.to_numpy()))
    # Entropy should be non-negative
    assert result.to_numpy()[0, 0] >= 0


def test_ampent_deterministic() -> None:
    """Ampent returns same result for identical inputs."""
    np.random.seed(42)
    data = _make_data(n_time=30, n_space=8)
    data_window = _make_data_window(n_time=5, n_space=4)

    feature = cb.feature.Ampent(data_window=data_window, band_width=0.3)

    result1 = feature.apply(data)
    result2 = feature.apply(data)

    np.testing.assert_array_almost_equal(
        result1.to_numpy(), result2.to_numpy()
    )


def test_ampent_constant_data() -> None:
    """Ampent handles constant data (zero entropy case)."""
    # Constant data has zero entropy (all values in one bin)
    arr = np.ones((20, 5))
    data = cb.Data.from_numpy(arr, dims=["time", "space"])
    data_window = _make_data_window(n_time=5, n_space=3)

    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    # Entropy of constant distribution is 0
    np.testing.assert_allclose(result.to_numpy()[0, 0], 0.0, atol=1e-10)


def test_ampent_uniform_data() -> None:
    """Ampent returns expected entropy for uniform distribution."""
    # Uniform distribution across bins
    np.random.seed(42)
    n_bins = 8
    samples_per_bin = 10
    n_space = n_bins * samples_per_bin

    # Create data with uniform distribution
    arr = np.zeros((10, n_space))
    for i in range(10):
        for bin_idx in range(n_bins):
            start = bin_idx * samples_per_bin
            end = start + samples_per_bin
            arr[i, start:end] = bin_idx

    data = cb.Data.from_numpy(arr, dims=["time", "space"])
    data_window = _make_data_window(n_time=5, n_space=4)

    feature = cb.feature.Ampent(data_window=data_window, band_width=1.0)
    result = feature.apply(data)

    # Uniform distribution over n_bins has entropy = log2(n_bins)
    expected_entropy = np.log2(n_bins)
    # Due to discretization, result should be close but not exact
    assert result.to_numpy()[0, 0] > 0
    assert result.to_numpy()[0, 0] <= expected_entropy + 0.1


def test_ampent_different_band_widths() -> None:
    """Ampent produces different results with different band_widths."""
    np.random.seed(42)
    data = _make_data(n_time=50, n_space=20)
    data_window = _make_data_window(n_time=10, n_space=5)

    feature_narrow = cb.feature.Ampent(data_window=data_window, band_width=0.1)
    feature_wide = cb.feature.Ampent(data_window=data_window, band_width=1.0)

    result_narrow = feature_narrow.apply(data)
    result_wide = feature_wide.apply(data)

    # Different bandwidths should produce different entropies
    # Wider bins -> fewer bins -> lower max entropy
    entropy_narrow = result_narrow.to_numpy()[0, 0]
    entropy_wide = result_wide.to_numpy()[0, 0]
    assert entropy_narrow != pytest.approx(entropy_wide, abs=1e-6)


def test_ampent_history_updated() -> None:
    """Ampent appends 'Ampent' to history."""
    data = _make_data()
    data_window = _make_data_window()
    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    result = feature.apply(data)

    assert result.history[-1] == "Ampent"


def test_ampent_metadata_preserved() -> None:
    """Ampent preserves subjectID, groupID, condition; sampling_rate becomes None."""
    data = _make_data(
        subjectID="s42", groupID="control", condition="task"
    )
    data_window = _make_data_window()
    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    result = feature.apply(data)

    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    # output_type = Data removes time dimension, so sampling_rate is None
    assert result.sampling_rate is None


def test_ampent_returns_data_instance() -> None:
    """Ampent.apply() always returns a Data instance."""
    data = _make_data()
    data_window = _make_data_window()
    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)


def test_ampent_does_not_mutate_input() -> None:
    """Ampent does not modify the input Data object."""
    data = _make_data()
    data_window = _make_data_window()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    _ = feature.apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_ampent_empty_row_handling() -> None:
    """Ampent handles edge case where histogram has zero counts."""
    # This tests the defensive guard in the code
    # Create data that might trigger the total == 0 case
    arr = np.zeros((5, 1))  # Single value per row
    data = cb.Data.from_numpy(arr, dims=["time", "space"])
    data_window = _make_data_window(n_time=3, n_space=2)

    feature = cb.feature.Ampent(data_window=data_window, band_width=1.0)
    result = feature.apply(data)

    # Should not raise, should return finite value
    assert np.isfinite(result.to_numpy()[0, 0])


def test_ampent_negative_band_width() -> None:
    """Ampent raises ValueError for negative band_width."""
    data_window = _make_data_window()
    with pytest.raises(ValueError):
        cb.feature.Ampent(data_window=data_window, band_width=-0.5)


def test_ampent_zero_band_width() -> None:
    """Ampent raises ValueError for zero band_width."""
    data_window = _make_data_window()
    with pytest.raises(ValueError):
        cb.feature.Ampent(data_window=data_window, band_width=0.0)


def test_ampent_single_row() -> None:
    """Ampent works with single-row data."""
    arr = np.random.randn(1, 10)
    data = cb.Data.from_numpy(arr, dims=["time", "space"])
    data_window = _make_data_window()

    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.shape == (1, 1)
    assert np.isfinite(result.to_numpy()[0, 0])


def test_ampent_single_column() -> None:
    """Ampent works with single-column data."""
    arr = np.random.randn(50, 1)
    data = cb.Data.from_numpy(arr, dims=["time", "space"])
    data_window = _make_data_window()

    feature = cb.feature.Ampent(data_window=data_window, band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.shape == (1, 1)
    assert np.isfinite(result.to_numpy()[0, 0])
```

## Action List

1. **[Severity: HIGH]** Create `tests/test_feature_ampent.py` with the complete test file above

2. **[Severity: MEDIUM]** Add validation in `ampent.py` for `band_width > 0` in `__post_init__`:

   ```python
   def __post_init__(self) -> None:
       if self.band_width <= 0:
           raise ValueError("band_width must be positive")
   ```

3. **[Severity: LOW]** The feature uses `data_window` parameter but doesn't actually use it in the computation - it only uses `self.band_width` and the input `data`. Consider removing this unused parameter or implementing the intended windowing behavior.

4. **[Severity: LOW]** Add docstring to the `Ampent` class explaining what amplitude entropy is and how it's computed.

---

**Note**: The feature currently has `data_window` as a required parameter but never uses it in the computation (line 21 uses `data.to_numpy()`, not `self.data_window`). This appears to be a bug or incomplete implementation that should be addressed.
