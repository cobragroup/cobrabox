# Test Plan: SpikeCount

**Feature**: `src/cobrabox/features/spikes_calc.py`
**Test file**: `tests/test_feature_spikes_calc.py` (does not exist)
**Date**: 2026-03-06
**Verdict**: MISSING

## Summary

`SpikeCount` is a `BaseFeature[Data]` that detects outliers using the IQR method (values outside ±1.5*IQR from Q1/Q3) and returns a scalar count. The feature sets `output_type = Data`, meaning the output has no dimensions and `sampling_rate` becomes `None`. Tests must verify the IQR-based spike detection, empty data handling, and proper metadata preservation.

## Coverage

Coverage: 0% (no test file exists)

## Proposed test file

```python
"""Tests for cb.feature.SpikeCount."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n_time: int = 100,
    n_space: int = 10,
    sampling_rate: float = 100.0,
    subjectID: str = "s1",
    groupID: str = "g1",
    condition: str = "rest",
) -> cb.Data:
    """Create test data with known distribution for spike testing."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((n_time, n_space))
    return cb.Data.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=sampling_rate,
        subjectID=subjectID,
        groupID=groupID,
        condition=condition,
    )


def _make_data_with_spikes(
    n_time: int = 100,
    n_space: int = 10,
    n_spikes: int = 5,
) -> cb.Data:
    """Create test data with known outliers for spike detection."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((n_time, n_space))
    # Add clear outliers (spikes) - values far outside normal range
    flat_arr = arr.flatten()
    spike_indices = rng.choice(len(flat_arr), size=n_spikes, replace=False)
    flat_arr[spike_indices] = rng.choice([-10.0, 10.0], size=n_spikes)
    arr = flat_arr.reshape((n_time, n_space))
    return cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_spike_count_basic() -> None:
    """SpikeCount returns a scalar Data object with no dimensions."""
    data = _make_data()
    result = cb.feature.SpikeCount().apply(data)
    assert isinstance(result, cb.Data)
    assert result.data.dims == ()
    assert result.data.shape == ()
    assert not np.isnan(result.to_numpy())
    assert result.to_numpy() >= 0


def test_spike_count_detects_outliers() -> None:
    """SpikeCount correctly detects outliers using IQR method."""
    data = _make_data_with_spikes(n_spikes=5)
    result = cb.feature.SpikeCount().apply(data)
    # With 5 inserted spikes, we expect at least 5 detected
    assert result.to_numpy() >= 5


def test_spike_count_no_spikes() -> None:
    """SpikeCount returns 0 for data with no outliers."""
    # Create data with very small variance, no outliers
    arr = np.ones((10, 10)) * 5.0
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    result = cb.feature.SpikeCount().apply(data)
    assert result.to_numpy() == 0.0


def test_spike_count_empty_data_raises() -> None:
    """SpikeCount raises ValueError when input data is empty."""
    arr = np.array([])
    data = cb.Data.from_numpy(arr, dims=[], sampling_rate=100.0)
    with pytest.raises(ValueError, match="empty"):
        cb.feature.SpikeCount().apply(data)


def test_spike_count_history_updated() -> None:
    """SpikeCount appends 'SpikeCount' to history."""
    data = _make_data()
    result = cb.feature.SpikeCount().apply(data)
    assert result.history[-1] == "SpikeCount"


def test_spike_count_metadata_preserved() -> None:
    """SpikeCount preserves subjectID, groupID, condition."""
    data = _make_data(
        sampling_rate=250.0, subjectID="s42", groupID="control", condition="task"
    )
    result = cb.feature.SpikeCount().apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"


def test_spike_count_sampling_rate_none() -> None:
    """SpikeCount sets sampling_rate to None (output_type = Data)."""
    data = _make_data(sampling_rate=250.0)
    result = cb.feature.SpikeCount().apply(data)
    assert result.sampling_rate is None


def test_spike_count_returns_data_instance() -> None:
    """SpikeCount.apply() always returns a Data instance."""
    data = _make_data()
    result = cb.feature.SpikeCount().apply(data)
    assert isinstance(result, cb.Data)


def test_spike_count_does_not_mutate_input() -> None:
    """SpikeCount does not modify the input Data object."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.SpikeCount().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_spike_count_1d_data() -> None:
    """SpikeCount works with 1D data."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(100)
    # Add some spikes
    arr[[10, 20, 30]] = [10.0, -10.0, 10.0]
    data = cb.Data.from_numpy(arr, dims=["time"], sampling_rate=100.0)
    result = cb.feature.SpikeCount().apply(data)
    assert result.to_numpy() >= 3


def test_spike_count_3d_data() -> None:
    """SpikeCount works with 3D data."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((10, 10, 10))
    data = cb.Data.from_numpy(arr, dims=["x", "y", "z"], sampling_rate=100.0)
    result = cb.feature.SpikeCount().apply(data)
    assert isinstance(result, cb.Data)
    assert result.data.shape == ()


def test_spike_count_iqr_calculation() -> None:
    """SpikeCount uses correct IQR bounds (1.5 * IQR)."""
    # Create data with known quartiles
    # Q1=25, Q2=50, Q3=75, IQR=50
    # Lower bound = 25 - 1.5*50 = -50
    # Upper bound = 75 + 1.5*50 = 150
    arr = np.concatenate([
        np.linspace(0, 100, 100),  # Normal data
        np.array([-100.0, 200.0]),  # Two clear outliers outside bounds
    ])
    data = cb.Data.from_numpy(arr, dims=["time"], sampling_rate=100.0)
    result = cb.feature.SpikeCount().apply(data)
    # Should detect exactly 2 spikes
    assert result.to_numpy() == 2.0
```

## Action List

1. [Severity: HIGH] Create `tests/test_feature_spikes_calc.py` with the complete test file above (no test file exists, coverage is 0%)
2. [Severity: MEDIUM] Consider adding `__post_init__` validation to `SpikeCount` if any parameters are added in the future
3. [Severity: LOW] The feature docstring says "Returns a scalar count" but returns `float` — consider if `int` would be more appropriate
