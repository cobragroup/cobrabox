"""Tests for the SpikeCount feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_spike_count_clean_data_no_outliers(rng: np.random.Generator) -> None:
    """SpikeCount returns 0 for clean data without outliers."""
    # Normal distribution data without extreme values
    arr = rng.standard_normal((100, 2)) * 10 + 50
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-01")

    out = cb.feature.SpikeCount().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.shape == ()
    assert out.data.dims == ()
    assert out.subjectID == "sub-01"
    assert out.history == ["SpikeCount"]
    # Clean data should have 0 or very few spikes
    assert out.to_numpy() < 5


def test_spike_count_with_outliers() -> None:
    """SpikeCount detects outliers beyond IQR bounds."""
    # Create data with known outliers
    arr = np.ones((100, 2)) * 50  # Base normal values
    arr[10, 0] = 200  # Extreme spike
    arr[50, 1] = -150  # Extreme dip

    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)

    out = cb.feature.SpikeCount().apply(data)

    assert isinstance(out, cb.Data)
    assert out.to_numpy() >= 2  # At least the 2 extreme values


def test_spike_count_preserves_metadata(rng: np.random.Generator) -> None:
    """SpikeCount preserves metadata from input Data."""
    arr = rng.standard_normal((50, 3))
    data = cb.Data.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-02",
        groupID="control",
        condition="task",
        extra={"task_name": "motor"},
    )

    out = cb.feature.SpikeCount().apply(data)

    assert out.subjectID == "sub-02"
    assert out.groupID == "control"
    assert out.condition == "task"
    assert out.extra.get("task_name") == "motor"
    assert out.history == ["SpikeCount"]


def test_spike_count_returns_scalar(rng: np.random.Generator) -> None:
    """SpikeCount returns scalar Data with shape ()."""
    arr = rng.standard_normal((30, 2))
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikeCount().apply(data)

    assert out.data.shape == ()
    assert out.data.dims == ()
    # to_numpy() returns a 0-d numpy array, extract scalar with item()
    val = out.to_numpy().item()
    assert isinstance(val, (int, float, np.integer, np.floating))


def test_spike_count_multivariate_data(rng: np.random.Generator) -> None:
    """SpikeCount works on multivariate data (multiple channels)."""
    # 3 channels with different spike patterns
    arr = rng.standard_normal((100, 3)) * 10
    arr[20:25, 0] = 500  # Spikes in channel 0
    arr[60, 2] = -300  # Spike in channel 2

    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikeCount().apply(data)

    assert isinstance(out, cb.Data)
    assert out.to_numpy() > 0


def test_spike_count_empty_data_raises() -> None:
    """SpikeCount raises ValueError for empty input data."""
    arr = np.array([]).reshape(0, 0)
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="empty"):
        cb.feature.SpikeCount().apply(data)


def test_spike_count_sampling_rate_none(rng: np.random.Generator) -> None:
    """SpikeCount sets sampling_rate to None when time dimension is removed."""
    arr = rng.standard_normal((50, 2))
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikeCount().apply(data)

    assert out.sampling_rate is None


def test_spike_count_does_not_mutate_input(rng: np.random.Generator) -> None:
    """SpikeCount does not modify the input Data object."""
    arr = rng.standard_normal((50, 2))
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01")

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.SpikeCount().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"


def test_spike_count_boundary_values() -> None:
    """Values exactly at IQR bounds are not counted as spikes."""
    # Create data where Q1=25, Q3=75, IQR=50
    # Bounds: low = 25 - 75 = -50, high = 75 + 75 = 150
    arr = np.array([25.0, 75.0, 0.0, 100.0, -50.0, 150.0])  # Values at bounds

    data = cb.Data.from_numpy(arr.reshape(-1, 1), dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikeCount().apply(data)

    # Values exactly at bounds should NOT be spikes
    # But anything beyond should be
    assert out.to_numpy() == 0
