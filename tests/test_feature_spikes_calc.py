"""Tests for the SpikesCalc feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_spikes_calc_clean_data_no_outliers(rng: np.random.Generator) -> None:
    """spikes_calc returns 0 for clean data without outliers."""
    # Normal distribution data without extreme values
    arr = rng.standard_normal((100, 2)) * 10 + 50
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-01")

    out = cb.feature.SpikesCalc().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.shape == ()
    assert out.data.dims == ()
    assert out.subjectID == "sub-01"
    assert out.history == ["SpikesCalc"]
    # Clean data should have 0 or very few spikes
    assert out.to_numpy() < 5


def test_spikes_calc_with_outliers() -> None:
    """SpikesCalc detects outliers beyond IQR bounds."""
    # Create data with known outliers
    arr = np.ones((100, 2)) * 50  # Base normal values
    arr[10, 0] = 200  # Extreme spike
    arr[50, 1] = -150  # Extreme dip

    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)

    out = cb.feature.SpikesCalc().apply(data)

    assert isinstance(out, cb.Data)
    assert out.to_numpy() >= 2  # At least the 2 extreme values


def test_spikes_calc_preserves_metadata(rng: np.random.Generator) -> None:
    """spikes_calc preserves metadata from input Data."""
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

    out = cb.feature.SpikesCalc().apply(data)

    assert out.subjectID == "sub-02"
    assert out.groupID == "control"
    assert out.condition == "task"
    assert out.extra.get("task_name") == "motor"
    assert out.history == ["SpikesCalc"]


def test_spikes_calc_returns_scalar(rng: np.random.Generator) -> None:
    """SpikesCalc returns scalar Data with shape ()."""
    arr = rng.standard_normal((30, 2))
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikesCalc().apply(data)

    assert out.data.shape == ()
    assert out.data.dims == ()
    # to_numpy() returns a 0-d numpy array, extract scalar with item()
    val = out.to_numpy().item()
    assert isinstance(val, (int, float, np.integer, np.floating))


def test_spikes_calc_multivariate_data(rng: np.random.Generator) -> None:
    """SpikesCalc works on multivariate data (multiple channels)."""
    # 3 channels with different spike patterns
    arr = rng.standard_normal((100, 3)) * 10
    arr[20:25, 0] = 500  # Spikes in channel 0
    arr[60, 2] = -300  # Spike in channel 2

    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikesCalc().apply(data)

    assert isinstance(out, cb.Data)
    assert out.to_numpy() > 0


def test_spikes_calc_empty_data_raises() -> None:
    """SpikesCalc raises ValueError for empty input data."""
    arr = np.array([]).reshape(0, 0)
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="empty"):
        cb.feature.SpikesCalc().apply(data)


def test_spikes_calc_sampling_rate_none(rng: np.random.Generator) -> None:
    """SpikesCalc sets sampling_rate to None when time dimension is removed."""
    arr = rng.standard_normal((50, 2))
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikesCalc().apply(data)

    assert out.sampling_rate is None


def test_spikes_calc_does_not_mutate_input(rng: np.random.Generator) -> None:
    """SpikesCalc does not modify the input Data object."""
    arr = rng.standard_normal((50, 2))
    data = cb.Data.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01")

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.SpikesCalc().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"


def test_spikes_calc_boundary_values() -> None:
    """Values exactly at IQR bounds are not counted as spikes."""
    # Create data where Q1=25, Q3=75, IQR=50
    # Bounds: low = 25 - 75 = -50, high = 75 + 75 = 150
    arr = np.array([25.0, 75.0, 0.0, 100.0, -50.0, 150.0])  # Values at bounds

    data = cb.Data.from_numpy(arr.reshape(-1, 1), dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SpikesCalc().apply(data)

    # Values exactly at bounds should NOT be spikes
    # But anything beyond should be
    assert out.to_numpy() == 0
