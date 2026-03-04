"""Tests for the spikes_calc feature behavior."""

from __future__ import annotations

import numpy as np

import cobrabox as cb


def test_spikes_calc_clean_data_no_outliers(rng: np.random.Generator) -> None:
    """spikes_calc returns 0 for clean data without outliers."""
    # Normal distribution data without extreme values
    arr = rng.standard_normal((100, 2)) * 10 + 50
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-01")

    out = cb.feature.spikes_calc(data, mandatory_arg=1)

    assert isinstance(out, cb.Data)
    assert out.data.shape == (1, 1)
    assert set(out.data.dims) == {"time", "space"}
    assert out.subjectID == "sub-01"
    assert out.history == ["spikes_calc"]
    # Clean data should have 0 or very few spikes (allow up to 5 for random draws)
    assert out.to_numpy().flat[0] <= 5


def test_spikes_calc_with_outliers() -> None:
    """spikes_calc detects outliers beyond IQR bounds."""
    # Create data with known outliers
    arr = np.ones((100, 2)) * 50  # Base normal values
    arr[10, 0] = 200  # Extreme spike
    arr[50, 1] = -150  # Extreme dip

    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)

    out = cb.feature.spikes_calc(data, mandatory_arg=1)

    assert isinstance(out, cb.Data)
    assert out.to_numpy().flat[0] >= 2  # At least the 2 extreme values


def test_spikes_calc_preserves_metadata(rng: np.random.Generator) -> None:
    """spikes_calc preserves metadata from input Data."""
    arr = rng.standard_normal((50, 3))
    data = cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-02",
        groupID="control",
        condition="task",
        extra={"task_name": "motor"},
    )

    out = cb.feature.spikes_calc(data, mandatory_arg=1)

    assert out.subjectID == "sub-02"
    assert out.groupID == "control"
    assert out.condition == "task"
    assert out.extra.get("task_name") == "motor"
    assert out.history == ["spikes_calc"]


def test_spikes_calc_returns_shape_1_1(rng: np.random.Generator) -> None:
    """spikes_calc returns Data with shape (1, 1)."""
    arr = rng.standard_normal((30, 2))
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.spikes_calc(data, mandatory_arg=1)

    assert out.data.shape == (1, 1)
    assert set(out.data.dims) == {"time", "space"}
    assert isinstance(out.to_numpy().flat[0], (int, float, np.integer, np.floating))


def test_spikes_calc_multivariate_data(rng: np.random.Generator) -> None:
    """spikes_calc works on multivariate data (multiple channels)."""
    # 3 channels with different spike patterns
    arr = rng.standard_normal((100, 3)) * 10
    arr[20:25, 0] = 500  # Spikes in channel 0
    arr[60, 2] = -300  # Spike in channel 2

    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.spikes_calc(data, mandatory_arg=1)

    assert isinstance(out, cb.Data)
    assert out.to_numpy().flat[0] > 0
