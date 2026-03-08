"""Tests for the SlidingWindowReduce feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_sliding_window_reduce_basic_mean() -> None:
    """SlidingWindowReduce with mean produces correct windowed means."""
    arr = np.arange(100).reshape(1, 100)  # 1 channel, 0-99
    data = cb.SignalData.from_numpy(arr, dims=["channel", "time"], sampling_rate=100.0)

    result = cb.feature.SlidingWindowReduce(
        window_size=10, step_size=10, dim="time", agg="mean"
    ).apply(data)

    assert result.data.shape == (1, 10)  # channel, window
    expected_means = np.arange(4.5, 100, 10).reshape(1, 10)
    np.testing.assert_allclose(result.to_numpy(), expected_means)


def test_sliding_window_reduce_returns_data_with_window_dim() -> None:
    """Result has 'window' dimension, not 'time'."""
    data = cb.SignalData.from_numpy(np.ones((3, 50)), dims=["channel", "time"], sampling_rate=100.0)

    result = cb.feature.SlidingWindowReduce(
        window_size=10, step_size=10, dim="time", agg="mean"
    ).apply(data)

    assert "time" not in result.data.dims
    assert "window" in result.data.dims
    assert result.data.sizes["window"] == 5  # (50 - 10) // 10 + 1


def test_sliding_window_reduce_all_aggregations() -> None:
    """All aggregation functions work correctly."""
    arr = np.arange(20).reshape(1, 20)  # 0-19
    data = cb.SignalData.from_numpy(arr, dims=["channel", "time"], sampling_rate=100.0)

    aggs = ["mean", "std", "sum", "min", "max"]
    for agg in aggs:
        result = cb.feature.SlidingWindowReduce(
            window_size=5,
            step_size=5,
            dim="time",
            agg=agg,  # type: ignore[arg-type]
        ).apply(data)

        assert result.data.shape == (1, 4)  # 4 windows
        assert "window" in result.data.dims

        # Verify values for first window (0-4)
        first_window = result.data.isel(window=0).values
        if agg == "mean":
            np.testing.assert_allclose(first_window, [2.0])
        elif agg == "sum":
            np.testing.assert_allclose(first_window, [10.0])
        elif agg == "min":
            np.testing.assert_allclose(first_window, [0.0])
        elif agg == "max":
            np.testing.assert_allclose(first_window, [4.0])


def test_sliding_window_reduce_overlapping_windows() -> None:
    """SlidingWindowReduce works with overlapping windows (step < window_size)."""
    arr = np.arange(20).reshape(1, 20)
    data = cb.SignalData.from_numpy(arr, dims=["channel", "time"], sampling_rate=100.0)

    result = cb.feature.SlidingWindowReduce(
        window_size=5, step_size=2, dim="time", agg="mean"
    ).apply(data)

    # (20 - 5) // 2 + 1 = 8 windows
    assert result.data.sizes["window"] == 8
    assert result.data.shape == (1, 8)


def test_sliding_window_reduce_preserves_metadata() -> None:
    """Metadata is preserved in the output (sampling_rate becomes None since time is reduced)."""
    data = cb.SignalData.from_numpy(
        np.ones((2, 50)),
        dims=["channel", "time"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
    )

    result = cb.feature.SlidingWindowReduce(
        window_size=10, step_size=10, dim="time", agg="mean"
    ).apply(data)

    assert result.subjectID == "sub-01"
    assert result.groupID == "patient"
    assert result.condition == "rest"
    # sampling_rate becomes None since time dimension is reduced
    assert result.sampling_rate is None


def test_sliding_window_reduce_updates_history() -> None:
    """History is updated with feature name."""
    data = cb.SignalData.from_numpy(np.ones((2, 50)), dims=["channel", "time"], sampling_rate=100.0)

    result = cb.feature.SlidingWindowReduce(
        window_size=10, step_size=10, dim="time", agg="mean"
    ).apply(data)

    assert result.history == ["SlidingWindowReduce"]


def test_sliding_window_reduce_returns_data_type() -> None:
    """Output type is Data (not SignalData) since time dimension is reduced."""
    data = cb.SignalData.from_numpy(np.ones((2, 50)), dims=["channel", "time"], sampling_rate=100.0)

    result = cb.feature.SlidingWindowReduce(
        window_size=10, step_size=10, dim="time", agg="mean"
    ).apply(data)

    assert isinstance(result, cb.Data)
    assert not isinstance(result, cb.SignalData)


def test_sliding_window_reduce_other_dimension() -> None:
    """Can reduce over dimensions other than 'time'."""
    arr = np.arange(60).reshape(3, 20)  # 3 channels, 20 timepoints
    data = cb.SignalData.from_numpy(arr, dims=["channel", "time"], sampling_rate=100.0)

    result = cb.feature.SlidingWindowReduce(
        window_size=2, step_size=2, dim="channel", agg="mean"
    ).apply(data)

    assert "channel" not in result.data.dims
    assert "window" in result.data.dims
    assert result.data.sizes["window"] == 1  # (3 - 2) // 2 + 1 = 1
    assert result.data.shape == (1, 20)  # window, time


def test_sliding_window_reduce_raises_when_window_size_less_than_one() -> None:
    """Error when window_size < 1."""
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        cb.feature.SlidingWindowReduce(window_size=0)


def test_sliding_window_reduce_raises_when_step_size_less_than_one() -> None:
    """Error when step_size < 1."""
    with pytest.raises(ValueError, match="step_size must be >= 1"):
        cb.feature.SlidingWindowReduce(step_size=0)


def test_sliding_window_reduce_raises_for_invalid_agg() -> None:
    """Error when agg is not one of the valid options."""
    with pytest.raises(ValueError, match="agg must be one of"):
        cb.feature.SlidingWindowReduce(agg="median")  # type: ignore[arg-type]


def test_sliding_window_reduce_raises_when_window_too_large() -> None:
    """Error when window_size exceeds dimension length."""
    data = cb.SignalData.from_numpy(np.ones((5, 10)), dims=["channel", "time"], sampling_rate=100.0)

    with pytest.raises(ValueError, match=r"window_size.*must be <=.*time"):
        cb.feature.SlidingWindowReduce(window_size=20, dim="time", agg="mean").apply(data)


def test_sliding_window_reduce_raises_for_unknown_dimension() -> None:
    """Error when dim is not present in the data."""
    data = cb.SignalData.from_numpy(np.ones((5, 10)), dims=["channel", "time"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="dim 'frequency' not found"):
        cb.feature.SlidingWindowReduce(window_size=5, dim="frequency", agg="mean").apply(data)


def test_sliding_window_reduce_does_not_mutate_input() -> None:
    """SlidingWindowReduce does not modify the input Data object."""
    data = cb.SignalData.from_numpy(np.ones((2, 50)), dims=["channel", "time"], sampling_rate=100.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.SlidingWindowReduce(window_size=10, step_size=10, dim="time", agg="mean").apply(
        data
    )

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
