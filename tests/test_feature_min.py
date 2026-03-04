"""Tests for the Min feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_feature_min_reduces_requested_dimension() -> None:
    """Min reduces only the requested dimension and updates history."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01")

    wdata = cb.feature.SlidingWindow(window_size=4, step_size=2).apply(data)
    out = cb.feature.Min(dim="window_index").apply(wdata)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 4)

    expected = np.min(np.stack([arr[0:4], arr[2:6], arr[4:8], arr[6:10]], axis=0), axis=1)
    np.testing.assert_allclose(out.to_numpy(), expected.T)
    assert out.subjectID == "sub-01"
    assert out.history == ["SlidingWindow", "Min"]


def test_feature_min_raises_for_unknown_dimension() -> None:
    """Min raises a clear error when dim is missing."""
    data = cb.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.Min(dim="band_index").apply(data)


def test_feature_min_single_channel_timeseries_returns_single_value() -> None:
    """Min over time on a single-channel signal returns exactly one value."""
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Min(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (1, 1)
    assert out.to_numpy().size == 1
    np.testing.assert_allclose(out.to_numpy(), np.array([[1.0]]))
    assert out.history == ["Min"]


def test_feature_min_finds_smallest_value_with_negative_numbers() -> None:
    """Min over time returns the true smallest value per channel."""
    arr = np.array([[2.0, -1.0], [-5.0, 4.0], [3.0, -7.0], [1.0, 0.0]])
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Min(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.shape == (2, 1)
    np.testing.assert_allclose(out.to_numpy(), np.array([[-5.0], [-7.0]]))
