"""Tests for the Max feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_feature_max_reduces_requested_dimension() -> None:
    """Max reduces only the requested dimension and updates history."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01")

    wdata = cb.feature.SlidingWindow(window_size=4, step_size=2).apply(data)
    out = cb.feature.Max(dim="window_index").apply(wdata)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 4)

    expected = np.max(np.stack([arr[0:4], arr[2:6], arr[4:8], arr[6:10]], axis=0), axis=1)
    np.testing.assert_allclose(out.to_numpy(), expected.T)
    assert out.subjectID == "sub-01"
    assert out.history == ["SlidingWindow", "Max"]


def test_feature_max_raises_for_unknown_dimension() -> None:
    """Max raises a clear error when dim is missing."""
    data = cb.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.Max(dim="band_index").apply(data)


def test_feature_max_single_channel_timeseries_returns_single_value() -> None:
    """Max over time on a single-channel signal returns exactly one value."""
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Max(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (1, 1)
    assert out.to_numpy().size == 1
    np.testing.assert_allclose(out.to_numpy(), np.array([[4.0]]))
    assert out.history == ["Max"]
