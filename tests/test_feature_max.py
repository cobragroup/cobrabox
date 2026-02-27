"""Tests for the max feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_feature_max_reduces_requested_dimension() -> None:
    """max reduces only the requested dimension and updates history."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01")

    wdata = cb.feature.sliding_window(data, window_size=4, step_size=2)
    out = cb.feature.max(wdata, dim="window_index")

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("time", "space")
    assert out.data.shape == (4, 2)

    expected = np.max(np.stack([arr[0:4], arr[2:6], arr[4:8], arr[6:10]], axis=0), axis=0)
    np.testing.assert_allclose(out.to_numpy(), expected)
    assert out.subjectID == "sub-01"
    assert out.history == ["sliding_window", "max"]


def test_feature_max_raises_for_unknown_dimension() -> None:
    """max raises a clear error when dim is missing."""
    data = cb.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.max(data, dim="band_index")


def test_feature_max_single_channel_timeseries_returns_single_value() -> None:
    """max over time on a single-channel signal returns exactly one value."""
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.max(data, dim="time")

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("time", "space")
    assert out.data.shape == (1, 1)
    assert out.to_numpy().size == 1
    np.testing.assert_allclose(out.to_numpy(), np.array([[4.0]]))
    assert out.history == ["max"]
