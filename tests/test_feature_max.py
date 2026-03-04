"""Tests for the Max feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_feature_max_reduces_extra_dimension() -> None:
    """Max reduces an extra dimension (run_index) and updates history."""
    import xarray as xr

    arr = np.arange(24, dtype=float).reshape(3, 4, 2)  # run_index, time, space
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.Data(xr_data, sampling_rate=100.0, subjectID="sub-01")

    out = cb.feature.Max(dim="run_index").apply(data)

    assert isinstance(out, cb.Data)
    assert "run_index" not in out.data.dims
    assert out.data.shape == (2, 4)
    np.testing.assert_allclose(out.to_numpy(), arr.max(axis=0).T)
    assert out.subjectID == "sub-01"
    assert out.history == ["Max"]


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
