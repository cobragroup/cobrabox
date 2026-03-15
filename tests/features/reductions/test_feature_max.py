"""Tests for the Max feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_max_reduces_extra_dimension() -> None:
    """Max reduces an extra dimension (run_index) and updates history."""
    import xarray as xr

    arr = np.arange(24, dtype=float).reshape(3, 4, 2)  # run_index, time, space
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(xr_data, sampling_rate=100.0, subjectID="sub-01")

    out = cb.feature.Max(dim="run_index").apply(data)

    assert isinstance(out, cb.Data)
    assert "run_index" not in out.data.dims
    assert out.data.shape == (2, 4)
    np.testing.assert_allclose(out.to_numpy(), arr.max(axis=0).T)
    assert out.subjectID == "sub-01"
    assert out.history == ["Max"]


def test_max_raises_for_unknown_dimension() -> None:
    """Max raises a clear error when dim is missing."""
    data = cb.SignalData.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.Max(dim="band_index").apply(data)


def test_max_single_channel_timeseries_returns_single_value() -> None:
    """Max over time on a single-channel signal returns exactly one value."""
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Max(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (1,)
    np.testing.assert_allclose(out.to_numpy(), np.array([4.0]))
    assert out.history == ["Max"]


def test_max_does_not_mutate_input() -> None:
    """Max.apply() leaves the input Data object unchanged."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s1",
        groupID="g1",
        condition="rest",
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Max(dim="time").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_max_preserves_metadata() -> None:
    """Max preserves subjectID, groupID, condition; sampling_rate becomes None when time removed."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.Max(dim="time").apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    # Max removes the time dimension, so sampling_rate should be None
    assert result.sampling_rate is None


def test_max_sampling_rate_preserved_when_time_kept() -> None:
    """Max preserves sampling_rate when time dimension is not the reduced dimension."""
    import xarray as xr

    arr = np.arange(24, dtype=float).reshape(3, 4, 2)  # run_index, time, space
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(xr_data, sampling_rate=100.0, subjectID="s1")

    result = cb.feature.Max(dim="run_index").apply(data)

    # Time dimension is preserved, so sampling_rate should be kept
    assert result.sampling_rate == 100.0
