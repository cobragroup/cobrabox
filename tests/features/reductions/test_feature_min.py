"""Tests for the Min feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_min_reduces_extra_dimension() -> None:
    """Min reduces an extra dimension (run_index) and updates history."""
    import xarray as xr

    arr = np.arange(24, dtype=float).reshape(3, 4, 2)  # run_index, time, space
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(
        xr_data, sampling_rate=100.0, subjectID="sub-01", groupID="group-A", condition="rest"
    )

    out = cb.feature.Min(dim="run_index").apply(data)

    assert isinstance(out, cb.Data)
    assert "run_index" not in out.data.dims
    assert out.data.shape == (2, 4)
    np.testing.assert_allclose(out.to_numpy(), arr.min(axis=0).T)
    # Metadata preservation
    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(100.0)
    assert out.history == ["Min"]


def test_min_raises_for_unknown_dimension() -> None:
    """Min raises a clear error when dim is missing."""
    data = cb.SignalData.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.Min(dim="band_index").apply(data)


def test_min_single_channel_timeseries_returns_single_value() -> None:
    """Min over time on a single-channel signal returns exactly one value."""
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Min(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (1, 1)
    assert out.to_numpy().size == 1
    np.testing.assert_allclose(out.to_numpy(), np.array([[1.0]]))
    assert out.history == ["Min"]


def test_min_finds_smallest_value_with_negative_numbers() -> None:
    """Min over time returns the true smallest value per channel."""
    arr = np.array([[2.0, -1.0], [-5.0, 4.0], [3.0, -7.0], [1.0, 0.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Min(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.shape == (2, 1)
    np.testing.assert_allclose(out.to_numpy(), np.array([[-5.0], [-7.0]]))


def test_min_does_not_mutate_input() -> None:
    """Min.apply() leaves the input Data object unchanged."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dims=["time", "space"], sampling_rate=100.0
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Min(dim="time").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_min_preserves_metadata() -> None:
    """Min preserves all metadata fields correctly."""
    data = cb.SignalData.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.Min(dim="time").apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    # sampling_rate is preserved from input
    assert result.sampling_rate == 250.0
