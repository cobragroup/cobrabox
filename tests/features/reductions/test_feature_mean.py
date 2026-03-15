"""Tests for the Mean feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_mean_reduces_extra_dimension() -> None:
    """Mean reduces an extra dimension (run_index) and updates history."""
    import xarray as xr

    arr = np.arange(24, dtype=float).reshape(3, 4, 2)  # run_index, time, space
    xr_data = xr.DataArray(arr, dims=["run_index", "time", "space"])
    data = cb.SignalData(
        xr_data, sampling_rate=100.0, subjectID="sub-01", groupID="group-A", condition="rest"
    )

    out = cb.feature.Mean(dim="run_index").apply(data)

    assert isinstance(out, cb.Data)
    assert "run_index" not in out.data.dims
    assert out.data.shape == (2, 4)
    np.testing.assert_allclose(out.to_numpy(), arr.mean(axis=0).T)
    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate == pytest.approx(100.0)
    assert out.history == ["Mean"]


def test_mean_raises_for_unknown_dimension() -> None:
    """Mean raises a clear error when dim is missing."""
    data = cb.SignalData.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.Mean(dim="band_index").apply(data)


def test_mean_single_channel_timeseries_returns_single_value() -> None:
    """Mean over time on a single-channel signal returns exactly one value."""
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Mean(dim="time").apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (1, 1)
    assert out.to_numpy().size == 1
    np.testing.assert_allclose(out.to_numpy(), np.array([[2.5]]))
    assert out.history == ["Mean"]


def test_mean_metadata_preserved() -> None:
    """Mean preserves subjectID, groupID, condition, and sampling_rate."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["run_index", "time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="rest",
    )

    out = cb.feature.Mean(dim="run_index").apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    # sampling_rate is preserved if time dim exists, otherwise None
    if "time" in out.data.dims:
        assert out.sampling_rate == pytest.approx(100.0)


def test_mean_does_not_mutate_input() -> None:
    """Mean.apply() leaves the input Data object unchanged."""
    arr = np.arange(24, dtype=float).reshape(3, 4, 2)
    data = cb.SignalData.from_numpy(
        arr, dims=["run_index", "time", "space"], sampling_rate=100.0, subjectID="sub-01"
    )

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Mean(dim="run_index").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"
