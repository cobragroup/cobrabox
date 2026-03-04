"""Tests for the sliding_window feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_feature_sliding_window_shapes_values_and_metadata() -> None:
    """sliding_window creates expected windows and preserves metadata/history."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
    )

    out = cb.feature.sliding_window(data, window_size=4, step_size=2)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "window_index", "time")
    assert out.data.shape == (2, 4, 4)
    np.testing.assert_allclose(out.data.isel(time=0).values, arr[0:4].T)
    np.testing.assert_allclose(out.data.isel(time=1).values, arr[2:6].T)

    assert out.subjectID == "sub-01"
    assert out.groupID == "patient"
    assert out.condition == "rest"
    assert out.sampling_rate == 100.0
    assert out.history == ["sliding_window"]


def test_feature_sliding_window_min_over_window_index_finds_smallest_per_window() -> None:
    """Reducing over window_index after sliding_window keeps one min per window."""
    arr = np.array([[5.0], [1.0], [7.0], [-2.0], [3.0], [0.0], [-9.0]])
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    wdata = cb.feature.sliding_window(data, window_size=3, step_size=1)
    out = cb.feature.min(wdata, dim="window_index")

    # Windows:
    # [5, 1, 7], [1, 7, -2], [7, -2, 3], [-2, 3, 0], [3, 0, -9]
    # Min per window: [ 1, -2, -2, -2, -9 ]
    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (1, 5)
    np.testing.assert_allclose(
        out.to_numpy().reshape(5, 1), np.array([[1.0], [-2.0], [-2.0], [-2.0], [-9.0]])
    )


def test_feature_sliding_window_raises_when_time_dim_missing() -> None:
    """sliding_window raises ValueError when the underlying DataArray lacks 'time'."""
    import xarray as xr

    from cobrabox.features.sliding_window import sliding_window

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])

    with pytest.raises(ValueError, match="must have 'time' dimension"):
        sliding_window.__wrapped__(_FakeData())  # type: ignore[attr-defined]


def test_feature_sliding_window_raises_when_window_too_large() -> None:
    """sliding_window raises ValueError when window_size exceeds the time axis length."""
    arr = np.ones((5, 2))
    data = cb.from_numpy(arr, dims=["time", "space"])
    with pytest.raises(ValueError, match="window_size"):
        cb.feature.sliding_window(data, window_size=10, step_size=1)


def test_feature_sliding_window_min_over_time_finds_smallest_per_local_index() -> None:
    """Reducing over time after sliding_window keeps one min per local index."""
    arr = np.array([[5.0], [1.0], [7.0], [-2.0], [3.0], [0.0], [-9.0]])
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    wdata = cb.feature.sliding_window(data, window_size=3, step_size=1)
    out = cb.feature.min(wdata, dim="time")

    # Windows over time:
    # [5, 1, 7], [1, 7, -2], [7, -2, 3], [-2, 3, 0], [3, 0, -9]
    # Min across windows at each local index: [ -2, -2, -9 ]
    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "window_index", "time")
    assert out.data.shape == (1, 3, 1)
    np.testing.assert_allclose(out.to_numpy().reshape(3, 1), np.array([[-2.0], [-2.0], [-9.0]]))
