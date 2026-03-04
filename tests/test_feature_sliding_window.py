"""Tests for the SlidingWindow splitter feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _windows(data: cb.Data, window_size: int = 4, step_size: int = 2) -> list[cb.Data]:
    return list(cb.feature.SlidingWindow(window_size=window_size, step_size=step_size)(data))


def test_sliding_window_yields_correct_number_of_windows() -> None:
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    windows = _windows(data)
    assert len(windows) == 4  # (10 - 4) // 2 + 1


def test_sliding_window_yields_correct_shape_and_values() -> None:
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    windows = _windows(data)

    assert windows[0].data.shape == (2, 4)
    np.testing.assert_allclose(windows[0].to_numpy(), arr[0:4].T)
    np.testing.assert_allclose(windows[1].to_numpy(), arr[2:6].T)
    np.testing.assert_allclose(windows[2].to_numpy(), arr[4:8].T)
    np.testing.assert_allclose(windows[3].to_numpy(), arr[6:10].T)


def test_sliding_window_each_window_is_data() -> None:
    data = cb.from_numpy(np.ones((10, 2)), dims=["time", "space"])
    for w in cb.feature.SlidingWindow(window_size=4, step_size=2)(data):
        assert isinstance(w, cb.Data)


def test_sliding_window_preserves_metadata() -> None:
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
    )
    for w in cb.feature.SlidingWindow(window_size=4, step_size=2)(data):
        assert w.subjectID == "sub-01"
        assert w.groupID == "patient"
        assert w.condition == "rest"
        assert w.sampling_rate == 100.0
        assert w.history == ["SlidingWindow"]


def test_sliding_window_raises_when_time_dim_missing() -> None:
    import xarray as xr

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])

    with pytest.raises(ValueError, match="must have 'time' dimension"):
        list(
            cb.feature.SlidingWindow()(  # type: ignore[arg-type]
                _FakeData()
            )
        )


def test_sliding_window_raises_when_window_too_large() -> None:
    data = cb.from_numpy(np.ones((5, 2)), dims=["time", "space"])
    with pytest.raises(ValueError, match="window_size"):
        list(cb.feature.SlidingWindow(window_size=10, step_size=1)(data))


def test_sliding_window_is_lazy() -> None:
    """Generator should not materialise all windows upfront."""
    data = cb.from_numpy(np.ones((100, 2)), dims=["time", "space"])
    gen = cb.feature.SlidingWindow(window_size=10, step_size=1)(data)
    import inspect

    assert inspect.isgenerator(gen)
