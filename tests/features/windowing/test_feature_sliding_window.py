"""Tests for the SlidingWindow splitter feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _windows(data: cb.Data, window_size: int = 4, step_size: int = 2) -> list[cb.Data]:
    return list(cb.feature.SlidingWindow(window_size=window_size, step_size=step_size)(data))


def test_sliding_window_yields_correct_number_of_windows() -> None:
    """SlidingWindow yields the expected number of windows."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    windows = _windows(data)
    assert len(windows) == 4  # (10 - 4) // 2 + 1


def test_sliding_window_yields_correct_shape_and_values() -> None:
    """Each window has correct shape and contains expected values."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    windows = _windows(data)

    assert windows[0].data.shape == (2, 4)
    np.testing.assert_allclose(windows[0].to_numpy(), arr[0:4].T)
    np.testing.assert_allclose(windows[1].to_numpy(), arr[2:6].T)
    np.testing.assert_allclose(windows[2].to_numpy(), arr[4:8].T)
    np.testing.assert_allclose(windows[3].to_numpy(), arr[6:10].T)


def test_sliding_window_each_window_is_data() -> None:
    """Each yielded window is a Data instance."""
    data = cb.SignalData.from_numpy(np.ones((10, 2)), dims=["time", "space"])
    for w in cb.feature.SlidingWindow(window_size=4, step_size=2)(data):
        assert isinstance(w, cb.Data)


def test_sliding_window_preserves_metadata() -> None:
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(
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


def test_sliding_window_raises_when_window_too_large() -> None:
    """SlidingWindow raises ValueError when window_size exceeds signal length."""
    data = cb.SignalData.from_numpy(np.ones((5, 2)), dims=["time", "space"])
    with pytest.raises(ValueError, match="window_size"):
        list(cb.feature.SlidingWindow(window_size=10, step_size=1)(data))


def test_sliding_window_raises_when_window_size_less_than_one() -> None:
    """SlidingWindow raises ValueError for window_size < 1."""
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        cb.feature.SlidingWindow(window_size=0)


def test_sliding_window_raises_when_step_size_less_than_one() -> None:
    """SlidingWindow raises ValueError for step_size < 1."""
    with pytest.raises(ValueError, match="step_size must be >= 1"):
        cb.feature.SlidingWindow(step_size=0)


def test_sliding_window_is_lazy() -> None:
    """Generator should not materialise all windows upfront."""
    data = cb.SignalData.from_numpy(np.ones((100, 2)), dims=["time", "space"])
    gen = cb.feature.SlidingWindow(window_size=10, step_size=1)(data)
    import inspect

    assert inspect.isgenerator(gen)


def test_sliding_window_does_not_mutate_input() -> None:
    """SlidingWindow does not modify the input Data object."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="patient",
        condition="rest",
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = list(cb.feature.SlidingWindow(window_size=4, step_size=2)(data))

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "sub-01"
    assert data.groupID == "patient"
    assert data.condition == "rest"
    assert data.sampling_rate == 100.0


def test_sliding_window_records_window_times_in_extra() -> None:
    """Each window carries its start/end time on the original axis (issue #118)."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    windows = _windows(data, window_size=4, step_size=2)

    # 100 Hz → sample i is at i/100 s; window k starts at sample 2k, ends at sample 2k+3
    assert [w.extra["window_start"] for w in windows] == pytest.approx([0.0, 0.02, 0.04, 0.06])
    assert [w.extra["window_end"] for w in windows] == pytest.approx([0.03, 0.05, 0.07, 0.09])


def test_sliding_window_times_survive_a_time_reducing_feature() -> None:
    """Window times outlive features that consume the time dimension."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    reduced = [cb.feature.LineLength().apply(w) for w in _windows(data)]

    assert [r.extra["window_start"] for r in reduced] == pytest.approx([0.0, 0.02, 0.04, 0.06])


def test_sliding_window_times_fall_back_to_indices_without_sampling_rate() -> None:
    """Without a sampling rate the time coord is sample indices, and so are window times."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"])
    windows = _windows(data, window_size=4, step_size=2)

    assert [w.extra["window_start"] for w in windows] == pytest.approx([0.0, 2.0, 4.0, 6.0])
    assert [w.extra["window_end"] for w in windows] == pytest.approx([3.0, 5.0, 7.0, 9.0])
