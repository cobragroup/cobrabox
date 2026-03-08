"""Tests for the ConcatAggregate aggregator feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _make_data(
    n_time: int = 10,
    n_space: int = 2,
    sampling_rate: float = 100.0,
    subjectID: str = "s1",
    groupID: str = "g1",
    condition: str = "rest",
) -> cb.SignalData:
    arr = np.random.default_rng(42).standard_normal((n_time, n_space))
    return cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=sampling_rate,
        subjectID=subjectID,
        groupID=groupID,
        condition=condition,
    )


def _make_windows(n_windows: int = 3, n_time: int = 5, n_space: int = 2) -> list[cb.Data]:
    windows = []
    for i in range(n_windows):
        arr = np.ones((n_time, n_space)) * i
        window = cb.SignalData.from_numpy(
            arr,
            dims=["time", "space"],
            sampling_rate=100.0,
            subjectID="win-sub",
            groupID="win-grp",
            condition="win-cond",
        )
        window = cb.feature.LineLength().apply(window)
        windows.append(window)
    return windows


def test_concat_aggregate_basic() -> None:
    """ConcatAggregate stacks windows along a new 'window' dimension."""
    original_data = _make_data(n_time=5, n_space=2)
    windows = _make_windows(n_windows=3)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)
    assert "window" in result.data.dims
    assert result.data.sizes["window"] == 3
    assert not np.any(np.isnan(result.data.values))


def test_concat_aggregate_shape() -> None:
    """Result has window dim prepended to per-window shape."""
    original_data = _make_data(n_space=4)
    windows = _make_windows(n_windows=5, n_space=4)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    # LineLength reduces time dim → shape is (n_space,), stacked → (n_windows, n_space)
    assert result.data.shape == (5, 4)


def test_concat_aggregate_integer_coordinates() -> None:
    """Window coordinates are integer indices 0, 1, 2, ..."""
    original_data = _make_data()
    windows = _make_windows(n_windows=4)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    np.testing.assert_array_equal(result.data.coords["window"].values, [0, 1, 2, 3])


def test_concat_aggregate_single_window() -> None:
    """ConcatAggregate works with a single window."""
    original_data = _make_data()
    windows = _make_windows(n_windows=1)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    assert result.data.sizes["window"] == 1


def test_concat_aggregate_empty_stream_raises() -> None:
    """ConcatAggregate raises ValueError when stream is empty."""
    original_data = _make_data()

    aggregator = cb.ConcatAggregate()
    with pytest.raises(ValueError, match="empty stream"):
        aggregator(original_data, iter([]))


def test_concat_aggregate_preserves_original_metadata() -> None:
    """ConcatAggregate preserves metadata from the original data argument."""
    original_data = _make_data(
        sampling_rate=250.0, subjectID="sub-42", groupID="patient", condition="task"
    )
    windows = _make_windows(n_windows=2)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    assert result.subjectID == "sub-42"
    assert result.groupID == "patient"
    assert result.condition == "task"
    # After LineLength the time dimension is gone, so sampling_rate is None
    assert result.sampling_rate is None


def test_concat_aggregate_propagates_window_history() -> None:
    """ConcatAggregate propagates per-window history and appends 'ConcatAggregate'."""
    original_data = _make_data()
    windows = _make_windows(n_windows=2)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    assert "LineLength" in result.history
    assert result.history[-1] == "ConcatAggregate"


def test_concat_aggregate_does_not_mutate_original() -> None:
    """ConcatAggregate does not modify the original data or windows."""
    original_data = _make_data()
    original_history = list(original_data.history)
    original_shape = original_data.data.shape

    windows = _make_windows(n_windows=2)
    window_histories = [list(w.history) for w in windows]
    window_shapes = [w.data.shape for w in windows]

    aggregator = cb.ConcatAggregate()
    _ = aggregator(original_data, iter(windows))

    assert original_data.history == original_history
    assert original_data.data.shape == original_shape
    for i, w in enumerate(windows):
        assert w.history == window_histories[i]
        assert w.data.shape == window_shapes[i]


def test_concat_aggregate_returns_data_instance() -> None:
    """ConcatAggregate returns a Data instance."""
    original_data = _make_data()
    windows = _make_windows(n_windows=2)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)


def test_concat_aggregate_via_chord() -> None:
    """ConcatAggregate works correctly in a Chord pipeline."""
    data = cb.SignalData.from_numpy(
        np.arange(20, dtype=float).reshape(10, 2),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="chord-test",
    )

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=4, step_size=2),
        pipeline=cb.feature.LineLength(),
        aggregate=cb.ConcatAggregate(),
    )
    result = chord.apply(data)

    assert isinstance(result, cb.Data)
    assert "window" in result.data.dims
    assert result.subjectID == "chord-test"
    assert "LineLength" in result.history
    assert "ConcatAggregate" in result.history
    assert "SlidingWindow" in result.history


def test_concat_aggregate_accessible_via_cb_feature() -> None:
    """ConcatAggregate is accessible via cb.feature namespace."""
    assert hasattr(cb.feature, "ConcatAggregate")
    assert cb.feature.ConcatAggregate is cb.ConcatAggregate


def test_concat_aggregate_preserves_sampling_rate_with_time_dim() -> None:
    """ConcatAggregate preserves sampling_rate when windows still have a time dimension."""
    original_data = _make_data(n_time=5, n_space=2, sampling_rate=250.0)
    raw_windows = []
    for i in range(3):
        arr = np.ones((5, 2)) * i
        w = cb.SignalData.from_numpy(
            arr, dims=["time", "space"], sampling_rate=250.0, subjectID="s1"
        )
        raw_windows.append(w)

    aggregator = cb.ConcatAggregate()
    result = aggregator(original_data, iter(raw_windows))

    # Result has (window, time, space) — Data sees time dim and preserves sampling_rate
    assert "time" in result.data.dims
    assert result.sampling_rate == pytest.approx(250.0)
