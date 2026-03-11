"""Tests for the MeanAggregate aggregator feature."""

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
) -> cb.Data:
    """Create a Data object for testing."""
    arr = np.random.default_rng(42).standard_normal((n_time, n_space))
    return cb.Data.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=sampling_rate,
        subjectID=subjectID,
        groupID=groupID,
        condition=condition,
    )


def _make_windows(
    n_windows: int = 3, n_time: int = 5, n_space: int = 2, base_value: float = 0.0
) -> list[cb.Data]:
    """Create a list of Data objects simulating windowed output."""
    windows = []
    for i in range(n_windows):
        arr = np.ones((n_time, n_space)) * (base_value + i)
        window = cb.SignalData.from_numpy(
            arr,
            dims=["time", "space"],
            sampling_rate=100.0,
            subjectID="win-sub",
            groupID="win-grp",
            condition="win-cond",
        )
        # Simulate pipeline history by applying LineLength first
        window = cb.feature.LineLength().apply(window)
        windows.append(window)
    return windows


def test_mean_aggregate_basic() -> None:
    """MeanAggregate averages values across windows correctly."""
    original_data = _make_data(n_time=5, n_space=2)
    windows = _make_windows(n_windows=3, base_value=1.0)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)
    assert result.data.shape == (2,)


def test_mean_aggregate_single_window() -> None:
    """MeanAggregate works with a single window."""
    original_data = _make_data(n_time=5, n_space=2)
    windows = _make_windows(n_windows=1, base_value=5.0)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)


def test_mean_aggregate_empty_stream_raises() -> None:
    """MeanAggregate raises ValueError when stream is empty."""
    original_data = _make_data()

    aggregator = cb.feature.MeanAggregate()
    with pytest.raises(ValueError, match="empty stream"):
        aggregator(original_data, iter([]))


def test_mean_aggregate_preserves_original_metadata() -> None:
    """MeanAggregate preserves metadata from the original data argument."""
    original_data = _make_data(
        sampling_rate=250.0, subjectID="sub-42", groupID="patient", condition="task"
    )
    windows = _make_windows(n_windows=2)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert result.subjectID == "sub-42"
    assert result.groupID == "patient"
    assert result.condition == "task"
    # sampling_rate is only meaningful for time-series data
    # After LineLength, windows don't have time dimension, so sampling_rate is None
    assert result.sampling_rate is None


def test_mean_aggregate_propagates_window_history() -> None:
    """MeanAggregate propagates per-window history and appends 'MeanAggregate'."""
    original_data = _make_data()
    windows = _make_windows(n_windows=2)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert "LineLength" in result.history
    assert result.history[-1] == "MeanAggregate"


def test_mean_aggregate_does_not_mutate_original() -> None:
    """MeanAggregate does not modify the original data or windows."""
    original_data = _make_data()
    original_history = list(original_data.history)
    original_shape = original_data.data.shape

    windows = _make_windows(n_windows=2)
    window_histories = [list(w.history) for w in windows]
    window_shapes = [w.data.shape for w in windows]

    aggregator = cb.feature.MeanAggregate()
    _ = aggregator(original_data, iter(windows))

    assert original_data.history == original_history
    assert original_data.data.shape == original_shape

    for i, w in enumerate(windows):
        assert w.history == window_histories[i]
        assert w.data.shape == window_shapes[i]


def test_mean_aggregate_returns_data_instance() -> None:
    """MeanAggregate returns a Data instance."""
    original_data = _make_data()
    windows = _make_windows(n_windows=2)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)


def test_mean_aggregate_via_chord() -> None:
    """MeanAggregate works correctly in a Chord pipeline."""
    data = cb.SignalData.from_numpy(
        np.arange(20, dtype=float).reshape(10, 2),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="chord-test",
    )

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=4, step_size=2),
        pipeline=cb.feature.LineLength(),
        aggregate=cb.feature.MeanAggregate(),
    )
    result = chord.apply(data)

    assert isinstance(result, cb.Data)
    assert result.subjectID == "chord-test"
    assert "LineLength" in result.history
    assert "MeanAggregate" in result.history
    assert "SlidingWindow" in result.history
