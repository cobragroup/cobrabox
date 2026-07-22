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
        window = cb.LineLength().apply(window)
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
        split=cb.SlidingWindow(window_size=4, step_size=2),
        pipeline=cb.LineLength(),
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
    assert cb.ConcatAggregate is cb.ConcatAggregate


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


def test_concat_aggregate_labels_window_axis_with_time() -> None:
    """Windows from SlidingWindow give the window axis real start times (issue #118)."""
    data = _make_data(n_time=100, n_space=3)
    windows = list(cb.feature.SlidingWindow(window_size=20, step_size=10)(data))

    result = cb.feature.ConcatAggregate()(data, iter(windows))

    # 100 Hz, step of 10 samples → a window every 0.1 s
    np.testing.assert_allclose(
        result.data.coords["window"].values, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    # window_end is the last sample of each window: start + 19 samples
    np.testing.assert_allclose(
        result.data.coords["window_end"].values,
        [0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89, 0.99],
    )


def test_concat_aggregate_window_times_survive_time_reducing_feature() -> None:
    """A chord whose per-window feature drops time still yields a time-labelled axis."""
    data = _make_data(n_time=100, n_space=3)
    chord = (
        cb.feature.SlidingWindow(window_size=20, step_size=10)
        | cb.feature.LineLength()
        | cb.feature.ConcatAggregate()
    )

    result = chord.apply(data)

    np.testing.assert_allclose(
        result.data.coords["window"].values, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )


def test_concat_aggregate_locates_an_event_on_the_window_axis() -> None:
    """The motivating use case: map an event time onto a window (issue #118)."""
    data = _make_data(n_time=1000, n_space=3)
    chord = (
        cb.feature.SlidingWindow(window_size=100, step_size=50)
        | cb.feature.LineLength()
        | cb.feature.ConcatAggregate()
    )

    result = chord.apply(data)

    starts = result.data.coords["window"].values
    onset = 4.2  # seizure onset in seconds
    index = int(np.searchsorted(starts, onset, side="right") - 1)
    assert starts[index] == pytest.approx(4.0)
    assert result.data.coords["window_end"].values[index] >= onset


def test_concat_aggregate_window_axis_is_not_reindexed_by_position() -> None:
    """The window coordinate carries time, so label-based selection uses seconds."""
    data = _make_data(n_time=100, n_space=3)
    chord = (
        cb.feature.SlidingWindow(window_size=20, step_size=10)
        | cb.feature.LineLength()
        | cb.feature.ConcatAggregate()
    )

    result = chord.apply(data)

    selected = result.data.sel(window=0.3)
    positional = result.data.isel(window=3)
    np.testing.assert_allclose(selected.values, positional.values)
