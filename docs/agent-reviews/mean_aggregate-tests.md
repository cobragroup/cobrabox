# Test Plan: mean_aggregate

**Feature**: `src/cobrabox/features/mean_aggregate.py`
**Test file**: `tests/test_feature_mean_aggregate.py` (does not exist)
**Date**: 2026-03-04
**Verdict**: MISSING

## Summary

MeanAggregate is an AggregatorFeature that combines a stream of per-window Data objects by averaging across the window dimension. It stacks windows along a temporary 'window' dimension, computes the mean, and propagates per-window pipeline history. Tests must cover basic aggregation, history propagation, metadata preservation, empty stream error handling, and non-mutation guarantees.

## Proposed test file

```python
"""Tests for the MeanAggregate aggregator feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n_time: int = 10,
    n_space: int = 2,
    sampling_rate: float = 100.0,
    subjectID: str = "s1",
    groupID: str = "g1",
    condition: str = "rest",
    history: list[str] | None = None,
) -> cb.Data:
    """Create a Data object for testing."""
    arr = np.random.randn(n_time, n_space)
    return cb.Data.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=sampling_rate,
        subjectID=subjectID,
        groupID=groupID,
        condition=condition,
        history=history or [],
    )


def _make_windows(
    n_windows: int = 3,
    n_time: int = 5,
    n_space: int = 2,
    base_value: float = 0.0,
) -> list[cb.Data]:
    """Create a list of Data objects simulating windowed output."""
    windows = []
    for i in range(n_windows):
        arr = np.ones((n_time, n_space)) * (base_value + i)
        window = cb.Data.from_numpy(
            arr,
            dims=["time", "space"],
            sampling_rate=100.0,
            subjectID="win-sub",
            groupID="win-grp",
            condition="win-cond",
            history=["SlidingWindow", "LineLength"],  # Simulated pipeline history
        )
        windows.append(window)
    return windows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mean_aggregate_basic() -> None:
    """MeanAggregate averages values across windows correctly."""
    original_data = _make_data(n_time=5, n_space=2)
    # Create 3 windows with values 1.0, 2.0, 3.0 → mean should be 2.0
    windows = _make_windows(n_windows=3, base_value=1.0)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)
    assert result.data.shape == (2, 5)  # (space, time) - same as each window
    np.testing.assert_allclose(result.to_numpy(), np.ones((2, 5)) * 2.0)


def test_mean_aggregate_single_window() -> None:
    """MeanAggregate works with a single window (returns that window's data)."""
    original_data = _make_data(n_time=5, n_space=2)
    windows = _make_windows(n_windows=1, base_value=5.0)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert isinstance(result, cb.Data)
    np.testing.assert_allclose(result.to_numpy(), np.ones((2, 5)) * 5.0)


def test_mean_aggregate_empty_stream_raises() -> None:
    """MeanAggregate raises ValueError when stream is empty."""
    original_data = _make_data()

    aggregator = cb.feature.MeanAggregate()
    with pytest.raises(ValueError, match="empty stream"):
        aggregator(original_data, iter([]))


def test_mean_aggregate_preserves_original_metadata() -> None:
    """MeanAggregate preserves metadata from the original data argument."""
    original_data = _make_data(
        sampling_rate=250.0,
        subjectID="sub-42",
        groupID="patient",
        condition="task",
    )
    windows = _make_windows(n_windows=2)

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    assert result.subjectID == "sub-42"
    assert result.groupID == "patient"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)


def test_mean_aggregate_propagates_window_history() -> None:
    """MeanAggregate propagates per-window history and appends 'MeanAggregate'."""
    original_data = _make_data(history=["OriginalOp"])
    windows = _make_windows(n_windows=2)
    # Each window has history ["SlidingWindow", "LineLength"]

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    # Result should have: original history + window ops (excluding duplicates) + MeanAggregate
    assert "OriginalOp" in result.history
    assert "SlidingWindow" in result.history
    assert "LineLength" in result.history
    assert result.history[-1] == "MeanAggregate"


def test_mean_aggregate_no_duplicate_history() -> None:
    """MeanAggregate avoids duplicating ops already in original data history."""
    # If original data already has some ops that windows also have, don't duplicate
    original_data = _make_data(history=["SlidingWindow"])
    windows = _make_windows(n_windows=1)  # Windows also have SlidingWindow in history

    aggregator = cb.feature.MeanAggregate()
    result = aggregator(original_data, iter(windows))

    # SlidingWindow should appear only once (from original data, not duplicated)
    sliding_count = result.history.count("SlidingWindow")
    assert sliding_count == 1


def test_mean_aggregate_does_not_mutate_original() -> None:
    """MeanAggregate does not modify the original data or windows."""
    original_data = _make_data(history=["Original"])
    original_history = list(original_data.history)
    original_shape = original_data.data.shape

    windows = _make_windows(n_windows=2)
    window_histories = [list(w.history) for w in windows]
    window_shapes = [w.data.shape for w in windows]

    aggregator = cb.feature.MeanAggregate()
    _ = aggregator(original_data, iter(windows))

    # Original data unchanged
    assert original_data.history == original_history
    assert original_data.data.shape == original_shape

    # Windows unchanged
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


def test_mean_aggregate_with_different_shapes_raises() -> None:
    """MeanAggregate raises error if windows have incompatible shapes."""
    original_data = _make_data()
    # Create windows with different shapes
    window1 = cb.Data.from_numpy(np.ones((5, 2)), dims=["time", "space"])
    window2 = cb.Data.from_numpy(np.ones((3, 2)), dims=["time", "space"])  # Different time dim

    aggregator = cb.feature.MeanAggregate()
    # xarray.concat with join="override" may handle this, but test behavior
    with pytest.raises(Exception):  # ValueError or xarray error
        aggregator(original_data, iter([window1, window2]))
```

## Action List

1. [Severity: HIGH] Create missing test file `tests/test_feature_mean_aggregate.py` with the proposed content above.
