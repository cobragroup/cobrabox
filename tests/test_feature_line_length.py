"""Tests for the LineLength feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_line_length_expected_values_and_history() -> None:
    """LineLength computes absolute temporal differences and wraps to Data."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    # Expected line length per channel:
    # ch0: |1-0| + |3-1| + |6-3| = 1 + 2 + 3 = 6
    # ch1: |3-1| + |2-3| + |2-2| = 2 + 1 + 0 = 3
    expected = np.array([[6.0], [3.0]])

    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=200.0,
        subjectID="sub-02",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.LineLength().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (2,)
    np.testing.assert_allclose(out.to_numpy(), expected.ravel())
    assert out.subjectID == "sub-02"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    # sampling_rate is not preserved for Data without time dimension
    assert out.sampling_rate is None
    assert out.history == ["LineLength"]


def test_line_length_single_channel_timeseries() -> None:
    """LineLength works for a single-channel (1D) signal represented as time x 1."""
    arr = np.array([[0.0], [2.0], [-1.0], [3.0]])
    # |2-0| + |-1-2| + |3-(-1)| = 2 + 3 + 4 = 9
    expected = np.array([[9.0]])

    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.LineLength().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (1,)
    np.testing.assert_allclose(out.to_numpy(), expected.ravel())
    assert out.history == ["LineLength"]


def test_line_length_via_chord() -> None:
    """LineLength applied per window via Chord produces one result per window."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=5, step_size=2),
        pipeline=cb.feature.LineLength(),
        aggregate=cb.feature.MeanAggregate(),
    )
    out = chord.apply(data)

    assert isinstance(out, cb.Data)
    assert "LineLength" in out.history
    assert "MeanAggregate" in out.history


def test_line_length_missing_time_raises() -> None:
    """LineLength raises ValueError when time dimension is missing."""
    import xarray as xr

    bad_xr = xr.DataArray(np.random.default_rng(42).standard_normal((100, 10)), dims=["t", "space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError, match="time"):
        cb.feature.LineLength()(raw)


def test_line_length_does_not_mutate_input() -> None:
    """LineLength.apply() leaves the input Data object unchanged."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-02"
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.LineLength().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_line_length_returns_data_instance() -> None:
    """LineLength.apply() always returns a Data instance."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)
    result = cb.feature.LineLength().apply(data)
    assert isinstance(result, cb.Data)
