"""Tests for the LineLength feature behavior."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb


def test_feature_line_length_expected_values_and_history() -> None:
    """LineLength computes absolute temporal differences and wraps to Data."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    # Expected line length per channel:
    # ch0: |1-0| + |3-1| + |6-3| = 1 + 2 + 3 = 6
    # ch1: |3-1| + |2-3| + |2-2| = 2 + 1 + 0 = 3
    expected = np.array([[6.0], [3.0]])

    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-02"
    )
    out = cb.feature.LineLength().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 1)
    np.testing.assert_allclose(out.to_numpy(), expected)
    assert out.subjectID == "sub-02"
    assert out.sampling_rate == 200.0
    assert out.history == ["LineLength"]


def test_feature_line_length_single_channel_timeseries() -> None:
    """LineLength works for a single-channel (1D) signal represented as time x 1."""
    arr = np.array([[0.0], [2.0], [-1.0], [3.0]])
    # |2-0| + |-1-2| + |3-(-1)| = 2 + 3 + 4 = 9
    expected = np.array([[9.0]])

    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.LineLength().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (1, 1)
    np.testing.assert_allclose(out.to_numpy(), expected)
    assert out.history == ["LineLength"]


def test_feature_line_length_raises_when_time_dim_missing() -> None:
    """LineLength raises ValueError when the underlying DataArray lacks 'time'."""

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])

    with pytest.raises(ValueError, match="must have 'time' dimension"):
        cb.feature.LineLength().__call__(_FakeData())  # type: ignore[arg-type]


def test_feature_line_length_via_chord() -> None:
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
