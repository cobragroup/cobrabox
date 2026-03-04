"""Tests for the line_length feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def test_feature_line_length_expected_values_and_history() -> None:
    """line_length computes absolute temporal differences and wraps to Data."""
    arr = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0], [6.0, 2.0]])
    # Expected line length per channel:
    # ch0: |1-0| + |3-1| + |6-3| = 1 + 2 + 3 = 6
    # ch1: |3-1| + |2-3| + |2-2| = 2 + 1 + 0 = 3
    expected = np.array([[6.0], [3.0]])

    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-02")
    out = cb.feature.line_length(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 1)
    np.testing.assert_allclose(out.to_numpy(), expected)
    assert out.subjectID == "sub-02"
    assert out.sampling_rate == 200.0
    assert out.history == ["line_length"]


def test_feature_line_length_single_channel_timeseries() -> None:
    """line_length works for a single-channel (1D) signal represented as time x 1."""
    arr = np.array([[0.0], [2.0], [-1.0], [3.0]])
    # |2-0| + |-1-2| + |3-(-1)| = 2 + 3 + 4 = 9
    expected = np.array([[9.0]])

    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.line_length(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (1, 1)
    np.testing.assert_allclose(out.to_numpy(), expected)
    assert out.history == ["line_length"]


def test_feature_line_length_raises_when_time_dim_missing() -> None:
    """line_length raises ValueError when the underlying DataArray lacks 'time'."""
    import xarray as xr

    from cobrabox.features.line_length import line_length

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])

    with pytest.raises(ValueError, match="must have 'time' dimension"):
        line_length.__wrapped__(_FakeData())  # type: ignore[attr-defined]


def test_feature_line_length_with_window_index_preserved() -> None:
    """line_length preserves window_index dimension when data comes from sliding_window."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    wdata = cb.feature.sliding_window(data, window_size=5, step_size=2)  # type: ignore[attr-defined]

    out = cb.feature.line_length(wdata)  # type: ignore[attr-defined]

    assert isinstance(out, cb.Data)
    assert "window_index" in out.data.dims
    # _copy_with_new_data adds a singleton time dim; verify window_index is intact
    assert out.data.sizes["time"] == 1
