"""Tests for the Nonreversibility feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb
from cobrabox.features.nonreversibility import Nonreversibility


def test_nonreversibility_output_shape_dims_and_metadata() -> None:
    """Nonreversibility returns Data with space='d_norm', correct dims, and metadata."""
    rng = np.random.default_rng(seed=0)
    arr = rng.standard_normal((200, 5))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="sub-01",
        groupID="ctrl",
        condition="rest",
        extra={"tag": "eeg"},
    )

    out = cb.feature.Nonreversibility().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (1,)
    assert list(out.data.coords["space"].values) == ["d_norm"]
    assert out.subjectID == "sub-01"
    assert out.groupID == "ctrl"
    assert out.condition == "rest"
    assert out.sampling_rate is None  # Data without time dimension
    assert out.extra.get("tag") == "eeg"
    assert out.history == ["Nonreversibility"]


def test_nonreversibility_scalar_is_nonnegative() -> None:
    """d_norm is always >= 0 (ratio of Frobenius norms)."""
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal((300, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Nonreversibility().apply(data)
    d = float(out.data.values.flat[0])

    assert d >= 0.0


def test_nonreversibility_raises_on_single_channel() -> None:
    """Nonreversibility raises ValueError when space dimension has fewer than 2 channels."""
    arr = np.random.default_rng(seed=7).standard_normal((100, 1))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="at least 2 time series"):
        cb.feature.Nonreversibility().apply(data)


def test_nonreversibility_raises_on_missing_time_dim() -> None:
    """Nonreversibility raises ValueError when 'time' dimension is absent."""

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])

        sampling_rate = None

    with pytest.raises(ValueError, match="must have 'time' dimension"):
        Nonreversibility()(_FakeData())  # type: ignore[arg-type]


def test_nonreversibility_raises_on_too_few_timepoints() -> None:
    """Nonreversibility raises ValueError when there is only 1 timepoint."""
    arr = np.ones((1, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="at least 2 timepoints"):
        cb.feature.Nonreversibility().apply(data)
