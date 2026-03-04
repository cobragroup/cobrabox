"""Tests for general Data container (no dimension requirements)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

RNG = np.random.default_rng(seed=42)


def test_data_from_numpy_basic() -> None:
    """Data.from_numpy creates a Data object with arbitrary dimensions."""
    a = RNG.standard_normal((100, 10))
    ds = cb.Data.from_numpy(a, dims=["time", "space"], sampling_rate=100.0)
    # Data preserves original dimension order
    assert ds.data.shape == (100, 10)
    assert ds.data.dims == ("time", "space")
    assert ds.sampling_rate == pytest.approx(100.0)
    np.testing.assert_array_almost_equal(ds.to_numpy(), a)


def test_data_from_numpy_1d() -> None:
    """Data.from_numpy accepts 1-D arrays."""
    a = RNG.standard_normal(20)
    ds = cb.Data.from_numpy(a, dims=["time"], sampling_rate=10.0)
    assert ds.data.shape == (20,)
    assert ds.data.dims == ("time",)


def test_data_from_numpy_no_time() -> None:
    """Data.from_numpy works without time dimension."""
    a = RNG.standard_normal((5, 3))
    ds = cb.Data.from_numpy(a, dims=["x", "y"])
    assert ds.data.shape == (5, 3)
    assert ds.data.dims == ("x", "y")
    assert ds.sampling_rate is None  # No time dimension


def test_data_from_xarray_basic() -> None:
    """Data.from_xarray wraps a DataArray with arbitrary dimensions."""
    ar = xr.DataArray(
        RNG.standard_normal((30, 6)),
        dims=["time", "space"],
        coords={"time": np.arange(30) / 100.0, "space": [f"ch{i}" for i in range(6)]},
    )
    ds = cb.Data.from_xarray(ar)
    # Data preserves original dimension order
    assert ds.data.shape == (30, 6)
    assert ds.data.dims == ("time", "space")
    assert ds.sampling_rate == pytest.approx(100.0)


def test_data_from_xarray_no_time() -> None:
    """Data.from_xarray works without time dimension."""
    ar = xr.DataArray(RNG.standard_normal((5, 3)), dims=["x", "y"])
    ds = cb.Data.from_xarray(ar)
    assert ds.data.shape == (5, 3)
    assert ds.data.dims == ("x", "y")
    assert ds.sampling_rate is None


def test_data_sampling_rate_none_without_time() -> None:
    """Data without time dimension has sampling_rate=None."""
    ar = xr.DataArray(np.ones((3, 2)), dims=["foo", "bar"])
    ds = cb.Data(ar)
    assert ds.sampling_rate is None


def test_data_invalid_sampling_rate() -> None:
    """Data rejects non-positive sampling_rate when provided."""
    with pytest.raises(ValueError, match="must be positive"):
        cb.Data.from_numpy(RNG.standard_normal((10, 5)), dims=["time", "space"], sampling_rate=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        cb.Data.from_numpy(RNG.standard_normal((10, 5)), dims=["time", "space"], sampling_rate=-1.0)


def test_data_from_numpy_dims_validation() -> None:
    """Data.from_numpy validates dims length matches array ndim."""
    with pytest.raises(ValueError, match="dims length must match"):
        cb.Data.from_numpy(RNG.standard_normal((10, 5, 2)), dims=["time", "space"])


def test_data_dtype_is_float64() -> None:
    """Data always stores float64 regardless of input dtype."""
    for dtype in [np.float32, np.int16, np.int32, np.float16]:
        a = np.ones((10, 4), dtype=dtype)
        ds = cb.Data.from_numpy(a, dims=["time", "space"])
        assert ds.data.dtype == np.float64, f"expected float64 for input dtype {dtype}"


def test_data_immutability() -> None:
    """Setting any attribute on a Data instance raises AttributeError."""
    ds = cb.Data.from_numpy(RNG.standard_normal((5, 2)), dims=["time", "space"])
    with pytest.raises(AttributeError, match="Cannot modify attribute"):
        ds.foo = "bar"


def test_data_to_pandas() -> None:
    """Data.to_pandas() returns a pandas DataFrame."""
    import pandas as pd

    ar = xr.DataArray(np.ones((4, 3)), dims=["time", "space"], name="signal")
    ds = cb.Data.from_xarray(ar)
    df = ds.to_pandas()
    assert isinstance(df, pd.DataFrame)
