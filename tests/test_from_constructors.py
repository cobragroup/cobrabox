"""Tests for Data constructors: from_numpy and from_xarray."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

RNG = np.random.default_rng(seed=42)


def test_from_numpy_basic() -> None:
    """from_numpy creates a Data object with correct shape and sampling_rate."""
    a = RNG.standard_normal((100, 10))
    ds = cb.from_numpy(a, dims=["time", "space"], sampling_rate=100.0)
    assert ds.data.shape == (10, 100)  # time is last
    assert ds.data.dims == ("space", "time")
    assert ds.sampling_rate == pytest.approx(100.0)
    np.testing.assert_array_almost_equal(ds.to_numpy(), a.T)


def test_from_numpy_with_metadata() -> None:
    """from_numpy accepts optional metadata."""
    a = RNG.standard_normal((50, 4))
    ds = cb.from_numpy(
        a, dims=["time", "space"], sampling_rate=50.0, subjectID="subj1", condition="rest"
    )
    assert ds.subjectID == "subj1"
    assert ds.condition == "rest"
    assert list(ds.data.coords["space"].values) == [0, 1, 2, 3]


def test_from_numpy_invalid_ndim() -> None:
    """from_numpy rejects arrays with fewer than 2 dimensions."""
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        cb.from_numpy(RNG.standard_normal(20), dims=["time"], sampling_rate=10.0)


def test_from_numpy_nd_array_allowed() -> None:
    """from_numpy accepts N-D arrays as long as dims include time and space."""
    a = RNG.standard_normal((10, 5, 3))
    ds = cb.from_numpy(a, dims=["time", "space", "dim_2"])
    assert ds.data.shape == (5, 3, 10)  # time is last
    assert ds.data.dims == ("space", "dim_2", "time")


def test_from_numpy_optional_sampling_rate() -> None:
    """from_numpy works without sampling_rate; ds.sampling_rate is then None."""
    a = RNG.standard_normal((20, 4))
    ds = cb.from_numpy(a, dims=["time", "space"])
    assert ds.data.shape == (4, 20)  # time is last
    assert ds.sampling_rate is None
    assert list(ds.data.coords["time"].values) == list(range(20))


def test_from_numpy_invalid_sampling_rate() -> None:
    """from_numpy rejects non-positive sampling_rate when provided."""
    with pytest.raises(ValueError, match="must be positive"):
        cb.from_numpy(RNG.standard_normal((10, 5)), dims=["time", "space"], sampling_rate=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        cb.from_numpy(RNG.standard_normal((10, 5)), dims=["time", "space"], sampling_rate=-1.0)


def test_from_numpy_dims_validation() -> None:
    """from_numpy validates dims length and required names."""
    with pytest.raises(ValueError, match="dims length must match array ndim"):
        cb.from_numpy(RNG.standard_normal((10, 5, 2)), dims=["time", "space"])
    with pytest.raises(ValueError, match="must include 'time'"):
        cb.from_numpy(RNG.standard_normal((10, 5, 2)), dims=["sample", "space", "z"])


def test_from_xarray_basic() -> None:
    """from_xarray wraps a DataArray with time and space dims."""
    ar = xr.DataArray(
        RNG.standard_normal((30, 6)),
        dims=["time", "space"],
        coords={"time": np.arange(30) / 100.0, "space": [f"ch{i}" for i in range(6)]},
    )
    ds = cb.from_xarray(ar)
    assert ds.data.shape == (6, 30)  # time is last
    assert ds.data.dims == ("space", "time")
    assert ds.sampling_rate == pytest.approx(100.0)
    np.testing.assert_array_almost_equal(ds.to_numpy(), ar.values.T)


def test_from_xarray_with_metadata() -> None:
    """from_xarray accepts optional metadata."""
    ar = xr.DataArray(
        RNG.standard_normal((10, 3)),
        dims=["time", "space"],
        coords={"time": np.arange(10) / 50.0, "space": ["x", "y", "z"]},
    )
    ds = cb.from_xarray(ar, subjectID="s1", groupID="g1")
    assert ds.subjectID == "s1"
    assert ds.groupID == "g1"


def test_asnumpy_gorka_style_returns_separate_arrays() -> None:
    """asnumpy(style='gorkastyle') returns (time, space, labels)."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    ds = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=10.0)

    time, space, labels = ds.to_numpy(style="gorkastyle")

    np.testing.assert_allclose(time, np.array([0.0, 0.1]))
    np.testing.assert_array_equal(space, np.array([0, 1]))
    np.testing.assert_allclose(labels, arr.T)  # stored as (space, time)


def test_asnumpy_invalid_style_raises() -> None:
    """asnumpy raises for unknown style names."""
    ds = cb.from_numpy(RNG.standard_normal((5, 2)), dims=["time", "space"])
    with pytest.raises(ValueError, match="Unknown style"):
        ds.to_numpy(style="something_else")


def test_asnumpy_gorka_style_alias_not_supported() -> None:
    """asnumpy no longer supports style='gorka_style' alias."""
    ds = cb.from_numpy(RNG.standard_normal((5, 2)), dims=["time", "space"])
    with pytest.raises(ValueError, match="Unknown style"):
        ds.to_numpy(style="gorka_style")


# --- dtype enforcement ---


def test_from_numpy_dtype_is_float64() -> None:
    """Data always stores float64 regardless of input dtype."""
    for dtype in [np.float32, np.int16, np.int32, np.float16]:
        a = np.ones((10, 4), dtype=dtype)
        ds = cb.from_numpy(a, dims=["time", "space"])
        assert ds.data.dtype == np.float64, f"expected float64 for input dtype {dtype}"


def test_from_xarray_dtype_is_float64() -> None:
    """from_xarray casts to float64."""
    ar = xr.DataArray(np.ones((10, 3), dtype=np.float32), dims=["time", "space"])
    ds = cb.from_xarray(ar)
    assert ds.data.dtype == np.float64


# --- time-last ordering ---


def test_time_is_last_dim_from_numpy() -> None:
    """time dimension is always the last dim after construction via from_numpy."""
    a = RNG.standard_normal((10, 5, 3))
    ds = cb.from_numpy(a, dims=["time", "space", "run_index"])
    assert ds.data.dims[-1] == "time"


def test_time_is_last_dim_from_xarray() -> None:
    """time dimension is always the last dim after construction via from_xarray."""
    ar = xr.DataArray(RNG.standard_normal((8, 4, 2)), dims=["time", "space", "band_index"])
    ds = cb.from_xarray(ar)
    assert ds.data.dims[-1] == "time"


def test_time_last_preserves_values() -> None:
    """Transposing time to last does not change the underlying values."""
    a = RNG.standard_normal((10, 4))
    ds = cb.from_numpy(a, dims=["time", "space"])
    # After transpose time is last; recover original (time, space) order for comparison
    recovered = ds.data.transpose("time", "space").values
    np.testing.assert_array_equal(recovered, a.astype(np.float64))


# --- Data.__init__ dimension validation ---


def test_data_init_missing_time_dim_raises() -> None:
    """Data.__init__ raises when DataArray lacks 'time' dimension."""
    ar = xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])
    with pytest.raises(ValueError, match="must have `time` dimension"):
        cb.from_xarray(ar)


def test_data_init_missing_space_dim_raises() -> None:
    """Data.__init__ raises when DataArray lacks 'space' dimension."""
    ar = xr.DataArray(np.ones((3, 2)), dims=["time", "foo"])
    with pytest.raises(ValueError, match="must have `space` dimension"):
        cb.from_xarray(ar)


# --- _infer_sampling_rate edge cases ---


def test_infer_sampling_rate_decreasing_time_returns_none() -> None:
    """Sampling rate is not inferred when time coordinates decrease."""
    ar = xr.DataArray(
        np.ones((3, 2)),
        dims=["time", "space"],
        coords={"time": [0.1, 0.05, 0.0], "space": [0, 1]},
    )
    ds = cb.from_xarray(ar)
    assert ds.sampling_rate is None


def test_infer_sampling_rate_uneven_time_returns_none() -> None:
    """Sampling rate is not inferred when time coordinates are unevenly spaced."""
    ar = xr.DataArray(
        np.ones((3, 2)),
        dims=["time", "space"],
        coords={"time": [0.0, 0.01, 0.05], "space": [0, 1]},
    )
    ds = cb.from_xarray(ar)
    assert ds.sampling_rate is None


# --- immutability ---


def test_immutability_guard_raises_on_setattr() -> None:
    """Setting any attribute on a Data instance raises AttributeError."""
    ds = cb.from_numpy(RNG.standard_normal((5, 2)), dims=["time", "space"])
    with pytest.raises(AttributeError, match="Cannot modify attribute"):
        ds.foo = "bar"  # type: ignore[attr-defined]


# --- to_pandas ---


def test_to_pandas_returns_dataframe() -> None:
    """to_pandas() returns a pandas DataFrame."""
    import pandas as pd
    import xarray as xr

    ar = xr.DataArray(
        np.ones((4, 3)),
        dims=["time", "space"],
        name="signal",
    )
    ds = cb.from_xarray(ar)
    df = ds.to_pandas()
    assert isinstance(df, pd.DataFrame)
