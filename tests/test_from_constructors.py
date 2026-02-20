"""Tests for Dataset constructors: from_numpy and from_xarray."""

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb


def test_from_numpy_basic():
    """from_numpy creates a Dataset with correct shape and sampling_rate."""
    a = np.random.randn(100, 10)
    ds = cb.from_numpy(a, sampling_rate=100.0)
    assert ds.data.shape == (100, 10)
    assert ds.data.dims == ("time", "space")
    assert ds.sampling_rate == pytest.approx(100.0)
    np.testing.assert_array_almost_equal(ds.asnumpy(), a)


def test_from_numpy_with_metadata():
    """from_numpy accepts optional metadata."""
    a = np.random.randn(50, 4)
    ds = cb.from_numpy(
        a,
        sampling_rate=50.0,
        space=["a", "b", "c", "d"],
        subjectID="subj1",
        condition="rest",
    )
    assert ds.subjectID == "subj1"
    assert ds.condition == "rest"
    assert list(ds.data.coords["space"].values) == ["a", "b", "c", "d"]


def test_from_numpy_invalid_ndim():
    """from_numpy rejects arrays with fewer than 2 dimensions."""
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        cb.from_numpy(np.random.randn(20), sampling_rate=10.0)


def test_from_numpy_nd_array_allowed():
    """from_numpy accepts N-D arrays as long as dims include time and space."""
    a = np.random.randn(10, 5, 3)
    ds = cb.from_numpy(a)
    assert ds.data.shape == (10, 5, 3)
    assert ds.data.dims == ("time", "space", "dim_2")


def test_from_numpy_optional_sampling_rate():
    """from_numpy works without sampling_rate; ds.sampling_rate is then None."""
    a = np.random.randn(20, 4)
    ds = cb.from_numpy(a)
    assert ds.data.shape == (20, 4)
    assert ds.sampling_rate is None
    assert list(ds.data.coords["time"].values) == list(range(20))


def test_from_numpy_invalid_sampling_rate():
    """from_numpy rejects non-positive sampling_rate when provided."""
    with pytest.raises(ValueError, match="must be positive"):
        cb.from_numpy(np.random.randn(10, 5), sampling_rate=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        cb.from_numpy(np.random.randn(10, 5), sampling_rate=-1.0)


def test_from_numpy_space_length():
    """from_numpy rejects space labels with wrong length."""
    with pytest.raises(ValueError, match="space length"):
        cb.from_numpy(np.random.randn(10, 5), sampling_rate=10.0, space=["a", "b"])


def test_from_numpy_time_and_space_arrays():
    """from_numpy accepts explicit time and space coordinates."""
    a = np.random.randn(8, 3)
    time = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
    space = ["Fp1", "Fp2", "C3"]
    ds = cb.from_numpy(a, time=time, space=space)
    assert list(ds.data.coords["time"].values) == list(time)
    assert list(ds.data.coords["space"].values) == space


def test_from_numpy_time_length():
    """from_numpy rejects time coordinates with wrong length."""
    with pytest.raises(ValueError, match="time length"):
        cb.from_numpy(np.random.randn(10, 5), time=np.arange(9))


def test_from_numpy_dims_validation():
    """from_numpy validates dims length and required names."""
    with pytest.raises(ValueError, match="dims length must match array ndim"):
        cb.from_numpy(np.random.randn(10, 5, 2), dims=["time", "space"])
    with pytest.raises(ValueError, match="must include 'time' and 'space'"):
        cb.from_numpy(np.random.randn(10, 5, 2), dims=["t", "s", "z"])


def test_from_xarray_basic():
    """from_xarray wraps a DataArray with time and space dims."""
    ar = xr.DataArray(
        np.random.randn(30, 6),
        dims=["time", "space"],
        coords={
            "time": np.arange(30) / 100.0,
            "space": [f"ch{i}" for i in range(6)],
        },
    )
    ds = cb.from_xarray(ar)
    assert ds.data.shape == (30, 6)
    assert ds.data.dims == ("time", "space")
    assert ds.sampling_rate == pytest.approx(100.0)
    np.testing.assert_array_almost_equal(ds.asnumpy(), ar.values)


def test_from_xarray_with_metadata():
    """from_xarray accepts optional metadata."""
    ar = xr.DataArray(
        np.random.randn(10, 3),
        dims=["time", "space"],
        coords={"time": np.arange(10) / 50.0, "space": ["x", "y", "z"]},
    )
    ds = cb.from_xarray(ar, subjectID="s1", groupID="g1")
    assert ds.subjectID == "s1"
    assert ds.groupID == "g1"


