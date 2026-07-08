"""Tests for the SVD feature behavior."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb


def _stack_to_matrix(da: xr.DataArray, dim: str) -> xr.DataArray:
    other_dims = [d for d in da.dims if d != dim]
    if other_dims:
        return da.stack(features=other_dims).transpose(dim, "features")
    return da.expand_dims(features=[0]).transpose(dim, "features")


def test_feature_svd_returns_dataarray_and_updates_history() -> None:
    arr = np.random.default_rng(0).normal(size=(20, 6)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "channel"], sampling_rate=1000.0, subjectID="sub-01")

    out = cb.feature.SVD(dim="time", n_components=5, center=True, return_unstacked_V=True).apply(
        data
    )

    assert isinstance(out, cb.Data)
    assert isinstance(out.data, xr.DataArray)
    assert out.data.name == "V"
    assert out.data.dims == ("component", "channel")
    assert out.data.shape == (5, 6)

    assert out.subjectID == "sub-01"
    assert out.history == ["SVD"]

    assert "svd" in out.data.attrs
    svd = out.data.attrs["svd"]
    assert svd["S"].shape == (5,)
    assert svd["Vh"].shape == (5, 6)
    assert svd["mean"] is not None
    assert svd["std"] is None


def test_feature_svd_unstacks_V_for_multidim_input() -> None:
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(12, 4, 3, 2)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "x", "y", "z"], sampling_rate=2.0)

    out = cb.feature.SVD(dim="time", n_components=4, center=True, return_unstacked_V=True).apply(
        data
    )

    assert isinstance(out.data, xr.DataArray)
    assert out.data.name == "V"
    assert out.data.dims == ("component", "x", "y", "z")
    assert out.data.shape == (4, 4, 3, 2)


def test_feature_svd_mask_reduces_feature_count() -> None:
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(25, 5, 4)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "x", "y"], sampling_rate=10.0)

    mask = xr.DataArray(rng.random(size=(5, 4)) > 0.5, dims=("x", "y"))

    out = cb.feature.SVD(
        dim="time", n_components=6, center=True, mask=mask, return_unstacked_V=False
    ).apply(data)

    assert out.data.name == "Vh"
    svd = out.data.attrs["svd"]

    n_keep = int(mask.values.astype(bool).sum())
    assert svd["Vh"].shape == (6, n_keep)
    assert svd["mean"] is not None
    assert svd["mean"].shape == (n_keep,)


def test_feature_svd_center_makes_feature_means_zero() -> None:
    rng = np.random.default_rng(3)
    arr = rng.normal(loc=10.0, scale=3.0, size=(40, 8)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    out = cb.feature.SVD(
        dim="time", n_components=5, center=True, zscore=False, return_unstacked_V=False
    ).apply(data)

    svd = out.data.attrs["svd"]
    X = _stack_to_matrix(data.data, "time")
    Xc = X - svd["mean"]

    means = Xc.mean("time").values
    assert np.allclose(means, 0.0, atol=1e-10)


def test_feature_svd_zscore_makes_means_zero_and_stds_one() -> None:
    rng = np.random.default_rng(4)
    arr = rng.normal(loc=5.0, scale=2.0, size=(60, 10)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "channel"], sampling_rate=250.0)

    out = cb.feature.SVD(dim="time", n_components=7, zscore=True, return_unstacked_V=False).apply(
        data
    )

    svd = out.data.attrs["svd"]
    assert svd["mean"] is not None
    assert svd["std"] is not None

    X = _stack_to_matrix(data.data, "time")
    Xz = (X - svd["mean"]) / svd["std"]

    means = Xz.mean("time").values
    stds = Xz.std("time").values

    assert np.allclose(means, 0.0, atol=1e-10)
    assert np.allclose(stds, 1.0, atol=1e-10)


def test_feature_svd_reconstruction_matches_centered_matrix_full_rank_case() -> None:
    rng = np.random.default_rng(5)
    arr = rng.normal(size=(30, 12)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    out = cb.feature.SVD(dim="time", n_components=12, center=True, return_unstacked_V=False).apply(
        data
    )

    svd = out.data.attrs["svd"]

    X = _stack_to_matrix(data.data, "time")
    Xc = X - svd["mean"]

    U = np.asarray(svd["U"].data)  # (time, k)
    S = np.asarray(svd["S"].data)  # (k,)
    Vh = np.asarray(svd["Vh"].data)  # (k, features)

    X_hat = (U * S) @ Vh
    rel_err = np.linalg.norm(Xc.values - X_hat) / np.linalg.norm(Xc.values)
    assert rel_err < 1e-10


def test_feature_svd_raises_for_unknown_dimension() -> None:
    data = cb.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=1000.0)

    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.SVD(dim="band_index").apply(data)


def test_feature_svd_n_components_non_positive_raises() -> None:
    data = cb.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=1000.0)

    with pytest.raises(ValueError, match="n_components must be > 0"):
        cb.feature.SVD(dim="time", n_components=0).apply(data)


def test_feature_svd_metadata_preserved() -> None:
    arr = np.random.default_rng(6).normal(size=(20, 2)).astype(float)
    data = cb.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=1000.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="task",
    )

    out = cb.feature.SVD(dim="time", n_components=2, center=True, return_unstacked_V=False).apply(
        data
    )

    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "task"
    assert out.sampling_rate is None


def test_feature_svd_does_not_mutate_input() -> None:
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(15, 4, 3)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "x", "y"], sampling_rate=10.0, subjectID="s1")

    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.SVD(dim="time", n_components=3, center=True).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "s1"


def test_feature_svd_output_u_mode() -> None:
    """SVD with output='U' returns U and stores V in attrs."""
    arr = np.random.default_rng(42).normal(size=(30, 5)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "channel"], sampling_rate=100.0)

    out = cb.feature.SVD(dim="time", n_components=3, center=True, output="U").apply(data)

    assert out.data.name == "U"
    assert out.data.dims == ("time", "component")
    assert out.data.shape == (30, 3)

    svd = out.data.attrs["svd"]
    assert svd["U"] is None
    assert svd["Vh"] is not None
    assert svd["Vh"].shape == (3, 5)
    assert svd["S"].shape == (3,)


def test_feature_svd_no_centering_no_zscore() -> None:
    """SVD with center=False, zscore=False skips normalization and stores None in attrs."""
    arr = np.random.default_rng(42).normal(size=(25, 4)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.SVD(
        dim="time", n_components=3, center=False, zscore=False, return_unstacked_V=False
    ).apply(data)

    svd = out.data.attrs["svd"]
    assert svd["mean"] is None
    assert svd["std"] is None
    assert svd["masked"] is False
    assert svd["center"] is False
    assert svd["zscore"] is False
