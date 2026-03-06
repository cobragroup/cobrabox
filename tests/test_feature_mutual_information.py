"""Tests for the line_length feature behavior."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb
from cobrabox.features import MutualInformation

r = 0.6
D = 6
L = 10000
gmi = -np.log2(1 - r**2) / 2
rng = np.random.default_rng(0)


@pytest.fixture
def high_dim_data() -> cb.Data:
    arr = np.zeros([2, 2, D, L])
    S = np.ones([D, D]) * r
    S[np.diag_indices(D)] = 1
    arr[0, 0] = rng.multivariate_normal(np.zeros(D), S, size=L).T
    arr[0, 1] = rng.normal(0, 1, size=[D, L])
    arr[1, 0] = rng.random([D, L])
    arr[1, 1, :3] = np.tile(np.repeat(np.arange(100), 100), (3, 1))
    arr[1, 1, 3:] = np.tile(np.tile(np.arange(100), 100), (3, 1))
    arr = xr.DataArray(arr, dims=["something", "sample", "space", "time"])
    return cb.from_xarray(arr)


@pytest.fixture
def low_dim_data() -> cb.Data:
    arr = np.zeros([D, L])
    S = np.ones([D, D]) * r
    S[np.diag_indices(D)] = 1
    arr = rng.multivariate_normal(np.zeros(D), S, size=L).T
    arr = xr.DataArray(arr, dims=["space", "time"])
    return cb.from_xarray(arr)


def test_entropy() -> None:
    v = np.zeros(10)
    v[:5] = 1
    assert np.isclose(MutualInformation()._vector_entropy(v), np.log(5))


def test_binning() -> None:
    v = np.arange(10)
    binned = MutualInformation(_n_bins=2)._get_binned(v)
    v2 = np.zeros_like(v)
    v2[5:] = 1
    assert (binned == v2).all()


def test_bad_inits(high_dim_data: cb.Data) -> None:
    with pytest.raises(ValueError, match="bins must be positive"):
        MutualInformation(bins=-1)
    with pytest.raises(ValueError, match="bins must be an integer"):
        MutualInformation(bins=2.5)
    with pytest.raises(ValueError, match="bins must be positive"):
        MutualInformation(bins=0)
    with pytest.raises(ValueError, match="dim must be a string"):
        MutualInformation(dim=123)
    with pytest.raises(ValueError, match="other_dim must be a string or None"):
        MutualInformation(other_dim=123)
    with pytest.raises(ValueError, match=r"Dimension.*not found in data"):
        MutualInformation(dim="not_a_dim").apply(high_dim_data)
    with pytest.raises(ValueError, match=r"Dimension.*not found in data"):
        MutualInformation(other_dim="not_a_dim").apply(high_dim_data)
    with pytest.raises(
        ValueError, match=r"self\.other_dim must be specified for data with more than 2 dimensions"
    ):
        MutualInformation().apply(high_dim_data)


def test_low_dim_equidistant_bins(low_dim_data: cb.Data) -> None:
    mi = MutualInformation(bins=5, equiprobable_bins=False)
    result = mi.apply(low_dim_data)
    assert isinstance(result, cb.Data)
    assert result.data.shape == (D, D)
    assert np.max((result.data - gmi) / gmi) < 0.05
    assert result.data.dims == ("space_from", "space_to")


def test_low_dim_equiprobable_bins(low_dim_data: cb.Data) -> None:
    mi = MutualInformation(bins=5, equiprobable_bins=True)
    result = mi.apply(low_dim_data)
    assert isinstance(result, cb.Data)
    assert result.data.shape == (D, D)
    assert np.max((result.data - gmi) / gmi) < 0.1
    assert result.data.dims == ("space_from", "space_to")


def test_high_dim_equiprobable_bins(high_dim_data: cb.Data) -> None:
    mi = MutualInformation(equiprobable_bins=True, dim="time", other_dim="space")
    result = mi.apply(high_dim_data)
    assert isinstance(result, cb.Data)
    assert result.data.shape == (2, 2, D, D)
    assert np.max((result.data.data[0, 0] - gmi) / gmi) < 0.15
    assert np.max(result.data.data[0, 1]) < 0.05
    assert np.max(result.data.data[1, 0]) < 0.05
    assert np.max(result.data.data[1, 1, :3, :3]) > 4
    assert np.max(result.data.data[1, 1, 3:, :3]) < 0.05
    assert result.data.dims == ("something", "sample", "space_from", "space_to")


def test_high_dim_equidistant_bins(high_dim_data: cb.Data) -> None:
    mi = MutualInformation(equiprobable_bins=False, dim="time", other_dim="space")
    result = mi.apply(high_dim_data)
    assert isinstance(result, cb.Data)
    assert result.data.shape == (2, 2, D, D)
    assert np.max((result.data.data[0, 0] - gmi) / gmi) < 0.1
    assert np.max(result.data.data[0, 1]) < 0.05
    assert np.max(result.data.data[1, 0]) < 0.05
    assert np.max(result.data.data[1, 1, :3, :3]) > 4
    assert np.max(result.data.data[1, 1, 3:, :3]) < 0.05
    assert result.data.dims == ("something", "sample", "space_from", "space_to")
