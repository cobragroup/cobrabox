"""Tests for the mutual_information feature behavior."""

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


def test_mutual_information_vector_entropy() -> None:
    """_vector_entropy computes correct entropy for a binned distribution."""
    v = np.zeros(10)
    v[:5] = 1
    assert np.isclose(MutualInformation()._vector_entropy(v), np.log(5))


def test_mutual_information_get_binned() -> None:
    """_get_binned correctly discretizes data into specified number of bins."""
    v = np.arange(10)
    binned = MutualInformation()._get_binned(v, n_bins=2)
    v2 = np.zeros_like(v)
    v2[5:] = 1
    assert (binned == v2).all()


def test_mutual_information_negative_bins_raises() -> None:
    """MutualInformation raises ValueError for negative bins."""
    with pytest.raises(ValueError, match="bins must be positive"):
        MutualInformation(bins=-1)


def test_mutual_information_non_integer_bins_raises() -> None:
    """MutualInformation raises ValueError for non-integer bins."""
    with pytest.raises(ValueError, match="bins must be an integer"):
        MutualInformation(bins=2.5)


def test_mutual_information_zero_bins_raises() -> None:
    """MutualInformation raises ValueError for zero bins."""
    with pytest.raises(ValueError, match="bins must be positive"):
        MutualInformation(bins=0)


def test_mutual_information_invalid_dim_type_raises() -> None:
    """MutualInformation raises ValueError when dim is not a string."""
    with pytest.raises(ValueError, match="dim must be a string"):
        MutualInformation(dim=123)  # type: ignore[arg-type]


def test_mutual_information_invalid_other_dim_type_raises() -> None:
    """MutualInformation raises ValueError when other_dim is not a string or None."""
    with pytest.raises(ValueError, match="other_dim must be a string or None"):
        MutualInformation(other_dim=123)  # type: ignore[arg-type]


def test_mutual_information_invalid_dim_raises(high_dim_data: cb.Data) -> None:
    """MutualInformation raises ValueError for invalid dimension."""
    with pytest.raises(ValueError, match=r"Dimension.*not found in data"):
        MutualInformation(dim="not_a_dim").apply(high_dim_data)


def test_mutual_information_invalid_other_dim_raises(high_dim_data: cb.Data) -> None:
    """MutualInformation raises ValueError for invalid other_dim."""
    with pytest.raises(ValueError, match=r"Dimension.*not found in data"):
        MutualInformation(other_dim="not_a_dim").apply(high_dim_data)


def test_mutual_information_high_dim_without_other_dim_raises(high_dim_data: cb.Data) -> None:
    """MutualInformation raises ValueError for high-dim data without other_dim."""
    with pytest.raises(
        ValueError, match=r"self\.other_dim must be specified for data with more than 2 dimensions"
    ):
        MutualInformation().apply(high_dim_data)


def test_low_dim_equidistant_bins(low_dim_data: cb.Data) -> None:
    """MutualInformation computes correct MI for 2D data with equidistant bins."""
    mi = MutualInformation(bins=5, equiprobable_bins=False)
    result = mi.apply(low_dim_data)
    assert isinstance(result, cb.Data)
    assert result.data.shape == (D, D)
    assert np.max((result.data - gmi) / gmi) < 0.05
    assert result.data.dims == ("space_from", "space_to")


def test_low_dim_equiprobable_bins(low_dim_data: cb.Data) -> None:
    """MutualInformation computes correct MI for 2D data with equiprobable bins."""
    mi = MutualInformation(bins=5, equiprobable_bins=True)
    result = mi.apply(low_dim_data)
    assert isinstance(result, cb.Data)
    assert result.data.shape == (D, D)
    assert np.max((result.data - gmi) / gmi) < 0.1
    assert result.data.dims == ("space_from", "space_to")


def test_high_dim_equiprobable_bins(high_dim_data: cb.Data) -> None:
    """MutualInformation handles 4D data correctly with equiprobable bins."""
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
    """MutualInformation handles 4D data correctly with equidistant bins."""
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


def test_mutual_information_history_updated() -> None:
    """MutualInformation appends 'MutualInformation' to history."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=100.0)
    result = MutualInformation().apply(data)
    assert result.history[-1] == "MutualInformation"


def test_mutual_information_metadata_preserved() -> None:
    """MutualInformation preserves subjectID, groupID, condition."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(
        arr,
        dims=["space", "time"],
        sampling_rate=100.0,
        subjectID="s42",
        groupID="control",
        condition="rest",
    )
    result = MutualInformation().apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "rest"


def test_mutual_information_sampling_rate_none() -> None:
    """MutualInformation sets sampling_rate to None since time dimension is removed."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=100.0)
    result = MutualInformation().apply(data)
    assert result.sampling_rate is None


def test_mutual_information_does_not_mutate_input() -> None:
    """MutualInformation does not modify the input Data object."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=100.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = MutualInformation().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
