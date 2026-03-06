"""Tests for the Covariance feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    arr: np.ndarray, *, dims: list[str] | None = None, sampling_rate: float | None = None
) -> cb.Data:
    """Create a plain Data from an array."""
    if dims is None:
        dims = ["space", "time"]
    return cb.Data.from_numpy(arr, dims=dims, sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_covariance_output_dims_and_shape() -> None:
    """Covariance returns Data with (space_to, space_from) dims and NxN matrix."""
    rng = np.random.default_rng(0)
    data = _make_data(rng.standard_normal((4, 300)))

    out = cb.feature.Covariance().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space_to", "space_from")
    assert out.data.sizes["space_to"] == 4
    assert out.data.sizes["space_from"] == 4


def test_covariance_output_is_square() -> None:
    """Output matrix shape is (n_channels, n_channels)."""
    rng = np.random.default_rng(1)
    n = 6
    data = _make_data(rng.standard_normal((n, 200)))

    out = cb.feature.Covariance().apply(data)

    assert out.data.shape == (n, n)


def test_covariance_channel_coords_preserved() -> None:
    """Both space_to and space_from carry the original channel labels."""
    rng = np.random.default_rng(2)
    labels = ["Fz", "Cz", "Pz"]
    arr = xr.DataArray(
        rng.standard_normal((3, 200)), dims=["space", "time"], coords={"space": labels}
    )
    data = cb.Data.from_xarray(arr)

    out = cb.feature.Covariance().apply(data)

    np.testing.assert_array_equal(out.data.coords["space_to"].values, labels)
    np.testing.assert_array_equal(out.data.coords["space_from"].values, labels)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_covariance_matches_numpy_cov() -> None:
    """Covariance output matches np.cov exactly."""
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((5, 400))
    data = _make_data(arr)

    out = cb.feature.Covariance().apply(data)

    np.testing.assert_allclose(out.data.values, np.cov(arr), atol=1e-12)


def test_covariance_diagonal_equals_sample_variance() -> None:
    """Diagonal must equal per-channel sample variance (ddof=1)."""
    rng = np.random.default_rng(4)
    arr = rng.standard_normal((4, 300))
    data = _make_data(arr)

    out = cb.feature.Covariance().apply(data)

    expected_var = np.var(arr, axis=1, ddof=1)
    np.testing.assert_allclose(np.diag(out.data.values), expected_var, atol=1e-12)


def test_covariance_matrix_is_symmetric() -> None:
    """C[i, j] == C[j, i] for all entries."""
    rng = np.random.default_rng(5)
    data = _make_data(rng.standard_normal((5, 300)))

    out = cb.feature.Covariance().apply(data)
    mat = out.data.values

    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


def test_covariance_diagonal_positive_for_nonzero_signals() -> None:
    """Diagonal (variance) must be strictly positive for non-constant signals."""
    rng = np.random.default_rng(6)
    data = _make_data(rng.standard_normal((4, 300)))

    out = cb.feature.Covariance().apply(data)

    assert np.all(np.diag(out.data.values) > 0)


def test_covariance_identical_channels_diagonal_matches_off_diagonal() -> None:
    """Identical channels have variance == covariance for each pair."""
    rng = np.random.default_rng(7)
    sig = rng.standard_normal(300)
    arr = np.stack([sig, sig, sig], axis=0)
    data = _make_data(arr)

    out = cb.feature.Covariance().apply(data)
    mat = out.data.values

    # All entries should be the same (var == cov for identical signals)
    np.testing.assert_allclose(mat, mat[0, 0] * np.ones((3, 3)), atol=1e-10)


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


def test_covariance_preserves_metadata_and_history() -> None:
    """Covariance propagates all metadata fields and sets history correctly."""
    rng = np.random.default_rng(8)
    data = cb.Data.from_numpy(
        rng.standard_normal((3, 200)),
        dims=["space", "time"],
        subjectID="sub-01",
        groupID="control",
        condition="rest",
        extra={"session": 2},
    )

    out = cb.feature.Covariance().apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "control"
    assert out.condition == "rest"
    assert out.history == ["Covariance"]
    assert out.sampling_rate is None
    assert out.extra.get("session") == 2


def test_covariance_history_appends_correctly() -> None:
    """History entry is appended and is 'Covariance'."""
    rng = np.random.default_rng(9)
    data = cb.Data.from_numpy(rng.standard_normal((3, 200)), dims=["space", "time"])

    out = cb.feature.Covariance().apply(data)

    assert out.history[-1] == "Covariance"


# ---------------------------------------------------------------------------
# Non-standard dim name
# ---------------------------------------------------------------------------


def test_covariance_custom_dim_name() -> None:
    """Covariance works on axes other than 'time', e.g. 'frequency'."""
    rng = np.random.default_rng(10)
    arr = rng.standard_normal((4, 100))
    data = _make_data(arr, dims=["electrode", "frequency"])

    out = cb.feature.Covariance(dim="frequency").apply(data)

    assert out.data.dims == ("electrode_to", "electrode_from")
    assert out.data.shape == (4, 4)


def test_covariance_correlates_along_non_default_dim() -> None:
    """When dim points to the first axis, the second axis becomes channels."""
    rng = np.random.default_rng(11)
    arr = rng.standard_normal((300, 5))
    data = _make_data(arr, dims=["time", "space"])

    out = cb.feature.Covariance(dim="time").apply(data)

    assert out.data.dims == ("space_to", "space_from")
    assert out.data.shape == (5, 5)


def test_covariance_custom_dim_matches_numpy_cov() -> None:
    """Covariance with custom dim produces same result as transposing and calling np.cov."""
    rng = np.random.default_rng(12)
    arr = rng.standard_normal((4, 100))
    data = _make_data(arr, dims=["electrode", "frequency"])

    out = cb.feature.Covariance(dim="frequency").apply(data)

    np.testing.assert_allclose(out.data.values, np.cov(arr), atol=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_covariance_raises_on_3d_input() -> None:
    """Covariance raises ValueError for 3D input."""
    rng = np.random.default_rng(13)
    arr = xr.DataArray(rng.standard_normal((2, 3, 100)), dims=["run", "space", "time"])
    data = cb.Data.from_xarray(arr)

    with pytest.raises(ValueError, match="exactly 2-dimensional"):
        cb.feature.Covariance().apply(data)


def test_covariance_raises_on_1d_input() -> None:
    """Covariance raises ValueError for 1D input."""
    rng = np.random.default_rng(14)
    arr = xr.DataArray(rng.standard_normal(100), dims=["time"])
    data = cb.Data.from_xarray(arr)

    with pytest.raises(ValueError, match="exactly 2-dimensional"):
        cb.feature.Covariance().apply(data)


def test_covariance_raises_when_dim_missing() -> None:
    """Covariance raises ValueError when dim is absent from data."""
    rng = np.random.default_rng(15)
    data = _make_data(rng.standard_normal((4, 300)), dims=["space", "frequency"])

    with pytest.raises(ValueError, match="'time'"):
        cb.feature.Covariance(dim="time").apply(data)


def test_covariance_raises_when_dim_missing_includes_hint() -> None:
    """Error message for missing dim suggests using dim= parameter."""
    rng = np.random.default_rng(16)
    data = _make_data(rng.standard_normal((4, 300)), dims=["electrode", "frequency"])

    with pytest.raises(ValueError, match="dim="):
        cb.feature.Covariance(dim="time").apply(data)


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------


def test_covariance_does_not_mutate_input() -> None:
    """Covariance.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(17)
    data = _make_data(rng.standard_normal((4, 300)))
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Covariance().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


# ---------------------------------------------------------------------------
# API accessibility
# ---------------------------------------------------------------------------


def test_covariance_accessible_via_feature_module() -> None:
    """Covariance is accessible as cb.feature.Covariance."""
    assert callable(cb.feature.Covariance)


def test_covariance_output_is_data_instance() -> None:
    """Output of apply() is always an instance of cb.Data."""
    rng = np.random.default_rng(18)
    data = _make_data(rng.standard_normal((3, 200)))

    out = cb.feature.Covariance().apply(data)

    assert isinstance(out, cb.Data)
