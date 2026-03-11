"""Tests for the Correlation feature."""

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


def test_correlation_output_dims_and_shape() -> None:
    """Correlation returns Data with (space_to, space_from) dims and NxN matrix."""
    rng = np.random.default_rng(0)
    data = _make_data(rng.standard_normal((4, 300)))

    out = cb.feature.Correlation().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space_to", "space_from")
    assert out.data.sizes["space_to"] == 4
    assert out.data.sizes["space_from"] == 4


def test_correlation_output_is_square() -> None:
    """Output matrix shape is (n_channels, n_channels)."""
    rng = np.random.default_rng(1)
    n = 6
    data = _make_data(rng.standard_normal((n, 200)))

    out = cb.feature.Correlation().apply(data)

    assert out.data.shape == (n, n)


def test_correlation_channel_coords_preserved() -> None:
    """Both space_to and space_from carry the original channel labels."""
    rng = np.random.default_rng(2)
    labels = ["Fz", "Cz", "Pz"]
    arr = xr.DataArray(
        rng.standard_normal((3, 200)), dims=["space", "time"], coords={"space": labels}
    )
    data = cb.Data.from_xarray(arr)

    out = cb.feature.Correlation().apply(data)

    np.testing.assert_array_equal(out.data.coords["space_to"].values, labels)
    np.testing.assert_array_equal(out.data.coords["space_from"].values, labels)


# ---------------------------------------------------------------------------
# Numerical correctness — Pearson
# ---------------------------------------------------------------------------


def test_correlation_pearson_matches_numpy_corrcoef() -> None:
    """Pearson output matches np.corrcoef exactly."""
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((5, 400))
    data = _make_data(arr)

    out = cb.feature.Correlation(method="pearson").apply(data)

    expected = np.corrcoef(arr)
    np.testing.assert_allclose(out.data.values, expected, atol=1e-12)


def test_correlation_pearson_diagonal_is_one() -> None:
    """Self-correlation diagonal must be exactly 1.0."""
    rng = np.random.default_rng(4)
    data = _make_data(rng.standard_normal((4, 300)))

    out = cb.feature.Correlation().apply(data)

    np.testing.assert_array_equal(np.diag(out.data.values), np.ones(4))


def test_correlation_pearson_matrix_is_symmetric() -> None:
    """C[i, j] == C[j, i] for all entries."""
    rng = np.random.default_rng(5)
    data = _make_data(rng.standard_normal((5, 300)))

    out = cb.feature.Correlation().apply(data)
    mat = out.data.values

    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


def test_correlation_pearson_values_in_minus_one_to_one() -> None:
    """All Pearson values lie in [-1, 1]."""
    rng = np.random.default_rng(6)
    data = _make_data(rng.standard_normal((8, 500)))

    out = cb.feature.Correlation().apply(data)
    mat = out.data.values

    assert np.all(mat >= -1.0 - 1e-10)
    assert np.all(mat <= 1.0 + 1e-10)


def test_correlation_pearson_identical_channels_give_one() -> None:
    """Perfectly correlated channels produce correlation of 1.0."""
    rng = np.random.default_rng(7)
    sig = rng.standard_normal(300)
    arr = np.stack([sig, sig, sig], axis=0)
    data = _make_data(arr)

    out = cb.feature.Correlation().apply(data)
    mat = out.data.values

    np.testing.assert_allclose(mat, np.ones((3, 3)), atol=1e-10)


def test_correlation_pearson_anti_correlated_gives_minus_one() -> None:
    """Anti-correlated channels produce correlation of -1.0."""
    rng = np.random.default_rng(8)
    sig = rng.standard_normal(300)
    arr = np.stack([sig, -sig], axis=0)
    data = _make_data(arr)

    out = cb.feature.Correlation().apply(data)

    np.testing.assert_allclose(out.data.values[0, 1], -1.0, atol=1e-10)
    np.testing.assert_allclose(out.data.values[1, 0], -1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Numerical correctness — Spearman
# ---------------------------------------------------------------------------


def test_correlation_spearman_matches_scipy() -> None:
    """Spearman output matches scipy.stats.spearmanr."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(9)
    arr = rng.standard_normal((4, 300))
    data = _make_data(arr)

    out = cb.feature.Correlation(method="spearman").apply(data)

    # spearmanr returns (n_samples, n_channels) convention — transpose arr
    expected = spearmanr(arr.T).statistic
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(out.data.values, expected, atol=1e-10)


def test_correlation_spearman_diagonal_is_one() -> None:
    """Spearman diagonal must be exactly 1.0."""
    rng = np.random.default_rng(10)
    data = _make_data(rng.standard_normal((4, 300)))

    out = cb.feature.Correlation(method="spearman").apply(data)

    np.testing.assert_array_equal(np.diag(out.data.values), np.ones(4))


def test_correlation_spearman_matrix_is_symmetric() -> None:
    """Spearman matrix is symmetric."""
    rng = np.random.default_rng(11)
    data = _make_data(rng.standard_normal((5, 300)))

    out = cb.feature.Correlation(method="spearman").apply(data)
    mat = out.data.values

    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


def test_pearson_and_spearman_differ_on_nonlinear_data() -> None:
    """Pearson and Spearman produce different results on non-linear relationships."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal(300)
    y = x**3 + rng.standard_normal(300) * 0.1
    arr = np.stack([x, y], axis=0)
    data = _make_data(arr)

    pearson = cb.feature.Correlation(method="pearson").apply(data)
    spearman = cb.feature.Correlation(method="spearman").apply(data)

    assert not np.allclose(pearson.data.values, spearman.data.values)


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


def test_correlation_preserves_metadata_and_history() -> None:
    """Correlation propagates all metadata fields and sets history correctly."""
    rng = np.random.default_rng(13)
    data = cb.Data.from_numpy(
        rng.standard_normal((3, 200)),
        dims=["space", "time"],
        subjectID="sub-01",
        groupID="control",
        condition="rest",
        extra={"session": 2},
    )

    out = cb.feature.Correlation().apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "control"
    assert out.condition == "rest"
    assert out.history == ["Correlation"]
    assert out.sampling_rate is None
    assert out.extra.get("session") == 2


def test_correlation_history_appends_to_existing() -> None:
    """History entry is appended when previous history exists."""
    rng = np.random.default_rng(14)
    data = cb.Data.from_numpy(rng.standard_normal((3, 200)), dims=["space", "time"])

    out = cb.feature.Correlation().apply(data)

    assert "Correlation" in out.history
    assert out.history[-1] == "Correlation"


# ---------------------------------------------------------------------------
# Non-standard dim name
# ---------------------------------------------------------------------------


def test_correlation_custom_dim_name() -> None:
    """Correlation works on axes other than 'time', e.g. 'frequency'."""
    # (electrode, frequency) array — correlate along frequency
    rng = np.random.default_rng(15)
    arr = rng.standard_normal((4, 100))
    data = _make_data(arr, dims=["electrode", "frequency"])

    out = cb.feature.Correlation(dim="frequency").apply(data)

    assert out.data.dims == ("electrode_to", "electrode_from")
    assert out.data.shape == (4, 4)


def test_correlation_correlates_along_non_default_dim() -> None:
    """When dim points to the first axis, the second axis becomes channels."""
    # (time, space) — correlate along time, output should be (space_to, space_from)
    rng = np.random.default_rng(16)
    arr = rng.standard_normal((300, 5))
    data = _make_data(arr, dims=["time", "space"])

    out = cb.feature.Correlation(dim="time").apply(data)

    assert out.data.dims == ("space_to", "space_from")
    assert out.data.shape == (5, 5)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_correlation_raises_on_3d_input() -> None:
    """Correlation raises ValueError for 3D input."""
    rng = np.random.default_rng(17)
    arr = xr.DataArray(rng.standard_normal((2, 3, 100)), dims=["run", "space", "time"])
    data = cb.Data.from_xarray(arr)

    with pytest.raises(ValueError, match="exactly 2-dimensional"):
        cb.feature.Correlation().apply(data)


def test_correlation_raises_on_1d_input() -> None:
    """Correlation raises ValueError for 1D input."""
    rng = np.random.default_rng(18)
    arr = xr.DataArray(rng.standard_normal(100), dims=["time"])
    data = cb.Data.from_xarray(arr)

    with pytest.raises(ValueError, match="exactly 2-dimensional"):
        cb.feature.Correlation().apply(data)


def test_correlation_raises_when_dim_missing() -> None:
    """Correlation raises ValueError when dim is absent from data."""
    rng = np.random.default_rng(19)
    data = _make_data(rng.standard_normal((4, 300)), dims=["space", "frequency"])

    with pytest.raises(ValueError, match="'time'"):
        cb.feature.Correlation(dim="time").apply(data)


def test_correlation_raises_when_dim_missing_includes_hint() -> None:
    """Error message for missing dim suggests using dim= parameter."""
    rng = np.random.default_rng(20)
    data = _make_data(rng.standard_normal((4, 300)), dims=["electrode", "frequency"])

    with pytest.raises(ValueError, match="dim="):
        cb.feature.Correlation(dim="time").apply(data)


def test_correlation_raises_on_invalid_method() -> None:
    """Correlation raises ValueError for an unknown method at construction time."""
    with pytest.raises(ValueError, match="'kendall'"):
        cb.feature.Correlation(method="kendall")


def test_correlation_raises_on_invalid_method_pearson_typo() -> None:
    """Typo in method name raises ValueError."""
    with pytest.raises(ValueError, match="pearson"):
        cb.feature.Correlation(method="Pearson")


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------


def test_correlation_does_not_mutate_input() -> None:
    """Correlation.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(21)
    data = _make_data(rng.standard_normal((4, 300)))
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Correlation().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


# ---------------------------------------------------------------------------
# API accessibility
# ---------------------------------------------------------------------------


def test_correlation_accessible_via_feature_module() -> None:
    """Correlation is accessible as cb.feature.Correlation."""
    assert callable(cb.feature.Correlation)


def test_correlation_is_data_instance() -> None:
    """Output of apply() is always an instance of cb.Data."""
    rng = np.random.default_rng(22)
    data = _make_data(rng.standard_normal((3, 200)))

    out = cb.feature.Correlation().apply(data)

    assert isinstance(out, cb.Data)
