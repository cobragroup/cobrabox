"""Tests for the Normalize feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _make_data(values: np.ndarray | None = None, dims: list[str] | None = None) -> cb.Data:
    """Create a Data object for testing."""
    if values is None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal((100, 4))
    if dims is None:
        dims = ["time", "space"]
    return cb.Data.from_numpy(values, dims=dims, subjectID="sub-01")


# ---------------------------------------------------------------------------
# MinMax
# ---------------------------------------------------------------------------


def test_minmax_range() -> None:
    """minmax output is in [0, 1]."""
    data = _make_data()
    out = cb.Normalize(method="minmax").apply(data)
    assert out.to_numpy().min() >= 0.0
    assert out.to_numpy().max() <= 1.0 + 1e-10


def test_minmax_along_dim() -> None:
    """minmax along a dimension normalizes each slice independently."""
    data = _make_data()
    out = cb.Normalize(method="minmax", dim="time").apply(data)
    # Each space slice should have its own [0, 1] range
    for i in range(out.data.sizes["space"]):
        col = out.data.isel(space=i).values
        np.testing.assert_allclose(col.min(), 0.0, atol=1e-10)
        np.testing.assert_allclose(col.max(), 1.0, atol=1e-10)


def test_minmax_constant_gives_zeros() -> None:
    """Constant input produces all zeros (no division by zero)."""
    data = _make_data(values=np.full((10, 3), 5.0))
    out = cb.Normalize(method="minmax").apply(data)
    np.testing.assert_allclose(out.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------


def test_zscore_mean_std() -> None:
    """zscore output has mean ~0 and std ~1."""
    data = _make_data()
    out = cb.Normalize(method="zscore").apply(data)
    arr = out.to_numpy()
    np.testing.assert_allclose(arr.mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(arr.std(), 1.0, atol=0.01)


def test_zscore_along_dim() -> None:
    """zscore along time normalizes each space column."""
    data = _make_data()
    out = cb.Normalize(method="zscore", dim="time").apply(data)
    for i in range(out.data.sizes["space"]):
        col = out.data.isel(space=i).values
        np.testing.assert_allclose(col.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(col.std(), 1.0, atol=0.02)


def test_zscore_constant_gives_zeros() -> None:
    """Constant input produces all zeros."""
    data = _make_data(values=np.full((10, 3), 7.0))
    out = cb.Normalize(method="zscore").apply(data)
    np.testing.assert_allclose(out.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# Robust
# ---------------------------------------------------------------------------


def test_robust_median_centered() -> None:
    """Robust normalization centers on median."""
    data = _make_data()
    out = cb.Normalize(method="robust").apply(data)
    arr = out.to_numpy()
    np.testing.assert_allclose(np.median(arr), 0.0, atol=0.1)


def test_robust_constant_gives_zeros() -> None:
    """Constant input produces all zeros."""
    data = _make_data(values=np.full((10, 3), 3.0))
    out = cb.Normalize(method="robust").apply(data)
    np.testing.assert_allclose(out.to_numpy(), 0.0)


def test_robust_outlier_resilient() -> None:
    """Robust is less affected by outliers than zscore."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
    data = _make_data(values=arr, dims=["space"])
    robust_out = cb.Normalize(method="robust").apply(data)
    zscore_out = cb.Normalize(method="zscore").apply(data)
    # The outlier (1000) should dominate zscore but not robust
    robust_max = np.abs(robust_out.to_numpy()).max()
    zscore_max = np.abs(zscore_out.to_numpy()).max()
    # Robust's max should be larger because IQR is small
    assert robust_max > zscore_max


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------


def test_l1_sums_to_one() -> None:
    """L1-normalized absolute values sum to 1."""
    data = _make_data(values=np.abs(np.random.default_rng(0).standard_normal((10, 3))))
    out = cb.Normalize(method="l1").apply(data)
    np.testing.assert_allclose(np.abs(out.to_numpy()).sum(), 1.0, atol=1e-10)


def test_l1_along_dim() -> None:
    """L1 along time makes each space column's abs values sum to 1."""
    rng = np.random.default_rng(0)
    data = _make_data(values=np.abs(rng.standard_normal((20, 3))))
    out = cb.Normalize(method="l1", dim="time").apply(data)
    for i in range(out.data.sizes["space"]):
        col = out.data.isel(space=i).values
        np.testing.assert_allclose(np.abs(col).sum(), 1.0, atol=1e-10)


def test_l1_zeros_stay_zeros() -> None:
    """All-zero input stays zeros."""
    data = _make_data(values=np.zeros((5, 3)))
    out = cb.Normalize(method="l1").apply(data)
    np.testing.assert_allclose(out.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def test_l2_unit_norm() -> None:
    """L2-normalized values have Euclidean norm 1."""
    data = _make_data()
    out = cb.Normalize(method="l2").apply(data)
    arr = out.to_numpy()
    np.testing.assert_allclose(np.sqrt((arr**2).sum()), 1.0, atol=1e-10)


def test_l2_along_dim() -> None:
    """L2 along time makes each space column a unit vector."""
    data = _make_data()
    out = cb.Normalize(method="l2", dim="time").apply(data)
    for i in range(out.data.sizes["space"]):
        col = out.data.isel(space=i).values
        np.testing.assert_allclose(np.sqrt((col**2).sum()), 1.0, atol=1e-10)


def test_l2_zeros_stay_zeros() -> None:
    """All-zero input stays zeros."""
    data = _make_data(values=np.zeros((5, 3)))
    out = cb.Normalize(method="l2").apply(data)
    np.testing.assert_allclose(out.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# Max-abs
# ---------------------------------------------------------------------------


def test_max_abs_range() -> None:
    """max_abs output is in [-1, 1]."""
    data = _make_data()
    out = cb.Normalize(method="max_abs").apply(data)
    arr = out.to_numpy()
    assert np.abs(arr).max() <= 1.0 + 1e-10


def test_max_abs_preserves_sign() -> None:
    """max_abs preserves the sign of the original values."""
    arr = np.array([[-3.0, 2.0], [1.0, -4.0]])
    data = _make_data(values=arr)
    out = cb.Normalize(method="max_abs").apply(data)
    out_arr = out.to_numpy()
    # Signs should match
    assert (np.sign(out_arr) == np.sign(arr)).all()


def test_max_abs_zeros_stay_zeros() -> None:
    """All-zero input stays zeros."""
    data = _make_data(values=np.zeros((5, 3)))
    out = cb.Normalize(method="max_abs").apply(data)
    np.testing.assert_allclose(out.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# Shape, metadata, history preservation
# ---------------------------------------------------------------------------


def test_preserves_shape() -> None:
    """Output shape matches input shape."""
    data = _make_data()
    out = cb.Normalize(method="zscore").apply(data)
    assert out.data.shape == data.data.shape
    assert out.data.dims == data.data.dims


def test_preserves_metadata() -> None:
    """Metadata is preserved."""
    arr = np.random.default_rng(0).standard_normal((50, 3))
    data = cb.Data.from_numpy(
        arr, dims=["time", "space"], subjectID="s1", groupID="g1", condition="rest"
    )
    out = cb.Normalize(method="minmax").apply(data)
    assert out.subjectID == "s1"
    assert out.groupID == "g1"
    assert out.condition == "rest"


def test_history_appended() -> None:
    """'Normalize' appears in history."""
    data = _make_data()
    out = cb.Normalize(method="zscore").apply(data)
    assert out.history[-1] == "Normalize"


def test_does_not_mutate_input() -> None:
    """Normalize does not modify the input Data object."""
    data = _make_data()
    original = data.to_numpy().copy()
    _ = cb.Normalize(method="minmax").apply(data)
    np.testing.assert_array_equal(data.to_numpy(), original)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_method_raises() -> None:
    """Unknown method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        cb.Normalize(method="bogus")  # type: ignore[arg-type]


def test_missing_dim_raises() -> None:
    """Non-existent dim raises ValueError."""
    data = _make_data()
    with pytest.raises(ValueError, match=r"dim.*not found"):
        cb.Normalize(method="zscore", dim="nonexistent").apply(data)


# ---------------------------------------------------------------------------
# Pipeline composability
# ---------------------------------------------------------------------------


def test_pipes_with_other_features() -> None:
    """Normalize composes in a pipeline."""
    rng = np.random.default_rng(0)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    pipeline = cb.LineLength() | cb.Normalize(method="minmax")
    result = pipeline.apply(data)
    assert "LineLength" in result.history
    assert "Normalize" in result.history
