"""Tests for the RecurrenceMatrix feature."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.features.time_domain.recurrence_matrix import FcMetric, RecMetric, RecurrenceMatrix


def _make_2d(n_time: int = 100, n_ch: int = 5, seed: int = 0) -> cb.SignalData:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_time, n_ch))
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)


def _make_3d(n_time: int = 30, n_ch: int = 4, seed: int = 0) -> cb.SignalData:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_ch, n_ch, n_time))
    return cb.SignalData.from_numpy(arr, dims=["space1", "space2", "time"], sampling_rate=100.0)


# ---------------------------------------------------------------------------
# 2-D: state-vector mode (no fc_options)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_recurrence_matrix_statevector_shape(rec_metric: RecMetric) -> None:
    """RecurrenceMatrix state-vector mode returns (T, T) Data with correct history."""
    data = _make_2d(n_time=50, n_ch=4)
    out = RecurrenceMatrix(rec_metric).apply(data)
    assert isinstance(out, cb.Data)
    assert out.data.dims == ("t1", "t2")
    assert out.data.shape == (50, 50)
    assert out.history == ["RecurrenceMatrix"]


def test_recurrence_matrix_statevector_cosine_diagonal_is_one() -> None:
    """RecurrenceMatrix cosine metric gives diagonal = 1."""
    out = RecurrenceMatrix("cosine").apply(_make_2d(40))
    np.testing.assert_allclose(np.diag(out.data.values), 1.0, atol=1e-12)


def test_recurrence_matrix_statevector_euclidean_diagonal_is_zero() -> None:
    """RecurrenceMatrix euclidean metric gives diagonal = 0."""
    out = RecurrenceMatrix("euclidean").apply(_make_2d(40))
    np.testing.assert_allclose(np.diag(out.data.values), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2-D: window/FC mode via fc_options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fc_metric", ["pearson", "spearman", "PLV", "AEC", "MI"])
def test_recurrence_matrix_window_fc_options_metric_only(fc_metric: FcMetric) -> None:
    """fc_options=[fc_metric] uses default window_size=10, overlap=0.5."""
    data = _make_2d(n_time=100, n_ch=5)
    step = max(1, int(10 * 0.5))
    n_windows = (100 - 10) // step + 1
    out = RecurrenceMatrix("cosine", [fc_metric]).apply(data)
    assert out.data.shape == (n_windows, n_windows)


def test_recurrence_matrix_window_fc_options_with_window_size() -> None:
    """fc_options=[fc_metric, window_size] uses default overlap=0.5."""
    data = _make_2d(n_time=100, n_ch=5)
    ws = 20
    step = max(1, int(ws * 0.5))
    n_windows = (100 - ws) // step + 1
    out = RecurrenceMatrix("cosine", ["pearson", ws]).apply(data)
    assert out.data.shape == (n_windows, n_windows)


def test_recurrence_matrix_window_fc_options_full() -> None:
    """fc_options=[fc_metric, window_size, overlap] — full control."""
    data = _make_2d(n_time=100, n_ch=5)
    ws, overlap = 20, 0.25
    step = max(1, int(ws * (1 - overlap)))
    n_windows = (100 - ws) // step + 1
    out = RecurrenceMatrix("cosine", ["pearson", ws, overlap]).apply(data)
    assert out.data.shape == (n_windows, n_windows)


def test_recurrence_matrix_window_small_warns() -> None:
    """RecurrenceMatrix emits UserWarning when window_size is below 5."""
    data = _make_2d(n_time=50, n_ch=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RecurrenceMatrix("cosine", ["pearson", 3]).apply(data)
    assert any("window_size" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# 3-D input: pre-computed FC time-series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_recurrence_matrix_3d_shape(rec_metric: RecMetric) -> None:
    """RecurrenceMatrix on 3-D input returns (T, T) Data with correct history."""
    data = _make_3d(n_time=30, n_ch=4)
    out = RecurrenceMatrix(rec_metric).apply(data)
    assert out.data.dims == ("t1", "t2")
    assert out.data.shape == (30, 30)
    assert out.history == ["RecurrenceMatrix"]


def test_recurrence_matrix_3d_cosine_diagonal_is_one() -> None:
    """RecurrenceMatrix cosine metric gives diagonal = 1 for 3-D input."""
    out = RecurrenceMatrix("cosine").apply(_make_3d(20, 4))
    np.testing.assert_allclose(np.diag(out.data.values), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_recurrence_matrix_statevector_symmetric(rec_metric: RecMetric) -> None:
    """RecurrenceMatrix output is symmetric (matrix equals its transpose)."""
    mat = RecurrenceMatrix(rec_metric).apply(_make_2d(60)).data.values
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_recurrence_matrix_3d_symmetric(rec_metric: RecMetric) -> None:
    """RecurrenceMatrix output is symmetric for 3-D input."""
    mat = RecurrenceMatrix(rec_metric).apply(_make_3d(20)).data.values
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_recurrence_matrix_metadata_preserved() -> None:
    """RecurrenceMatrix preserves subjectID, groupID, condition; sampling_rate becomes None."""
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((50, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=200.0,
        subjectID="sub-01",
        groupID="ctrl",
        condition="rest",
    )
    out = RecurrenceMatrix().apply(data)
    assert out.subjectID == "sub-01"
    assert out.groupID == "ctrl"
    assert out.condition == "rest"
    assert out.sampling_rate is None


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_recurrence_matrix_invalid_rec_metric_raises() -> None:
    """RecurrenceMatrix raises ValueError for unknown rec_metric."""
    with pytest.raises(ValueError, match="rec_metric"):
        RecurrenceMatrix("manhattan")  # type: ignore[arg-type]


def test_recurrence_matrix_invalid_fc_metric_in_options_raises() -> None:
    """RecurrenceMatrix raises ValueError for unknown fc_metric in fc_options."""
    with pytest.raises(ValueError, match="fc_options"):
        RecurrenceMatrix("cosine", ["kendall"])


def test_recurrence_matrix_too_many_fc_options_raises() -> None:
    """RecurrenceMatrix raises ValueError when fc_options has more than three elements."""
    with pytest.raises(ValueError, match="fc_options"):
        RecurrenceMatrix("cosine", ["pearson", 10, 0.5, "extra"])


def test_recurrence_matrix_invalid_overlap_raises() -> None:
    """RecurrenceMatrix raises ValueError when overlap is outside (0, 1)."""
    with pytest.raises(ValueError, match="overlap"):
        RecurrenceMatrix("cosine", ["pearson", 10, 1.0])


def test_recurrence_matrix_window_size_too_large_raises() -> None:
    """RecurrenceMatrix raises ValueError when window_size exceeds time dimension."""
    data = _make_2d(n_time=30)
    with pytest.raises(ValueError, match="window_size"):
        RecurrenceMatrix("cosine", ["pearson", 30]).apply(data)


def test_recurrence_matrix_missing_time_dim_raises() -> None:
    """RecurrenceMatrix raises ValueError when input lacks 'time' dimension."""
    arr = np.ones((5, 5))
    data = cb.Data.from_numpy(arr, dims=["space", "freq"])
    with pytest.raises(ValueError, match="time"):
        RecurrenceMatrix().apply(data)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_recurrence_matrix_window_size_zero_raises() -> None:
    """RecurrenceMatrix raises ValueError when window_size is 0 (< 1)."""
    with pytest.raises(ValueError, match="window_size"):
        RecurrenceMatrix("cosine", ["pearson", 0])


def test_recurrence_matrix_fc_mi() -> None:
    """RecurrenceMatrix with MI fc_metric produces a square output."""
    data = _make_2d(n_time=80, n_ch=3)
    out = RecurrenceMatrix("cosine", ["MI", 20, 0.5]).apply(data)
    assert isinstance(out, cb.Data)
    assert out.data.dims == ("t1", "t2")
    assert out.data.shape[0] == out.data.shape[1]
    assert not np.any(np.isnan(out.data.values))


def test_recurrence_matrix_invalid_spatial_dims_raises() -> None:
    """RecurrenceMatrix raises ValueError for input with 3 spatial dimensions."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((2, 3, 4, 10))
    data = cb.SignalData.from_numpy(arr, dims=["a", "b", "c", "time"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="spatial dimensions"):
        RecurrenceMatrix().apply(data)


def test_recurrence_matrix_3d_nonsquare_raises() -> None:
    """RecurrenceMatrix raises ValueError when 3-D spatial dims are not equal."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 4, 20))
    data = cb.SignalData.from_numpy(arr, dims=["space1", "space2", "time"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="spatial dimensions must be equal"):
        RecurrenceMatrix().apply(data)


def test_recurrence_matrix_no_mutation() -> None:
    """RecurrenceMatrix.apply() does not modify the input Data object."""
    data = _make_2d(n_time=50, n_ch=4)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = np.copy(data.to_numpy())
    _ = RecurrenceMatrix().apply(data)
    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_recurrence_matrix_output_finite() -> None:
    """RecurrenceMatrix produces finite (non-NaN, non-inf) values in state-vector mode."""
    data = _make_2d(n_time=40, n_ch=5)
    out = RecurrenceMatrix("cosine").apply(data)
    assert np.all(np.isfinite(out.data.values))
