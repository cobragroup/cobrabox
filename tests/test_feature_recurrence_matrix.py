"""Tests for the RecurrenceMatrix feature."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.features.recurrence_matrix import RecurrenceMatrix


def _make_2d(n_time: int = 100, n_ch: int = 5, seed: int = 0) -> cb.SignalData:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_time, n_ch))
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)


def _make_3d(n_time: int = 30, n_ch: int = 4, seed: int = 0) -> cb.Data:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_ch, n_ch, n_time))
    return cb.Data.from_numpy(arr, dims=["space1", "space2", "time"])


# ---------------------------------------------------------------------------
# 2-D: state-vector mode (no fc_options)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_statevector_shape(rec_metric):
    data = _make_2d(n_time=50, n_ch=4)
    out = RecurrenceMatrix(rec_metric).apply(data)
    assert isinstance(out, cb.Data)
    assert out.data.dims == ("t1", "t2")
    assert out.data.shape == (50, 50)
    assert out.history == ["RecurrenceMatrix"]


def test_statevector_cosine_diagonal_is_one():
    out = RecurrenceMatrix("cosine").apply(_make_2d(40))
    np.testing.assert_allclose(np.diag(out.data.values), 1.0, atol=1e-12)


def test_statevector_euclidean_diagonal_is_zero():
    out = RecurrenceMatrix("euclidean").apply(_make_2d(40))
    np.testing.assert_allclose(np.diag(out.data.values), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2-D: window/FC mode via fc_options
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fc_metric", ["pearson", "spearman", "PLV", "AEC"])
def test_window_fc_options_metric_only(fc_metric):
    """fc_options=[fc_metric] uses default window_size=10, overlap=0.5."""
    data = _make_2d(n_time=100, n_ch=5)
    step = max(1, int(10 * 0.5))
    n_windows = (100 - 10) // step + 1
    out = RecurrenceMatrix("cosine", [fc_metric]).apply(data)
    assert out.data.shape == (n_windows, n_windows)


def test_window_fc_options_with_window_size():
    """fc_options=[fc_metric, window_size] uses default overlap=0.5."""
    data = _make_2d(n_time=100, n_ch=5)
    ws = 20
    step = max(1, int(ws * 0.5))
    n_windows = (100 - ws) // step + 1
    out = RecurrenceMatrix("cosine", ["pearson", ws]).apply(data)
    assert out.data.shape == (n_windows, n_windows)


def test_window_fc_options_full():
    """fc_options=[fc_metric, window_size, overlap] — full control."""
    data = _make_2d(n_time=100, n_ch=5)
    ws, overlap = 20, 0.25
    step = max(1, int(ws * (1 - overlap)))
    n_windows = (100 - ws) // step + 1
    out = RecurrenceMatrix("cosine", ["pearson", ws, overlap]).apply(data)
    assert out.data.shape == (n_windows, n_windows)


def test_window_small_warns():
    data = _make_2d(n_time=50, n_ch=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RecurrenceMatrix("cosine", ["pearson", 3]).apply(data)
    assert any("window_size" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# 3-D input: pre-computed FC time-series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_3d_shape(rec_metric):
    data = _make_3d(n_time=30, n_ch=4)
    out = RecurrenceMatrix(rec_metric).apply(data)
    assert out.data.dims == ("t1", "t2")
    assert out.data.shape == (30, 30)


def test_3d_cosine_diagonal_is_one():
    out = RecurrenceMatrix("cosine").apply(_make_3d(20, 4))
    np.testing.assert_allclose(np.diag(out.data.values), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_statevector_symmetric(rec_metric):
    mat = RecurrenceMatrix(rec_metric).apply(_make_2d(60)).data.values
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


@pytest.mark.parametrize("rec_metric", ["cosine", "correlation", "euclidean"])
def test_3d_symmetric(rec_metric):
    mat = RecurrenceMatrix(rec_metric).apply(_make_3d(20)).data.values
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

def test_metadata_preserved():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((50, 3))
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=200.0,
        subjectID="sub-01", groupID="ctrl",
    )
    out = RecurrenceMatrix().apply(data)
    assert out.subjectID == "sub-01"
    assert out.groupID == "ctrl"
    assert out.sampling_rate is None


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_rec_metric_raises():
    with pytest.raises(ValueError, match="rec_metric"):
        RecurrenceMatrix("manhattan")


def test_invalid_fc_metric_in_options_raises():
    with pytest.raises(ValueError, match="fc_options"):
        RecurrenceMatrix("cosine", ["kendall"])


def test_too_many_fc_options_raises():
    with pytest.raises(ValueError, match="fc_options"):
        RecurrenceMatrix("cosine", ["pearson", 10, 0.5, "extra"])


def test_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        RecurrenceMatrix("cosine", ["pearson", 10, 1.0])


def test_window_size_too_large_raises():
    data = _make_2d(n_time=30)
    with pytest.raises(ValueError, match="window_size"):
        RecurrenceMatrix("cosine", ["pearson", 30]).apply(data)


def test_missing_time_dim_raises():
    arr = np.ones((5, 5))
    data = cb.Data.from_numpy(arr, dims=["space", "freq"])
    with pytest.raises(ValueError, match="'time' dimension"):
        RecurrenceMatrix().apply(data)
