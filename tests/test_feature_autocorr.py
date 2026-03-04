"""Tests for the autocorr feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _expected_autocorr(arr_1d: np.ndarray, lag: int) -> float:
    """Reference expected value (matches implementation using np.correlate)."""
    x = np.asarray(arr_1d, dtype=float).reshape(-1)

    if np.all(np.isnan(x)):
        return np.nan

    x = x - np.nanmean(x)
    x = np.nan_to_num(x)

    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]

    if corr[0] == 0:
        return np.nan

    corr = corr / corr[0]
    return float(corr[lag])


def test_feature_autocorr_reduces_requested_dimension() -> None:
    """autocorr reduces only the requested dimension and updates history."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0, subjectID="sub-01")

    out = cb.feature.autocorr(data, dim="time", fs=1000.0, lag_steps=1)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 1)

    expected = np.array(
        [[_expected_autocorr(arr[:, 0], lag=1)], [_expected_autocorr(arr[:, 1], lag=1)]],
        dtype=float,
    )

    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)

    assert out.subjectID == "sub-01"
    assert out.history == ["autocorr"]


def test_feature_autocorr_default_5ms_matches_explicit_steps() -> None:
    """Default lag (5 ms) matches lag_steps computed from sampling_rate."""
    arr = np.random.default_rng(0).normal(size=(50, 2)).astype(float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    out_default = cb.feature.autocorr(data, dim="time", fs=1000.0)
    out_steps = cb.feature.autocorr(data, dim="time", fs=1000.0, lag_steps=5)

    np.testing.assert_allclose(out_default.to_numpy(), out_steps.to_numpy(), equal_nan=True)

    assert out_default.history == ["autocorr"]
    assert out_steps.history == ["autocorr"]


def test_feature_autocorr_raises_for_unknown_dimension() -> None:
    """autocorr raises a clear error when dim is missing."""
    data = cb.from_numpy(np.ones((5, 3)), dims=["time", "space"], sampling_rate=1000.0)

    with pytest.raises(ValueError, match="dim 'band_index' not found"):
        cb.feature.autocorr(data, dim="band_index", fs=1000.0)


def test_feature_autocorr_raises_when_both_lag_inputs_provided() -> None:
    """autocorr raises when both lag_steps and lag_ms are provided."""
    data = cb.from_numpy(np.ones((10, 1)), dims=["time", "space"], sampling_rate=1000.0)

    with pytest.raises(ValueError, match="Specify either 'lag_steps' or 'lag_ms', not both"):
        cb.feature.autocorr(data, dim="time", fs=1000.0, lag_steps=5, lag_ms=5.0)


def test_feature_autocorr_raises_for_lag_out_of_range() -> None:
    """autocorr raises when lag is out of range."""
    data = cb.from_numpy(np.ones((5, 1)), dims=["time", "space"], sampling_rate=1000.0)

    with pytest.raises(ValueError, match="lag must be between 1 and 4"):
        cb.feature.autocorr(data, dim="time", fs=1000.0, lag_steps=5)


def test_feature_autocorr_constant_signal_returns_nan() -> None:
    """Constant signal yields NaN autocorrelation (zero-energy demeaned signal)."""
    arr = np.ones((30, 2), dtype=float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    out = cb.feature.autocorr(data, dim="time", fs=1000.0, lag_steps=1)

    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 1)
    assert np.all(np.isnan(out.to_numpy()))
    assert out.history == ["autocorr"]


def test_feature_autocorr_all_nan_returns_nan() -> None:
    """All-NaN input yields NaN autocorrelation."""
    arr = np.full((10, 2), np.nan, dtype=float)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=1000.0)

    out = cb.feature.autocorr(data, dim="time", fs=1000.0, lag_steps=1)

    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 1)
    assert np.all(np.isnan(out.to_numpy()))

    assert out.history == ["autocorr"]
