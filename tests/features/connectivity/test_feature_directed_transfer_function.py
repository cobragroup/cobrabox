"""Tests for the DirectedTransferFunction feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _causal_signal(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + 0.3 * Y[t - 1] + rng.normal() * 0.1
    return np.column_stack([X, Y])


def _data(arr: np.ndarray) -> cb.SignalData:
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=128.0)


def test_output_dims_and_shape() -> None:
    data = _data(_causal_signal(n=200))
    result = cb.DirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    assert result.data.dims == ("space_to", "space_from", "frequency")
    assert result.data.shape == (2, 2, 32)


def test_values_in_unit_interval() -> None:
    data = _data(_causal_signal(n=200))
    result = cb.DirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    vals = result.data.values
    assert np.all((vals >= 0) & (vals <= 1.0 + 1e-9))


def test_rows_sum_to_unity() -> None:
    """DTF² sums to 1 across each row (row-normalization invariant)."""
    data = _data(_causal_signal(n=400))
    result = cb.DirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    row_sums = (result.data.values**2).sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_matrix_is_asymmetric() -> None:
    """DTF differs from its transpose for genuinely directed signals."""
    data = _data(_causal_signal(n=400))
    result = cb.DirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    vals = result.data.values
    assert not np.allclose(vals, vals.transpose(1, 0, 2), atol=1e-3)


def test_requires_sampling_rate() -> None:
    arr = _causal_signal(n=100)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=128.0)
    data.data.attrs["sampling_rate"] = None
    no_sr = cb.SignalData.from_numpy(arr, dims=["time", "space"])
    no_sr.data.attrs.pop("sampling_rate", None)
    with pytest.raises(ValueError, match="sampling_rate"):
        cb.DirectedTransferFunction().apply(no_sr)


def test_rejects_single_channel() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((200, 1))
    data = _data(arr)
    with pytest.raises(ValueError, match="at least 2 channels"):
        cb.DirectedTransferFunction().apply(data)


def test_invalid_var_order() -> None:
    with pytest.raises(ValueError, match="var_order"):
        cb.DirectedTransferFunction(var_order=0)


def test_invalid_n_freqs() -> None:
    with pytest.raises(ValueError, match="n_freqs"):
        cb.DirectedTransferFunction(n_freqs=0)


def test_history_appended() -> None:
    data = _data(_causal_signal(n=200))
    result = cb.DirectedTransferFunction(var_order=2, n_freqs=16).apply(data)
    assert result.history[-1] == "DirectedTransferFunction"
