"""Tests for the DirectDirectedTransferFunction (dDTF) feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _causal_chain(n: int = 400, seed: int = 0) -> np.ndarray:
    """Create a 3-channel chain: Z -> X -> Y (X mediates Z's influence on Y)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    X = np.zeros(n)
    Y = np.zeros(n)
    for t in range(1, n):
        X[t] = 0.5 * Z[t - 1] + rng.normal() * 0.1
        Y[t] = 0.5 * X[t - 1] + rng.normal() * 0.1
    return np.column_stack([X, Y, Z])


def _causal_pair(n: int = 400, seed: int = 0) -> np.ndarray:
    """Create a 2-channel signal: Y -> X (direct)."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=n)
    X = np.zeros(n)
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + 0.3 * Y[t - 1] + rng.normal() * 0.1
    return np.column_stack([X, Y])


def _data(arr: np.ndarray) -> cb.SignalData:
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=128.0)


def test_output_dims_and_shape() -> None:
    data = _data(_causal_pair(n=200))
    result = cb.DirectDirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    assert result.data.dims == ("space_to", "space_from", "frequency")
    assert result.data.shape == (2, 2, 32)


def test_output_dims_three_channels() -> None:
    data = _data(_causal_chain(n=300))
    result = cb.DirectDirectedTransferFunction(var_order=2, n_freqs=16).apply(data)
    assert result.data.shape == (3, 3, 16)


def test_values_non_negative() -> None:
    data = _data(_causal_pair(n=200))
    result = cb.DirectDirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    assert np.all(result.data.values >= -1e-12)


def test_values_bounded_above() -> None:
    data = _data(_causal_pair(n=200))
    result = cb.DirectDirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    assert np.all(result.data.values <= 1.0 + 1e-9)


def test_ddtf_suppresses_indirect_path() -> None:
    """In Z -> X -> Y, dDTF should show weaker Z->Y than DTF does."""
    data = _data(_causal_chain(n=800))
    dtf_result = cb.DirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    ddtf_result = cb.DirectDirectedTransferFunction(var_order=2, n_freqs=32).apply(data)

    # Z is channel 2, Y is channel 1
    # DTF Z->Y (indirect, through X) should be larger than dDTF Z->Y
    dtf_zy = dtf_result.data.sel(space_to=1, space_from=2).values.mean()
    ddtf_zy = ddtf_result.data.sel(space_to=1, space_from=2).values.mean()

    assert ddtf_zy < dtf_zy, (
        f"dDTF Z->Y ({ddtf_zy:.4f}) should be smaller than DTF Z->Y ({dtf_zy:.4f}) "
        "because Z->Y is indirect (mediated by X)"
    )


def test_matrix_is_asymmetric() -> None:
    data = _data(_causal_pair(n=400))
    result = cb.DirectDirectedTransferFunction(var_order=2, n_freqs=32).apply(data)
    vals = result.data.values
    assert not np.allclose(vals, vals.transpose(1, 0, 2), atol=1e-3)


def test_requires_sampling_rate() -> None:
    arr = _causal_pair(n=100)
    no_sr = cb.SignalData.from_numpy(arr, dims=["time", "space"])
    no_sr.data.attrs.pop("sampling_rate", None)
    with pytest.raises(ValueError, match="sampling_rate"):
        cb.DirectDirectedTransferFunction().apply(no_sr)


def test_rejects_single_channel() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((200, 1))
    data = _data(arr)
    with pytest.raises(ValueError, match="at least 2 channels"):
        cb.DirectDirectedTransferFunction().apply(data)


def test_invalid_var_order() -> None:
    with pytest.raises(ValueError, match="var_order"):
        cb.DirectDirectedTransferFunction(var_order=0)


def test_invalid_n_freqs() -> None:
    with pytest.raises(ValueError, match="n_freqs"):
        cb.DirectDirectedTransferFunction(n_freqs=0)


def test_history_appended() -> None:
    data = _data(_causal_pair(n=200))
    result = cb.DirectDirectedTransferFunction(var_order=2, n_freqs=16).apply(data)
    assert result.history[-1] == "DirectDirectedTransferFunction"


def test_functional_api() -> None:
    data = _data(_causal_pair(n=200))
    result = cb.direct_directed_transfer_function(data, var_order=2, n_freqs=16)
    assert isinstance(result, cb.Data)
    assert result.data.dims == ("space_to", "space_from", "frequency")


def test_serialization_round_trip() -> None:
    feature = cb.DirectDirectedTransferFunction(var_order=3, n_freqs=64)
    yaml_str = feature.to_yaml()
    reloaded = cb.deserialize(yaml_str)
    assert isinstance(reloaded, cb.Pipeline)
    nf = reloaded.features[0]
    assert isinstance(nf, cb.DirectDirectedTransferFunction)
    assert nf.var_order == 3
    assert nf.n_freqs == 64
