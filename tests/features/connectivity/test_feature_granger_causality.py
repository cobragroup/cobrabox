"""Tests for the consolidated GrangerCausality matrix feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _create_causal_signal(
    n_samples: int = 200, causal_strength: float = 0.3, seed: int = 42
) -> np.ndarray:
    """Create 2-channel signal where Y causes X (so X is sink, Y is source).

    Layout: column 0 = X, column 1 = Y. In matrix notation,
    ``result[space_to=X, space_from=Y]`` should be positive (Y causes X).
    """
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=n_samples)
    X = np.zeros(n_samples)
    for t in range(1, n_samples):
        X[t] = 0.5 * X[t - 1] + causal_strength * Y[t - 1] + rng.normal() * 0.1
    return np.column_stack([X, Y])


def _signal_data(arr: np.ndarray) -> cb.SignalData:
    return cb.SignalData.from_numpy(arr, dims=["time", "space"])


def test_returns_full_matrix_with_default_coords() -> None:
    data = _signal_data(_create_causal_signal(n_samples=200))
    result = cb.feature.GrangerCausality(lag=2).apply(data)
    assert result.data.shape == (2, 2)
    assert result.data.dims == ("space_to", "space_from")


def test_diagonal_is_nan() -> None:
    data = _signal_data(_create_causal_signal(n_samples=200))
    result = cb.feature.GrangerCausality(lag=2).apply(data)
    assert np.isnan(result.data.values[0, 0])
    assert np.isnan(result.data.values[1, 1])


def test_detects_known_causal_direction() -> None:
    """Y → X is encoded; result[X, Y] should exceed result[Y, X]."""
    data = _signal_data(_create_causal_signal(n_samples=200))
    result = cb.feature.GrangerCausality(lag=2).apply(data)
    # space_to=0 is X (sink), space_from=1 is Y (source) → strong
    assert result.data.values[0, 1] > 0
    # reverse direction (X → Y) should be weaker
    assert result.data.values[1, 0] < result.data.values[0, 1]


def test_coords_subset_restricts_output() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((200, 4))
    data = _signal_data(arr)
    result = cb.feature.GrangerCausality(coords=[0, 2], lag=2).apply(data)
    assert result.data.shape == (2, 2)
    assert list(result.data.coords["space_to"].values) == [0, 2]


def test_coords_empty_raises() -> None:
    with pytest.raises(ValueError, match="coords cannot be an empty list"):
        cb.feature.GrangerCausality(coords=[])


def test_maxlag_adds_lag_index_dim() -> None:
    data = _signal_data(_create_causal_signal(n_samples=200))
    result = cb.feature.GrangerCausality(maxlag=4).apply(data)
    assert "lag_index" in result.data.dims
    assert result.data.sizes["lag_index"] == 4


def test_lag_takes_precedence_over_maxlag() -> None:
    data = _signal_data(_create_causal_signal(n_samples=200))
    result = cb.feature.GrangerCausality(lag=2, maxlag=10).apply(data)
    assert "lag_index" not in result.data.dims


def test_invalid_lag_raises() -> None:
    with pytest.raises(ValueError, match="lag must be >= 1"):
        cb.feature.GrangerCausality(lag=0)


def test_invalid_maxlag_raises() -> None:
    with pytest.raises(ValueError, match="maxlag must be >= 1"):
        cb.feature.GrangerCausality(maxlag=0)


def test_invalid_coord_raises() -> None:
    data = _signal_data(_create_causal_signal(n_samples=200))
    with pytest.raises(ValueError, match="not found in space dimension"):
        cb.feature.GrangerCausality(coords=[99]).apply(data)


def test_history_appended() -> None:
    data = _signal_data(_create_causal_signal(n_samples=200))
    result = cb.feature.GrangerCausality(lag=2).apply(data)
    assert result.history[-1] == "GrangerCausality"


def test_metadata_preserved() -> None:
    arr = _create_causal_signal(n_samples=200)
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], subjectID="sub-01", groupID="g1", condition="rest"
    )
    result = cb.feature.GrangerCausality(lag=2).apply(data)
    assert result.subjectID == "sub-01"
    assert result.groupID == "g1"
    assert result.condition == "rest"


def test_does_not_mutate_input() -> None:
    data = _signal_data(_create_causal_signal(n_samples=200))
    snapshot = data.data.values.copy()
    _ = cb.feature.GrangerCausality(lag=2).apply(data)
    np.testing.assert_array_equal(data.data.values, snapshot)
