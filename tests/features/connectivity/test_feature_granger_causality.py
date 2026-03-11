from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _create_causal_signal(
    n_samples: int = 200, causal_strength: float = 0.3, seed: int = 42
) -> np.ndarray:
    """Create 2-channel signal where Y causes X."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=n_samples)
    X = np.zeros(n_samples)
    for t in range(1, n_samples):
        X[t] = 0.5 * X[t - 1] + causal_strength * Y[t - 1] + rng.normal() * 0.1
    return np.column_stack([X, Y])


def test_granger_causality_correct_value() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert isinstance(result, cb.Data)
    assert isinstance(result.data.values.item(), float)
    assert result.data.values.item() > 0


def test_granger_causality_detects_causality() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.data.values.item() > 0


def test_granger_causality_directionality() -> None:
    rng = np.random.default_rng(42)
    Y = rng.normal(size=200)
    X = np.zeros(200)
    for t in range(1, 200):
        X[t] = 0.5 * X[t - 1] + 0.3 * Y[t - 1] + rng.normal() * 0.1
    data = cb.data.SignalData.from_numpy(np.column_stack([X, Y]), dims=["time", "space"])

    feature_fwd = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    fwd_result = feature_fwd.apply(data)

    feature_bwd = cb.feature.GrangerCausality(coord_x=1, coord_y=0, lag=2)
    bwd_result = feature_bwd.apply(data)

    assert fwd_result.data.values.item() > 0
    assert bwd_result.data.values.item() < fwd_result.data.values.item()


def test_granger_causality_single_lag() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.data.shape == ()
    assert isinstance(result.data.values.item(), float)


def test_granger_causality_multiple_lags() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, maxlag=4)
    result = feature.apply(data)
    assert "lag_index" in result.data.dims
    assert len(result.data) == 4


def test_granger_causality_lag_precedence() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2, maxlag=10)
    result = feature.apply(data)
    assert result.data.shape == ()


def test_granger_causality_matrix_shape() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2)
    result = feature.apply(data)
    assert result.data.shape == (2, 2)


def test_granger_causality_matrix_diagonal_nan() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2)
    result = feature.apply(data)
    assert np.isnan(result.data.values[0, 0])
    assert np.isnan(result.data.values[1, 1])


def test_granger_causality_matrix_directional() -> None:
    rng = np.random.default_rng(42)
    Y = rng.normal(size=200)
    X = np.zeros(200)
    for t in range(1, 200):
        X[t] = 0.5 * X[t - 1] + 0.3 * Y[t - 1] + rng.normal() * 0.1
    data = cb.data.SignalData.from_numpy(np.column_stack([X, Y]), dims=["time", "space"])

    feature = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2)
    result = feature.apply(data)

    assert result.data.values[0, 1] > 0.1
    assert result.data.values[1, 0] < result.data.values[0, 1]


def test_granger_causality_matrix_default_coords() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausalityMatrix(lag=2)
    result = feature.apply(data)
    assert result.data.shape == (2, 2)


def test_granger_causality_returns_positive_for_causality() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200, causal_strength=0.5), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.data.values.item() > 0


def test_granger_causality_no_causality_small() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=200)
    Y = rng.normal(size=200)
    data = cb.data.SignalData.from_numpy(np.column_stack([X, Y]), dims=["time", "space"])

    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.data.values.item() >= 0


def test_granger_causality_requires_time_dimension() -> None:
    rng = np.random.default_rng(42)
    data = cb.from_numpy(rng.normal(size=(10, 3)), dims=["other", "space"])
    with pytest.raises(ValueError, match=r"Dimensions.*do not exist"):
        cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)


def test_granger_causality_requires_space_dimension() -> None:
    rng = np.random.default_rng(42)
    data = cb.data.SignalData.from_numpy(rng.normal(size=(10, 5)), dims=["time", "other"])
    with pytest.raises(KeyError):
        cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)


def test_granger_causality_invalid_coord() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    with pytest.raises(IndexError):
        cb.feature.GrangerCausality(coord_x=0, coord_y=5, lag=2).apply(data)


def test_granger_causality_invalid_lag() -> None:
    with pytest.raises(ValueError, match="lag must be >= 1"):
        cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=0)


def test_granger_causality_invalid_maxlag() -> None:
    with pytest.raises(ValueError, match="maxlag must be >= 1"):
        cb.feature.GrangerCausality(coord_x=0, coord_y=1, maxlag=0)


def test_granger_causality_small_samples() -> None:
    rng = np.random.default_rng(42)
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(30, 2)), dims=["time", "space"], sampling_rate=100.0
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=5)
    result = feature.apply(data)
    assert isinstance(result, cb.Data)
    assert isinstance(result.data.values.item(), float)


def test_granger_causality_small_dataset() -> None:
    rng = np.random.default_rng(42)
    Y = rng.normal(size=50)
    X = np.zeros(50)
    for t in range(1, 50):
        X[t] = 0.5 * X[t - 1] + 0.4 * Y[t - 1] + rng.normal() * 0.1
    data = cb.data.SignalData.from_numpy(np.column_stack([X, Y]), dims=["time", "space"])

    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.data.values.item() > 0


def test_granger_causality_detects_known_causality() -> None:
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200, causal_strength=0.5), dims=["time", "space"]
    )

    feature_fwd = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    fwd_result = feature_fwd.apply(data)

    feature_bwd = cb.feature.GrangerCausality(coord_x=1, coord_y=0, lag=2)
    bwd_result = feature_bwd.apply(data)

    assert fwd_result.data.values.item() > 1.0
    assert bwd_result.data.values.item() < fwd_result.data.values.item()


def test_granger_causality_history_tracking() -> None:
    """GrangerCausality appends class name to history."""
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.history[-1] == "GrangerCausality"


def test_granger_causality_preserves_metadata() -> None:
    """GrangerCausality preserves subjectID, groupID, condition, sampling_rate."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="test_subject",
        groupID="test_group",
        condition="test_condition",
        sampling_rate=100.0,
    )
    feature = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2)
    result = feature.apply(data)
    assert result.subjectID == "test_subject"
    assert result.groupID == "test_group"
    assert result.condition == "test_condition"
    assert result.sampling_rate is None  # time dimension removed


def test_granger_causality_no_mutation() -> None:
    """GrangerCausality does not modify input Data."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="s1",
        groupID="g1",
        condition="rest",
        sampling_rate=100.0,
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
    assert data.subjectID == "s1"
    assert data.groupID == "g1"
    assert data.condition == "rest"
    assert data.sampling_rate == 100.0


def test_granger_causality_returns_data_instance() -> None:
    """GrangerCausality.apply() returns Data instance."""
    data = cb.from_numpy(_create_causal_signal(n_samples=200), dims=["time", "space"])
    result = cb.feature.GrangerCausality(coord_x=0, coord_y=1, lag=2).apply(data)
    assert isinstance(result, cb.Data)


def test_granger_causality_matrix_multiple_lags() -> None:
    """GrangerCausalityMatrix supports maxlag for testing multiple lags."""
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    feature = cb.feature.GrangerCausalityMatrix(coords=[0, 1], maxlag=4)
    result = feature.apply(data)
    assert result.data.shape == (2, 2, 4)
    assert "lag_index" in result.data.dims
    assert list(result.data.coords["lag_index"].values) == [1, 2, 3, 4]


def test_granger_causality_matrix_invalid_lag() -> None:
    """GrangerCausalityMatrix raises ValueError for lag < 1."""
    with pytest.raises(ValueError, match="lag must be >= 1"):
        cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=0)


def test_granger_causality_matrix_invalid_maxlag() -> None:
    """GrangerCausalityMatrix raises ValueError for maxlag < 1."""
    with pytest.raises(ValueError, match="maxlag must be >= 1"):
        cb.feature.GrangerCausalityMatrix(coords=[0, 1], maxlag=0)


def test_granger_causality_matrix_returns_data_instance() -> None:
    """GrangerCausalityMatrix.apply() returns Data instance."""
    data = cb.data.SignalData.from_numpy(
        _create_causal_signal(n_samples=200), dims=["time", "space"]
    )
    result = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
    assert isinstance(result, cb.Data)


def test_granger_causality_matrix_history_updated() -> None:
    """GrangerCausalityMatrix appends class name to history."""
    data = cb.from_numpy(_create_causal_signal(n_samples=200), dims=["time", "space"])
    result = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
    assert result.history[-1] == "GrangerCausalityMatrix"


def test_granger_causality_matrix_preserves_metadata() -> None:
    """GrangerCausalityMatrix preserves subjectID, groupID, condition."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="s1",
        groupID="g1",
        condition="rest",
        sampling_rate=100.0,
    )
    result = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "g1"
    assert result.condition == "rest"
    assert result.sampling_rate is None  # time dimension removed


def test_granger_causality_matrix_no_mutation() -> None:
    """GrangerCausalityMatrix does not modify input Data."""
    data = cb.SignalData.from_numpy(
        _create_causal_signal(n_samples=200),
        dims=["time", "space"],
        subjectID="s1",
        groupID="g1",
        condition="rest",
        sampling_rate=100.0,
    )
    original_history = list(data.history)
    original_shape = data.data.shape

    _ = cb.feature.GrangerCausalityMatrix(coords=[0, 1], lag=2).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape


def test_granger_causality_matrix_empty_coords() -> None:
    """GrangerCausalityMatrix raises ValueError for empty coords list."""
    with pytest.raises(ValueError, match="coords cannot be an empty list"):
        cb.feature.GrangerCausalityMatrix(coords=[], lag=2)
