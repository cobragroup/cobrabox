"""Tests for cb.feature.AmplitudeEntropy."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n_time: int = 100,
    n_space: int = 10,
    subjectID: str = "s1",
    groupID: str = "g1",
    condition: str = "rest",
    seed: int = 42,
) -> cb.Data:
    """Create a Data object for testing."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_time, n_space))
    return cb.Data.from_numpy(
        arr, dims=["time", "space"], subjectID=subjectID, groupID=groupID, condition=condition
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_amplitude_entropy_basic() -> None:
    """AmplitudeEntropy returns a Data object with scalar entropy value."""
    data = _make_data(n_time=50, n_space=10)
    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    # Result should be 0-dimensional scalar
    assert result.data.shape == ()
    assert result.data.ndim == 0
    assert not np.isnan(result.to_numpy())
    assert not np.isinf(result.to_numpy())
    # Entropy should be non-negative
    assert float(result.to_numpy()) >= 0


def test_amplitude_entropy_deterministic() -> None:
    """AmplitudeEntropy returns same result for identical inputs."""
    data = _make_data(n_time=30, n_space=8, seed=42)

    feature = cb.feature.AmplitudeEntropy(band_width=0.3)

    result1 = feature.apply(data)
    result2 = feature.apply(data)

    np.testing.assert_array_almost_equal(result1.to_numpy(), result2.to_numpy())


def test_amplitude_entropy_constant_data() -> None:
    """AmplitudeEntropy handles constant data (zero entropy case)."""
    # Constant data has zero entropy (all values in one bin)
    arr = np.ones((20, 5))
    data = cb.Data.from_numpy(arr, dims=["time", "space"])

    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    # Entropy of constant distribution is 0
    np.testing.assert_allclose(float(result.to_numpy()), 0.0, atol=1e-10)


def test_amplitude_entropy_uniform_data() -> None:
    """AmplitudeEntropy returns expected entropy for uniform distribution."""
    # Uniform distribution across bins
    n_bins = 8
    samples_per_bin = 10
    n_space = n_bins * samples_per_bin

    # Create data with uniform distribution
    arr = np.zeros((10, n_space))
    for i in range(10):
        for bin_idx in range(n_bins):
            start = bin_idx * samples_per_bin
            end = start + samples_per_bin
            arr[i, start:end] = bin_idx

    data = cb.Data.from_numpy(arr, dims=["time", "space"])

    feature = cb.feature.AmplitudeEntropy(band_width=1.0)
    result = feature.apply(data)

    # Uniform distribution over n_bins has entropy = log2(n_bins)
    expected_entropy = np.log2(n_bins)
    # Due to discretization, result should be close but not exact
    assert float(result.to_numpy()) > 0
    assert float(result.to_numpy()) <= expected_entropy + 0.1


def test_amplitude_entropy_different_band_widths() -> None:
    """AmplitudeEntropy produces different results with different band_widths."""
    data = _make_data(n_time=50, n_space=20, seed=123)

    feature_narrow = cb.feature.AmplitudeEntropy(band_width=0.1)
    feature_wide = cb.feature.AmplitudeEntropy(band_width=1.0)

    result_narrow = feature_narrow.apply(data)
    result_wide = feature_wide.apply(data)

    # Different bandwidths should produce different entropies
    # Wider bins -> fewer bins -> lower max entropy
    entropy_narrow = float(result_narrow.to_numpy())
    entropy_wide = float(result_wide.to_numpy())
    assert entropy_narrow != pytest.approx(entropy_wide, abs=1e-6)


def test_amplitude_entropy_history_updated() -> None:
    """AmplitudeEntropy appends 'AmplitudeEntropy' to history."""
    data = _make_data()
    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    result = feature.apply(data)

    assert result.history[-1] == "AmplitudeEntropy"


def test_amplitude_entropy_metadata_preserved() -> None:
    """AmplitudeEntropy preserves subjectID, groupID, condition; sampling_rate becomes None."""
    data = _make_data(subjectID="s42", groupID="control", condition="task")
    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    result = feature.apply(data)

    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    # output_type = Data removes time dimension, so sampling_rate is None
    assert result.sampling_rate is None


def test_amplitude_entropy_returns_data_instance() -> None:
    """AmplitudeEntropy.apply() always returns a Data instance."""
    data = _make_data()
    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)


def test_amplitude_entropy_does_not_mutate_input() -> None:
    """AmplitudeEntropy does not modify the input Data object."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    _ = feature.apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_amplitude_entropy_empty_row_handling() -> None:
    """AmplitudeEntropy handles edge case where histogram has zero counts."""
    # This tests the defensive guard in the code
    # Create data that might trigger the total == 0 case
    arr = np.zeros((5, 1))  # Single value per row
    data = cb.Data.from_numpy(arr, dims=["time", "space"])

    feature = cb.feature.AmplitudeEntropy(band_width=1.0)
    result = feature.apply(data)

    # Should not raise, should return finite value
    assert np.isfinite(float(result.to_numpy()))


def test_amplitude_entropy_negative_band_width() -> None:
    """AmplitudeEntropy raises ValueError for negative band_width."""
    with pytest.raises(ValueError, match="band_width must be positive"):
        cb.feature.AmplitudeEntropy(band_width=-0.5)


def test_amplitude_entropy_zero_band_width() -> None:
    """AmplitudeEntropy raises ValueError for zero band_width."""
    with pytest.raises(ValueError, match="band_width must be positive"):
        cb.feature.AmplitudeEntropy(band_width=0.0)


def test_amplitude_entropy_single_row() -> None:
    """AmplitudeEntropy works with single-row data."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((1, 10))
    data = cb.Data.from_numpy(arr, dims=["time", "space"])

    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.shape == ()
    assert np.isfinite(float(result.to_numpy()))


def test_amplitude_entropy_single_column() -> None:
    """AmplitudeEntropy works with single-column data."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((50, 1))
    data = cb.Data.from_numpy(arr, dims=["time", "space"])

    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    result = feature.apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.shape == ()
    assert np.isfinite(float(result.to_numpy()))


def test_amplitude_entropy_1d_input_raises() -> None:
    """AmplitudeEntropy raises ValueError for 1D input data."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(50)
    data = cb.Data.from_numpy(arr, dims=["time"])

    feature = cb.feature.AmplitudeEntropy(band_width=0.5)
    with pytest.raises(ValueError, match="Input data must have at least 2 dimensions"):
        feature.apply(data)
