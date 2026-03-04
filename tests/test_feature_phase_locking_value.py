"""Tests for phase locking value feature behavior."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

import cobrabox as cb

rng = np.random.default_rng(42)


def _manual_phase_locking_value(x: np.ndarray, y: np.ndarray) -> float:
    """Compute phase locking value manually for verification."""
    return np.abs(np.mean(np.exp(1j * (np.angle(hilbert(x)) - np.angle(hilbert(y))))))


def test_phase_locking_value_returns_float() -> None:
    """phase_locking_value returns a valid Data object with correct value."""
    data = cb.from_numpy(rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

    result = cb.feature.phase_locking_value(data, 0, 1)

    assert isinstance(result, cb.Data)
    assert result.data.dims == ("space", "time")
    assert result.data.shape == (1, 1)

    expected = _manual_phase_locking_value(
        data.data.sel(space=0).values, data.data.sel(space=1).values
    )
    np.testing.assert_allclose(result.data.values.item(), expected, rtol=1e-5)


def test_phase_locking_value_matrix_returns_square_matrix() -> None:
    """phase_locking_value_matrix returns correct square shape."""
    data = cb.from_numpy(rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

    result = cb.feature.phase_locking_value_matrix(data, [0, 1, 2])

    assert isinstance(result, cb.Data)
    matrix_values = result.data.values[0, :, :, 0]
    assert matrix_values.shape == (3, 3)


def test_phase_locking_value_diagonal_is_one() -> None:
    """Partial correlation of coordinate with itself equals 1.0."""
    data = cb.from_numpy(rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

    result = cb.feature.phase_locking_value(data, 0, 0)

    np.testing.assert_allclose(result.data.values.item(), 1.0)


def test_phase_locking_value_raises_invalid_coordinate() -> None:
    """Raises ValueError when coordinate is not found in space dimension."""
    data = cb.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="coordinate '99' not found"):
        cb.feature.phase_locking_value(data, 99, 1)


def test_phase_locking_value_matrix_raises_empty_coords() -> None:
    """Raises ValueError when coords list is empty."""
    data = cb.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="coords must have at least one coordinate"):
        cb.feature.phase_locking_value_matrix(data, [])


def test_phase_locking_value_preserves_history() -> None:
    """History is updated correctly after calling phase_locking_value."""
    data = cb.from_numpy(rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

    result = cb.feature.phase_locking_value(data, 0, 1)

    assert "phase_locking_value" in result.history


def test_phase_locking_value_matrix_preserves_history() -> None:
    """History is updated correctly after calling phase_locking_value_matrix."""
    data = cb.from_numpy(rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

    result = cb.feature.phase_locking_value_matrix(data, [0, 1])

    assert "phase_locking_value_matrix" in result.history
