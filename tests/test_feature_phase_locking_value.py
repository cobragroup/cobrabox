"""Tests for phase locking value feature behavior."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

import cobrabox as cb

rng = np.random.default_rng(42)


def _manual_phase_locking_value(x: np.ndarray, y: np.ndarray) -> float:
    """Compute phase locking value manually for verification."""
    return float(np.abs(np.mean(np.exp(1j * (np.angle(hilbert(x)) - np.angle(hilbert(y)))))))


def test_phase_locking_value_returns_float() -> None:
    """PhaseLockingValue returns a valid Data object with correct value."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.dims == ()
    assert result.data.shape == ()

    expected = _manual_phase_locking_value(
        data.data.sel(space=0).values, data.data.sel(space=1).values
    )
    np.testing.assert_allclose(float(result.data.values), expected, rtol=1e-5)


def test_phase_locking_value_matrix_returns_square_matrix() -> None:
    """PhaseLockingValueMatrix returns correct square shape."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.dims == ("coord_i", "coord_j")
    assert result.data.shape == (3, 3)


def test_phase_locking_value_diagonal_is_one() -> None:
    """PLV of a coordinate with itself equals 1.0."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=0).apply(data)

    np.testing.assert_allclose(float(result.data.values), 1.0)


def test_phase_locking_value_raises_invalid_coordinate() -> None:
    """Raises ValueError when coordinate is not found in space dimension."""
    data = cb.data.SignalData.from_numpy(
        np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0
    )

    with pytest.raises(ValueError, match="coordinate '99' not found"):
        cb.feature.PhaseLockingValue(coord_x=99, coord_y=1).apply(data)


def test_phase_locking_value_matrix_raises_empty_coords() -> None:
    """Raises ValueError when coords list is empty."""
    data = cb.data.SignalData.from_numpy(
        np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0
    )

    with pytest.raises(ValueError, match="coords must have at least one coordinate"):
        cb.feature.PhaseLockingValueMatrix(coords=[]).apply(data)


def test_phase_locking_value_preserves_history() -> None:
    """History is updated correctly after calling PhaseLockingValue."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)

    assert result.history == ["PhaseLockingValue"]


def test_phase_locking_value_raises_when_no_space_dim() -> None:
    """Raises ValueError when data has no space dimension."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "channels"])
    with pytest.raises(ValueError, match="dimension 'space' not found"):
        cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)


def test_phase_locking_value_raises_when_coord_y_not_found() -> None:
    """Raises ValueError when coord_y is missing (coord_x is valid)."""
    data = cb.data.SignalData.from_numpy(
        np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0
    )
    with pytest.raises(ValueError, match="coordinate '99' not found"):
        cb.feature.PhaseLockingValue(coord_x=0, coord_y=99).apply(data)


def test_phase_locking_value_matrix_raises_when_no_space_dim() -> None:
    """Raises ValueError when data has no space dimension."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "channels"])
    with pytest.raises(ValueError, match="dimension 'space' not found"):
        cb.feature.PhaseLockingValueMatrix(coords=[0, 1]).apply(data)


def test_phase_locking_value_matrix_raises_when_coord_not_found() -> None:
    """Raises ValueError when a coord in coords is not in space."""
    data = cb.data.SignalData.from_numpy(
        np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0
    )
    with pytest.raises(ValueError, match="coordinate '99' not found"):
        cb.feature.PhaseLockingValueMatrix(coords=[99, 1]).apply(data)


def test_phase_locking_value_matrix_preserves_history() -> None:
    """History is updated correctly after calling PhaseLockingValueMatrix."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1]).apply(data)

    assert result.history == ["PhaseLockingValueMatrix"]
