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


def test_phase_locking_value_metadata_preserved() -> None:
    """PhaseLockingValue preserves subjectID, groupID, condition; sampling_rate is None."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate is None  # time dimension removed


def test_phase_locking_value_matrix_metadata_preserved() -> None:
    """PhaseLockingValueMatrix preserves subjectID, groupID, condition; sampling_rate is None."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s1",
        groupID="patient",
        condition="rest",
    )
    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1]).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "patient"
    assert result.condition == "rest"
    assert result.sampling_rate is None  # time dimension removed


def test_phase_locking_value_does_not_mutate_input() -> None:
    """PhaseLockingValue does not modify the input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_phase_locking_value_matrix_does_not_mutate_input() -> None:
    """PhaseLockingValueMatrix does not modify the input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PhaseLockingValueMatrix(coords=[0, 1]).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_phase_locking_value_matrix_symmetric() -> None:
    """PLV matrix should be symmetric (PLV[i,j] == PLV[j,i])."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)
    matrix = result.to_numpy()

    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10)


def test_phase_locking_value_in_range() -> None:
    """PLV values should be in [0, 1]."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValue(coord_x=0, coord_y=1).apply(data)
    value = float(result.data.values)

    assert 0.0 <= value <= 1.0


def test_phase_locking_value_matrix_in_range() -> None:
    """All PLV matrix values should be in [0, 1]."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)
    matrix = result.to_numpy()

    assert np.all((matrix >= 0.0) & (matrix <= 1.0))


def test_phase_locking_value_with_string_coords() -> None:
    """PhaseLockingValue works with string coordinate labels."""
    arr = rng.normal(size=(100, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    # Relabel space dimension with strings
    data = cb.Data.from_xarray(
        data.data.assign_coords(space=["ch0", "ch1", "ch2"]), sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValue(coord_x="ch0", coord_y="ch1").apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.shape == ()


def test_phase_locking_value_matrix_diagonal_all_ones() -> None:
    """Diagonal of PLV matrix should be all 1.0 (self-correlation)."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PhaseLockingValueMatrix(coords=[0, 1, 2]).apply(data)
    matrix = result.to_numpy()

    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_phase_locking_value_via_chord() -> None:
    """PhaseLockingValue works correctly within a Chord pipeline."""
    data = cb.data.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=50, step_size=25),
        pipeline=cb.feature.PhaseLockingValue(coord_x=0, coord_y=1),
        aggregate=cb.feature.MeanAggregate(),
    )
    result = chord.apply(data)

    assert isinstance(result, cb.Data)
    assert "PhaseLockingValue" in result.history
    assert "MeanAggregate" in result.history
    assert "SlidingWindow" in result.history
