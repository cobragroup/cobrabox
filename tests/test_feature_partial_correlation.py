"""Tests for partial correlation feature behavior."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb

rng = np.random.default_rng(42)


def _manual_partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Compute partial correlation manually for verification."""
    all_vars = np.column_stack([x, y, z])
    corr = np.corrcoef(all_vars, rowvar=False)
    prec = np.linalg.inv(corr)
    return -prec[0, 1] / np.sqrt(prec[0, 0] * prec[1, 1])


def test_partial_correlation_returns_float() -> None:
    """PartialCorrelation returns a valid Data object with correct value."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.dims == ("space", "time")
    assert result.data.shape == (1, 1)

    expected = _manual_partial_correlation(
        data.data.sel(space=0).values, data.data.sel(space=1).values, data.data.sel(space=2).values
    )
    np.testing.assert_allclose(result.data.values.item(), expected, rtol=1e-5)


def test_partial_correlation_matrix_returns_square_matrix() -> None:
    """PartialCorrelationMatrix returns correct square shape."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1, 2], control_vars=[3]).apply(data)

    assert isinstance(result, cb.Data)
    matrix_values = result.data.values
    assert matrix_values.shape == (3, 3)


def test_partial_correlation_diagonal_is_one() -> None:
    """Partial correlation of coordinate with itself equals 1.0."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=0, control_vars=[2]).apply(data)

    np.testing.assert_allclose(result.data.values.item(), 1.0)


def test_partial_correlation_raises_empty_control_vars() -> None:
    """Raises ValueError when control_vars is empty."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="control_vars must have at least one coordinate"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[]).apply(data)


def test_partial_correlation_raises_invalid_coordinate() -> None:
    """Raises ValueError when coordinate is not found in space dimension."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="coordinate '99' not found"):
        cb.feature.PartialCorrelation(coord_x=99, coord_y=1, control_vars=[2]).apply(data)


def test_partial_correlation_raises_invalid_control_coordinate() -> None:
    """Raises ValueError when control variable coordinate is not found."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="control coordinate '99' not found"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[99]).apply(data)


def test_partial_correlation_matrix_raises_empty_coords() -> None:
    """Raises ValueError when coords list is empty."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="coords must have at least one coordinate"):
        cb.feature.PartialCorrelationMatrix(coords=[], control_vars=[2]).apply(data)


def test_partial_correlation_preserves_history() -> None:
    """History is updated correctly after calling PartialCorrelation."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)

    assert "PartialCorrelation" in result.history


def test_partial_correlation_matrix_preserves_history() -> None:
    """History is updated correctly after calling PartialCorrelationMatrix."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)

    assert "PartialCorrelationMatrix" in result.history


def test_partial_correlation_with_multiple_controls() -> None:
    """Works correctly with multiple control variables."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 5)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2, 3]).apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.values.shape == (1, 1)
