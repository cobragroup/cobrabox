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

    assert result.history[-1] == "PartialCorrelation"


def test_partial_correlation_matrix_preserves_history() -> None:
    """History is updated correctly after calling PartialCorrelationMatrix."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)

    assert result.history[-1] == "PartialCorrelationMatrix"


def test_partial_correlation_raises_when_no_space_dim() -> None:
    """Raises ValueError when data has no space dimension."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "channels"])
    with pytest.raises(ValueError, match="dimension 'space' not found"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)


def test_partial_correlation_raises_when_coord_y_not_found() -> None:
    """Raises ValueError when coord_y is missing (coord_x is valid)."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="coordinate '99' not found"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=99, control_vars=[2]).apply(data)


def test_partial_correlation_matrix_raises_when_no_space_dim() -> None:
    """Raises ValueError when data has no space dimension."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "channels"])
    with pytest.raises(ValueError, match="dimension 'space' not found"):
        cb.feature.PartialCorrelationMatrix(coords=[0], control_vars=[2]).apply(data)


def test_partial_correlation_matrix_raises_empty_control_vars() -> None:
    """Raises ValueError when control_vars is empty (coords is non-empty)."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="control_vars must have at least one coordinate"):
        cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[]).apply(data)


def test_partial_correlation_matrix_raises_invalid_coord() -> None:
    """Raises ValueError when a coord in coords is not in space."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="coordinate '99' not found"):
        cb.feature.PartialCorrelationMatrix(coords=[99, 1], control_vars=[2]).apply(data)


def test_partial_correlation_matrix_raises_invalid_control_var() -> None:
    """Raises ValueError when a coord in control_vars is not in space."""
    data = cb.SignalData.from_numpy(np.ones((10, 3)), dims=["time", "space"], sampling_rate=100.0)
    with pytest.raises(ValueError, match="control coordinate '99' not found"):
        cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[99]).apply(data)


def test_partial_correlation_with_multiple_controls() -> None:
    """Works correctly with multiple control variables."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 5)), dims=["time", "space"], sampling_rate=100.0
    )

    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2, 3]).apply(data)

    assert isinstance(result, cb.Data)
    assert result.data.values.shape == (1, 1)


def test_partial_correlation_raises_singular_matrix() -> None:
    """PartialCorrelation raises ValueError when correlation matrix is singular."""
    # Create perfectly correlated data (linearly dependent columns)
    # When variables are perfectly correlated, correlation matrix becomes singular
    base = np.arange(10, dtype=float)
    data = cb.SignalData.from_numpy(
        np.column_stack([base, base * 2, base * 3, base * 4]),
        dims=["time", "space"],
        sampling_rate=100.0,
    )
    with pytest.raises(ValueError, match="singular"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)


def test_partial_correlation_raises_missing_time_dim() -> None:
    """PartialCorrelation raises ValueError when time dimension is missing."""
    import xarray as xr

    # Create DataArray without time dimension
    xr_data = xr.DataArray(np.ones((5, 3)), dims=["space", "channels"])
    # Bypass Data validation to test feature guard
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="time"):
        cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(raw)


def test_partial_correlation_matrix_raises_missing_time_dim() -> None:
    """PartialCorrelationMatrix raises ValueError when time dimension is missing."""
    import xarray as xr

    xr_data = xr.DataArray(np.ones((5, 3)), dims=["space", "channels"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="time"):
        cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(raw)


def test_partial_correlation_metadata_preserved() -> None:
    """PartialCorrelation preserves subjectID, groupID, condition."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s1",
        groupID="g1",
        condition="rest",
    )
    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "g1"
    assert result.condition == "rest"


def test_partial_correlation_sampling_rate_none() -> None:
    """PartialCorrelation sets sampling_rate to None (output_type = Data)."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)
    assert result.sampling_rate is None


def test_partial_correlation_does_not_mutate_input() -> None:
    """PartialCorrelation does not modify input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PartialCorrelation(coord_x=0, coord_y=1, control_vars=[2]).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_partial_correlation_matrix_metadata_preserved() -> None:
    """PartialCorrelationMatrix preserves subjectID, groupID, condition."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s1",
        groupID="g1",
        condition="rest",
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)
    assert result.subjectID == "s1"
    assert result.groupID == "g1"
    assert result.condition == "rest"


def test_partial_correlation_matrix_sampling_rate_none() -> None:
    """PartialCorrelationMatrix sets sampling_rate to None (output_type = Data)."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)
    assert result.sampling_rate is None


def test_partial_correlation_matrix_does_not_mutate_input() -> None:
    """PartialCorrelationMatrix does not modify input Data object."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.PartialCorrelationMatrix(coords=[0, 1], control_vars=[2]).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_partial_correlation_matrix_diagonal_is_one() -> None:
    """PartialCorrelationMatrix diagonal equals 1.0 (self-correlation)."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1, 2], control_vars=[3]).apply(data)
    matrix = result.data.values
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_partial_correlation_matrix_is_symmetric() -> None:
    """PartialCorrelationMatrix output is symmetric."""
    data = cb.SignalData.from_numpy(
        rng.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0
    )
    result = cb.feature.PartialCorrelationMatrix(coords=[0, 1, 2], control_vars=[3]).apply(data)
    matrix = result.data.values
    np.testing.assert_allclose(matrix, matrix.T)
