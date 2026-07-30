"""Tests for the consolidated PartialCorrelation matrix feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _data(arr: np.ndarray) -> cb.SignalData:
    return cb.SignalData.from_numpy(arr, dims=["time", "space"])


def test_returns_full_matrix_by_default() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((400, 3))
    data = _data(arr)
    result = cb.PartialCorrelation().apply(data)
    assert result.data.dims == ("space_to", "space_from")
    assert result.data.shape == (3, 3)


def test_diagonal_is_one() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((400, 3))
    data = _data(arr)
    result = cb.PartialCorrelation().apply(data)
    np.testing.assert_allclose(np.diag(result.data.values), 1.0)


def test_matrix_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((400, 3))
    data = _data(arr)
    result = cb.PartialCorrelation().apply(data)
    np.testing.assert_allclose(result.data.values, result.data.values.T, atol=1e-10)


def test_coords_restricts_output() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((400, 4))
    data = _data(arr)
    result = cb.PartialCorrelation(coords=[0, 2]).apply(data)
    assert result.data.shape == (2, 2)
    assert list(result.data.coords["space_to"].values) == [0, 2]


def test_control_vars_extend_precision_matrix() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((400, 4))
    data = _data(arr)
    result = cb.PartialCorrelation(coords=[0, 1], control_vars=[3]).apply(data)
    assert result.data.shape == (2, 2)


def test_coords_empty_raises() -> None:
    with pytest.raises(ValueError, match="coords cannot be an empty list"):
        cb.PartialCorrelation(coords=[])


def test_invalid_coord_raises() -> None:
    rng = np.random.default_rng(0)
    data = _data(rng.standard_normal((100, 3)))
    with pytest.raises(ValueError, match="not found in space dimension"):
        cb.PartialCorrelation(coords=[99]).apply(data)


def test_invalid_control_var_raises() -> None:
    rng = np.random.default_rng(0)
    data = _data(rng.standard_normal((100, 3)))
    with pytest.raises(ValueError, match="not found in space dimension"):
        cb.PartialCorrelation(coords=[0], control_vars=[99]).apply(data)


def test_singular_matrix_raises() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    arr = np.column_stack([x, x, rng.standard_normal(200)])
    data = _data(arr)
    with pytest.raises(ValueError, match="singular"):
        cb.PartialCorrelation().apply(data)


def test_history_appended() -> None:
    rng = np.random.default_rng(0)
    data = _data(rng.standard_normal((200, 3)))
    result = cb.PartialCorrelation().apply(data)
    assert result.history[-1] == "PartialCorrelation"


def test_metadata_preserved() -> None:
    rng = np.random.default_rng(0)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((200, 3)), dims=["time", "space"], subjectID="sub-01"
    )
    result = cb.PartialCorrelation().apply(data)
    assert result.subjectID == "sub-01"


def test_does_not_mutate_input() -> None:
    rng = np.random.default_rng(0)
    data = _data(rng.standard_normal((200, 3)))
    snapshot = data.data.values.copy()
    _ = cb.PartialCorrelation().apply(data)
    np.testing.assert_array_equal(data.data.values, snapshot)
