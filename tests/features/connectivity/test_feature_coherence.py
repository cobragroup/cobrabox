"""Tests for the Coherence feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(arr: np.ndarray, *, sampling_rate: float = 100.0) -> cb.SignalData:
    """Create Data from a (time, space) array."""
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_coherence_output_dims_and_shape() -> None:
    """Coherence returns Data with (space, space_from) dims and NxN matrix."""
    rng = np.random.default_rng(0)
    data = _make_data(rng.standard_normal((300, 4)))

    out = cb.feature.Coherence().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space_to", "space_from")
    assert out.data.sizes["space_to"] == 4
    assert out.data.sizes["space_from"] == 4


def test_coherence_space_coords_are_preserved() -> None:
    """Both space and space_from carry the original channel coordinates."""
    arr_xr = xr.DataArray(
        np.random.default_rng(1).standard_normal((300, 3)),
        dims=["time", "space"],
        coords={"space": ["Fz", "Cz", "Pz"], "time": np.arange(300, dtype=float) / 100.0},
    )
    data = cb.data.SignalData.from_xarray(arr_xr)

    out = cb.feature.Coherence().apply(data)

    np.testing.assert_array_equal(out.data.coords["space_to"].values, ["Fz", "Cz", "Pz"])
    np.testing.assert_array_equal(out.data.coords["space_from"].values, ["Fz", "Cz", "Pz"])


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_coherence_identical_channels_give_unity_coherence() -> None:
    """Identical signals have magnitude-squared coherence of exactly 1.0."""
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(400)
    # Three channels that are all the same signal
    arr = np.stack([sig, sig, sig], axis=1)
    data = _make_data(arr, sampling_rate=200.0)

    out = cb.feature.Coherence().apply(data)
    mat = out.data.values  # (3, 3)

    for i in range(3):
        for j in range(3):
            if i != j:
                np.testing.assert_allclose(mat[i, j], 1.0, atol=1e-10)


def test_coherence_diagonal_is_nan() -> None:
    """Self-coherence (diagonal) is NaN."""
    rng = np.random.default_rng(3)
    data = _make_data(rng.standard_normal((300, 5)))

    out = cb.feature.Coherence().apply(data)
    mat = out.data.values

    assert np.all(np.isnan(np.diag(mat)))


def test_coherence_matrix_is_symmetric() -> None:
    """C[i, j] == C[j, i] for all off-diagonal entries."""
    rng = np.random.default_rng(4)
    data = _make_data(rng.standard_normal((400, 6)))

    out = cb.feature.Coherence().apply(data)
    mat = out.data.values

    mask = ~np.isnan(mat)
    np.testing.assert_allclose(mat[mask], mat.T[mask])


def test_coherence_values_are_in_unit_range() -> None:
    """All off-diagonal coherence values lie in [0, 1]."""
    rng = np.random.default_rng(5)
    data = _make_data(rng.standard_normal((512, 8)))

    out = cb.feature.Coherence().apply(data)
    mat = out.data.values
    off_diag = mat[~np.isnan(mat)]

    assert np.all(off_diag >= 0.0)
    assert np.all(off_diag <= 1.0)


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


def test_coherence_preserves_metadata_and_history() -> None:
    """Coherence propagates subjectID, groupID, condition, sampling_rate, extra, and history."""
    rng = np.random.default_rng(6)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((200, 3)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="control",
        condition="rest",
        extra={"session": 1},
    )

    out = cb.feature.Coherence().apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "control"
    assert out.condition == "rest"
    # sampling_rate is not preserved for Data without time dimension
    assert out.sampling_rate is None
    assert out.history == ["Coherence"]
    assert out.extra.get("session") == 1


# ---------------------------------------------------------------------------
# Extra dimensions
# ---------------------------------------------------------------------------


def test_coherence_with_run_index_preserves_extra_dim() -> None:
    """Coherence is computed per run when data has a run_index dimension."""
    rng = np.random.default_rng(7)
    n_runs, n_time, n_space = 3, 300, 4
    arr = rng.standard_normal((n_runs, n_time, n_space))
    arr_xr = xr.DataArray(
        arr,
        dims=["run_index", "time", "space"],
        coords={
            "run_index": np.arange(n_runs),
            "time": np.arange(n_time, dtype=float) / 100.0,
            "space": [f"ch{k}" for k in range(n_space)],
        },
    )
    data = cb.SignalData.from_xarray(arr_xr, sampling_rate=100.0)

    out = cb.feature.Coherence().apply(data)

    assert "run_index" in out.data.dims
    assert out.data.sizes["run_index"] == n_runs
    assert out.data.sizes["space_to"] == n_space
    assert out.data.sizes["space_from"] == n_space
    # Each run's diagonal must be NaN
    for r in range(n_runs):
        mat = out.data.isel(run_index=r).values
        assert np.all(np.isnan(np.diag(mat)))


# ---------------------------------------------------------------------------
# nperseg parameter
# ---------------------------------------------------------------------------


def test_coherence_custom_nperseg_produces_valid_output() -> None:
    """A custom nperseg produces results in [0, 1] with the expected structure."""
    rng = np.random.default_rng(8)
    data = _make_data(rng.standard_normal((200, 3)))

    out = cb.feature.Coherence(nperseg=32).apply(data)

    assert isinstance(out, cb.Data)
    mat = out.data.values
    off_diag = mat[~np.isnan(mat)]
    assert np.all(off_diag >= 0.0)
    assert np.all(off_diag <= 1.0)


def test_coherence_results_depend_on_nperseg() -> None:
    """Different nperseg values produce different (but both valid) coherence estimates."""
    rng = np.random.default_rng(9)
    data = _make_data(rng.standard_normal((512, 3)))

    out32 = cb.feature.Coherence(nperseg=32).apply(data)
    out128 = cb.feature.Coherence(nperseg=128).apply(data)

    # Values will differ because segment length affects the estimate
    assert not np.allclose(out32.data.values, out128.data.values, equal_nan=True)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_coherence_raises_when_fewer_than_two_channels() -> None:
    """Coherence raises ValueError for data with a single spatial channel."""
    data = _make_data(np.ones((100, 1)))

    with pytest.raises(ValueError, match="at least 2 spatial channels"):
        cb.feature.Coherence().apply(data)


def test_coherence_raises_when_nperseg_exceeds_n_time() -> None:
    """Coherence raises ValueError when nperseg is larger than the time axis."""
    data = _make_data(np.ones((50, 3)))

    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Coherence(nperseg=100).apply(data)


def test_coherence_raises_when_nperseg_is_less_than_two() -> None:
    """Coherence raises ValueError when nperseg < 2."""
    data = _make_data(np.ones((50, 3)))

    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Coherence(nperseg=1).apply(data)


# ---------------------------------------------------------------------------
# API accessibility
# ---------------------------------------------------------------------------


def test_coherence_accessible_via_feature_module() -> None:
    """Coherence is accessible as cb.feature.Coherence."""
    assert callable(cb.feature.Coherence)


def test_coherence_missing_space_dim_raises() -> None:
    """Coherence raises ValueError when data lacks 'space' dimension."""
    arr = np.random.default_rng(100).standard_normal((100, 5))
    xr_data = xr.DataArray(arr, dims=["time", "channel"])
    raw = cb.SignalData.__new__(cb.SignalData)
    object.__setattr__(raw, "_data", xr_data)
    object.__setattr__(raw, "_sampling_rate", 100.0)

    with pytest.raises(ValueError, match="space"):
        cb.feature.Coherence().apply(raw)


def test_coherence_default_nperseg_too_small_raises() -> None:
    """Coherence raises ValueError when n_time < 2 causes default nperseg < 2."""
    rng = np.random.default_rng(101)
    data = _make_data(rng.standard_normal((1, 3)))

    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Coherence().apply(data)


def test_coherence_does_not_mutate_input() -> None:
    """Coherence.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(102)
    data = _make_data(rng.standard_normal((200, 3)))
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Coherence().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
