"""Tests for the InwardStrength feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb


def _asymmetric_matrix(n: int = 4, *, with_frequency: bool = False, seed: int = 0) -> cb.Data:
    """Build a directed (asymmetric) connectivity matrix as a Data object."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 1, size=(n, n))
    if with_frequency:
        n_f = 32
        arr = np.stack([base + 0.01 * f for f in range(n_f)], axis=-1)
        arr += 0.5 * rng.standard_normal(arr.shape)
        coords = {
            "space_to": np.arange(n),
            "space_from": np.arange(n),
            "frequency": np.linspace(0.0, 100.0, n_f),
        }
        dims = ("space_to", "space_from", "frequency")
    else:
        arr = base + 0.5 * rng.standard_normal((n, n))
        coords = {"space_to": np.arange(n), "space_from": np.arange(n)}
        dims = ("space_to", "space_from")
    da = xr.DataArray(arr, dims=dims, coords=coords)
    return cb.Data(da)


# ---------------------------------------------------------------------------
# Basic output shape and dims
# ---------------------------------------------------------------------------


def test_returns_vector() -> None:
    """InwardStrength produces a (space,) vector."""
    mat = _asymmetric_matrix(n=4)
    result = cb.InwardStrength().apply(mat)
    assert result.data.dims == ("space",)
    assert result.data.shape == (4,)


def test_with_frequency_band() -> None:
    """freq_band averages over frequency before computing strength."""
    mat = _asymmetric_matrix(n=4, with_frequency=True)
    result = cb.InwardStrength(freq_band=(20.0, 60.0)).apply(mat)
    assert result.data.dims == ("space",)
    assert result.data.shape == (4,)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_matches_manual_computation() -> None:
    """InwardStrength matches manual row-mean (excluding diagonal)."""
    arr = np.array([[1.0, 0.2, 0.3], [0.4, 1.0, 0.6], [0.7, 0.8, 1.0]])
    da = xr.DataArray(
        arr,
        dims=("space_to", "space_from"),
        coords={"space_to": [0, 1, 2], "space_from": [0, 1, 2]},
    )
    mat = cb.Data(da)
    result = cb.InwardStrength().apply(mat)

    # inward[i] = mean of row i excluding diagonal
    expected = np.array(
        [
            np.mean([0.2, 0.3]),  # row 0
            np.mean([0.4, 0.6]),  # row 1
            np.mean([0.7, 0.8]),  # row 2
        ]
    )
    np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-10)


def test_space_coords_preserved() -> None:
    """Space coordinates from space_to are used in output."""
    arr = np.random.default_rng(0).standard_normal((3, 3))
    da = xr.DataArray(
        arr,
        dims=("space_to", "space_from"),
        coords={"space_to": ["Fp1", "Fp2", "C3"], "space_from": ["Fp1", "Fp2", "C3"]},
    )
    mat = cb.Data(da)
    result = cb.InwardStrength().apply(mat)
    assert list(result.data.coords["space"].values) == ["Fp1", "Fp2", "C3"]


# ---------------------------------------------------------------------------
# Metadata and history
# ---------------------------------------------------------------------------


def test_history_appended() -> None:
    mat = _asymmetric_matrix(n=3)
    result = cb.InwardStrength().apply(mat)
    assert result.history[-1] == "InwardStrength"


def test_does_not_mutate_input() -> None:
    mat = _asymmetric_matrix(n=3)
    original = mat.data.values.copy()
    _ = cb.InwardStrength().apply(mat)
    np.testing.assert_array_equal(mat.data.values, original)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_rejects_time_series() -> None:
    rng = np.random.default_rng(0)
    sig = cb.SignalData.from_numpy(
        rng.standard_normal((200, 4)), dims=["time", "space"], sampling_rate=200.0
    )
    with pytest.raises(ValueError, match="matrix-only"):
        cb.InwardStrength().apply(sig)


def test_missing_space_dims_raises() -> None:
    arr = xr.DataArray(np.zeros((4, 4)), dims=("a", "b"))
    bad = cb.Data(arr)
    with pytest.raises(ValueError, match="'space_to' and 'space_from'"):
        cb.InwardStrength().apply(bad)


def test_freq_band_required_when_frequency_present() -> None:
    mat = _asymmetric_matrix(n=3, with_frequency=True)
    with pytest.raises(ValueError, match="freq_band=None"):
        cb.InwardStrength().apply(mat)


def test_freq_band_outside_range_raises() -> None:
    mat = _asymmetric_matrix(n=3, with_frequency=True)
    with pytest.raises(ValueError, match="outside the available"):
        cb.InwardStrength(freq_band=(500.0, 600.0)).apply(mat)


def test_invalid_freq_band_raises() -> None:
    with pytest.raises(ValueError, match="fmin < fmax"):
        cb.InwardStrength(freq_band=(50.0, 10.0))


def test_freq_band_without_frequency_raises() -> None:
    mat = _asymmetric_matrix(n=3)
    with pytest.raises(ValueError, match="freq_band"):
        cb.InwardStrength(freq_band=(1.0, 40.0)).apply(mat)


def test_coords_fallback_uses_arange() -> None:
    arr = np.random.default_rng(0).standard_normal((3, 3))
    da = xr.DataArray(arr, dims=("space_to", "space_from"))
    mat = cb.Data(da)
    result = cb.InwardStrength().apply(mat)

    np.testing.assert_array_equal(result.data.coords["space"].values, np.arange(3))


# ---------------------------------------------------------------------------
# Pipeline composability
# ---------------------------------------------------------------------------


def test_pipes_after_directed_connectivity() -> None:
    """PartialDirectedCoherence | InwardStrength end-to-end."""
    rng = np.random.default_rng(0)
    sig = cb.SignalData.from_numpy(
        rng.standard_normal((200, 3)), dims=["time", "space"], sampling_rate=128.0
    )
    pipeline = cb.PartialDirectedCoherence() | cb.InwardStrength(freq_band=(0.0, 30.0))
    result = pipeline.apply(sig)
    assert result.data.dims == ("space",)
    assert result.data.shape == (3,)
