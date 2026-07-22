"""Tests for the matrix-only ReciprocalConnectivity feature."""

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
        # Ensure asymmetry survives
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


def test_matrix_input_no_frequency_returns_vector() -> None:
    mat = _asymmetric_matrix(n=4, with_frequency=False)
    result = cb.feature.ReciprocalConnectivity(freq_band=None).apply(mat)
    assert result.data.dims == ("space",)
    assert result.data.shape == (4,)


def test_matrix_input_with_frequency_band_averages() -> None:
    mat = _asymmetric_matrix(n=4, with_frequency=True)
    result = cb.feature.ReciprocalConnectivity(freq_band=(20.0, 60.0)).apply(mat)
    assert result.data.dims == ("space",)
    assert result.data.shape == (4,)


def test_rejects_time_series_input() -> None:
    rng = np.random.default_rng(0)
    sig = cb.SignalData.from_numpy(
        rng.standard_normal((200, 4)), dims=["time", "space"], sampling_rate=200.0
    )
    with pytest.raises(ValueError, match="matrix-only"):
        cb.feature.ReciprocalConnectivity(freq_band=None).apply(sig)


def test_missing_space_dims_raises() -> None:
    arr = xr.DataArray(np.zeros((4, 4)), dims=("a", "b"))
    bad = cb.Data(arr)
    with pytest.raises(ValueError, match="'space_to' and 'space_from'"):
        cb.feature.ReciprocalConnectivity(freq_band=None).apply(bad)


def test_symmetric_input_warns_not_raises() -> None:
    arr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    da = xr.DataArray(
        arr,
        dims=("space_to", "space_from"),
        coords={"space_to": np.arange(3), "space_from": np.arange(3)},
    )
    mat = cb.Data(da)
    with pytest.warns(UserWarning, match="symmetric"):
        result = cb.feature.ReciprocalConnectivity(freq_band=None).apply(mat)
    # symmetric input → all-zero output
    np.testing.assert_allclose(result.data.values, 0.0, atol=1e-10)


def test_freq_band_required_when_frequency_present() -> None:
    mat = _asymmetric_matrix(n=3, with_frequency=True)
    with pytest.raises(ValueError, match="freq_band=None"):
        cb.feature.ReciprocalConnectivity(freq_band=None).apply(mat)


def test_freq_band_outside_range_raises() -> None:
    mat = _asymmetric_matrix(n=3, with_frequency=True)
    with pytest.raises(ValueError, match="outside the available"):
        cb.feature.ReciprocalConnectivity(freq_band=(500.0, 600.0)).apply(mat)


def test_invalid_freq_band_raises() -> None:
    with pytest.raises(ValueError, match="fmin < fmax"):
        cb.feature.ReciprocalConnectivity(freq_band=(50.0, 10.0))


def test_history_appended() -> None:
    mat = _asymmetric_matrix(n=4, with_frequency=False)
    result = cb.feature.ReciprocalConnectivity(freq_band=None).apply(mat)
    assert result.history[-1] == "ReciprocalConnectivity"


def test_pipes_after_directed_connectivity() -> None:
    """Smoke test: PartialDirectedCoherence | ReciprocalConnectivity end-to-end."""
    rng = np.random.default_rng(0)
    sig = cb.SignalData.from_numpy(
        rng.standard_normal((200, 3)), dims=["time", "space"], sampling_rate=128.0
    )
    pipeline = cb.feature.PartialDirectedCoherence() | cb.feature.ReciprocalConnectivity(
        freq_band=(0.0, 30.0)
    )
    result = pipeline.apply(sig)
    assert result.data.dims == ("space",)
    assert result.data.shape == (3,)
