"""Tests for the Nonreversibility feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb
from cobrabox.features.time_domain.nonreversibility import Nonreversibility


def test_nonreversibility_output_shape_dims_and_metadata() -> None:
    """Nonreversibility returns Data with space='dc_norm', correct dims, and metadata."""
    rng = np.random.default_rng(seed=0)
    arr = rng.standard_normal((200, 5))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="sub-01",
        groupID="ctrl",
        condition="rest",
        extra={"tag": "eeg"},
    )

    out = cb.feature.Nonreversibility().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (1,)
    assert list(out.data.coords["space"].values) == ["dc_norm"]
    assert out.subjectID == "sub-01"
    assert out.groupID == "ctrl"
    assert out.condition == "rest"
    assert out.sampling_rate is None  # Data without time dimension
    assert out.extra.get("tag") == "eeg"
    assert out.history == ["Nonreversibility"]


def test_nonreversibility_scalar_is_nonnegative() -> None:
    """dc_norm is always in [0, 1) (normalised ratio of Frobenius norms)."""
    rng = np.random.default_rng(seed=42)
    arr = rng.standard_normal((300, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Nonreversibility().apply(data)
    dc = float(out.data.values.flat[0])

    assert 0.0 <= dc < 1.0


def test_nonreversibility_raises_on_single_channel() -> None:
    """Nonreversibility raises ValueError when space dimension has fewer than 2 channels."""
    arr = np.random.default_rng(seed=7).standard_normal((100, 1))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="at least 2 time series"):
        cb.feature.Nonreversibility().apply(data)


def test_nonreversibility_raises_on_missing_space_dim() -> None:
    """Nonreversibility raises ValueError when 'space' dimension is absent."""

    class _FakeSignalData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["time", "freq"])

        sampling_rate = 100.0

    with pytest.raises(ValueError, match="'space' dimension"):
        Nonreversibility()(_FakeSignalData())  # type: ignore[arg-type]


def test_nonreversibility_raises_on_too_few_timepoints() -> None:
    """Nonreversibility raises ValueError when there is only 1 timepoint."""
    arr = np.ones((1, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    with pytest.raises(ValueError, match="at least 2 timepoints"):
        cb.feature.Nonreversibility().apply(data)


def test_nonreversibility_spectral_radius_rescaling() -> None:
    """dc_norm stays in [0, 1) even when the VAR(1) matrix has spectral radius >= 1."""
    rng = np.random.default_rng(seed=99)
    # Construct a signal with a near-unit-root AR structure so that the
    # fitted VAR(1) coefficient matrix has spectral radius >= 1 before rescaling.
    n_time, n_space = 300, 3
    X = np.zeros((n_time, n_space))
    X[0] = rng.standard_normal(n_space)
    for t in range(1, n_time):
        X[t] = 1.05 * X[t - 1] + 0.1 * rng.standard_normal(n_space)
    data = cb.SignalData.from_numpy(X, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Nonreversibility().apply(data)
    dc = float(out.data.values.flat[0])

    assert 0.0 <= dc < 1.0


def test_nonreversibility_zero_denominator_returns_zero() -> None:
    """dc_norm returns 0.0 when denominator is effectively zero (flat signal)."""
    # An all-zeros signal produces A = 0 and B = 0, so denom = ||0||_F + ||0||_F = 0.
    arr = np.zeros((50, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    out = cb.feature.Nonreversibility().apply(data)
    dc = float(out.data.values.flat[0])

    assert dc == pytest.approx(0.0)


def test_nonreversibility_does_not_mutate_input() -> None:
    """Nonreversibility.apply() leaves the input SignalData object unchanged."""
    rng = np.random.default_rng(seed=1)
    arr = rng.standard_normal((100, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = np.copy(data.to_numpy())

    _ = cb.feature.Nonreversibility().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_nonreversibility_public_api() -> None:
    """Nonreversibility is accessible via cb.feature.Nonreversibility."""
    assert hasattr(cb.feature, "Nonreversibility")
    rng = np.random.default_rng(seed=3)
    arr = rng.standard_normal((100, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.Nonreversibility().apply(data)
    assert isinstance(out, cb.Data)
