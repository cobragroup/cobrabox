"""Tests for FourierTransform / InverseFourierTransform."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _signal(arr: np.ndarray, sampling_rate: float | None = 200.0) -> cb.SignalData:
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


def test_forward_transform_default_magnitude() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((128, 3)))
    result = cb.FourierTransform().apply(data)
    assert "frequency" in result.data.dims
    assert result.data.dtype == np.float64
    # All magnitudes non-negative
    assert np.all(result.data.values >= 0)


def test_forward_transform_complex_coefficients() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((128, 3)))
    result = cb.FourierTransform(return_magnitude=False).apply(data)
    assert np.iscomplexobj(result.data.values)


def test_frequency_axis_uses_sampling_rate() -> None:
    data = _signal(np.zeros((128, 1)), sampling_rate=256.0)
    result = cb.FourierTransform().apply(data)
    freqs = result.data.coords["frequency"].values
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(128.0)  # Nyquist = sr/2


def test_requires_time_dim() -> None:
    arr = np.zeros((3, 3))
    bad = cb.Data.from_numpy(arr, dims=["a", "b"])
    with pytest.raises(ValueError, match="time"):
        cb.FourierTransform().apply(bad)


def test_roundtrip_with_inverse() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((128, 2))
    data = _signal(arr)
    forward = cb.FourierTransform(return_magnitude=False).apply(data)
    back = cb.InverseFourierTransform(n=128, sampling_rate=200.0).apply(forward)
    np.testing.assert_allclose(back.data.transpose("time", "space").values, arr, atol=1e-9)


def test_inverse_requires_frequency_dim() -> None:
    arr = np.zeros((3, 3))
    bad = cb.Data.from_numpy(arr, dims=["a", "b"])
    with pytest.raises(ValueError, match="frequency"):
        cb.InverseFourierTransform().apply(bad)


def test_inverse_requires_complex_input() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((64, 2)))
    forward_mag = cb.FourierTransform(return_magnitude=True).apply(data)
    with pytest.raises(ValueError, match="complex"):
        cb.InverseFourierTransform().apply(forward_mag)


def test_inverse_default_length_doubles_minus_one() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((64, 2))
    data = _signal(arr)
    forward = cb.FourierTransform(return_magnitude=False).apply(data)
    back = cb.InverseFourierTransform().apply(forward)
    # numpy.fft.irfft default n = 2*(len-1)
    assert back.data.sizes["time"] == 64


def test_history_appended() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((64, 2)))
    forward = cb.FourierTransform().apply(data)
    assert forward.history[-1] == "FourierTransform"
