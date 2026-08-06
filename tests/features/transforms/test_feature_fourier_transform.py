"""Tests for FourierTransform / InverseFourierTransform / PowerSpectralDensity."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _signal(arr: np.ndarray, sampling_rate: float | None = 200.0) -> cb.SignalData:
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# --- FourierTransform (default: raw complex coefficients) ---


def test_forward_transform_default_complex() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((128, 3)))
    result = cb.FourierTransform().apply(data)
    assert "frequency" in result.data.dims
    assert np.iscomplexobj(result.data.values)


def test_forward_transform_norm_real() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((128, 3)))
    result = cb.FourierTransform(norm="real").apply(data)
    assert result.data.dtype == np.float64
    assert np.all(result.data.values >= 0)


def test_forward_transform_norm_psd() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((128, 3)))
    result = cb.FourierTransform(norm="psd").apply(data)
    assert result.data.dtype == np.float64
    assert np.all(result.data.values >= 0)


def test_invalid_norm_raises() -> None:
    with pytest.raises(ValueError, match="norm"):
        cb.FourierTransform(norm="bad")


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


# --- Cutoff ---


def test_cutoff_limits_frequencies() -> None:
    data = _signal(np.zeros((128, 1)), sampling_rate=256.0)
    result = cb.FourierTransform(cutoff=50.0).apply(data)
    freqs = result.data.coords["frequency"].values
    assert freqs[-1] <= 50.0


def test_negative_cutoff_raises() -> None:
    with pytest.raises(ValueError, match="cutoff must be positive"):
        cb.FourierTransform(cutoff=-1.0)


def test_cutoff_above_nyquist_warns() -> None:
    data = _signal(np.zeros((128, 1)), sampling_rate=100.0)  # Nyquist = 50
    with pytest.warns(UserWarning, match="larger than Nyquist"):
        cb.FourierTransform(cutoff=60.0).apply(data)


def test_cutoff_does_not_mutate_instance() -> None:
    data = _signal(np.zeros((128, 1)), sampling_rate=100.0)
    ft = cb.FourierTransform(cutoff=60.0)
    with pytest.warns(UserWarning, match="larger than Nyquist"):
        ft.apply(data)
    assert ft.cutoff == 60.0  # original value preserved


# --- Missing sampling_rate ---


def test_no_sampling_rate_warns() -> None:
    data = _signal(np.zeros((128, 1)), sampling_rate=None)
    with pytest.warns(UserWarning, match="sampling_rate"):
        cb.FourierTransform().apply(data)


# --- Roundtrip ---


def test_roundtrip_with_inverse() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((128, 2))
    data = _signal(arr)
    forward = cb.FourierTransform().apply(data)
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
    forward_mag = cb.FourierTransform(norm="real").apply(data)
    with pytest.raises(ValueError, match="complex"):
        cb.InverseFourierTransform().apply(forward_mag)


def test_inverse_default_length_doubles_minus_one() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((64, 2))
    data = _signal(arr)
    forward = cb.FourierTransform().apply(data)
    back = cb.InverseFourierTransform().apply(forward)
    # numpy.fft.irfft default n = 2*(len-1)
    assert back.data.sizes["time"] == 64


def test_history_appended() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((64, 2)))
    forward = cb.FourierTransform().apply(data)
    assert forward.history[-1] == "FourierTransform"


# --- PSD normalization: Parseval's theorem ---


def test_psd_parseval() -> None:
    """Total PSD power ≈ signal variance (Parseval's theorem)."""
    rng = np.random.default_rng(42)
    sr = 256.0
    n = 1024
    arr = rng.standard_normal((n, 1))
    data = _signal(arr, sampling_rate=sr)
    psd = cb.FourierTransform(norm="psd").apply(data)
    freqs = psd.data.coords["frequency"].values
    df = freqs[1] - freqs[0]
    total_power = float((psd.data.values * df).sum())
    expected_var = float(np.var(arr))
    np.testing.assert_allclose(total_power, expected_var, rtol=0.05)


# --- PowerSpectralDensity alias ---


def test_psd_alias_matches_fourier_transform() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((128, 3)))
    from_ft = cb.FourierTransform(norm="psd").apply(data)
    from_psd = cb.PowerSpectralDensity().apply(data)
    np.testing.assert_allclose(from_psd.data.values, from_ft.data.values)


def test_psd_functional_api() -> None:
    rng = np.random.default_rng(0)
    data = _signal(rng.standard_normal((128, 3)))
    result = cb.power_spectral_density(data)
    assert "frequency" in result.data.dims
    assert result.data.dtype == np.float64
