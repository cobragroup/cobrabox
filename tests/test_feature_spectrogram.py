"""Tests for the spectrogram feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(arr: np.ndarray, *, sampling_rate: float = 256.0) -> cb.SignalData:
    """Create Data from a (time, space) array."""
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_spectrogram_output_dims() -> None:
    """spectrogram returns Data with (space, frequency, time) dims."""
    rng = np.random.default_rng(0)
    data = _make_data(rng.standard_normal((512, 3)))

    out = cb.feature.Spectrogram().apply(data)

    assert isinstance(out, cb.Data)
    assert set(out.data.dims) == {"space", "frequency", "time"}


def test_spectrogram_space_dim_preserved() -> None:
    """The space dimension and its coordinates are unchanged."""
    arr_xr = xr.DataArray(
        np.random.default_rng(1).standard_normal((512, 4)),
        dims=["time", "space"],
        coords={"space": ["Fz", "Cz", "Pz", "Oz"], "time": np.arange(512) / 256.0},
    )
    data = cb.data.SignalData.from_xarray(arr_xr)

    out = cb.feature.Spectrogram().apply(data)

    np.testing.assert_array_equal(out.data.coords["space"].values, ["Fz", "Cz", "Pz", "Oz"])


def test_spectrogram_frequency_coords_are_nonneg_and_bounded() -> None:
    """Frequency axis is non-negative and bounded by Nyquist."""
    rng = np.random.default_rng(2)
    fs = 256.0
    data = _make_data(rng.standard_normal((512, 2)), sampling_rate=fs)

    out = cb.feature.Spectrogram().apply(data)

    freqs = out.data.coords["frequency"].values
    assert np.all(freqs >= 0.0)
    assert freqs[-1] <= fs / 2.0 + 1e-9


def test_spectrogram_time_coords_are_positive() -> None:
    """Spectrogram time axis contains positive values (window centres)."""
    rng = np.random.default_rng(3)
    data = _make_data(rng.standard_normal((512, 2)))

    out = cb.feature.Spectrogram().apply(data)

    t = out.data.coords["time"].values
    assert np.all(t > 0)


def test_spectrogram_output_shape_matches_scipy() -> None:
    """Output (n_space, n_freq, n_t) matches what scipy would return."""
    from scipy.signal import spectrogram as _sp

    rng = np.random.default_rng(4)
    n_time, n_space, fs, seg = 512, 3, 256.0, 64
    arr = rng.standard_normal((n_time, n_space))
    data = _make_data(arr, sampling_rate=fs)

    out = cb.feature.Spectrogram(nperseg=seg).apply(data)

    f, t, _ = _sp(arr[:, 0], fs=fs, nperseg=seg, window="hann")
    assert out.data.sizes["frequency"] == len(f)
    assert out.data.sizes["time"] == len(t)
    assert out.data.sizes["space"] == n_space


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_spectrogram_log_scaling_matches_scipy() -> None:
    """log scaling matches 10*log10(scipy PSD) channel-by-channel."""
    from scipy.signal import spectrogram as _sp

    rng = np.random.default_rng(5)
    fs, seg = 256.0, 64
    arr = rng.standard_normal((512, 2))
    data = _make_data(arr, sampling_rate=fs)

    out = cb.feature.Spectrogram(nperseg=seg, scaling="log").apply(data)

    for ch in range(2):
        _, _, Sxx = _sp(arr[:, ch], fs=fs, nperseg=seg, window="hann", scaling="density")
        expected = 10.0 * np.log10(np.maximum(Sxx, np.finfo(np.float64).tiny))
        np.testing.assert_allclose(out.data.isel(space=ch).values, expected, rtol=1e-10)


def test_spectrogram_density_scaling_matches_scipy() -> None:
    """density scaling matches raw scipy PSD."""
    from scipy.signal import spectrogram as _sp

    rng = np.random.default_rng(6)
    fs, seg = 128.0, 32
    arr = rng.standard_normal((256, 2))
    data = _make_data(arr, sampling_rate=fs)

    out = cb.feature.Spectrogram(nperseg=seg, scaling="density").apply(data)

    for ch in range(2):
        _, _, Sxx = _sp(arr[:, ch], fs=fs, nperseg=seg, window="hann", scaling="density")
        np.testing.assert_allclose(out.data.isel(space=ch).values, Sxx, rtol=1e-10)


def test_spectrogram_spectrum_scaling_matches_scipy() -> None:
    """spectrum scaling matches raw scipy power spectrum."""
    from scipy.signal import spectrogram as _sp

    rng = np.random.default_rng(7)
    fs, seg = 128.0, 32
    arr = rng.standard_normal((256, 2))
    data = _make_data(arr, sampling_rate=fs)

    out = cb.feature.Spectrogram(nperseg=seg, scaling="spectrum").apply(data)

    for ch in range(2):
        _, _, Sxx = _sp(arr[:, ch], fs=fs, nperseg=seg, window="hann", scaling="spectrum")
        np.testing.assert_allclose(out.data.isel(space=ch).values, Sxx, rtol=1e-10)


def test_spectrogram_magnitude_scaling_matches_scipy() -> None:
    """|STFT| magnitude scaling matches |scipy.signal.stft| output."""
    from scipy.signal import stft as _stft

    rng = np.random.default_rng(8)
    fs, seg = 128.0, 32
    arr = rng.standard_normal((256, 2))
    data = _make_data(arr, sampling_rate=fs)

    out = cb.feature.Spectrogram(nperseg=seg, scaling="magnitude").apply(data)

    for ch in range(2):
        _, _, Zxx = _stft(arr[:, ch], fs=fs, nperseg=seg, window="hann")
        np.testing.assert_allclose(out.data.isel(space=ch).values, np.abs(Zxx), rtol=1e-10)


def test_spectrogram_log_no_neg_inf() -> None:
    """log scaling never produces -inf, even for near-zero signals."""
    arr = np.zeros((256, 2))
    arr[0, 0] = 1e-20
    data = _make_data(arr)

    out = cb.feature.Spectrogram(scaling="log").apply(data)

    assert np.all(np.isfinite(out.data.values))


def test_spectrogram_density_values_nonneg() -> None:
    """density scaling is always >= 0."""
    rng = np.random.default_rng(9)
    data = _make_data(rng.standard_normal((512, 4)))

    out = cb.feature.Spectrogram(scaling="density").apply(data)

    assert np.all(out.data.values >= 0.0)


def test_spectrogram_pure_tone_has_peak_at_correct_freq() -> None:
    """A pure sine wave produces a spectrogram peak near the sine frequency."""
    fs = 256.0
    t = np.arange(1024) / fs
    freq = 40.0
    sig = np.sin(2 * np.pi * freq * t)
    arr = sig[:, np.newaxis]  # (1024, 1)
    data = _make_data(arr, sampling_rate=fs)

    out = cb.feature.Spectrogram(nperseg=256, scaling="density").apply(data)
    psd = out.data.isel(space=0).values  # (n_freq, n_t)
    mean_psd = psd.mean(axis=-1)
    peak_freq = out.data.coords["frequency"].values[np.argmax(mean_psd)]

    assert abs(peak_freq - freq) <= 2.0  # within 2 Hz


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


def test_spectrogram_preserves_metadata() -> None:
    """spectrogram propagates subjectID, groupID, condition, sampling_rate, extra, history."""
    rng = np.random.default_rng(10)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((256, 2)),
        dims=["time", "space"],
        sampling_rate=128.0,
        subjectID="sub-01",
        groupID="ctrl",
        condition="rest",
        extra={"session": 2},
    )

    out = cb.feature.Spectrogram().apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "ctrl"
    assert out.condition == "rest"
    assert out.sampling_rate == 128.0
    assert out.history == ["Spectrogram"]
    assert out.extra.get("session") == 2


# ---------------------------------------------------------------------------
# Extra dimensions
# ---------------------------------------------------------------------------


def test_spectrogram_preserves_extra_dim() -> None:
    """spectrogram is computed per-window when data has a window_index dimension."""
    rng = np.random.default_rng(11)
    n_windows, n_time, n_space = 4, 128, 3
    arr = rng.standard_normal((n_windows, n_time, n_space))
    arr_xr = xr.DataArray(
        arr,
        dims=["window_index", "time", "space"],
        coords={
            "window_index": np.arange(n_windows),
            "time": np.arange(n_time, dtype=float) / 128.0,
            "space": [f"ch{k}" for k in range(n_space)],
        },
    )
    data = cb.data.SignalData.from_xarray(arr_xr)

    out = cb.feature.Spectrogram(nperseg=32).apply(data)

    assert "window_index" in out.data.dims
    assert out.data.sizes["window_index"] == n_windows
    assert out.data.sizes["space"] == n_space
    assert "frequency" in out.data.dims
    assert "time" in out.data.dims


# ---------------------------------------------------------------------------
# Parameter behaviour
# ---------------------------------------------------------------------------


def test_spectrogram_custom_nperseg_changes_freq_resolution() -> None:
    """Larger nperseg → more frequency bins."""
    rng = np.random.default_rng(12)
    data = _make_data(rng.standard_normal((512, 2)))

    out32 = cb.feature.Spectrogram(nperseg=32).apply(data)
    out128 = cb.feature.Spectrogram(nperseg=128).apply(data)

    assert out32.data.sizes["frequency"] < out128.data.sizes["frequency"]


def test_spectrogram_noverlap_changes_time_resolution() -> None:
    """Greater overlap → more time bins."""
    rng = np.random.default_rng(13)
    data = _make_data(rng.standard_normal((512, 2)))

    out_lo = cb.feature.Spectrogram(nperseg=64, noverlap=0).apply(data)
    out_hi = cb.feature.Spectrogram(nperseg=64, noverlap=60).apply(data)

    assert out_lo.data.sizes["time"] < out_hi.data.sizes["time"]


def test_spectrogram_different_windows_produce_different_results() -> None:
    """Using a Hamming vs Hann window gives different spectrogram values."""
    rng = np.random.default_rng(14)
    data = _make_data(rng.standard_normal((512, 2)))

    out_hann = cb.feature.Spectrogram(window="hann").apply(data)
    out_hamming = cb.feature.Spectrogram(window="hamming").apply(data)

    assert not np.allclose(out_hann.data.values, out_hamming.data.values)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_spectrogram_raises_on_invalid_scaling() -> None:
    """spectrogram raises ValueError for unknown scaling."""
    data = _make_data(np.ones((64, 2)))

    with pytest.raises(ValueError, match="scaling"):
        cb.feature.Spectrogram(scaling="invalid").apply(data)


def test_spectrogram_raises_when_nperseg_exceeds_n_time() -> None:
    """spectrogram raises ValueError when nperseg > n_time."""
    data = _make_data(np.ones((32, 2)))

    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Spectrogram(nperseg=64).apply(data)


def test_spectrogram_raises_when_nperseg_is_less_than_two() -> None:
    """spectrogram raises ValueError when nperseg < 2."""
    data = _make_data(np.ones((64, 2)))

    with pytest.raises(ValueError, match="nperseg"):
        cb.feature.Spectrogram(nperseg=1).apply(data)


def test_spectrogram_raises_when_noverlap_gte_nperseg() -> None:
    """spectrogram raises ValueError when noverlap >= nperseg."""
    data = _make_data(np.ones((128, 2)))

    with pytest.raises(ValueError, match="noverlap"):
        cb.feature.Spectrogram(nperseg=32, noverlap=32).apply(data)


# ---------------------------------------------------------------------------
# API accessibility
# ---------------------------------------------------------------------------


def test_spectrogram_accessible_via_feature_module() -> None:
    """Spectrogram is accessible as cb.feature.Spectrogram."""
    assert callable(cb.feature.Spectrogram)
