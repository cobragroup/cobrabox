"""Tests for the NotchFilter feature."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

import cobrabox as cb


def _make_data(
    n_time: int = 1000, n_space: int = 3, sampling_rate: float = 250.0, subject: str = "sub-01"
) -> cb.SignalData:
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((n_time, n_space))
    return cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sampling_rate, subjectID=subject
    )


def _make_sine_data(
    freqs_hz: list[float],
    sampling_rate: float = 250.0,
    duration: float = 4.0,
    n_space: int = 1,
    seed: int = 0,
) -> cb.SignalData:
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    sig = np.zeros_like(t)
    for f in freqs_hz:
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 2 * np.pi)
        sig += np.sin(2 * np.pi * f * t + phase)
    arr = np.tile(sig[:, None], (1, n_space))
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Basic API and shape tests
# ---------------------------------------------------------------------------


def test_notchfilter_history_updated() -> None:
    data = _make_data()
    result = cb.NotchFilter(freq=50.0).apply(data)
    assert result.history[-1] == "NotchFilter"


def test_notchfilter_returns_signal_data() -> None:
    data = _make_data()
    result = cb.NotchFilter(freq=50.0).apply(data)
    assert isinstance(result, cb.SignalData)


def test_notchfilter_output_shape_matches_input() -> None:
    data = _make_data(n_time=500, n_space=4)
    result = cb.NotchFilter(freq=50.0).apply(data)

    assert result.data.shape == data.data.shape
    assert list(result.data.dims) == list(data.data.dims)


def test_notchfilter_metadata_preserved() -> None:
    rng = np.random.default_rng(42)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((200, 3)),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.NotchFilter(freq=50.0).apply(data)

    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)


def test_notchfilter_does_not_mutate_input() -> None:
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.NotchFilter(freq=50.0).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


# ---------------------------------------------------------------------------
# Coordinate preservation
# ---------------------------------------------------------------------------


def test_notchfilter_preserves_time_coords() -> None:
    data = _make_data(sampling_rate=256.0)
    result = cb.NotchFilter(freq=50.0).apply(data)

    np.testing.assert_array_equal(
        result.data.coords["time"].values, data.data.coords["time"].values
    )


def test_notchfilter_preserves_space_coords() -> None:
    arr = np.random.default_rng(0).standard_normal((200, 4))
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=256.0, subjectID="s1"
    )
    result = cb.NotchFilter(freq=50.0).apply(data)

    assert result.data.sizes["space"] == 4
    assert list(result.data.dims) == list(data.data.dims)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_notchfilter_attenuates_target_frequency() -> None:
    sr = 500.0
    duration = 10.0
    trim = int(sr)  # discard filter transient

    data = _make_sine_data(freqs_hz=[50.0], sampling_rate=sr, duration=duration)
    out = cb.NotchFilter(freq=50.0, q=30.0).apply(data)

    arr = data.to_numpy()  # (space, time)
    arr_out = out.to_numpy()
    in_rms = float(np.sqrt(np.mean(arr[:, trim:] ** 2)))
    out_rms = float(np.sqrt(np.mean(arr_out[:, trim:] ** 2)))

    assert out_rms < in_rms * 0.3, (
        f"Notch should attenuate 50 Hz: in={in_rms:.4f}, out={out_rms:.4f}"
    )


def test_notchfilter_preserves_other_frequencies() -> None:
    sr = 500.0
    duration = 10.0
    trim = int(sr)

    data = _make_sine_data(freqs_hz=[5.0], sampling_rate=sr, duration=duration)
    out = cb.NotchFilter(freq=50.0, q=30.0).apply(data)

    arr = data.to_numpy()
    arr_out = out.to_numpy()
    in_rms = float(np.sqrt(np.mean(arr[:, trim:] ** 2)))
    out_rms = float(np.sqrt(np.mean(arr_out[:, trim:] ** 2)))

    assert out_rms > in_rms * 0.7, (
        f"5 Hz sine should pass through 50 Hz notch: in={in_rms:.4f}, out={out_rms:.4f}"
    )


def test_notchfilter_matches_manual_scipy() -> None:
    sr = 250.0
    data = _make_data(n_time=500, n_space=2, sampling_rate=sr)
    freq = 50.0
    q = 30.0

    out = cb.NotchFilter(freq=freq, q=q).apply(data)

    b, a = signal.iirnotch(freq, q, fs=sr)
    expected = signal.lfilter(b, a, data.to_numpy(), axis=-1)

    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-12)


def test_notchfilter_mixed_sine_attenuation() -> None:
    sr = 500.0
    duration = 10.0
    trim = int(sr)

    data = _make_sine_data(freqs_hz=[5.0, 50.0], sampling_rate=sr, duration=duration)
    out = cb.NotchFilter(freq=50.0, q=30.0).apply(data)

    out_5hz = signal.lfilter(
        *signal.iirnotch(50.0, 30.0, fs=sr),
        _make_sine_data(freqs_hz=[5.0], sampling_rate=sr, duration=duration).to_numpy(),
        axis=-1,
    )
    rms_out = float(np.sqrt(np.mean(out.to_numpy()[:, trim:] ** 2)))
    rms_5hz = float(np.sqrt(np.mean(out_5hz[:, trim:] ** 2)))

    assert rms_out == pytest.approx(rms_5hz, rel=0.05), (
        f"Notch output RMS ({rms_out:.4f}) should approx 5 Hz-only RMS ({rms_5hz:.4f})"
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_notchfilter_missing_freq_raises_type_error() -> None:
    with pytest.raises(TypeError):
        cb.NotchFilter()  # type: ignore[call-arg]


def test_notchfilter_zero_freq_raises() -> None:
    with pytest.raises(ValueError, match="freq"):
        cb.NotchFilter(freq=0.0)


def test_notchfilter_negative_freq_raises() -> None:
    with pytest.raises(ValueError, match="freq"):
        cb.NotchFilter(freq=-10.0)


def test_notchfilter_negative_q_raises() -> None:
    with pytest.raises(ValueError, match="q"):
        cb.NotchFilter(freq=50.0, q=0.0)


def test_notchfilter_freq_exceeds_nyquist_raises() -> None:
    data = _make_data(sampling_rate=100.0)
    with pytest.raises(ValueError, match="Nyquist"):
        cb.NotchFilter(freq=60.0).apply(data)


def test_notchfilter_missing_sampling_rate_raises() -> None:
    import xarray as xr

    rng = np.random.default_rng(42)
    arr = rng.standard_normal((100, 3))
    xr_da = xr.DataArray(arr, dims=["time", "space"])
    data = cb.SignalData.from_xarray(xr_da, subjectID="s1")
    with pytest.raises(ValueError, match="sampling_rate"):
        cb.NotchFilter(freq=50.0).apply(data)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_notchfilter_serialization_round_trip() -> None:
    feature = cb.NotchFilter(freq=50.0, q=30.0)
    yaml_str = feature.to_yaml()
    reloaded = cb.deserialize(yaml_str)
    assert isinstance(reloaded, cb.Pipeline)
    assert len(reloaded.features) == 1
    nf = reloaded.features[0]
    assert isinstance(nf, cb.NotchFilter)
    assert nf.freq == 50.0
    assert nf.q == 30.0
