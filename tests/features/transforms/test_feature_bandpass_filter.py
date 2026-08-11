"""Tests for the BandpassFilter feature."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

import cobrabox as cb


def _make_data(
    n_time: int = 1000, n_space: int = 3, sampling_rate: float = 250.0, subject: str = "sub-01"
) -> cb.SignalData:
    """Create a simple Data object with white noise for testing."""
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
) -> cb.Data:
    """Create a Data object whose signal is a sum of pure sinusoids."""
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    sig = np.zeros_like(t)
    for f in freqs_hz:
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 2 * np.pi)
        sig += np.sin(2 * np.pi * f * t + phase)
    arr = np.tile(sig[:, None], (1, n_space))
    return cb.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Basic API and shape tests
# ---------------------------------------------------------------------------


def test_bandpass_history_updated() -> None:
    data = _make_data()
    result = cb.BandpassFilter(bands=[[8, 12]]).apply(data)
    assert result.history[-1] == "BandpassFilter"


def test_bandpass_returns_data_instance() -> None:
    data = _make_data()
    result = cb.BandpassFilter(bands=[[8, 12]]).apply(data)
    assert isinstance(result, cb.Data)


def test_bandpass_output_shape_matches_input() -> None:
    data = _make_data(n_time=500, n_space=4)
    result = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    assert result.data.shape == data.data.shape
    assert list(result.data.dims) == list(data.data.dims)


def test_bandpass_no_band_dimension() -> None:
    """The output has no extra band dimension."""
    data = _make_data()
    result = cb.BandpassFilter(bands=[[8, 12], [12, 30]]).apply(data)
    assert "band" not in result.data.dims


def test_bandpass_metadata_preserved() -> None:
    rng = np.random.default_rng(42)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((200, 3)),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)


def test_bandpass_does_not_mutate_input() -> None:
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


# ---------------------------------------------------------------------------
# Coordinate preservation
# ---------------------------------------------------------------------------


def test_bandpass_preserves_time_coords() -> None:
    data = _make_data(sampling_rate=100.0)
    result = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    np.testing.assert_array_equal(
        result.data.coords["time"].values, data.data.coords["time"].values
    )


def test_bandpass_preserves_space_coords_when_present() -> None:
    import xarray as xr

    arr = np.random.default_rng(0).standard_normal((200, 4))
    xr_da = xr.DataArray(
        arr,
        dims=["time", "space"],
        coords={"time": np.arange(200) / 100.0, "space": ["Fp1", "Fp2", "C3", "C4"]},
    )
    data = cb.from_xarray(xr_da, subjectID="s1")
    out = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    assert list(out.data.coords["space"].values) == ["Fp1", "Fp2", "C3", "C4"]


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_bandpass_single_band_matches_manual_scipy() -> None:
    sr = 250.0
    data = _make_data(n_time=500, n_space=2, sampling_rate=sr)

    out = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    b, a = signal.butter(3, [8, 12], btype="band", fs=sr)
    expected = signal.lfilter(b, a, data.to_numpy(), axis=-1)

    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-12)


def test_bandpass_multi_band_equals_sum() -> None:
    """The output for multiple bands equals the sum of the individual filters."""
    sr = 250.0
    data = _make_data(n_time=500, n_space=2, sampling_rate=sr)
    bands = [[8, 12], [12, 30]]

    out = cb.BandpassFilter(bands=bands).apply(data)

    total = None
    for low, high in bands:
        b, a = signal.butter(3, [low, high], btype="band", fs=sr)
        filtered = signal.lfilter(b, a, data.to_numpy(), axis=-1)
        total = filtered if total is None else total + filtered

    np.testing.assert_allclose(out.to_numpy(), total, atol=1e-12)


def test_bandpass_sine_concentrated_in_band() -> None:
    """A single-band filter passes its target sine and rejects out-of-band sines."""
    sr = 500.0
    duration = 10.0
    trim = int(sr)  # discard filter transient

    data = _make_sine_data(freqs_hz=[10.0, 50.0], sampling_rate=sr, duration=duration)
    out = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    arr = data.to_numpy()  # (time, space)
    arr_out = out.to_numpy()  # (space, time) after SignalData transpose
    rms_out = float(np.sqrt(np.mean(arr_out[..., trim:] ** 2)))

    in_rms = float(np.sqrt(np.mean(arr[trim:, :] ** 2)))
    assert rms_out > 0, "filtered output should retain the in-band sine"
    assert rms_out < in_rms, "out-of-band sine should be attenuated"


def test_bandpass_single_band_rejects_out_of_band() -> None:
    """A 50 Hz sine should be strongly attenuated by an 8-12 Hz bandpass."""
    sr = 500.0
    duration = 10.0
    trim = int(sr)

    data = _make_sine_data(freqs_hz=[50.0], sampling_rate=sr, duration=duration)
    out = cb.BandpassFilter(bands=[[8, 12]]).apply(data)

    arr = data.to_numpy()  # (time, space)
    arr_out = out.to_numpy()  # (space, time)
    in_rms = float(np.sqrt(np.mean(arr[trim:, :] ** 2)))
    out_rms = float(np.sqrt(np.mean(arr_out[..., trim:] ** 2)))

    assert out_rms < in_rms * 0.3, (
        f"50 Hz should be attenuated by 8-12 Hz bandpass: in={in_rms:.4f}, out={out_rms:.4f}"
    )


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


def test_bandpass_output_is_valid_data_for_further_features() -> None:
    data = _make_data(n_time=200, n_space=2)
    out = cb.BandpassFilter(bands=[[8, 12], [12, 30]]).apply(data)

    reduced = cb.Mean(dim="space").apply(out)
    assert "space" not in reduced.data.dims
    assert "BandpassFilter" in reduced.history
    assert "Mean" in reduced.history


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_bandpass_zero_order_raises() -> None:
    with pytest.raises(ValueError, match="ord"):
        cb.BandpassFilter(bands=[[8, 12]], ord=0)


def test_bandpass_negative_order_raises() -> None:
    with pytest.raises(ValueError, match="ord"):
        cb.BandpassFilter(bands=[[8, 12]], ord=-1)


def test_bandpass_empty_bands_raises() -> None:
    with pytest.raises(ValueError, match="bands"):
        cb.BandpassFilter(bands=[])


def test_bandpass_missing_bands_raises_type_error() -> None:
    with pytest.raises(TypeError):
        cb.BandpassFilter()  # type: ignore[call-arg]


def test_bandpass_invalid_band_range_raises() -> None:
    with pytest.raises(ValueError, match="low frequency"):
        cb.BandpassFilter(bands=[[20, 10]])


def test_bandpass_negative_frequency_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cb.BandpassFilter(bands=[[-5, 10]])


def test_bandpass_band_wrong_number_of_frequencies_raises() -> None:
    with pytest.raises(ValueError, match="exactly 2"):
        cb.BandpassFilter(bands=[[1, 10, 20]])


def test_bandpass_band_exceeds_nyquist_raises() -> None:
    data = _make_data(sampling_rate=100.0)
    with pytest.raises(ValueError, match="Nyquist"):
        cb.BandpassFilter(bands=[[40, 60]]).apply(data)


def test_bandpass_missing_sampling_rate_raises() -> None:
    import xarray as xr

    rng = np.random.default_rng(42)
    arr = rng.standard_normal((100, 3))
    xr_da = xr.DataArray(arr, dims=["time", "space"])
    data = cb.SignalData.from_xarray(xr_da, subjectID="s1")
    with pytest.raises(ValueError, match="sampling_rate"):
        cb.BandpassFilter(bands=[[8, 12]]).apply(data)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_bandpass_serialization_round_trip() -> None:
    feature = cb.BandpassFilter(bands=[[8, 12], [12, 30]], ord=4)
    yaml_str = feature.to_yaml()
    reloaded = cb.deserialize(yaml_str)
    assert isinstance(reloaded, cb.Pipeline)
    assert len(reloaded.features) == 1
    nf = reloaded.features[0]
    assert isinstance(nf, cb.BandpassFilter)
    assert nf.bands == [[8, 12], [12, 30]]
    assert nf.ord == 4
