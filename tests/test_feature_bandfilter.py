"""Tests for the BandFilter feature."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

import cobrabox as cb


def _make_data(
    n_time: int = 1000, n_space: int = 3, sampling_rate: float = 250.0, subject: str = "sub-01"
) -> cb.Data:
    """Create a simple Data object with white noise for testing."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((n_time, n_space))
    return cb.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sampling_rate, subjectID=subject
    )


def _make_sine_data(
    freqs_hz: list[float], sampling_rate: float = 250.0, duration: float = 4.0, n_space: int = 1
) -> cb.Data:
    """Create a Data object whose signal is a sum of pure sinusoids."""
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    sig = np.zeros_like(t)
    for f in freqs_hz:
        sig += np.sin(2 * np.pi * f * t)
    arr = np.tile(sig[:, None], (1, n_space))
    return cb.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Basic API and shape tests
# ---------------------------------------------------------------------------


def test_bandfilter_default_band_coords() -> None:
    """Default bands have the standard EEG band names as coordinates."""
    data = _make_data()
    out = cb.feature.BandFilter().apply(data)

    expected_names = ["delta", "theta", "alpha", "beta", "gamma"]
    assert list(out.data.coords["band"].values) == expected_names


def test_bandfilter_default_band_coords_keep_orig() -> None:
    """Default bands have the standard EEG band names as coordinates."""
    data = _make_data()
    out = cb.feature.BandFilter(keep_orig=True).apply(data)

    expected_names = ["original", "delta", "theta", "alpha", "beta", "gamma"]
    assert list(out.data.coords["band"].values) == expected_names


def test_bandfilter_custom_bands() -> None:
    """Custom bands dict is respected in shape and coordinate labels."""
    data = _make_data()
    out = cb.feature.BandFilter(bands={"low": [1, 10], "high": [30, 60]}).apply(data)

    assert out.data.sizes["band"] == 2
    assert list(out.data.coords["band"].values) == ["low", "high"]


def test_bandfilter_single_band() -> None:
    """A single-band dict still produces a band dimension of size 1."""
    data = _make_data()
    out = cb.feature.BandFilter(bands={"alpha": [8, 12]}).apply(data)

    assert out.data.sizes["band"] == 1
    assert list(out.data.coords["band"].values) == ["alpha"]


# ---------------------------------------------------------------------------
# Coordinate preservation
# ---------------------------------------------------------------------------


def test_bandfilter_preserves_time_coords() -> None:
    """Time coordinates survive the round-trip through filtering."""
    data = _make_data(sampling_rate=100.0)
    out = cb.feature.BandFilter(bands={"alpha": [8, 12]}).apply(data)

    np.testing.assert_array_equal(out.data.coords["time"].values, data.data.coords["time"].values)


def test_bandfilter_preserves_space_coords_when_present() -> None:
    """Space coordinates are kept when the input has them."""
    import xarray as xr

    arr = np.random.default_rng(0).standard_normal((200, 4))
    xr_da = xr.DataArray(
        arr,
        dims=["time", "space"],
        coords={"time": np.arange(200) / 100.0, "space": ["Fp1", "Fp2", "C3", "C4"]},
    )
    data = cb.from_xarray(xr_da, subjectID="s1")
    out = cb.feature.BandFilter(bands={"alpha": [8, 12]}).apply(data)

    assert list(out.data.coords["space"].values) == ["Fp1", "Fp2", "C3", "C4"]


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_bandfilter_matches_manual_scipy_filter() -> None:
    """Each band slice matches a manual scipy butter+lfilter call."""
    data = _make_data(n_time=500, n_space=2, sampling_rate=250.0)
    bands = {"alpha": [8, 12], "beta": [12, 30]}
    out = cb.feature.BandFilter(bands=bands).apply(data)

    # data.to_numpy() returns (time, space); lfilter along axis=0 (time).
    # out.data.sel(band=name) has dims (space, time) so transpose for comparison.
    arr_in = data.to_numpy()  # shape (time, space)
    for name, freqs in bands.items():
        b, a = signal.butter(3, freqs, btype="band", fs=250.0)
        expected = signal.lfilter(b, a, arr_in, axis=0).T  # → (space, time)
        actual = out.data.sel(band=name).values  # (space, time)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_bandfilter_output_is_not_all_zeros() -> None:
    """Filtered output should contain non-zero values for broadband input."""
    data = _make_data()
    out = cb.feature.BandFilter(bands={"alpha": [8, 12]}).apply(data)

    assert not np.allclose(out.to_numpy(), 0.0)


@pytest.mark.parametrize(
    ("pass_freq_hz", "reject_freqs_hz", "pass_band"),
    [(2.0, [10.0, 20.0], "delta"), (10.0, [2.0, 20.0], "alpha"), (20.0, [2.0, 10.0], "beta")],
)
def test_bandfilter_sine_concentrated_in_correct_band(
    pass_freq_hz: float, reject_freqs_hz: list[float], pass_band: str
) -> None:
    """Each band passes its target sine and rejects the out-of-band sines.

    The mixed signal (all three sines) is filtered, then the same band is also
    applied to a signal built from *only* the out-of-band sines. The target
    sine's contribution should dominate the out-of-band leakage in that band.
    """
    bands = {"delta": [1, 4], "alpha": [8, 12], "beta": [12, 30]}
    sr = 250.0
    duration = 4.0
    trim = int(sr)  # discard first second to avoid filter transient

    # Filter the full mixed signal — contains target + out-of-band sines
    mixed = _make_sine_data(
        freqs_hz=[pass_freq_hz, *reject_freqs_hz], sampling_rate=sr, duration=duration, n_space=1
    )
    out_mixed = cb.feature.BandFilter(bands=bands).apply(mixed)
    rms_mixed = float(np.sqrt(np.mean(out_mixed.data.sel(band=pass_band).values[..., trim:] ** 2)))

    # Filter a signal built from *only* the out-of-band sines through the same band
    out_of_band = _make_sine_data(
        freqs_hz=reject_freqs_hz, sampling_rate=sr, duration=duration, n_space=1
    )
    out_reject = cb.feature.BandFilter(bands=bands).apply(out_of_band)
    rms_reject = float(
        np.sqrt(np.mean(out_reject.data.sel(band=pass_band).values[..., trim:] ** 2))
    )

    assert rms_mixed > rms_reject * 3, (
        f"{pass_freq_hz} Hz sine in '{pass_band}': mixed RMS ({rms_mixed:.4f}) should be "
        f"at least 3x out-of-band-only RMS ({rms_reject:.4f})"
    )


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


def test_bandfilter_output_is_valid_data_for_further_features() -> None:
    """The output can be fed into another feature (e.g. Mean)."""
    data = _make_data(n_time=200, n_space=2)
    out = cb.feature.BandFilter(bands={"alpha": [8, 12], "beta": [12, 30]}).apply(data)

    # Mean over band dimension should collapse it
    reduced = cb.feature.Mean(dim="band").apply(out)
    assert "band" not in reduced.data.dims
    assert "BandFilter" in reduced.history
    assert "Mean" in reduced.history
