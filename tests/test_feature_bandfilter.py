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


def test_bandfilter_history_updated() -> None:
    """BandFilter appends 'BandFilter' to history."""
    data = _make_data()
    result = cb.feature.BandFilter().apply(data)
    assert result.history[-1] == "BandFilter"


def test_bandfilter_metadata_preserved() -> None:
    """BandFilter preserves subjectID, groupID, condition, and sampling_rate."""
    rng = np.random.default_rng(42)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((100, 3)),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.BandFilter().apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(250.0)


def test_bandfilter_returns_data_instance() -> None:
    """BandFilter.apply() always returns a Data instance."""
    data = _make_data()
    result = cb.feature.BandFilter().apply(data)
    assert isinstance(result, cb.Data)


def test_bandfilter_does_not_mutate_input() -> None:
    """BandFilter does not modify the input Data object."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.BandFilter().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_bandfilter_missing_sampling_rate_raises() -> None:
    """BandFilter raises ValueError when input has no sampling_rate."""
    import xarray as xr

    rng = np.random.default_rng(42)
    arr = rng.standard_normal((100, 3))
    xr_da = xr.DataArray(arr, dims=["time", "space"])
    data = cb.Data.from_xarray(xr_da, subjectID="s1")
    assert data.sampling_rate is None
    with pytest.raises(ValueError, match="sampling_rate"):
        cb.feature.BandFilter().apply(data)


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


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_bandfilter_zero_order_raises() -> None:
    """BandFilter raises ValueError for ord of 0."""
    with pytest.raises(ValueError, match="ord"):
        cb.feature.BandFilter(ord=0)


def test_bandfilter_negative_order_raises() -> None:
    """BandFilter raises ValueError for negative ord."""
    with pytest.raises(ValueError, match="ord"):
        cb.feature.BandFilter(ord=-1)


def test_bandfilter_empty_bands_raises() -> None:
    """BandFilter raises ValueError when bands dict is empty."""
    with pytest.raises(ValueError, match="bands"):
        cb.feature.BandFilter(bands={})


def test_bandfilter_invalid_band_range_raises() -> None:
    """BandFilter raises ValueError when band low >= high frequency."""
    data = _make_data()
    with pytest.raises(ValueError, match="Band"):
        cb.feature.BandFilter(bands={"bad": [20, 10]}).apply(data)


def test_bandfilter_negative_frequency_raises() -> None:
    """BandFilter raises ValueError for negative frequencies."""
    data = _make_data()
    with pytest.raises(ValueError, match="frequencies must be non-negative"):
        cb.feature.BandFilter(bands={"bad": [-5, 10]}).apply(data)


def test_bandfilter_band_wrong_number_of_frequencies_raises() -> None:
    """BandFilter raises ValueError when band doesn't have exactly 2 frequencies."""
    data = _make_data()
    with pytest.raises(ValueError, match="exactly 2"):
        cb.feature.BandFilter(bands={"bad": [1, 10, 20]}).apply(data)
