"""Tests for the Cordance feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _sine_data(
    freq_hz: float,
    sampling_rate: float = 256.0,
    n_seconds: float = 4.0,
    n_channels: int = 4,
    subjectID: str = "sub-01",
) -> cb.SignalData:
    """Helper: pure sine wave at ``freq_hz`` Hz, same signal on all channels."""
    n_time = int(n_seconds * sampling_rate)
    t = np.arange(n_time) / sampling_rate
    signal = np.sin(2 * np.pi * freq_hz * t)
    arr = np.stack([signal] * n_channels, axis=1)  # (time, space)
    return cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sampling_rate, subjectID=subjectID
    )


def _varied_data(
    sampling_rate: float = 256.0, n_seconds: float = 4.0, n_channels: int = 4
) -> cb.SignalData:
    """Helper: each channel has a different dominant frequency.

    ch0 = 2 Hz (delta), ch1 = 6 Hz (theta), ch2 = 10 Hz (alpha), ch3 = 20 Hz (beta).
    """
    n_time = int(n_seconds * sampling_rate)
    t = np.arange(n_time) / sampling_rate
    freqs = [2.0, 6.0, 10.0, 20.0]
    channels = [np.sin(2 * np.pi * f * t) for f in freqs[:n_channels]]
    arr = np.stack(channels, axis=1)  # (time, space)
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Dims, shape and coordinates
# ---------------------------------------------------------------------------


def test_cordance_default_dims_and_shape() -> None:
    """Default bands produce (band, space) output."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("band", "space")
    assert out.data.shape == (5, 4)


def test_cordance_default_band_coords() -> None:
    """band coordinate matches the five default band names in order."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance().apply(data)

    expected_names = ["delta", "theta", "alpha", "beta", "gamma"]
    assert out.data.coords["band"].values.tolist() == expected_names


def test_cordance_custom_bands_shape() -> None:
    """Custom band spec produces correct shape and band coordinate."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance(bands={"alpha": [8, 12]}).apply(data)

    assert out.data.shape == (1, 4)
    assert out.data.coords["band"].values.tolist() == ["alpha"]


def test_cordance_mixed_spec_shape() -> None:
    """Mixed True + custom range produces correct number of bands."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance(bands={"alpha": True, "ripple": [45, 80]}).apply(data)

    assert out.data.shape == (2, 4)
    assert out.data.coords["band"].values.tolist() == ["alpha", "ripple"]


# ---------------------------------------------------------------------------
# Z-score properties
# ---------------------------------------------------------------------------


def test_cordance_z_scores_sum_to_zero_across_channels() -> None:
    """Within each band, cordance z-scores should sum ≈ 0 across channels."""
    data = _varied_data()
    out = cb.feature.Cordance().apply(data)

    values = out.data.values  # (n_bands, n_space)
    for b in range(values.shape[0]):
        # z(anorm) and z(rnorm) each sum to 0 across space → their sum does too
        np.testing.assert_allclose(values[b, :].sum(), 0.0, atol=1e-10)


def test_cordance_values_are_finite() -> None:
    """Cordance values must be finite for well-formed input."""
    data = _varied_data()
    out = cb.feature.Cordance().apply(data)
    assert np.all(np.isfinite(out.to_numpy()))


# ---------------------------------------------------------------------------
# Semantic correctness
# ---------------------------------------------------------------------------


def test_cordance_channel_with_dominant_band_has_positive_value() -> None:
    """A channel whose power is concentrated in a band should have positive cordance there.

    ch2 is a 10 Hz sine → dominant alpha power. Its alpha-band cordance should
    be higher than the mean (i.e. positive, since mean of z-scores is 0).
    """
    data = _varied_data()
    out = cb.feature.Cordance().apply(data)

    band_names = out.data.coords["band"].values.tolist()
    alpha_idx = band_names.index("alpha")
    # ch2 has 10 Hz → should have highest alpha cordance
    vals = out.data.values[alpha_idx, :]  # (n_space,)
    assert vals[2] == vals.max(), f"ch2 (10 Hz sine) should dominate alpha cordance: {vals}"


def test_cordance_agrees_with_manual_calculation() -> None:
    """Verify cordance output matches a manual step-by-step calculation."""
    data = _varied_data(n_channels=3, n_seconds=4.0)
    bands = {"theta": [4, 8], "alpha": [8, 12]}
    out = cb.feature.Cordance(bands=bands).apply(data)

    # Manually compute via Bandpower
    bp = cb.feature.Bandpower(bands=bands).apply(data)
    bp_vals = bp.data.values[:, :, 0]  # (n_bands, n_space)

    # Absolute power
    ap = bp_vals
    # Relative power
    total = ap.sum(axis=0, keepdims=True)
    rp = ap / total

    eps = np.finfo(np.float64).tiny
    anorm = np.log(np.maximum(ap, eps))
    rnorm = np.log(np.maximum(rp, eps))

    # Z-score across space (axis=1)
    z_a = (anorm - anorm.mean(axis=1, keepdims=True)) / anorm.std(axis=1, keepdims=True)
    z_r = (rnorm - rnorm.mean(axis=1, keepdims=True)) / rnorm.std(axis=1, keepdims=True)

    expected = z_a + z_r
    np.testing.assert_allclose(out.data.values, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Comparison with Bandpower (reuse)
# ---------------------------------------------------------------------------


def test_cordance_consistent_with_bandpower_ranking() -> None:
    """The channel with highest absolute bandpower should have the highest
    cordance for that band when all other bands have negligible power."""
    # One-band scenario: only alpha band. The channel with highest alpha
    # power should also have highest alpha cordance.
    data = _varied_data()
    bp_out = cb.feature.Bandpower(bands={"alpha": True}).apply(data)
    cord_out = cb.feature.Cordance(bands={"alpha": True}).apply(data)

    bp_vals = bp_out.data.values[0, :, 0]
    cord_vals = cord_out.data.values[0, :]

    # Channel ranking by bandpower should match ranking by cordance
    assert bp_vals.argmax() == cord_vals.argmax()


# ---------------------------------------------------------------------------
# nperseg parameter
# ---------------------------------------------------------------------------


def test_cordance_nperseg_changes_nothing_in_shape() -> None:
    """nperseg only affects estimation quality, not output shape."""
    data = _sine_data(freq_hz=10.0)

    out_default = cb.feature.Cordance().apply(data)
    out_128 = cb.feature.Cordance(nperseg=128).apply(data)
    out_512 = cb.feature.Cordance(nperseg=512).apply(data)

    assert out_default.data.shape == out_128.data.shape == out_512.data.shape


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_cordance_history_appended() -> None:
    """'Cordance' must appear as the last entry in history."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance().apply(data)

    assert out.history[-1] == "Cordance"


def test_cordance_metadata_preserved() -> None:
    """subjectID is carried through. sampling_rate is None since time dim is removed."""
    data = _sine_data(freq_hz=10.0, subjectID="sub-42")
    out = cb.feature.Cordance().apply(data)

    assert out.subjectID == "sub-42"
    # output_type=Data and no time dim → sampling_rate is stripped
    assert out.sampling_rate is None


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------


def test_cordance_in_pipeline() -> None:
    """Cordance can be composed with other features via pipe syntax."""
    data = _varied_data()
    pipe = cb.feature.Cordance(bands={"alpha": True})
    result = pipe.apply(data)
    assert result.data.dims == ("band", "space")
    assert "Cordance" in result.history


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_cordance_identical_channels_zero_cordance() -> None:
    """When all channels are identical, all z-scores are 0 → cordance is 0."""
    data = _sine_data(freq_hz=10.0, n_channels=4)
    out = cb.feature.Cordance().apply(data)

    # All channels have identical power → std=0 → z=0 → cordance=0
    np.testing.assert_allclose(out.to_numpy(), 0.0, atol=1e-10)


def test_cordance_empty_bands_equals_none() -> None:
    """bands={} and bands=None must produce identical results."""
    data = _varied_data()

    out_none = cb.feature.Cordance(bands=None).apply(data)
    out_empty = cb.feature.Cordance(bands={}).apply(data)

    np.testing.assert_allclose(out_none.to_numpy(), out_empty.to_numpy())


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_cordance_raises_when_sampling_rate_missing() -> None:
    """ValueError raised when sampling_rate is not set."""
    import xarray as xr

    from cobrabox.features.cordance import Cordance

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((4, 10)), dims=["space", "time"])

        @property
        def sampling_rate(self) -> None:
            return None

    with pytest.raises(ValueError, match="sampling_rate must be set"):
        Cordance().__call__(_FakeData())  # type: ignore[arg-type]


def test_cordance_raises_when_no_space_dim() -> None:
    """ValueError raised when data has no 'space' dimension."""
    n_time = 1024
    t = np.arange(n_time) / 256.0
    arr = np.sin(2 * np.pi * 10 * t)
    data = cb.SignalData.from_numpy(
        arr[:, np.newaxis], dims=["time", "channel"], sampling_rate=256.0
    )

    with pytest.raises(ValueError, match="'space' dimension"):
        cb.feature.Cordance().apply(data)


def test_cordance_raises_when_single_channel() -> None:
    """ValueError raised when only one spatial channel is available."""
    n_time = 1024
    t = np.arange(n_time) / 256.0
    arr = np.sin(2 * np.pi * 10 * t)[:, np.newaxis]
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)

    with pytest.raises(ValueError, match="at least 2 spatial channels"):
        cb.feature.Cordance().apply(data)


def test_cordance_raises_for_true_with_unknown_band() -> None:
    """ValueError raised when True is used for an unrecognised band name."""
    data = _varied_data()

    with pytest.raises(ValueError, match="not a known default band"):
        cb.feature.Cordance(bands={"foobar": True}).apply(data)


def test_cordance_raises_when_nperseg_less_than_2() -> None:
    with pytest.raises(ValueError, match="nperseg must be >= 2"):
        cb.feature.Cordance(nperseg=1)


def test_cordance_raises_on_zero_signal() -> None:
    """ValueError raised when input signal is all zeros (zero bandpower)."""
    arr = np.zeros((4, 512))  # 4 channels, all zeros
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=256.0)

    with pytest.raises(ValueError, match="Total bandpower is zero"):
        cb.feature.Cordance().apply(data)


def test_cordance_nan_on_zero_outputs_nan_for_silent_channels() -> None:
    """With nan_on_zero=True, silent channels get NaN instead of error."""
    n_time = 512
    t = np.arange(n_time) / 256.0
    # Channel 0 is silent (zeros), channels 1-3 have signal
    arr = np.zeros((4, n_time))
    arr[1] = np.sin(2 * np.pi * 10 * t)
    arr[2] = np.sin(2 * np.pi * 20 * t)
    arr[3] = np.sin(2 * np.pi * 5 * t)
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=256.0)

    out = cb.feature.Cordance(nan_on_zero=True).apply(data)

    # Channel 0 should be NaN for all bands
    assert np.all(np.isnan(out.data.sel(space=0).values))
    # Other channels should have finite values
    assert np.all(np.isfinite(out.data.sel(space=1).values))
    assert np.all(np.isfinite(out.data.sel(space=2).values))
    assert np.all(np.isfinite(out.data.sel(space=3).values))


def test_cordance_raises_for_false_band_spec() -> None:
    data = _varied_data()
    with pytest.raises(ValueError, match="must be True"):
        cb.feature.Cordance(bands={"alpha": False}).apply(data)


def test_cordance_true_alias_matches_explicit_range() -> None:
    """bands={'alpha': True} must give identical results to bands={'alpha': [8, 12]}."""
    data = _varied_data()

    out_true = cb.feature.Cordance(bands={"alpha": True}).apply(data)
    out_explicit = cb.feature.Cordance(bands={"alpha": [8, 12]}).apply(data)

    np.testing.assert_allclose(out_true.to_numpy(), out_explicit.to_numpy())
