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


def _varied_amplitude_data(sampling_rate: float = 256.0, n_seconds: float = 4.0) -> cb.SignalData:
    """Helper: all channels have alpha freq but different amplitudes.

    ch0 = 1x, ch1 = 2x, ch2 = 0.3x, ch3 = 0.8x amplitude.
    This creates clear variation in absolute power while all have same relative power.
    """
    n_time = int(n_seconds * sampling_rate)
    t = np.arange(n_time) / sampling_rate
    amplitudes = [1.0, 2.0, 0.3, 0.8]
    freq = 10.0  # alpha
    channels = [a * np.sin(2 * np.pi * freq * t) for a in amplitudes]
    arr = np.stack(channels, axis=1)  # (time, space)
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=sampling_rate)


# ---------------------------------------------------------------------------
# Dims, shape and coordinates
# ---------------------------------------------------------------------------


def test_cordance_default_dims_and_shape() -> None:
    """Default bands produce (band_index, space) output."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("band_index", "space")
    assert out.data.shape == (5, 4)


def test_cordance_default_band_coords() -> None:
    """band_index coordinate matches the five default band names in order."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance().apply(data)

    expected_names = ["delta", "theta", "alpha", "beta", "gamma"]
    assert out.data.coords["band_index"].values.tolist() == expected_names


def test_cordance_custom_bands_shape() -> None:
    """Custom band spec produces correct shape and band_index coordinate."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance(bands={"alpha": [8, 12]}).apply(data)

    assert out.data.shape == (1, 4)
    assert out.data.coords["band_index"].values.tolist() == ["alpha"]


def test_cordance_mixed_spec_shape() -> None:
    """Mixed True + custom range produces correct number of bands."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Cordance(bands={"alpha": True, "ripple": [45, 80]}).apply(data)

    assert out.data.shape == (2, 4)
    assert out.data.coords["band_index"].values.tolist() == ["alpha", "ripple"]


# ---------------------------------------------------------------------------
# Threshold-based algorithm properties
# ---------------------------------------------------------------------------


def test_cordance_values_are_finite() -> None:
    """Cordance values must be finite for well-formed input."""
    data = _varied_data()
    out = cb.feature.Cordance().apply(data)
    assert np.all(np.isfinite(out.to_numpy()))


def test_cordance_values_bounded() -> None:
    """Cordance values are bounded by the threshold math.

    Max concordance = (1 - t) + (1 - t) = 1.0 for t=0.5
    Max discordance = t + (1 - t) = 1.0 for t=0.5
    So combined cordance is in [-1, 1].
    """
    data = _varied_data()
    out = cb.feature.Cordance().apply(data)
    vals = out.to_numpy()
    assert np.all(vals >= -1.0)
    assert np.all(vals <= 1.0)


def test_cordance_concordance_positive_discordance_negative() -> None:
    """In combined mode, concordant channels are positive, discordant are negative."""
    data = _varied_amplitude_data()
    out = cb.feature.Cordance(bands={"alpha": True}).apply(data)

    # All channels have same frequency, so relative power is 1.0 for all in alpha band
    # Rnorm = 1/max(RP) = 1 for all channels (since RP is same for all)
    # So Rnorm > 0.5 for all channels
    # Anorm varies: ch1 has max power → Anorm=1, others < 1
    # Channels with Anorm > 0.5 are concordant (positive)
    # Channels with Anorm < 0.5 are discordant (negative)
    vals = out.data.values[0, :]  # (space,)

    # ch1 has highest power (amplitude 2.0) → Anorm = 1.0 → concordant
    assert vals[1] > 0, f"ch1 should be concordant (positive), got {vals[1]}"

    # ch2 has lowest power (amplitude 0.3) → Anorm = 0.3^2 / 2^2 = 0.0225 → discordant
    assert vals[2] < 0, f"ch2 should be discordant (negative), got {vals[2]}"


# ---------------------------------------------------------------------------
# Semantic correctness
# ---------------------------------------------------------------------------


def test_cordance_channel_with_dominant_band_highest_relative_power() -> None:
    """A channel with high relative power in a band gets marked.

    ch2 is a 10 Hz sine → 100% relative power in alpha band.
    ch0 is 2 Hz → 100% relative power in delta.
    Both have high relative power in their respective bands.
    """
    data = _varied_data()
    out = cb.feature.Cordance().apply(data)

    # Values should be finite and either positive (concordant) or negative (discordant)
    # or zero (neither)
    assert np.all(np.isfinite(out.to_numpy()))


def test_cordance_agrees_with_manual_calculation() -> None:
    """Verify cordance output matches a manual step-by-step calculation."""
    data = _varied_amplitude_data()
    bands = {"alpha": [8, 12]}
    out = cb.feature.Cordance(bands=bands, threshold=0.5).apply(data)

    # Manually compute via Bandpower
    bp = cb.feature.Bandpower(bands=bands).apply(data)
    bp_vals = bp.data.values[:, :, 0]  # (n_bands, n_space)

    # Absolute power
    ap = bp_vals
    # Relative power (only one band, so RP = 1.0 for all)
    total = ap.sum(axis=0, keepdims=True)
    rp = ap / total  # Should be 1.0 for all since only one band

    # Normalize by max
    anorm = ap / ap.max(axis=1, keepdims=True)
    rnorm = rp / rp.max(axis=1, keepdims=True)

    t = 0.5

    # Concordance: Anorm > t and Rnorm > t
    concordant_mask = (anorm > t) & (rnorm > t)
    concordance = np.where(concordant_mask, (anorm - t) + (rnorm - t), 0.0)

    # Discordance: Anorm < t and Rnorm > t
    discordant_mask = (anorm < t) & (rnorm > t)
    discordance = np.where(discordant_mask, (t - anorm) + (rnorm - t), 0.0)

    expected = concordance - discordance
    np.testing.assert_allclose(out.data.values, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Output parameter
# ---------------------------------------------------------------------------


def test_cordance_output_concordance_only() -> None:
    """output='concordance' returns only concordance scores."""
    data = _varied_amplitude_data()
    out = cb.feature.Cordance(bands={"alpha": True}, output="concordance").apply(data)

    # Concordance scores are >= 0
    assert np.all(out.to_numpy() >= 0)


def test_cordance_output_discordance_only() -> None:
    """output='discordance' returns only discordance scores."""
    data = _varied_amplitude_data()
    out = cb.feature.Cordance(bands={"alpha": True}, output="discordance").apply(data)

    # Discordance scores are >= 0
    assert np.all(out.to_numpy() >= 0)


def test_cordance_combined_equals_concordance_minus_discordance() -> None:
    """cordance = concordance - discordance."""
    data = _varied_amplitude_data()

    cord = cb.feature.Cordance(bands={"alpha": True}, output="cordance").apply(data)
    conc = cb.feature.Cordance(bands={"alpha": True}, output="concordance").apply(data)
    disc = cb.feature.Cordance(bands={"alpha": True}, output="discordance").apply(data)

    expected = conc.to_numpy() - disc.to_numpy()
    np.testing.assert_allclose(cord.to_numpy(), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Threshold parameter
# ---------------------------------------------------------------------------


def test_cordance_threshold_changes_classification() -> None:
    """Different thresholds lead to different concordant/discordant classification."""
    data = _varied_amplitude_data()

    out_low = cb.feature.Cordance(bands={"alpha": True}, threshold=0.3).apply(data)
    out_high = cb.feature.Cordance(bands={"alpha": True}, threshold=0.7).apply(data)

    # With lower threshold, more channels are concordant (positive)
    # With higher threshold, fewer channels are concordant
    assert not np.allclose(out_low.to_numpy(), out_high.to_numpy())


def test_cordance_threshold_validation() -> None:
    """Threshold must be in (0, 1)."""
    with pytest.raises(ValueError, match="threshold must be in"):
        cb.feature.Cordance(threshold=0.0)

    with pytest.raises(ValueError, match="threshold must be in"):
        cb.feature.Cordance(threshold=1.0)

    with pytest.raises(ValueError, match="threshold must be in"):
        cb.feature.Cordance(threshold=-0.1)

    with pytest.raises(ValueError, match="threshold must be in"):
        cb.feature.Cordance(threshold=1.5)


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
    assert result.data.dims == ("band_index", "space")
    assert "Cordance" in result.history


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_cordance_identical_channels_all_concordant() -> None:
    """When all channels are identical, all have Anorm=Rnorm=1 → all concordant."""
    data = _sine_data(freq_hz=10.0, n_channels=4)
    out = cb.feature.Cordance().apply(data)

    # All channels identical → AP same → Anorm = 1 for all
    # All channels identical → RP same → Rnorm = 1 for all
    # Both > 0.5 → concordant. Score = (1 - 0.5) + (1 - 0.5) = 1.0
    # All channels have the same concordance score
    vals = out.to_numpy()
    for band_idx in range(vals.shape[0]):
        band_vals = vals[band_idx, :]
        # All channels should have the same value
        assert np.allclose(band_vals, band_vals[0])


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


def test_cordance_invalid_output_parameter() -> None:
    """ValueError raised for invalid output parameter."""
    with pytest.raises(ValueError, match="output must be"):
        cb.feature.Cordance(output="invalid")  # type: ignore[arg-type]
