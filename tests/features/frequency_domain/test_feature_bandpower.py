"""Tests for the Bandpower feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb


def _sine_data(
    freq_hz: float,
    sampling_rate: float = 256.0,
    n_seconds: float = 4.0,
    n_channels: int = 2,
    subjectID: str = "sub-01",
) -> cb.SignalData:
    """Helper: pure sine wave at ``freq_hz`` Hz."""
    n_time = int(n_seconds * sampling_rate)
    t = np.arange(n_time) / sampling_rate
    signal = np.sin(2 * np.pi * freq_hz * t)
    arr = np.stack([signal] * n_channels, axis=1)  # (time, space)
    return cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sampling_rate, subjectID=subjectID
    )


# ---------------------------------------------------------------------------
# Dims, shape and coordinates
# ---------------------------------------------------------------------------


def test_bandpower_default_dims_and_shape() -> None:
    """Default bands produce (band_index, space, time=1) output."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Bandpower().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("band_index", "space", "time")
    assert out.data.shape == (5, 2, 1)


def test_bandpower_default_band_index_coords() -> None:
    """band_index coordinate matches the five default band names in order."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Bandpower().apply(data)

    expected_names = ["delta", "theta", "alpha", "beta", "gamma"]
    assert out.data.coords["band_index"].values.tolist() == expected_names


def test_bandpower_custom_range_shape() -> None:
    """Custom band spec produces correct shape and band_index coordinate."""
    data = _sine_data(freq_hz=50.0, sampling_rate=512.0)
    out = cb.feature.Bandpower(bands={"ripple": [45, 80]}).apply(data)

    assert out.data.shape == (1, 2, 1)
    assert out.data.coords["band_index"].values.tolist() == ["ripple"]


def test_bandpower_mixed_spec_shape() -> None:
    """Mixed True + custom range produces correct number of bands."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Bandpower(bands={"alpha": True, "ripple": [45, 80]}).apply(data)

    assert out.data.shape == (2, 2, 1)
    assert out.data.coords["band_index"].values.tolist() == ["alpha", "ripple"]


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_bandpower_alpha_dominates_for_10hz_sine() -> None:
    """10 Hz sine should have highest power in the alpha band [8, 12]."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Bandpower().apply(data)

    band_names = out.data.coords["band_index"].values.tolist()
    alpha_idx = band_names.index("alpha")

    # For each channel the alpha power should be strictly the largest
    powers = out.data.values[:, :, 0]  # (band_index, space)
    n_channels = powers.shape[1]
    for ch in range(n_channels):
        assert powers[alpha_idx, ch] == powers[:, ch].max(), (
            f"Channel {ch}: alpha power not dominant. Powers: {powers[:, ch]}"
        )


def test_bandpower_true_alias_matches_explicit_range() -> None:
    """bands={'alpha': True} must give identical results to bands={'alpha': [8, 12]}."""
    data = _sine_data(freq_hz=10.0)

    out_true = cb.feature.Bandpower(bands={"alpha": True}).apply(data)
    out_explicit = cb.feature.Bandpower(bands={"alpha": [8, 12]}).apply(data)

    np.testing.assert_allclose(out_true.to_numpy(), out_explicit.to_numpy())


def test_bandpower_all_positive_values() -> None:
    """Power values must be non-negative for any signal."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((512, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.Bandpower().apply(data)

    assert (out.data.values >= 0).all()


def test_bandpower_empty_bands_equals_none() -> None:
    """bands={} and bands=None must produce identical results."""
    data = _sine_data(freq_hz=10.0)

    out_none = cb.feature.Bandpower(bands=None).apply(data)
    out_empty = cb.feature.Bandpower(bands={}).apply(data)

    np.testing.assert_allclose(out_none.to_numpy(), out_empty.to_numpy())


# ---------------------------------------------------------------------------
# nperseg parameter
# ---------------------------------------------------------------------------


def test_bandpower_nperseg_changes_nothing_in_shape() -> None:
    """nperseg only affects estimation quality, not output shape."""
    data = _sine_data(freq_hz=10.0)

    out_default = cb.feature.Bandpower().apply(data)
    out_128 = cb.feature.Bandpower(nperseg=128).apply(data)
    out_512 = cb.feature.Bandpower(nperseg=512).apply(data)

    assert out_default.data.shape == out_128.data.shape == out_512.data.shape


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_bandpower_history_appended() -> None:
    """'Bandpower' must appear as the last entry in history."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.Bandpower().apply(data)

    assert out.history[-1] == "Bandpower"


def test_bandpower_metadata_preserved() -> None:
    """subjectID, groupID, condition, and sampling_rate are carried through."""
    data = cb.SignalData.from_numpy(
        _sine_data(freq_hz=10.0).to_numpy(),
        dims=["time", "space"],
        sampling_rate=256.0,
        subjectID="sub-42",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.Bandpower().apply(data)

    assert out.subjectID == "sub-42"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    assert out.sampling_rate == 256.0


def test_bandpower_does_not_mutate_input() -> None:
    """Bandpower.apply() leaves the input Data object unchanged."""
    data = _sine_data(freq_hz=10.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Bandpower().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_bandpower_raises_when_sampling_rate_missing() -> None:
    """ValueError raised when sampling_rate is not set."""
    import xarray as xr

    from cobrabox.features.frequency_domain.bandpower import Bandpower

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((2, 10)), dims=["space", "time"])

        @property
        def sampling_rate(self) -> None:
            return None

    with pytest.raises(ValueError, match="sampling_rate must be set"):
        Bandpower().__call__(_FakeData())  # type: ignore[arg-type]


def test_bandpower_raises_for_true_with_unknown_band() -> None:
    """ValueError raised when True is used for an unrecognised band name."""
    data = _sine_data(freq_hz=10.0)

    with pytest.raises(ValueError, match="not a known default band"):
        cb.feature.Bandpower(bands={"foobar": True}).apply(data)


def test_bandpower_raises_when_nperseg_less_than_2() -> None:
    """Bandpower raises ValueError when nperseg is less than 2."""
    with pytest.raises(ValueError, match="nperseg must be >= 2"):
        cb.feature.Bandpower(nperseg=1)


def test_bandpower_raises_for_false_band_spec() -> None:
    """Bandpower raises ValueError when band spec is False."""
    data = _sine_data(freq_hz=10.0)
    with pytest.raises(ValueError, match="must be True"):
        cb.feature.Bandpower(bands={"alpha": False}).apply(data)


def test_bandpower_transposes_when_time_not_last() -> None:
    """When time is not the last dim, Bandpower transposes before computing."""
    import xarray as xr

    from cobrabox.features.frequency_domain.bandpower import Bandpower

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((10, 4)), dims=["time", "space"])

        @property
        def sampling_rate(self) -> float:
            return 256.0

    out = Bandpower().__call__(_FakeData())  # type: ignore[arg-type]
    assert "band_index" in out.dims


def test_bandpower_zeros_when_no_freq_bins_in_band() -> None:
    """Bands with no matching frequency bins return zero power."""
    # Low sampling rate → Nyquist = 5 Hz; band [100, 200] has no bins
    data = _sine_data(freq_hz=1.0, sampling_rate=10.0, n_seconds=4.0)
    out = cb.feature.Bandpower(bands={"ultra": [100.0, 200.0]}).apply(data)
    assert (out.to_numpy() == 0.0).all()
