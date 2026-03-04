"""Tests for the bandpower feature."""

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
) -> cb.Data:
    """Helper: pure sine wave at ``freq_hz`` Hz."""
    n_time = int(n_seconds * sampling_rate)
    t = np.arange(n_time) / sampling_rate
    signal = np.sin(2 * np.pi * freq_hz * t)
    arr = np.stack([signal] * n_channels, axis=1)  # (time, space)
    return cb.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sampling_rate, subjectID=subjectID
    )


# ---------------------------------------------------------------------------
# Dims, shape and coordinates
# ---------------------------------------------------------------------------


def test_bandpower_default_dims_and_shape() -> None:
    """Default bands produce (band_index, space, time=1) output."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.bandpower(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("band_index", "space", "time")
    assert out.data.shape == (5, 2, 1)


def test_bandpower_default_band_index_coords() -> None:
    """band_index coordinate matches the five default band names in order."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.bandpower(data)

    expected_names = ["delta", "theta", "alpha", "beta", "gamma"]
    assert out.data.coords["band_index"].values.tolist() == expected_names


def test_bandpower_custom_range_shape() -> None:
    """Custom band spec produces correct shape and band_index coordinate."""
    data = _sine_data(freq_hz=50.0, sampling_rate=512.0)
    out = cb.feature.bandpower(data, bands={"ripple": [45, 80]})

    assert out.data.shape == (1, 2, 1)
    assert out.data.coords["band_index"].values.tolist() == ["ripple"]


def test_bandpower_mixed_spec_shape() -> None:
    """Mixed True + custom range produces correct number of bands."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.bandpower(data, bands={"alpha": True, "ripple": [45, 80]})

    assert out.data.shape == (2, 2, 1)
    assert out.data.coords["band_index"].values.tolist() == ["alpha", "ripple"]


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_bandpower_alpha_dominates_for_10hz_sine() -> None:
    """10 Hz sine should have highest power in the alpha band [8, 12]."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.bandpower(data)

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

    out_true = cb.feature.bandpower(data, bands={"alpha": True})
    out_explicit = cb.feature.bandpower(data, bands={"alpha": [8, 12]})

    np.testing.assert_allclose(out_true.to_numpy(), out_explicit.to_numpy())


def test_bandpower_all_positive_values() -> None:
    """Power values must be non-negative for any signal."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((512, 4))
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.bandpower(data)

    assert (out.data.values >= 0).all()


def test_bandpower_empty_bands_equals_none() -> None:
    """bands={} and bands=None must produce identical results."""
    data = _sine_data(freq_hz=10.0)

    out_none = cb.feature.bandpower(data, bands=None)
    out_empty = cb.feature.bandpower(data, bands={})

    np.testing.assert_allclose(out_none.to_numpy(), out_empty.to_numpy())


# ---------------------------------------------------------------------------
# nperseg parameter
# ---------------------------------------------------------------------------


def test_bandpower_nperseg_changes_nothing_in_shape() -> None:
    """nperseg only affects estimation quality, not output shape."""
    data = _sine_data(freq_hz=10.0)

    out_default = cb.feature.bandpower(data)
    out_128 = cb.feature.bandpower(data, nperseg=128)
    out_512 = cb.feature.bandpower(data, nperseg=512)

    assert out_default.data.shape == out_128.data.shape == out_512.data.shape


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_bandpower_history_appended() -> None:
    """'bandpower' must appear as the last entry in history."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.bandpower(data)

    assert out.history[-1] == "bandpower"


def test_bandpower_metadata_preserved() -> None:
    """subjectID and sampling_rate are carried through."""
    data = _sine_data(freq_hz=10.0, subjectID="sub-42")
    out = cb.feature.bandpower(data)

    assert out.subjectID == "sub-42"
    assert out.sampling_rate == 256.0


def test_bandpower_squeeze_removes_singleton_time() -> None:
    """Squeezing the singleton time axis yields (band_index, space)."""
    data = _sine_data(freq_hz=10.0)
    out = cb.feature.bandpower(data)

    squeezed = out.data.squeeze("time")
    assert squeezed.dims == ("band_index", "space")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_bandpower_raises_when_time_dim_missing() -> None:
    """ValueError raised when data lacks 'time' dimension."""
    import xarray as xr

    from cobrabox.features.bandpower import bandpower

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])

        @property
        def sampling_rate(self) -> float:
            return 256.0

    with pytest.raises(ValueError, match="must have 'time' dimension"):
        bandpower.__wrapped__(_FakeData())  # type: ignore[attr-defined]


def test_bandpower_raises_when_sampling_rate_missing() -> None:
    """ValueError raised when sampling_rate is not set."""
    import xarray as xr

    from cobrabox.features.bandpower import bandpower

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((2, 10)), dims=["space", "time"])

        @property
        def sampling_rate(self) -> None:
            return None

    with pytest.raises(ValueError, match="sampling_rate must be set"):
        bandpower.__wrapped__(_FakeData())  # type: ignore[attr-defined]


def test_bandpower_raises_for_true_with_unknown_band() -> None:
    """ValueError raised when True is used for an unrecognised band name."""
    data = _sine_data(freq_hz=10.0)

    with pytest.raises(ValueError, match="not a known default band"):
        cb.feature.bandpower(data, bands={"foobar": True})
