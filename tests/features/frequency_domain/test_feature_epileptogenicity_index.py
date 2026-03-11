"""Tests for the EpileptogenicityIndex feature.

Signal design rationale
-----------------------
A detectable rapid discharge needs to dominate the gamma band relative to the
theta/alpha background.  The synthetic helper ``_gamma_onset_signal`` produces:

  - Background: amplitude-1 sine at 5 Hz (theta band) → ER ≈ 0
  - Burst: amplitude-10 sine at 40 Hz added from ``onset`` seconds → ER >> 0

With the default threshold of 30 the Page-Hinkley alarm fires within one or two
seconds of the burst onset, making behaviour straightforward to assert.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FS = 256.0
_N_SEC = 30


def _make_data(arr: np.ndarray, fs: float = _FS, **kwargs: object) -> cb.SignalData:
    """Wrap a (time, space) numpy array in a Data object."""
    return cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=fs, **kwargs)


def _gamma_onset_signal(
    n_seconds: float = _N_SEC,
    fs: float = _FS,
    onset: float = 10.0,
    gamma_hz: float = 40.0,
    gamma_amp: float = 10.0,
) -> np.ndarray:
    """1-D signal: theta background everywhere, gamma burst from ``onset`` seconds."""
    t = np.arange(int(n_seconds * fs)) / fs
    sig = np.sin(2 * np.pi * 5.0 * t)  # theta background
    mask = t >= onset
    sig[mask] += gamma_amp * np.sin(2 * np.pi * gamma_hz * t[mask])
    return sig.astype(np.float64)


def _two_channel_data(onset_ch0: float = 10.0, onset_ch1: float = 20.0) -> cb.SignalData:
    """Two-channel Data: channel 0 fires at onset_ch0 s, channel 1 at onset_ch1 s."""
    sig0 = _gamma_onset_signal(onset=onset_ch0)
    sig1 = _gamma_onset_signal(onset=onset_ch1)
    arr = np.stack([sig0, sig1], axis=1)  # (time, space)
    return _make_data(arr, subjectID="sub-test")


# ---------------------------------------------------------------------------
# Output shape, dims and coordinates
# ---------------------------------------------------------------------------


def test_epileptogenicity_index_dims() -> None:
    """Output has exactly (space,) dimensions."""
    out = cb.feature.EpileptogenicityIndex().apply(_two_channel_data())
    assert set(out.data.dims) == {"space"}


def test_epileptogenicity_index_output_shape() -> None:
    """Output shape is (n_channels,)."""
    out = cb.feature.EpileptogenicityIndex().apply(_two_channel_data())
    assert out.data.shape == (2,)


def test_epileptogenicity_index_space_coords_preserved() -> None:
    """Space coordinates from the input are carried through unchanged."""
    arr = np.stack([_gamma_onset_signal(onset=10.0)] * 3, axis=1)
    space_vals = np.array(["ch0", "ch1", "ch2"])
    xr_da = xr.DataArray(
        arr,
        dims=["time", "space"],
        coords={"time": np.arange(arr.shape[0]) / _FS, "space": space_vals},
    )
    data = cb.data.SignalData.from_xarray(xr_da)
    out = cb.feature.EpileptogenicityIndex().apply(data)
    np.testing.assert_array_equal(out.data.coords["space"].values, space_vals)


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------


def test_epileptogenicity_index_values_in_unit_interval() -> None:
    """All EI values must be in [0, 1]."""
    out = cb.feature.EpileptogenicityIndex().apply(_two_channel_data())
    vals = out.data.values
    assert np.all(vals >= 0.0)
    assert np.all(vals <= 1.0)


def test_epileptogenicity_index_max_is_one() -> None:
    """After normalisation the maximum EI value must equal 1."""
    out = cb.feature.EpileptogenicityIndex().apply(_two_channel_data())
    assert np.isclose(out.data.values.max(), 1.0)


def test_epileptogenicity_index_early_channel_scores_higher() -> None:
    """Channel firing earlier must have higher EI than a channel firing later."""
    out = cb.feature.EpileptogenicityIndex().apply(
        _two_channel_data(onset_ch0=10.0, onset_ch1=20.0)
    )
    ei = out.data.values  # (space,)
    # Channel 0 fires 10 s earlier → should have higher EI
    assert ei[0] > ei[1], f"Expected ei[0]={ei[0]:.4f} > ei[1]={ei[1]:.4f}"


def test_epileptogenicity_index_no_discharge_channel_near_zero() -> None:
    """Channel with no rapid discharge should receive near-zero EI."""
    # Channel 0: gamma burst; channel 1: pure theta (no discharge)
    sig_burst = _gamma_onset_signal(onset=10.0)
    sig_theta = np.sin(2 * np.pi * 5.0 * np.arange(int(_N_SEC * _FS)) / _FS)
    arr = np.stack([sig_burst, sig_theta], axis=1)
    data = _make_data(arr)

    out = cb.feature.EpileptogenicityIndex().apply(data)
    ei = out.data.values  # (space,)

    assert ei[0] > 0.5, f"Burst channel EI should be high, got {ei[0]:.4f}"
    assert ei[1] < 0.1, f"Theta-only channel EI should be near zero, got {ei[1]:.4f}"


def test_epileptogenicity_index_flat_signal_all_zero() -> None:
    """Constant signal has no rapid discharge; all EI values should be zero."""
    arr = np.ones((int(_N_SEC * _FS), 2))
    data = _make_data(arr)

    out = cb.feature.EpileptogenicityIndex().apply(data)
    assert np.allclose(out.data.values, 0.0)


def test_epileptogenicity_index_three_channels_ordering() -> None:
    """With three channels firing at t=10, 15, 20 s, EI order should be ch0>ch1>ch2."""
    sigs = [_gamma_onset_signal(onset=t) for t in (10.0, 15.0, 20.0)]
    arr = np.stack(sigs, axis=1)
    data = _make_data(arr)

    out = cb.feature.EpileptogenicityIndex().apply(data)
    ei = out.data.values
    assert ei[0] > ei[1] > ei[2], f"Expected decreasing EI, got {ei}"


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_epileptogenicity_index_history_appended() -> None:
    """'EpileptogenicityIndex' must appear as the last entry in history."""
    out = cb.feature.EpileptogenicityIndex().apply(_two_channel_data())
    assert out.history[-1] == "EpileptogenicityIndex"


def test_epileptogenicity_index_metadata_preserved() -> None:
    """subjectID, groupID, condition are preserved; sampling_rate not preserved
    for Data without time."""
    arr = np.stack([_gamma_onset_signal(onset=10.0)] * 2, axis=1)
    data = _make_data(arr, subjectID="sub-99", groupID="patients", condition="seizure")
    out = cb.feature.EpileptogenicityIndex().apply(data)

    assert out.subjectID == "sub-99"
    assert out.groupID == "patients"
    assert out.condition == "seizure"
    # sampling_rate is not preserved for Data without time dimension
    assert out.sampling_rate is None


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


def test_epileptogenicity_index_window_duration_accepted_and_shape_unchanged() -> None:
    """Different window_duration values should not crash and must keep output shape."""
    data = _two_channel_data()
    for wd in (0.25, 0.5, 1.0, 2.0):
        out = cb.feature.EpileptogenicityIndex(window_duration=wd).apply(data)
        assert out.data.shape == (2,), f"Unexpected shape for window_duration={wd}"
        assert np.all(out.data.values >= 0.0)
        assert np.all(out.data.values <= 1.0)


def test_epileptogenicity_index_very_high_threshold_suppresses_detection() -> None:
    """With threshold=1e9 no channel fires; all EI values should be zero."""
    data = _two_channel_data()
    out = cb.feature.EpileptogenicityIndex(threshold=1e9).apply(data)
    # No detection → all N_di = last sample → EI numerator from background ≈ 0
    assert np.all(out.data.values >= 0.0)
    assert np.all(out.data.values <= 1.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_epileptogenicity_index_raises_without_time_dim() -> None:
    """ValueError raised when data lacks the 'time' dimension."""
    from cobrabox.features.frequency_domain.epileptogenicity_index import EpileptogenicityIndex

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((3, 2)), dims=["foo", "space"])

        @property
        def sampling_rate(self) -> float:
            return 256.0

    with pytest.raises(ValueError, match="exactly 'time' and 'space'"):
        EpileptogenicityIndex()(_FakeData())  # type: ignore[arg-type]


def test_epileptogenicity_index_raises_without_sampling_rate() -> None:
    """ValueError raised when sampling_rate is not set."""
    from cobrabox.features.frequency_domain.epileptogenicity_index import EpileptogenicityIndex

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((2, 10)), dims=["space", "time"])

        @property
        def sampling_rate(self) -> None:
            return None

    with pytest.raises(ValueError, match="sampling_rate must be set"):
        EpileptogenicityIndex()(_FakeData())  # type: ignore[arg-type]


def test_epileptogenicity_index_raises_with_extra_dims() -> None:
    """ValueError raised when data has dimensions beyond 'time' and 'space'."""
    from cobrabox.features.frequency_domain.epileptogenicity_index import EpileptogenicityIndex

    class _FakeData:
        @property
        def data(self) -> xr.DataArray:
            return xr.DataArray(np.ones((2, 3, 10)), dims=["window_index", "space", "time"])

        @property
        def sampling_rate(self) -> float:
            return 256.0

    with pytest.raises(ValueError, match="exactly 'time' and 'space'"):
        EpileptogenicityIndex()(_FakeData())  # type: ignore[arg-type]


def test_epileptogenicity_index_raises_when_signal_shorter_than_window() -> None:
    """ValueError raised when the signal is shorter than one ER window."""
    arr = np.ones((10, 2))  # 10 samples at 256 Hz = 0.039 s  <  default 1.0 s window
    data = _make_data(arr)

    with pytest.raises(ValueError, match="shorter than window"):
        cb.feature.EpileptogenicityIndex().apply(data)


def test_epileptogenicity_index_returns_data_instance() -> None:
    """EpileptogenicityIndex.apply() returns a Data instance."""
    data = _two_channel_data()
    result = cb.feature.EpileptogenicityIndex().apply(data)
    assert isinstance(result, cb.Data)


def test_epileptogenicity_index_does_not_mutate_input() -> None:
    """EpileptogenicityIndex.apply() leaves input Data unchanged."""
    data = _two_channel_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.EpileptogenicityIndex().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
