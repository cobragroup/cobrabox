"""Tests for the PartialDirectedCoherence feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    arr: np.ndarray, *, sampling_rate: float = 250.0, space: list[str] | None = None
) -> cb.SignalData:
    """Create SignalData from a (space, time) 2-D NumPy array."""
    coords: dict = {}
    if space is not None:
        coords["space"] = space
    xr_arr = xr.DataArray(arr, dims=["space", "time"], coords=coords)
    return cb.SignalData.from_xarray(xr_arr, sampling_rate=sampling_rate)


def _coupled_signals(
    n_times: int, sr: float, drive_freq: float, *, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Return (2, n_times) array where channel 0 drives channel 1 at drive_freq Hz."""
    if rng is None:
        rng = np.random.default_rng(42)
    t = np.arange(n_times) / sr
    driver = np.sin(2 * np.pi * drive_freq * t) + 0.1 * rng.standard_normal(n_times)
    driven = np.roll(driver, 5) + 0.1 * rng.standard_normal(n_times)  # lagged copy
    return np.stack([driver, driven])


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------


def test_pdc_is_registered() -> None:
    """PartialDirectedCoherence is auto-discovered by the feature module."""
    assert hasattr(cb.feature, "PartialDirectedCoherence")


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_pdc_output_dims() -> None:
    """PDC returns Data with dims (space_to, space_from, frequency)."""
    rng = np.random.default_rng(0)
    data = _make_data(rng.standard_normal((4, 500)))
    out = cb.feature.PartialDirectedCoherence().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space_to", "space_from", "frequency")


def test_pdc_output_shape() -> None:
    """PDC output shape is (n_ch, n_ch, n_freqs)."""
    rng = np.random.default_rng(1)
    n_ch, n_freqs = 3, 64
    data = _make_data(rng.standard_normal((n_ch, 400)))
    out = cb.feature.PartialDirectedCoherence(n_freqs=n_freqs).apply(data)

    assert out.data.shape == (n_ch, n_ch, n_freqs)


def test_pdc_space_coords_preserved() -> None:
    """Space coordinates from the input are propagated to both space_to and space_from."""
    labels = ["Fz", "Cz", "Pz"]
    rng = np.random.default_rng(2)
    data = _make_data(rng.standard_normal((3, 400)), space=labels)
    out = cb.feature.PartialDirectedCoherence().apply(data)

    np.testing.assert_array_equal(out.data.coords["space_to"].values, labels)
    np.testing.assert_array_equal(out.data.coords["space_from"].values, labels)


def test_pdc_frequency_coord_range() -> None:
    """Frequency coordinates span [0, sr/2]."""
    sr = 250.0
    rng = np.random.default_rng(3)
    data = _make_data(rng.standard_normal((2, 500)), sampling_rate=sr)
    out = cb.feature.PartialDirectedCoherence().apply(data)

    freqs = out.data.coords["frequency"].values
    assert freqs[0] == pytest.approx(0.0)
    assert freqs[-1] == pytest.approx(sr / 2.0)


def test_pdc_returns_data_not_signal_data() -> None:
    """output_type=Data: result is Data, not SignalData (no time dim)."""
    rng = np.random.default_rng(4)
    data = _make_data(rng.standard_normal((3, 300)))
    out = cb.feature.PartialDirectedCoherence().apply(data)

    assert type(out) is cb.Data
    assert "time" not in out.data.dims


def test_pdc_history_updated() -> None:
    """apply() appends 'PartialDirectedCoherence' to history."""
    rng = np.random.default_rng(5)
    data = _make_data(rng.standard_normal((2, 300)))
    out = cb.feature.PartialDirectedCoherence().apply(data)

    assert out.history[-1] == "PartialDirectedCoherence"


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_pdc_values_in_unit_interval() -> None:
    """All PDC values lie in [0, 1]."""
    rng = np.random.default_rng(6)
    data = _make_data(rng.standard_normal((4, 600)))
    out = cb.feature.PartialDirectedCoherence().apply(data)

    vals = out.data.values
    assert np.all(vals >= 0.0 - 1e-10)
    assert np.all(vals <= 1.0 + 1e-10)


def test_pdc_column_sums_to_one() -> None:
    """At each frequency, the squared PDC values sum to 1 over the sink dimension."""
    rng = np.random.default_rng(7)
    data = _make_data(rng.standard_normal((3, 600)))
    out = cb.feature.PartialDirectedCoherence().apply(data)

    # pdc[space_to, space_from, frequency]; sum over 'space_to' at each (space_from, freq) ≈ 1
    vals = out.data.values  # (K, K, n_freqs)
    col_sums_sq = (vals**2).sum(axis=0)  # (K, n_freqs)
    np.testing.assert_allclose(col_sums_sq, 1.0, atol=1e-6)


def test_pdc_directed_coupling_detected() -> None:
    """PDC detects the direction of a planted coupling at the driving frequency."""
    sr = 250.0
    drive_freq = 40.0
    arr = _coupled_signals(n_times=2000, sr=sr, drive_freq=drive_freq)
    data = _make_data(arr, sampling_rate=sr)

    out = cb.feature.PartialDirectedCoherence(var_order=5, n_freqs=256).apply(data)
    freqs = out.data.coords["frequency"].values

    # Find index closest to the drive frequency
    f_idx = np.argmin(np.abs(freqs - drive_freq))
    # channel 0 → channel 1: pdc[space_to=1, space_from=0, f]
    pdc_01 = out.data.values[1, 0, f_idx]
    # channel 1 → channel 0: pdc[space_to=0, space_from=1, f]
    pdc_10 = out.data.values[0, 1, f_idx]

    assert pdc_01 > pdc_10, (
        f"Expected PDC(0→1)={pdc_01:.3f} > PDC(1→0)={pdc_10:.3f} at {drive_freq} Hz"
    )


def test_pdc_fixed_var_order() -> None:
    """var_order parameter is respected (no crash, correct output dims)."""
    rng = np.random.default_rng(8)
    data = _make_data(rng.standard_normal((3, 400)))
    out = cb.feature.PartialDirectedCoherence(var_order=3).apply(data)

    assert out.data.dims == ("space_to", "space_from", "frequency")


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


def test_pdc_requires_sampling_rate() -> None:
    """Raises ValueError when data.sampling_rate is None."""
    arr = xr.DataArray(np.random.default_rng(9).standard_normal((3, 200)), dims=["space", "time"])
    data = cb.SignalData.from_xarray(arr, sampling_rate=None)

    with pytest.raises(ValueError, match="sampling_rate"):
        cb.feature.PartialDirectedCoherence().apply(data)


def test_pdc_requires_at_least_2_channels() -> None:
    """Raises ValueError for single-channel input."""
    rng = np.random.default_rng(10)
    data = _make_data(rng.standard_normal((1, 300)))

    with pytest.raises(ValueError, match="2 channels"):
        cb.feature.PartialDirectedCoherence().apply(data)


def test_pdc_invalid_n_freqs_raises() -> None:
    """n_freqs < 1 raises ValueError at construction."""
    with pytest.raises(ValueError, match="n_freqs"):
        cb.feature.PartialDirectedCoherence(n_freqs=0)


def test_pdc_invalid_var_order_raises() -> None:
    """var_order < 1 raises ValueError at construction."""
    with pytest.raises(ValueError, match="var_order"):
        cb.feature.PartialDirectedCoherence(var_order=0)


def test_pdc_metadata_preserved() -> None:
    """PDC preserves subjectID, groupID, condition; sampling_rate is None (no time dim)."""
    rng = np.random.default_rng(11)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((400, 3)),
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )

    out = cb.feature.PartialDirectedCoherence().apply(data)

    assert out.subjectID == "s42"
    assert out.groupID == "control"
    assert out.condition == "task"
    assert out.sampling_rate is None  # output_type=Data removes time dim


def test_pdc_does_not_mutate_input() -> None:
    """PDC.apply() leaves the input Data object unchanged."""
    rng = np.random.default_rng(12)
    arr = rng.standard_normal((3, 400))
    data = _make_data(arr, space=["Fz", "Cz", "Pz"])

    original_history = list(data.history)
    original_shape = data.data.shape
    original_data_array = data.data.values.copy()

    _ = cb.feature.PartialDirectedCoherence().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.data.values, original_data_array)


def test_pdc_invalid_ndim_raises() -> None:
    """PDC raises ValueError for non-2-D input (3-D array)."""
    rng = np.random.default_rng(13)
    # Create 3-D data (space x time x extra)
    arr = rng.standard_normal((3, 100, 2))
    xr_arr = xr.DataArray(arr, dims=["space", "time", "trial"])
    data = cb.SignalData.from_xarray(xr_arr, sampling_rate=250.0)

    with pytest.raises(ValueError, match="2-D input"):
        cb.feature.PartialDirectedCoherence().apply(data)
