"""Tests for the ReciprocalConnectivity feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal_data(
    arr: np.ndarray, *, sampling_rate: float = 250.0, space: list[str] | None = None
) -> cb.SignalData:
    """Create SignalData from a (space, time) 2-D NumPy array."""
    coords: dict = {}
    if space is not None:
        coords["space"] = space
    xr_arr = xr.DataArray(arr, dims=["space", "time"], coords=coords)
    return cb.SignalData.from_xarray(xr_arr, sampling_rate=sampling_rate)


def _make_pdc_matrix(
    vals: np.ndarray, *, space: list[str] | None = None, freqs: np.ndarray | None = None
) -> cb.Data:
    """Create a Data object from a pre-computed (space_to, space_from[, frequency]) matrix."""
    n_ch = vals.shape[0]
    space_vals = space if space is not None else list(range(n_ch))
    coords: dict = {"space_to": space_vals, "space_from": space_vals}
    if vals.ndim == 3 and freqs is not None:
        coords["frequency"] = freqs
        dims = ["space_to", "space_from", "frequency"]
    else:
        dims = ["space_to", "space_from"]
    xr_arr = xr.DataArray(vals, dims=dims, coords=coords)
    return cb.Data.from_xarray(xr_arr)


def _coupled_signals(n_times: int, sr: float, drive_freq: float = 40.0) -> np.ndarray:
    """Return (2, n_times) where channel 0 drives channel 1."""
    rng = np.random.default_rng(42)
    t = np.arange(n_times) / sr
    driver = np.sin(2 * np.pi * drive_freq * t) + 0.1 * rng.standard_normal(n_times)
    driven = np.roll(driver, 5) + 0.1 * rng.standard_normal(n_times)
    return np.stack([driver, driven])


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------


def test_reciprocal_connectivity_is_registered() -> None:
    """ReciprocalConnectivity is auto-discovered by the feature module."""
    assert hasattr(cb.feature, "ReciprocalConnectivity")


# ---------------------------------------------------------------------------
# Output structure — time-series path
# ---------------------------------------------------------------------------


def test_reciprocal_connectivity_from_timeseries_output_dims() -> None:
    """From time-series: output has dim ('space',) only."""
    rng = np.random.default_rng(0)
    data = _make_signal_data(rng.standard_normal((3, 800)))
    out = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)


def test_reciprocal_connectivity_from_timeseries_output_shape() -> None:
    """Output vector length equals number of channels."""
    rng = np.random.default_rng(1)
    n_ch = 4
    data = _make_signal_data(rng.standard_normal((n_ch, 800)))
    out = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)

    assert out.data.shape == (n_ch,)


def test_reciprocal_connectivity_from_timeseries_space_coords_preserved() -> None:
    """Space coordinates are carried through to the output."""
    labels = ["Fz", "Cz", "Pz"]
    rng = np.random.default_rng(2)
    data = _make_signal_data(rng.standard_normal((3, 800)), space=labels)
    out = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)

    np.testing.assert_array_equal(out.data.coords["space"].values, labels)


def test_reciprocal_connectivity_history_updated() -> None:
    """apply() appends 'ReciprocalConnectivity' to history."""
    rng = np.random.default_rng(3)
    data = _make_signal_data(rng.standard_normal((2, 800)))
    out = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)

    assert out.history[-1] == "ReciprocalConnectivity"


# ---------------------------------------------------------------------------
# Numerical correctness — time-series path
# ---------------------------------------------------------------------------


def test_reciprocal_connectivity_driver_is_negative_sink_is_positive() -> None:
    """Known driver (ch 0) has negative RC; known sink (ch 1) has positive RC."""
    sr = 250.0
    arr = _coupled_signals(n_times=3000, sr=sr, drive_freq=40.0)
    data = _make_signal_data(arr, sampling_rate=sr)

    out = cb.feature.ReciprocalConnectivity(freq_band=(30.0, 55.0), var_order=5).apply(data)
    rc = out.data.values  # (2,)

    assert rc[0] < 0, f"Driver ch0 expected negative RC, got {rc[0]:.4f}"
    assert rc[1] > 0, f"Sink  ch1 expected positive RC, got {rc[1]:.4f}"


def test_reciprocal_connectivity_normalize_changes_values_not_shape() -> None:
    """normalize=True produces different values but same shape."""
    # Use coupled signals so RC values are nonzero (not near 0 for all channels)
    arr = _coupled_signals(n_times=3000, sr=250.0, drive_freq=40.0)
    data = _make_signal_data(arr, sampling_rate=250.0)

    out_raw = cb.feature.ReciprocalConnectivity(
        freq_band=(30.0, 55.0), var_order=5, normalize=False
    ).apply(data)
    out_norm = cb.feature.ReciprocalConnectivity(
        freq_band=(30.0, 55.0), var_order=5, normalize=True
    ).apply(data)

    assert out_raw.data.shape == out_norm.data.shape
    assert not np.allclose(out_raw.data.values, out_norm.data.values)


# ---------------------------------------------------------------------------
# Pre-computed matrix path
# ---------------------------------------------------------------------------


def test_reciprocal_connectivity_from_precomputed_2d_matrix() -> None:
    """From a pre-computed asymmetric 2-D matrix (no freq dim)."""
    # mat[i, j] = flow from j → i  (space_to=i, space_from=j)
    # mat[0, 1] = 0.3: flow from ch1 → ch0
    # mat[1, 0] = 0.7: flow from ch0 → ch1
    # ch0: in=0.3, out=0.7 → RC = -0.4  (net source)
    # ch1: in=0.7, out=0.3 → RC = +0.4  (net sink)
    mat = np.array([[0.0, 0.3], [0.7, 0.0]])
    data = _make_pdc_matrix(mat)

    out = cb.feature.ReciprocalConnectivity(freq_band=None).apply(data)

    assert out.data.dims == ("space",)
    assert out.data.shape == (2,)
    np.testing.assert_allclose(out.data.values, [-0.4, 0.4], atol=1e-10)
    assert out.data.values[0] < 0, "ch0 sends more than it receives → should be negative"
    assert out.data.values[1] > 0, "ch1 receives more than it sends → should be positive"


def test_reciprocal_connectivity_from_precomputed_3d_matrix_with_freq_band() -> None:
    """From a pre-computed 3-D matrix with frequency averaging."""
    n_ch, n_freqs = 3, 64
    rng = np.random.default_rng(5)
    mat = rng.random((n_ch, n_ch, n_freqs))
    # Make it asymmetric by zeroing upper triangle
    for k in range(n_freqs):
        np.fill_diagonal(mat[:, :, k], 0.0)
    mat[0, 1, :] = 0.9  # strong flow from ch1 to ch0
    freqs = np.linspace(0.0, 125.0, n_freqs)
    data = _make_pdc_matrix(mat, freqs=freqs)

    out = cb.feature.ReciprocalConnectivity(freq_band=(30.0, 80.0)).apply(data)

    assert out.data.dims == ("space",)
    assert out.data.shape == (n_ch,)


def test_reciprocal_connectivity_precomputed_space_coords_preserved() -> None:
    """Space coords on pre-computed 2-D matrix are propagated to output."""
    labels = ["A", "B", "C"]
    rng = np.random.default_rng(6)
    mat = rng.random((3, 3))
    np.fill_diagonal(mat, 0.0)
    # Ensure it is asymmetric
    mat[0, 1] = 0.9
    mat[1, 0] = 0.1
    data = _make_pdc_matrix(mat, space=labels)

    out = cb.feature.ReciprocalConnectivity(freq_band=None).apply(data)

    np.testing.assert_array_equal(out.data.coords["space"].values, labels)


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


def test_reciprocal_connectivity_unsupported_connectivity_raises() -> None:
    """Unsupported connectivity measure raises ValueError for time-series input."""
    rng = np.random.default_rng(7)
    data = _make_signal_data(rng.standard_normal((2, 500)))

    with pytest.raises(ValueError, match="'dtf'"):
        cb.feature.ReciprocalConnectivity(connectivity="dtf", freq_band=(10.0, 60.0)).apply(data)


def test_reciprocal_connectivity_symmetric_matrix_raises() -> None:
    """A symmetric pre-computed matrix raises ValueError."""
    mat = np.array([[0.0, 0.5], [0.5, 0.0]])
    data = _make_pdc_matrix(mat)

    with pytest.raises(ValueError, match="symmetric"):
        cb.feature.ReciprocalConnectivity(freq_band=None).apply(data)


def test_reciprocal_connectivity_freq_band_set_but_no_freq_dim_raises() -> None:
    """Setting freq_band on a matrix without frequency dim raises ValueError."""
    mat = np.array([[0.0, 0.7], [0.3, 0.0]])
    data = _make_pdc_matrix(mat)

    with pytest.raises(ValueError, match="freq_band"):
        cb.feature.ReciprocalConnectivity(freq_band=(30.0, 80.0)).apply(data)


def test_reciprocal_connectivity_freq_band_outside_range_raises() -> None:
    """freq_band outside the available frequency range raises ValueError."""
    n_ch, n_freqs = 2, 64
    rng = np.random.default_rng(8)
    mat = rng.random((n_ch, n_ch, n_freqs))
    mat[0, 1, :] = 0.9  # asymmetric
    freqs = np.linspace(0.0, 60.0, n_freqs)  # max freq = 60 Hz
    data = _make_pdc_matrix(mat, freqs=freqs)

    with pytest.raises(ValueError, match="outside the available"):
        cb.feature.ReciprocalConnectivity(freq_band=(70.0, 100.0)).apply(data)


def test_reciprocal_connectivity_freq_band_none_but_freq_dim_present_raises() -> None:
    """freq_band=None with a frequency dimension raises ValueError."""
    n_ch, n_freqs = 2, 64
    rng = np.random.default_rng(9)
    mat = rng.random((n_ch, n_ch, n_freqs))
    mat[0, 1, :] = 0.9
    freqs = np.linspace(0.0, 125.0, n_freqs)
    data = _make_pdc_matrix(mat, freqs=freqs)

    with pytest.raises(ValueError, match=r"frequency.*dimension"):
        cb.feature.ReciprocalConnectivity(freq_band=None).apply(data)


def test_reciprocal_connectivity_invalid_freq_band_fmin_ge_fmax_raises() -> None:
    """freq_band where fmin >= fmax raises ValueError at construction."""
    with pytest.raises(ValueError, match="fmin < fmax"):
        cb.feature.ReciprocalConnectivity(freq_band=(80.0, 30.0))


def test_reciprocal_connectivity_precomputed_missing_space_dims_raises() -> None:
    """Pre-computed matrix without expected dims raises ValueError."""
    # A Data object with only a 'space' dim (not 'space_to'/'space_from') — no 'time' either
    xr_arr = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=["space"])
    data = cb.Data.from_xarray(xr_arr)

    with pytest.raises(ValueError, match="'space_from'"):
        cb.feature.ReciprocalConnectivity(freq_band=None).apply(data)


def test_reciprocal_connectivity_metadata_preserved() -> None:
    """ReciprocalConnectivity preserves subjectID, groupID, condition; sampling_rate is None."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((800, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=250.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )

    out = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)

    assert out.subjectID == "s42"
    assert out.groupID == "control"
    assert out.condition == "task"
    # output_type = Data removes time dimension, so sampling_rate is None
    assert out.sampling_rate is None


def test_reciprocal_connectivity_does_not_mutate_input() -> None:
    """ReciprocalConnectivity.apply() leaves the input Data object unchanged."""
    labels = ["Fz", "Cz"]
    rng = np.random.default_rng(43)
    arr = rng.standard_normal((2, 800))
    data = _make_signal_data(arr, space=labels)

    original_history = list(data.history)
    original_shape = data.data.shape
    original_data_array = data.data.values.copy()

    _ = cb.feature.ReciprocalConnectivity(freq_band=(10.0, 60.0)).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.data.values, original_data_array)


def test_reciprocal_connectivity_no_space_coords_fallback() -> None:
    """When space_to coords are missing, output uses integer indices."""
    mat = np.array([[0.0, 0.3], [0.7, 0.0]])
    # Create Data without space_to/space_from coordinate values
    xr_arr = xr.DataArray(mat, dims=["space_to", "space_from"])
    data = cb.Data.from_xarray(xr_arr)

    out = cb.feature.ReciprocalConnectivity(freq_band=None).apply(data)

    assert out.data.dims == ("space",)
    # Should use integer indices 0, 1 as fallback
    np.testing.assert_array_equal(out.data.coords["space"].values, [0, 1])
