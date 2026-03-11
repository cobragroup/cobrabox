"""Tests for the EMD (Empirical Mode Decomposition) feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb


def _make_data(
    n_time: int = 500, n_space: int = 3, sampling_rate: float = 100.0, subject: str = "sub-01"
) -> cb.SignalData:
    """Create a simple SignalData object with noise for testing."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((n_time, n_space))
    return cb.data.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sampling_rate, subjectID=subject
    )


def _make_sine_data(
    freqs_hz: list[float], sampling_rate: float = 100.0, duration: float = 5.0
) -> cb.SignalData:
    """Create a SignalData object whose signal is a sum of pure sinusoids."""
    t = np.arange(int(sampling_rate * duration)) / sampling_rate
    sig = np.zeros_like(t)
    for f in freqs_hz:
        sig += np.sin(2 * np.pi * f * t)
    return cb.data.SignalData.from_numpy(
        sig[:, None], dims=["time", "space"], sampling_rate=sampling_rate
    )


# ---------------------------------------------------------------------------
# Basic API and shape tests
# ---------------------------------------------------------------------------


def test_emd_history_updated() -> None:
    """EMD appends 'EMD' to history."""
    data = _make_data()
    result = cb.feature.EMD().apply(data)
    assert result.history[-1] == "EMD"


def test_emd_metadata_preserved() -> None:
    """EMD preserves subjectID, groupID, condition, and sampling_rate."""
    rng = np.random.default_rng(42)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((100, 3)),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="s42",
        groupID="control",
        condition="task",
    )
    result = cb.feature.EMD(max_imfs=3).apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "task"
    assert result.sampling_rate == pytest.approx(100.0)


def test_emd_returns_data_instance() -> None:
    """EMD.apply() always returns a Data instance."""
    data = _make_data()
    result = cb.feature.EMD(max_imfs=3).apply(data)
    assert isinstance(result, cb.Data)


def test_emd_does_not_mutate_input() -> None:
    """EMD does not modify the input Data object."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.EMD(max_imfs=3).apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)


def test_emd_adds_imf_dimension() -> None:
    """EMD adds an 'imf' dimension to the output."""
    data = _make_data()
    result = cb.feature.EMD(max_imfs=3).apply(data)
    assert "imf" in result.data.dims


def test_emd_imf_coords_labelled_correctly() -> None:
    """IMF coordinates are labelled imf0, imf1, ... with residual as last."""
    data = _make_data()
    result = cb.feature.EMD(max_imfs=3).apply(data)
    imf_coords = list(result.data.coords["imf"].values)

    # Should have imf0, imf1, ..., residual
    # The actual number depends on what EMD extracts
    assert imf_coords[-1] == "residual"
    assert all(c.startswith("imf") for c in imf_coords[:-1])


def test_emd_residual_always_included() -> None:
    """Residual is always included as the last IMF."""
    data = _make_data()
    result = cb.feature.EMD(max_imfs=3).apply(data)
    imf_coords = list(result.data.coords["imf"].values)
    assert imf_coords[-1] == "residual"
    assert all(c.startswith("imf") for c in imf_coords[:-1])


def test_emd_keep_orig_includes_original() -> None:
    """When keep_orig=True, original signal is included as first IMF."""
    data = _make_data()
    result = cb.feature.EMD(max_imfs=3, keep_orig=True).apply(data)
    imf_coords = list(result.data.coords["imf"].values)

    # Original should be first, residual should be last
    assert imf_coords[0] == "original"
    assert imf_coords[-1] == "residual"
    assert all(c.startswith("imf") for c in imf_coords[1:-1])


def test_emd_keep_orig_false_excludes_original() -> None:
    """When keep_orig=False (default), original signal is not included."""
    data = _make_data()
    result = cb.feature.EMD(max_imfs=3, keep_orig=False).apply(data)
    imf_coords = list(result.data.coords["imf"].values)
    assert "original" not in imf_coords


def test_emd_keep_orig_original_matches_input() -> None:
    """The 'original' IMF matches the input signal."""
    data = _make_data(n_space=1)
    result = cb.feature.EMD(max_imfs=3, keep_orig=True).apply(data)

    original_from_result = result.data.sel(imf="original").values
    original_from_input = data.data.values.squeeze()
    np.testing.assert_allclose(original_from_result.squeeze(), original_from_input)


def test_emd_preserves_time_coords() -> None:
    """Time coordinates survive the decomposition."""
    data = _make_data(sampling_rate=100.0)
    result = cb.feature.EMD(max_imfs=3).apply(data)
    np.testing.assert_array_equal(
        result.data.coords["time"].values, data.data.coords["time"].values
    )


def test_emd_preserves_space_coords() -> None:
    """Space coordinates are kept when the input has them."""
    arr = np.random.default_rng(0).standard_normal((200, 4))
    xr_da = xr.DataArray(
        arr,
        dims=["time", "space"],
        coords={"time": np.arange(200) / 100.0, "space": ["Fp1", "Fp2", "C3", "C4"]},
    )
    data = cb.data.SignalData.from_xarray(xr_da, subjectID="s1")
    result = cb.feature.EMD(max_imfs=3).apply(data)
    assert list(result.data.coords["space"].values) == ["Fp1", "Fp2", "C3", "C4"]


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_emd_output_not_all_zeros() -> None:
    """EMD output should contain non-zero values."""
    data = _make_data()
    result = cb.feature.EMD(max_imfs=3).apply(data)
    assert not np.allclose(result.to_numpy(), 0.0)


def test_emd_imfs_sum_to_original() -> None:
    """IMFs (including residual) should sum back to the original signal."""
    data = _make_data(n_space=1)
    result = cb.feature.EMD(max_imfs=5).apply(data)

    # Sum all IMFs along the imf dimension
    reconstructed = result.data.sum(dim="imf")
    original = data.data

    # The reconstruction should be close to the original
    np.testing.assert_allclose(reconstructed.values, original.values, rtol=1e-10, atol=1e-10)


def test_emd_extracts_oscillatory_components() -> None:
    """EMD should separate a mixed signal into oscillatory IMFs."""
    # Create a signal with two distinct frequencies
    data = _make_sine_data([5.0, 20.0], sampling_rate=100.0, duration=5.0)
    result = cb.feature.EMD(max_imfs=4).apply(data)

    # Should have extracted at least 2 IMFs
    n_imfs = result.data.sizes["imf"]
    assert n_imfs >= 2


# ---------------------------------------------------------------------------
# Different sift methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sift", "mask_sift", "iterated_mask_sift"])
def test_emd_different_methods_work(method: str) -> None:
    """Different sift methods should all produce valid output."""
    data = _make_data(n_time=200, n_space=1)
    result = cb.feature.EMD(max_imfs=3, method=method).apply(data)  # type: ignore[arg-type]
    assert "imf" in result.data.dims
    assert result.data.sizes["imf"] >= 1


# ---------------------------------------------------------------------------
# Multidimensional data
# ---------------------------------------------------------------------------


def test_emd_3d_data() -> None:
    """EMD handles 3D data (time, space, channel)."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((100, 3, 2))
    xr_da = xr.DataArray(arr, dims=["time", "space", "channel"])
    data = cb.data.SignalData.from_xarray(xr_da, subjectID="s1")
    result = cb.feature.EMD(max_imfs=3).apply(data)

    assert "imf" in result.data.dims
    assert "space" in result.data.dims
    assert "channel" in result.data.dims


def test_emd_1d_data() -> None:
    """EMD handles 1D time-only data."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(200)
    data = cb.data.SignalData.from_numpy(arr, dims=["time"], sampling_rate=100.0, subjectID="s1")
    result = cb.feature.EMD(max_imfs=3).apply(data)

    assert "imf" in result.data.dims
    assert "time" in result.data.dims


# ---------------------------------------------------------------------------
# n_imfs coordinate tracking
# ---------------------------------------------------------------------------


def test_emd_1d_data_n_imfs_in_attrs() -> None:
    """For 1D data, n_imfs is stored in attrs as a dict with 'signal' key."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(200)
    data = cb.data.SignalData.from_numpy(arr, dims=["time"], sampling_rate=100.0, subjectID="s1")
    result = cb.feature.EMD(max_imfs=3).apply(data)

    # n_imfs should be in attrs as a dict
    assert "n_imfs" in result.data.attrs
    n_imfs = result.data.attrs["n_imfs"]
    assert isinstance(n_imfs, dict)
    assert "signal" in n_imfs
    assert n_imfs["signal"] > 0

    # n_imfs should equal number of imf coords minus residual (and minus original if present)
    imf_coords = list(result.data.coords["imf"].values)
    expected_n_imfs = len([c for c in imf_coords if c.startswith("imf")])
    assert n_imfs["signal"] == expected_n_imfs


def test_emd_1d_data_n_imfs_correct_with_keep_orig() -> None:
    """n_imfs count is correct even when keep_orig=True."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(200)
    data = cb.data.SignalData.from_numpy(arr, dims=["time"], sampling_rate=100.0, subjectID="s1")
    result = cb.feature.EMD(max_imfs=3, keep_orig=True).apply(data)

    n_imfs = result.data.attrs["n_imfs"]["signal"]
    imf_coords = list(result.data.coords["imf"].values)
    # Should not count 'original' or 'residual'
    expected_n_imfs = len([c for c in imf_coords if c.startswith("imf")])
    assert n_imfs == expected_n_imfs


def test_emd_multidim_n_imfs_in_attrs() -> None:
    """For multi-dimensional data, n_imfs is stored in attrs as a dict."""
    data = _make_data(n_time=200, n_space=3)
    result = cb.feature.EMD(max_imfs=3).apply(data)

    # n_imfs should be in attrs as a dict
    assert "n_imfs" in result.data.attrs
    n_imfs = result.data.attrs["n_imfs"]
    assert isinstance(n_imfs, dict)

    # Should have 3 entries (one per space coordinate)
    assert len(n_imfs) == 3
    # All values should be positive integers
    assert all(v > 0 for v in n_imfs.values())


def test_emd_multidim_n_imfs_keys_match_coords() -> None:
    """n_imfs dict keys match the coordinate values."""
    arr = np.random.default_rng(0).standard_normal((200, 4))
    xr_da = xr.DataArray(
        arr,
        dims=["time", "space"],
        coords={"time": np.arange(200) / 100.0, "space": ["Fp1", "Fp2", "C3", "C4"]},
    )
    data = cb.data.SignalData.from_xarray(xr_da, subjectID="s1")
    result = cb.feature.EMD(max_imfs=3).apply(data)

    n_imfs = result.data.attrs["n_imfs"]
    # Keys should be the space coordinate values
    assert set(n_imfs.keys()) == {"Fp1", "Fp2", "C3", "C4"}


def test_emd_3d_data_n_imfs_dict() -> None:
    """n_imfs dict works with 3D data (time, space, channel)."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((100, 2, 2))
    xr_da = xr.DataArray(
        arr, dims=["time", "space", "channel"], coords={"space": ["A", "B"], "channel": ["X", "Y"]}
    )
    data = cb.data.SignalData.from_xarray(xr_da, subjectID="s1")
    result = cb.feature.EMD(max_imfs=3).apply(data)

    # n_imfs should be in attrs as a dict
    assert "n_imfs" in result.data.attrs
    n_imfs = result.data.attrs["n_imfs"]
    assert isinstance(n_imfs, dict)

    # Should have 4 entries (2 space x 2 channel), with "/" separator
    assert len(n_imfs) == 4
    assert set(n_imfs.keys()) == {"A/X", "A/Y", "B/X", "B/Y"}


# ---------------------------------------------------------------------------
# Chaining
# ---------------------------------------------------------------------------


def test_emd_output_is_valid_for_further_features() -> None:
    """The output can be fed into another feature (e.g. Mean)."""
    data = _make_data(n_time=200, n_space=2)
    result = cb.feature.EMD(max_imfs=3).apply(data)

    # Mean over imf dimension should collapse it
    reduced = cb.feature.Mean(dim="imf").apply(result)
    assert "imf" not in reduced.data.dims
    assert "EMD" in reduced.history
    assert "Mean" in reduced.history


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_emd_invalid_method_raises() -> None:
    """EMD raises ValueError for invalid method."""
    with pytest.raises(ValueError, match="Invalid method"):
        cb.feature.EMD(method="invalid_method")  # type: ignore[arg-type]


def test_emd_zero_max_imfs_raises() -> None:
    """EMD raises ValueError for max_imfs of 0."""
    with pytest.raises(ValueError, match="max_imfs must be positive"):
        cb.feature.EMD(max_imfs=0)


def test_emd_negative_max_imfs_raises() -> None:
    """EMD raises ValueError for negative max_imfs."""
    with pytest.raises(ValueError, match="max_imfs must be positive"):
        cb.feature.EMD(max_imfs=-1)
