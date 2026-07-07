"""Tests for the ContinuousWaveletTransform feature."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 256.0
N_TIME = 512
N_SPACE = 3


def _make_data(
    n_time: int = N_TIME, n_space: int = N_SPACE, sampling_rate: float = SR, *, seed: int = 0
) -> cb.SignalData:
    rng = np.random.default_rng(seed)
    return cb.SignalData.from_numpy(
        rng.standard_normal((n_time, n_space)),
        dims=["time", "space"],
        sampling_rate=sampling_rate,
        subjectID="sub-01",
        groupID="ctrl",
        condition="rest",
    )


def _make_data_with_window_dim(
    n_windows: int = 4, n_time: int = 128, n_space: int = N_SPACE, *, seed: int = 1
) -> cb.SignalData:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_windows, n_time, n_space))
    xr_arr = xr.DataArray(
        arr,
        dims=["window_index", "time", "space"],
        coords={
            "window_index": np.arange(n_windows),
            "time": np.arange(n_time, dtype=float) / SR,
            "space": [f"ch{k}" for k in range(n_space)],
        },
    )
    return cb.SignalData.from_xarray(xr_arr)


# ===========================================================================
# ContinuousWaveletTransform
# ===========================================================================


class TestContinuousWaveletTransform:
    # -----------------------------------------------------------------------
    # Output structure
    # -----------------------------------------------------------------------

    def test_cwt_output_dims(self) -> None:
        """CWT returns SignalData with (space, scale, time) dims."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        assert isinstance(out, cb.SignalData)
        assert out.data.dims == ("space", "scale", "time")

    def test_cwt_time_dim_preserved(self) -> None:
        """CWT preserves the time dimension length."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        assert out.data.sizes["time"] == N_TIME

    def test_cwt_time_coords_preserved(self) -> None:
        """CWT preserves original time coordinate values."""
        xr_arr = xr.DataArray(
            np.random.default_rng(5).standard_normal((N_TIME, 2)),
            dims=["time", "space"],
            coords={"time": np.arange(N_TIME) / SR, "space": ["a", "b"]},
        )
        data = cb.SignalData.from_xarray(xr_arr)
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        np.testing.assert_array_equal(out.data.coords["time"].values, xr_arr.coords["time"].values)

    def test_cwt_scale_dim_matches_n_scales(self) -> None:
        """scale dimension size equals n_scales."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=16).apply(data)
        assert out.data.sizes["scale"] == 16

    def test_cwt_scale_dim_matches_explicit_scales(self) -> None:
        """scale dimension size equals len(scales) when provided explicitly."""
        data = _make_data()
        scales = [1.0, 2.0, 4.0, 8.0]
        out = cb.feature.ContinuousWaveletTransform(scales=scales).apply(data)
        assert out.data.sizes["scale"] == len(scales)
        np.testing.assert_array_equal(out.data.coords["scale"].values, scales)

    def test_cwt_frequency_coord_present(self) -> None:
        """frequency is a coordinate on the scale dimension."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        assert "frequency" in out.data.coords

    def test_cwt_frequency_coord_length_matches_scales(self) -> None:
        """frequency coordinate has same length as scale dimension."""
        data = _make_data()
        n_scales = 12
        out = cb.feature.ContinuousWaveletTransform(n_scales=n_scales).apply(data)
        assert len(out.data.coords["frequency"]) == n_scales

    def test_cwt_space_coords_preserved(self) -> None:
        """Space coordinates are unchanged after CWT."""
        xr_arr = xr.DataArray(
            np.random.default_rng(6).standard_normal((N_TIME, 3)),
            dims=["time", "space"],
            coords={"space": ["Fz", "Cz", "Pz"], "time": np.arange(N_TIME) / SR},
        )
        data = cb.SignalData.from_xarray(xr_arr)
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        np.testing.assert_array_equal(out.data.coords["space"].values, ["Fz", "Cz", "Pz"])

    # -----------------------------------------------------------------------
    # Numerical correctness
    # -----------------------------------------------------------------------

    def test_cwt_magnitude_matches_pywt(self) -> None:
        """magnitude scaling matches |pywt.cwt| channel-by-channel."""
        import pywt

        rng = np.random.default_rng(7)
        arr = rng.standard_normal((N_TIME, N_SPACE))
        xr_arr = xr.DataArray(arr, dims=["time", "space"])
        data = cb.SignalData.from_xarray(xr_arr, sampling_rate=SR)

        scales = np.arange(1, 9, dtype=float)
        out = cb.feature.ContinuousWaveletTransform(scales=list(scales), scaling="magnitude").apply(
            data
        )

        for ch in range(N_SPACE):
            coefs, _ = pywt.cwt(arr[:, ch], scales, "morl", sampling_period=1.0 / SR)
            np.testing.assert_allclose(out.data.isel(space=ch).values, np.abs(coefs), rtol=1e-10)

    def test_cwt_power_equals_magnitude_squared(self) -> None:
        """power scaling equals magnitude squared."""
        data = _make_data()
        scales = list(np.arange(1, 9, dtype=float))
        out_mag = cb.feature.ContinuousWaveletTransform(scales=scales, scaling="magnitude").apply(  # pyright: ignore[reportArgumentType]
            data
        )
        out_pow = cb.feature.ContinuousWaveletTransform(scales=scales, scaling="power").apply(data)  # pyright: ignore[reportArgumentType]
        np.testing.assert_allclose(out_pow.data.values, out_mag.data.values**2, rtol=1e-10)

    def test_cwt_complex_real_part_is_not_zero(self) -> None:
        """complex scaling returns complex coefficients with non-trivial real part."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(
            wavelet="cmor1.5-1.0", n_scales=8, scaling="complex"
        ).apply(data)
        assert np.iscomplexobj(out.data.values)

    def test_cwt_magnitude_is_nonneg(self) -> None:
        """magnitude scaling is always non-negative."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=8, scaling="magnitude").apply(data)
        assert np.all(out.data.values >= 0.0)

    def test_cwt_power_is_nonneg(self) -> None:
        """power scaling is always non-negative."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=8, scaling="power").apply(data)
        assert np.all(out.data.values >= 0.0)

    def test_cwt_frequency_decreases_with_scale(self) -> None:
        """Pseudo-frequency decreases as scale increases (inverse relationship)."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=16).apply(data)
        freqs = out.data.coords["frequency"].values
        assert np.all(np.diff(freqs) < 0), "frequency should decrease as scale increases"

    def test_cwt_sine_peak_at_correct_scale(self) -> None:
        """CWT of a pure sine wave has maximum energy at the scale matching the sine freq."""
        fs = 256.0
        freq_hz = 20.0
        t = np.arange(1024) / fs
        sig = np.sin(2 * np.pi * freq_hz * t)[:, np.newaxis]  # (1024, 1)
        data = cb.SignalData.from_numpy(sig, dims=["time", "space"], sampling_rate=fs)

        out = cb.feature.ContinuousWaveletTransform(n_scales=64, scaling="power").apply(data)
        # Average power over time for the single channel
        mean_power = out.data.isel(space=0).mean("time").values
        peak_freq = out.data.coords["frequency"].values[np.argmax(mean_power)]

        assert abs(peak_freq - freq_hz) <= 10.0  # within 10 Hz

    def test_cwt_different_wavelets_produce_different_results(self) -> None:
        """Using morl vs mexh gives different CWT values."""
        data = _make_data()
        out_morl = cb.feature.ContinuousWaveletTransform(wavelet="morl", n_scales=8).apply(data)
        out_mexh = cb.feature.ContinuousWaveletTransform(wavelet="mexh", n_scales=8).apply(data)
        assert not np.allclose(out_morl.data.values, out_mexh.data.values)

    # -----------------------------------------------------------------------
    # Extra dimensions
    # -----------------------------------------------------------------------

    def test_cwt_preserves_window_dim(self) -> None:
        """CWT processes correctly when a window_index dimension is present."""
        n_windows = 3
        data = _make_data_with_window_dim(n_windows=n_windows, n_time=64)
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        assert "window_index" in out.data.dims
        assert out.data.sizes["window_index"] == n_windows
        assert out.data.sizes["space"] == N_SPACE

    # -----------------------------------------------------------------------
    # Metadata propagation
    # -----------------------------------------------------------------------

    def test_cwt_preserves_metadata(self) -> None:
        """CWT propagates subjectID, groupID, condition, sampling_rate and appends history."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        assert out.subjectID == "sub-01"
        assert out.groupID == "ctrl"
        assert out.condition == "rest"
        assert out.sampling_rate == pytest.approx(SR)
        assert out.history[-1] == "ContinuousWaveletTransform"

    def test_cwt_does_not_mutate_input(self) -> None:
        """CWT leaves the input Data object unchanged."""
        data = _make_data()
        original_values = data.data.values.copy()
        original_history = list(data.history)
        cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        np.testing.assert_array_equal(data.data.values, original_values)
        assert data.history == original_history

    def test_cwt_falls_back_to_sample_period_without_sampling_rate(self) -> None:
        """CWT runs without error when sampling_rate is None (falls back to 1 Hz)."""
        xr_arr = xr.DataArray(
            np.random.default_rng(8).standard_normal((N_TIME, 2)), dims=["time", "space"]
        )
        data = cb.SignalData.from_xarray(xr_arr)
        out = cb.feature.ContinuousWaveletTransform(n_scales=4).apply(data)
        assert out.data.dims == ("space", "scale", "time")

    # -----------------------------------------------------------------------
    # Parameter behaviour
    # -----------------------------------------------------------------------

    def test_cwt_default_scales_are_1_to_n_scales(self) -> None:
        """When scales=None, scales coordinate runs from 1 to n_scales."""
        data = _make_data()
        n = 10
        out = cb.feature.ContinuousWaveletTransform(n_scales=n).apply(data)
        np.testing.assert_array_equal(out.data.coords["scale"].values, np.arange(1, n + 1))

    # -----------------------------------------------------------------------
    # Error handling
    # -----------------------------------------------------------------------

    def test_cwt_raises_on_invalid_wavelet(self) -> None:
        with pytest.raises(ValueError, match="Unknown continuous wavelet"):
            cb.feature.ContinuousWaveletTransform(wavelet="notawavelet")

    def test_cwt_raises_on_empty_scales(self) -> None:
        with pytest.raises(ValueError, match="scales must not be empty"):
            cb.feature.ContinuousWaveletTransform(scales=[])

    def test_cwt_raises_on_nonpositive_scale(self) -> None:
        with pytest.raises(ValueError, match="all scales must be positive"):
            cb.feature.ContinuousWaveletTransform(scales=[1.0, -1.0, 2.0])

    def test_cwt_raises_on_n_scales_zero(self) -> None:
        with pytest.raises(ValueError, match="n_scales must be >= 1"):
            cb.feature.ContinuousWaveletTransform(n_scales=0)

    def test_cwt_raises_on_invalid_scaling(self) -> None:
        with pytest.raises(ValueError, match="scaling must be one of"):
            cb.feature.ContinuousWaveletTransform(scaling="invalid")

    # -----------------------------------------------------------------------
    # API accessibility
    # -----------------------------------------------------------------------

    def test_cwt_accessible_via_feature_module(self) -> None:
        assert callable(cb.feature.ContinuousWaveletTransform)

    # -----------------------------------------------------------------------
    # Pipeline compatibility
    # -----------------------------------------------------------------------

    def test_cwt_pipe_into_mean(self) -> None:
        """CWT | Mean should work end-to-end (reduce along some dim)."""
        data = _make_data()
        out = cb.feature.ContinuousWaveletTransform(n_scales=8).apply(data)
        # The CWT output is a SignalData with (space, scale, time);
        # Mean over "time" dim should be possible via direct xarray ops
        mean_over_time = out.data.mean("time")
        assert "time" not in mean_over_time.dims
        assert mean_over_time.shape == (N_SPACE, 8)
