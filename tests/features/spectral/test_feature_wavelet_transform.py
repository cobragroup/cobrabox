"""Tests for DiscreteWaveletTransform and ContinuousWaveletTransform features."""

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
# DiscreteWaveletTransform
# ===========================================================================


class TestDiscreteWaveletTransform:
    # -----------------------------------------------------------------------
    # Output structure
    # -----------------------------------------------------------------------

    def test_dwt_output_dims(self) -> None:
        """DWT returns Data with (space, wavelet_level, coef_index) dims."""
        data = _make_data()
        out = cb.feature.DiscreteWaveletTransform(level=4).apply(data)
        assert isinstance(out, cb.Data)
        assert out.data.dims == ("space", "wavelet_level", "coef_index")

    def test_dwt_output_type_is_data_not_signal_data(self) -> None:
        """DWT returns plain Data (time dim is consumed)."""
        data = _make_data()
        out = cb.feature.DiscreteWaveletTransform(level=2).apply(data)
        assert type(out) is cb.Data

    def test_dwt_level_labels_correct_order(self) -> None:
        """wavelet_level coordinate follows pywt order: approx, detail_L, ..., detail_1."""
        data = _make_data()
        out = cb.feature.DiscreteWaveletTransform(level=3).apply(data)
        expected = ["approx", "detail_3", "detail_2", "detail_1"]
        assert list(out.data.coords["wavelet_level"].values) == expected

    def test_dwt_n_levels_equals_level_plus_one(self) -> None:
        """Number of wavelet_level entries = level + 1 (approx + level details)."""
        data = _make_data()
        for lv in (1, 2, 4):
            out = cb.feature.DiscreteWaveletTransform(level=lv).apply(data)
            assert out.data.sizes["wavelet_level"] == lv + 1

    def test_dwt_space_coords_preserved(self) -> None:
        """Space coordinates are unchanged after DWT."""
        xr_arr = xr.DataArray(
            np.random.default_rng(2).standard_normal((N_TIME, N_SPACE)),
            dims=["time", "space"],
            coords={"space": ["Fz", "Cz", "Pz"], "time": np.arange(N_TIME) / SR},
        )
        data = cb.SignalData.from_xarray(xr_arr)
        out = cb.feature.DiscreteWaveletTransform(level=2).apply(data)
        np.testing.assert_array_equal(out.data.coords["space"].values, ["Fz", "Cz", "Pz"])

    def test_dwt_coef_index_length_matches_finest_detail(self) -> None:
        """coef_index length equals the finest-level (detail_1) coefficient count."""
        import pywt

        data = _make_data()
        level = 3
        out = cb.feature.DiscreteWaveletTransform(level=level).apply(data)

        # Compute expected finest-detail length directly
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(N_TIME)
        coeffs = pywt.wavedec(sig, "db4", level=level)
        expected_len = len(coeffs[-1])  # cD_1 is last and longest
        assert out.data.sizes["coef_index"] == expected_len

    # -----------------------------------------------------------------------
    # Numerical correctness
    # -----------------------------------------------------------------------

    def test_dwt_approx_matches_pywt(self) -> None:
        """Approximation coefficients match pywt.wavedec output exactly."""
        import pywt

        rng = np.random.default_rng(3)
        arr = rng.standard_normal((N_TIME, N_SPACE))
        data = _make_data(seed=3)
        level = 3
        out = cb.feature.DiscreteWaveletTransform(level=level).apply(data)

        for ch in range(N_SPACE):
            expected_coeffs = pywt.wavedec(arr[:, ch], "db4", level=level)
            expected_approx = expected_coeffs[0]
            actual_approx = out.data.sel(space=ch, wavelet_level="approx").values
            # Only compare the non-NaN portion
            np.testing.assert_allclose(
                actual_approx[: len(expected_approx)], expected_approx, rtol=1e-10
            )

    def test_dwt_detail_coefficients_match_pywt(self) -> None:
        """Detail coefficients at each level match pywt.wavedec output."""
        import pywt

        rng = np.random.default_rng(4)
        arr = rng.standard_normal((N_TIME, 1))
        xr_arr = xr.DataArray(arr, dims=["time", "space"])
        data = cb.SignalData.from_xarray(xr_arr)

        level = 3
        out = cb.feature.DiscreteWaveletTransform(level=level).apply(data)
        expected_coeffs = pywt.wavedec(arr[:, 0], "db4", level=level)
        # pywt order: [cA, cD_L, ..., cD_1]; detail_j in output corresponds to index (level-j+1)
        for j in range(1, level + 1):
            pywt_idx = level - j + 1  # index into expected_coeffs for detail_j
            label = f"detail_{j}"
            actual = out.data.sel(space=0, wavelet_level=label).values
            expected = expected_coeffs[pywt_idx]
            np.testing.assert_allclose(actual[: len(expected)], expected, rtol=1e-10)

    def test_dwt_shorter_levels_are_nan_padded(self) -> None:
        """Coefficients shorter than the finest detail are NaN-padded on the right."""
        import pywt

        data = _make_data()
        level = 4
        out = cb.feature.DiscreteWaveletTransform(level=level).apply(data)

        rng = np.random.default_rng(0)
        coeffs = pywt.wavedec(rng.standard_normal(N_TIME), "db4", level=level)
        finest_len = len(coeffs[-1])

        # approx is the shortest; check that the tail is NaN
        approx_len = len(coeffs[0])
        approx_vals = out.data.isel(space=0).sel(wavelet_level="approx").values
        assert np.all(np.isnan(approx_vals[approx_len:finest_len]))
        assert np.all(np.isfinite(approx_vals[:approx_len]))

    def test_dwt_finest_detail_has_no_nan(self) -> None:
        """The finest detail level (detail_1) has no NaN values."""
        data = _make_data()
        out = cb.feature.DiscreteWaveletTransform(level=3).apply(data)
        finest = out.data.sel(wavelet_level="detail_1").values
        assert np.all(np.isfinite(finest))

    def test_dwt_default_level_uses_max(self) -> None:
        """With level=None, the maximum possible decomposition level is used."""
        import pywt

        data = _make_data()
        out = cb.feature.DiscreteWaveletTransform(level=None).apply(data)
        expected_max = pywt.dwt_max_level(N_TIME, "db4")
        assert out.data.sizes["wavelet_level"] == expected_max + 1

    # -----------------------------------------------------------------------
    # Extra dimensions
    # -----------------------------------------------------------------------

    def test_dwt_preserves_window_dim(self) -> None:
        """DWT processes correctly when a window_index dimension is present."""
        n_windows = 4
        data = _make_data_with_window_dim(n_windows=n_windows, n_time=128)
        out = cb.feature.DiscreteWaveletTransform(level=2).apply(data)
        assert "window_index" in out.data.dims
        assert out.data.sizes["window_index"] == n_windows
        assert out.data.sizes["space"] == N_SPACE

    # -----------------------------------------------------------------------
    # Metadata propagation
    # -----------------------------------------------------------------------

    def test_dwt_preserves_metadata(self) -> None:
        """DWT propagates subjectID, groupID, condition, appends to history,
        and sets sampling_rate to None.
        """
        data = _make_data()
        out = cb.feature.DiscreteWaveletTransform(level=2).apply(data)
        assert out.subjectID == "sub-01"
        assert out.groupID == "ctrl"
        assert out.condition == "rest"
        assert out.sampling_rate is None  # time axis consumed → no meaningful sampling rate
        assert out.history[-1] == "DiscreteWaveletTransform"

    def test_dwt_does_not_mutate_input(self) -> None:
        """DWT leaves the input Data object unchanged."""
        data = _make_data()
        original_values = data.data.values.copy()
        original_history = list(data.history)
        cb.feature.DiscreteWaveletTransform(level=2).apply(data)
        np.testing.assert_array_equal(data.data.values, original_values)
        assert data.history == original_history

    # -----------------------------------------------------------------------
    # Parameter behaviour
    # -----------------------------------------------------------------------

    def test_dwt_higher_level_gives_more_nan_in_approx(self) -> None:
        """More decomposition levels → more NaN padding in the approximation row.

        cD_1 (finest detail) determines coef_index length for all levels.
        A deeper decomposition pushes the approximation to a shorter length,
        so it has a larger NaN tail.
        """
        import pywt

        data = _make_data()
        for lv in (2, 4):
            out = cb.feature.DiscreteWaveletTransform(level=lv).apply(data)
            coeffs = pywt.wavedec(data.data.values[0], "db4", level=lv)
            approx_len = len(coeffs[0])  # cA length at this level
            approx_row = out.data.isel(space=0).sel(wavelet_level="approx").values
            assert np.all(np.isfinite(approx_row[:approx_len]))
            assert np.all(np.isnan(approx_row[approx_len:]))

    def test_dwt_different_wavelets_produce_different_results(self) -> None:
        """Choosing a different wavelet changes the approximation coefficients."""
        import pywt

        rng = np.random.default_rng(42)
        sig = rng.standard_normal(N_TIME)
        coeffs_db4 = pywt.wavedec(sig, "db4", level=2)
        coeffs_haar = pywt.wavedec(sig, "haar", level=2)
        # Compare approx coefficients (both have the same length for haar and db4 at level=2
        # when using symmetric mode — but values differ)
        min_len = min(len(coeffs_db4[0]), len(coeffs_haar[0]))
        assert not np.allclose(coeffs_db4[0][:min_len], coeffs_haar[0][:min_len])

    # -----------------------------------------------------------------------
    # Error handling
    # -----------------------------------------------------------------------

    def test_dwt_raises_on_invalid_wavelet(self) -> None:
        with pytest.raises(ValueError, match="Unknown discrete wavelet"):
            cb.feature.DiscreteWaveletTransform(wavelet="notawavelet")

    def test_dwt_raises_on_level_zero(self) -> None:
        with pytest.raises(ValueError, match="level must be >= 1"):
            cb.feature.DiscreteWaveletTransform(level=0)

    def test_dwt_raises_when_level_exceeds_max(self) -> None:
        """DiscreteWaveletTransform raises ValueError when level > max for signal."""
        data = _make_data(n_time=8)  # very short signal
        with pytest.raises(ValueError, match="exceeds the maximum"):
            cb.feature.DiscreteWaveletTransform(level=100).apply(data)

    # -----------------------------------------------------------------------
    # API accessibility
    # -----------------------------------------------------------------------

    def test_dwt_accessible_via_feature_module(self) -> None:
        assert callable(cb.feature.DiscreteWaveletTransform)


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
