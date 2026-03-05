"""Tests for the Hilbert feature."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 256.0  # sampling rate (Hz)
N_TIME = 512
N_SPACE = 3
FREQ_HZ = 10.0  # sine frequency used in numerical tests


def _make_data(
    freq_hz: float = FREQ_HZ, n_space: int = N_SPACE, n_time: int = N_TIME
) -> cb.SignalData:
    """Return a SignalData with pure sine waves (one per channel)."""
    t = np.arange(n_time) / SR
    signal = np.stack([np.sin(2 * np.pi * freq_hz * t) for _ in range(n_space)], axis=0)
    return cb.SignalData.from_numpy(
        signal, dims=["space", "time"], sampling_rate=SR, subjectID="test-subject"
    )


def _make_data_no_sr() -> cb.SignalData:
    """SignalData without a sampling rate (needed for frequency-mode error test)."""
    t = np.arange(N_TIME) / SR
    signal = np.stack([np.sin(2 * np.pi * FREQ_HZ * t) for _ in range(N_SPACE)], axis=0)
    return cb.SignalData.from_numpy(signal, dims=["space", "time"])


# ---------------------------------------------------------------------------
# Output shape / dims
# ---------------------------------------------------------------------------


def test_output_is_signal_data() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert isinstance(out, cb.SignalData)


def test_output_dims_preserved() -> None:
    data = _make_data()
    for feat in ("envelope", "phase", "frequency"):
        out = cb.feature.Hilbert(feature=feat).apply(data)
        assert out.data.dims == data.data.dims, f"dims changed for feature={feat!r}"


def test_output_shape_preserved() -> None:
    data = _make_data()
    for feat in ("envelope", "phase", "frequency"):
        out = cb.feature.Hilbert(feature=feat).apply(data)
        assert out.data.shape == data.data.shape, f"shape changed for feature={feat!r}"


def test_analytic_dims() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="analytic").apply(data)
    assert out.data.dims[0] == "component"
    assert out.data.dims[1:] == data.data.dims


def test_analytic_shape() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="analytic").apply(data)
    assert out.data.shape == (2, N_SPACE, N_TIME)


def test_analytic_component_coords() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="analytic").apply(data)
    np.testing.assert_array_equal(out.data.coords["component"].values, ["real", "imag"])


def test_default_feature_is_analytic() -> None:
    data = _make_data()
    out_default = cb.feature.Hilbert().apply(data)
    out_explicit = cb.feature.Hilbert(feature="analytic").apply(data)
    np.testing.assert_array_equal(out_default.data.values, out_explicit.data.values)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_envelope_pure_sine() -> None:
    """Envelope of a unit-amplitude sine ≈ 1 everywhere (away from edges)."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    envelope = out.data.values  # shape (space, time)

    # Trim edge artefacts — check the middle 80 %
    trim = N_TIME // 10
    np.testing.assert_allclose(
        envelope[:, trim:-trim], np.ones((N_SPACE, N_TIME - 2 * trim)), atol=1e-3
    )


def test_envelope_nonnegative() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert np.all(out.data.values >= 0)


def test_phase_range() -> None:
    """Instantaneous phase must be within [-pi, pi]."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="phase").apply(data)
    phase = out.data.values
    assert np.all(phase >= -np.pi - 1e-10)
    assert np.all(phase <= np.pi + 1e-10)


def test_frequency_pure_sine() -> None:
    """Instantaneous frequency of a pure sine ≈ FREQ_HZ (away from edges)."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="frequency").apply(data)
    freq = out.data.values  # shape (space, time)

    trim = N_TIME // 10
    np.testing.assert_allclose(
        freq[:, trim:-trim],
        np.full((N_SPACE, N_TIME - 2 * trim), FREQ_HZ),
        atol=0.5,  # allow 0.5 Hz tolerance
    )


def test_analytic_is_float64() -> None:
    """Analytic mode returns real-valued float64 (stacked, not complex)."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="analytic").apply(data)
    assert out.data.dtype == np.float64


def test_analytic_real_component_equals_input() -> None:
    """The 'real' component of the analytic output equals the original signal."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="analytic").apply(data)
    np.testing.assert_allclose(out.data.sel(component="real").values, data.data.values, atol=1e-10)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_history_appended() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert out.history[-1] == "Hilbert"


def test_subject_id_preserved() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert out.subjectID == "test-subject"


def test_sampling_rate_preserved() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert out.sampling_rate == SR


def test_coords_preserved() -> None:
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    for dim in data.data.dims:
        if dim in data.data.coords:
            np.testing.assert_array_equal(out.data.coords[dim].values, data.data.coords[dim].values)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_feature_raises() -> None:
    with pytest.raises(ValueError, match="Invalid feature"):
        cb.feature.Hilbert(feature="invalid")  # type: ignore[arg-type]


def test_frequency_without_sampling_rate_raises() -> None:
    data = _make_data_no_sr()
    with pytest.raises(ValueError, match="sampling_rate"):
        cb.feature.Hilbert(feature="frequency").apply(data)


# ---------------------------------------------------------------------------
# Pipeline compatibility
# ---------------------------------------------------------------------------


def test_pipe_with_line_length() -> None:
    """Hilbert | LineLength should work end-to-end."""
    data = _make_data()
    pipeline = cb.feature.Hilbert(feature="envelope") | cb.feature.LineLength()
    out = pipeline.apply(data)
    assert isinstance(out, cb.Data)
    assert "time" not in out.data.dims
