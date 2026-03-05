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


def test_hilbert_output_is_signal_data() -> None:
    """Hilbert returns a SignalData instance."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert isinstance(out, cb.SignalData)


def test_hilbert_output_dims_preserved() -> None:
    """Hilbert preserves all dimensions for all feature modes."""
    data = _make_data()
    for feat in ("analytic", "envelope", "phase", "frequency"):
        out = cb.feature.Hilbert(feature=feat).apply(data)
        assert out.data.dims == data.data.dims, f"dims changed for feature={feat!r}"


def test_hilbert_output_shape_preserved() -> None:
    """Hilbert preserves array shape for all feature modes."""
    data = _make_data()
    for feat in ("analytic", "envelope", "phase", "frequency"):
        out = cb.feature.Hilbert(feature=feat).apply(data)
        assert out.data.shape == data.data.shape, f"shape changed for feature={feat!r}"


def test_hilbert_analytic_dtype_is_complex() -> None:
    """Hilbert analytic mode returns complex128 dtype."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="analytic").apply(data)
    assert np.iscomplexobj(out.data.values)
    assert out.data.dtype == np.complex128


def test_hilbert_analytic_real_part_equals_input() -> None:
    """Real part of the analytic signal equals the original signal."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="analytic").apply(data)
    np.testing.assert_allclose(out.data.values.real, data.data.values, atol=1e-10)


def test_hilbert_default_feature_is_analytic() -> None:
    """Hilbert default feature parameter is 'analytic'."""
    data = _make_data()
    out_default = cb.feature.Hilbert().apply(data)
    out_explicit = cb.feature.Hilbert(feature="analytic").apply(data)
    np.testing.assert_array_equal(out_default.data.values, out_explicit.data.values)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_hilbert_envelope_pure_sine() -> None:
    """Envelope of a unit-amplitude sine ≈ 1 everywhere (away from edges)."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    envelope = out.data.values  # shape (space, time)

    # Trim edge artefacts — check the middle 80 %
    trim = N_TIME // 10
    np.testing.assert_allclose(
        envelope[:, trim:-trim], np.ones((N_SPACE, N_TIME - 2 * trim)), atol=1e-3
    )


def test_hilbert_envelope_nonnegative() -> None:
    """Hilbert envelope is always non-negative."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert np.all(out.data.values >= 0)


def test_hilbert_phase_range() -> None:
    """Instantaneous phase must be within [-pi, pi]."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="phase").apply(data)
    phase = out.data.values
    assert np.all(phase >= -np.pi - 1e-10)
    assert np.all(phase <= np.pi + 1e-10)


def test_hilbert_frequency_pure_sine() -> None:
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


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def test_hilbert_history_appended() -> None:
    """Hilbert appends 'Hilbert' to history."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert out.history[-1] == "Hilbert"


def test_hilbert_subject_id_preserved() -> None:
    """Hilbert preserves subjectID metadata."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert out.subjectID == "test-subject"


def test_hilbert_sampling_rate_preserved() -> None:
    """Hilbert preserves sampling_rate metadata."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert out.sampling_rate == SR


def test_hilbert_coords_preserved() -> None:
    """Hilbert preserves coordinate values for all dimensions."""
    data = _make_data()
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    for dim in data.data.dims:
        if dim in data.data.coords:
            np.testing.assert_array_equal(out.data.coords[dim].values, data.data.coords[dim].values)


def test_hilbert_group_id_and_condition_preserved() -> None:
    """Hilbert preserves groupID and condition metadata."""
    rng = np.random.default_rng(42)
    data = cb.SignalData.from_numpy(
        rng.standard_normal((3, 100)),
        dims=["space", "time"],
        sampling_rate=SR,
        subjectID="test-subject",
        groupID="test-group",
        condition="test-condition",
    )
    out = cb.feature.Hilbert(feature="envelope").apply(data)
    assert out.groupID == "test-group"
    assert out.condition == "test-condition"


def test_hilbert_does_not_mutate_input() -> None:
    """Hilbert.apply() leaves the input Data object unchanged."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.data.values.copy()

    _ = cb.feature.Hilbert(feature="envelope").apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.data.values, original_values)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_hilbert_invalid_feature_raises() -> None:
    """Hilbert raises ValueError for invalid feature parameter."""
    with pytest.raises(ValueError, match="Invalid feature"):
        cb.feature.Hilbert(feature="invalid")  # type: ignore[arg-type]


def test_hilbert_frequency_without_sampling_rate_raises() -> None:
    """Hilbert raises ValueError for frequency mode without sampling_rate."""
    data = _make_data_no_sr()
    with pytest.raises(ValueError, match="sampling_rate"):
        cb.feature.Hilbert(feature="frequency").apply(data)


def test_hilbert_missing_time_raises() -> None:
    """Hilbert raises ValueError when 'time' dimension is missing."""
    import xarray as xr

    # Build a Data without time dimension
    rng = np.random.default_rng(42)
    bad_xr = xr.DataArray(rng.standard_normal((10, 10)), dims=["space", "trial"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", bad_xr)
    with pytest.raises(ValueError):  # noqa: PT011
        cb.feature.Hilbert(feature="envelope").apply(raw)


# ---------------------------------------------------------------------------
# Pipeline compatibility
# ---------------------------------------------------------------------------


def test_hilbert_pipe_with_line_length() -> None:
    """Hilbert | LineLength should work end-to-end."""
    data = _make_data()
    pipeline = cb.feature.Hilbert(feature="envelope") | cb.feature.LineLength()
    out = pipeline.apply(data)
    assert isinstance(out, cb.Data)
    assert "time" not in out.data.dims
