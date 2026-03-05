"""Tests for FractalDimHiguchi (and FractalDimKatz placeholder)."""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.features.fractal_dimension import FractalDimHiguchi, FractalDimKatz

# ---------------------------------------------------------------------------
# FractalDimHiguchi
# ---------------------------------------------------------------------------


def test_higuchi_output_type_dims_history(rng: np.random.Generator) -> None:
    """HFD removes the time dimension, returns Data, and records history."""
    arr = rng.standard_normal((256, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=256.0,
        subjectID="sub-01",
        groupID="g1",
        condition="rest",
    )
    out = cb.feature.FractalDimHiguchi().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (3,)
    assert out.subjectID == "sub-01"
    assert out.groupID == "g1"
    assert out.condition == "rest"
    assert out.sampling_rate is None
    assert out.history == ["FractalDimHiguchi"]


def test_higuchi_linear_signal_fd_equals_one() -> None:
    """A perfectly linear signal has FD = 1 (analytically exact)."""
    arr = np.arange(1, 257, dtype=float).reshape(-1, 1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.FractalDimHiguchi().apply(data)

    assert float(out.to_numpy()[0]) == pytest.approx(1.0, abs=1e-10)


def test_higuchi_known_value_matches_static_method() -> None:
    """Feature output matches direct call to _higuchi_1d for a known signal."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200)
    expected = FractalDimHiguchi._higuchi_1d(signal, k_max=10)

    arr = signal.reshape(-1, 1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)
    out = cb.feature.FractalDimHiguchi().apply(data)

    assert float(out.to_numpy()[0]) == pytest.approx(expected)


def test_higuchi_random_more_complex_than_sine() -> None:
    """Random noise should have a higher FD than a smooth sine wave."""
    rng = np.random.default_rng(42)
    t = np.arange(512)
    sine = np.sin(2 * np.pi * t / 64)
    noise = rng.standard_normal(512)

    def _hfd(sig: np.ndarray) -> float:
        arr = sig.reshape(-1, 1)
        data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
        return float(cb.feature.FractalDimHiguchi().apply(data).to_numpy()[0])

    assert _hfd(noise) > _hfd(sine)


def test_higuchi_fd_in_expected_range(rng: np.random.Generator) -> None:
    """HFD for random EEG-like data should be clearly above 1 and finite."""
    arr = rng.standard_normal((512, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    values = cb.feature.FractalDimHiguchi().apply(data).to_numpy()

    # Random noise should be clearly more complex than a line (FD > 1)
    assert np.all(values > 1.5)
    # Should be finite and not wildly outside the fractal dimension range
    assert np.all(np.isfinite(values))


def test_higuchi_multichannel_computed_independently() -> None:
    """Each channel is processed independently; results differ across channels."""
    t = np.arange(256)
    ch0 = np.sin(2 * np.pi * t / 32)  # smooth sine
    ch1 = np.random.default_rng(7).standard_normal(256)  # noise
    arr = np.stack([ch0, ch1], axis=1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.FractalDimHiguchi().apply(data)

    assert out.data.shape == (2,)
    # Noise channel must be more complex than the sine channel
    assert out.to_numpy()[1] > out.to_numpy()[0]


def test_higuchi_custom_k_max() -> None:
    """k_max parameter changes the estimate (more intervals → finer estimate)."""
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((512, 1))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)

    out_k5 = float(cb.feature.FractalDimHiguchi(k_max=5).apply(data).to_numpy()[0])
    out_k20 = float(cb.feature.FractalDimHiguchi(k_max=20).apply(data).to_numpy()[0])

    # Both should be reasonable FDs: finite, positive, clearly above 1
    assert np.isfinite(out_k5)
    assert out_k5 > 1.0
    assert np.isfinite(out_k20)
    assert out_k20 > 1.0


def test_higuchi_n_steps_zero_path() -> None:
    """n_steps==0 fallback is exercised when N == k_max + 1."""
    # At k=10, m=10: n_steps = (11-10)//10 = 0 — exercises the zero-step branch
    arr = np.arange(11, dtype=float).reshape(-1, 1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.FractalDimHiguchi(k_max=10).apply(data)

    assert isinstance(out, cb.Data)
    assert np.isfinite(float(out.to_numpy()[0]))


def test_higuchi_does_not_mutate_input(rng: np.random.Generator) -> None:
    """FractalDimHiguchi.apply() leaves the input Data unchanged."""
    arr = rng.standard_normal((200, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    _ = cb.feature.FractalDimHiguchi().apply(data)
    assert data.history == original_history
    assert data.data.shape == original_shape


def test_higuchi_raises_for_invalid_k_max() -> None:
    """k_max < 2 must raise ValueError at construction time."""
    with pytest.raises(ValueError, match="k_max must be >= 2"):
        cb.feature.FractalDimHiguchi(k_max=1)


def test_higuchi_raises_when_signal_too_short() -> None:
    """Signal length <= k_max must raise ValueError, not silently return nan."""
    arr = np.ones((10, 1))  # length 10, k_max defaults to 10 → N <= k_max
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    with pytest.raises(ValueError, match="Signal length"):
        cb.feature.FractalDimHiguchi().apply(data)


def test_higuchi_via_chord() -> None:
    """HFD applied per window via Chord propagates history correctly."""
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((200, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=100, step_size=50),
        pipeline=cb.feature.FractalDimHiguchi(),
        aggregate=cb.feature.MeanAggregate(),
    )
    out = chord.apply(data)

    assert isinstance(out, cb.Data)
    assert "FractalDimHiguchi" in out.history
    assert "MeanAggregate" in out.history


# ---------------------------------------------------------------------------
# FractalDimKatz
# ---------------------------------------------------------------------------


def test_katz_output_type_dims_history(rng: np.random.Generator) -> None:
    """KFD removes the time dimension, returns Data, and records history."""
    arr = rng.standard_normal((256, 3))
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=256.0,
        subjectID="sub-02",
        groupID="g2",
        condition="task",
    )
    out = cb.feature.FractalDimKatz().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (3,)
    assert out.subjectID == "sub-02"
    assert out.groupID == "g2"
    assert out.condition == "task"
    assert out.sampling_rate is None
    assert out.history == ["FractalDimKatz"]


def test_katz_linear_signal_fd_equals_one() -> None:
    """A linear signal has KFD = 1 (analytically exact)."""
    arr = np.arange(1, 257, dtype=float).reshape(-1, 1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.FractalDimKatz().apply(data)

    assert float(out.to_numpy()[0]) == pytest.approx(1.0, abs=1e-10)


def test_katz_known_value_matches_static_method() -> None:
    """Feature output matches direct call to _katz_1d for a known signal."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(200)
    expected = FractalDimKatz._katz_1d(signal)

    arr = signal.reshape(-1, 1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)
    out = cb.feature.FractalDimKatz().apply(data)

    assert float(out.to_numpy()[0]) == pytest.approx(expected)


def test_katz_random_more_complex_than_sine() -> None:
    """Random noise should have a higher KFD than a smooth sine wave."""
    rng = np.random.default_rng(42)
    t = np.arange(512)
    sine = np.sin(2 * np.pi * t / 64)
    noise = rng.standard_normal(512)

    def _kfd(sig: np.ndarray) -> float:
        arr = sig.reshape(-1, 1)
        data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
        return float(cb.feature.FractalDimKatz().apply(data).to_numpy()[0])

    assert _kfd(noise) > _kfd(sine)


def test_katz_via_chord() -> None:
    """KFD applied per window via Chord propagates history correctly."""
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((200, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=200.0)

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=100, step_size=50),
        pipeline=cb.feature.FractalDimKatz(),
        aggregate=cb.feature.MeanAggregate(),
    )
    out = chord.apply(data)

    assert isinstance(out, cb.Data)
    assert "FractalDimKatz" in out.history
    assert "MeanAggregate" in out.history


def test_katz_does_not_mutate_input(rng: np.random.Generator) -> None:
    """FractalDimKatz.apply() leaves the input Data unchanged."""
    arr = rng.standard_normal((200, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    _ = cb.feature.FractalDimKatz().apply(data)
    assert data.history == original_history
    assert data.data.shape == original_shape
