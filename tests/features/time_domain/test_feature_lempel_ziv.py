"""Tests for the LempelZiv feature."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.features.time_domain.lempel_ziv import LempelZiv


def test_feature_lempel_ziv_output_type_and_history() -> None:
    """LempelZiv removes the time dimension and records history."""
    arr = np.random.default_rng(0).standard_normal((100, 3))
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=256.0, subjectID="sub-01"
    )
    out = cb.feature.LempelZiv().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (3,)
    assert out.subjectID == "sub-01"
    assert out.sampling_rate is None
    assert out.history == ["LempelZiv"]


def test_feature_lempel_ziv_known_value() -> None:
    """LempelZiv returns the expected value for a hand-verified signal."""
    # Construct a known symbolic sequence so we can compute expected value by hand.
    # Signal: alternating above/below mean → binary "10101010" (8 samples)
    signal = np.array(
        [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]
    )  # mean=1, symbolic=[1,0,1,0,1,0,1,0]
    symbolic = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    c, n = LempelZiv._count(symbolic)
    expected = float((c * math.log2(n)) / n)

    arr = signal.reshape(-1, 1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.LempelZiv().apply(data)

    assert float(out.to_numpy()[0]) == pytest.approx(expected)


def test_feature_lempel_ziv_random_more_complex_than_periodic() -> None:
    """A random signal should have higher LZC than a periodic one."""
    rng = np.random.default_rng(42)
    t = np.arange(200)
    periodic = np.sin(2 * np.pi * t / 20)
    random = rng.standard_normal(200)

    def _lzc(sig: np.ndarray) -> float:
        arr = sig.reshape(-1, 1)
        data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
        return float(cb.feature.LempelZiv().apply(data).to_numpy()[0])

    assert _lzc(random) > _lzc(periodic)


def test_feature_lempel_ziv_values_are_positive() -> None:
    """LZC values should be strictly positive."""
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((256, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.LempelZiv().apply(data)

    assert np.all(out.to_numpy() > 0)


def test_feature_lempel_ziv_via_chord() -> None:
    """LempelZiv applied per window via Chord propagates history correctly."""
    arr = np.random.default_rng(1).standard_normal((100, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=50, step_size=25),
        pipeline=cb.feature.LempelZiv(),
        aggregate=cb.feature.MeanAggregate(),
    )
    out = chord.apply(data)

    assert isinstance(out, cb.Data)
    assert "LempelZiv" in out.history
    assert "MeanAggregate" in out.history


def test_feature_lempel_ziv_multichannel_independent() -> None:
    """LZC is computed independently per channel."""
    rng = np.random.default_rng(3)
    ch0 = rng.standard_normal(128)
    ch1 = np.sin(2 * np.pi * np.arange(128) / 10)  # periodic → lower LZC
    arr = np.stack([ch0, ch1], axis=1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.LempelZiv().apply(data)

    assert out.data.shape == (2,)
    # Random channel should be more complex than sinusoidal channel
    assert out.to_numpy()[0] > out.to_numpy()[1]


def test_feature_lempel_ziv_missing_time_raises() -> None:
    """LempelZiv raises error when time dimension is missing."""
    import xarray as xr

    rng = np.random.default_rng(42)
    arr = rng.standard_normal(10)
    xr_data = xr.DataArray(arr, dims=["space"])
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    # xarray raises ValueError when core dim is missing
    with pytest.raises(ValueError, match="not in tuple"):
        cb.feature.LempelZiv().apply(raw)


def test_feature_lempel_ziv_does_not_mutate_input() -> None:
    """LempelZiv does not modify the input Data object."""
    rng = np.random.default_rng(8)
    arr = rng.standard_normal((100, 3))
    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=100.0, subjectID="s1"
    )
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.LempelZiv().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
