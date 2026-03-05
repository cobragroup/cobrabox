"""Tests for the AmpVar feature behavior."""

from __future__ import annotations

import numpy as np

import cobrabox as cb


def test_feature_amp_var_expected_values_and_history() -> None:
    """AmpVar computes standard deviation over time and wraps to Data."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    # Expected std per channel (ddof=0):
    # ch0: std([1, 3, 5, 7]) = sqrt(5) ≈ 2.2360679...
    # ch1: std([2, 4, 6, 8]) = sqrt(5) ≈ 2.2360679...
    expected = np.array([np.std([1.0, 3.0, 5.0, 7.0]), np.std([2.0, 4.0, 6.0, 8.0])])

    data = cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=200.0, subjectID="sub-01"
    )
    out = cb.feature.AmpVar().apply(data)

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space",)
    assert out.data.shape == (2,)
    np.testing.assert_allclose(out.to_numpy(), expected)
    assert out.subjectID == "sub-01"
    assert out.sampling_rate is None
    assert out.history == ["AmpVar"]


def test_feature_amp_var_constant_signal_is_zero() -> None:
    """A constant signal has zero amplitude variation."""
    arr = np.ones((10, 3))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.AmpVar().apply(data)

    assert out.data.shape == (3,)
    np.testing.assert_allclose(out.to_numpy(), 0.0)


def test_feature_amp_var_single_channel() -> None:
    """AmpVar works for a single-channel signal."""
    arr = np.array([[1.0], [2.0], [3.0]])
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    out = cb.feature.AmpVar().apply(data)

    assert out.data.dims == ("space",)
    assert out.data.shape == (1,)
    np.testing.assert_allclose(out.to_numpy(), [np.std([1.0, 2.0, 3.0])])


def test_feature_amp_var_via_chord() -> None:
    """AmpVar applied per window via Chord produces an aggregated result."""
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=5, step_size=2),
        pipeline=cb.feature.AmpVar(),
        aggregate=cb.feature.MeanAggregate(),
    )
    out = chord.apply(data)

    assert isinstance(out, cb.Data)
    assert "AmpVar" in out.history
    assert "MeanAggregate" in out.history
