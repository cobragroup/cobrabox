"""Tests for the placeholder time axis guard (issue #118).

A Chord returns the input container type, so when its aggregator hands back data whose
time dimension was consumed per-window, SignalData inserts a length-1 'time' axis to
keep its contract. Time-domain features must refuse to run on that axis rather than
silently returning a degenerate result.
"""

from __future__ import annotations

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.base_feature import _requires_real_time
from cobrabox.data import has_placeholder_time


def _data(n_time: int = 100, n_space: int = 3) -> cb.SignalData:
    arr = np.random.default_rng(0).standard_normal((n_space, n_time))
    return cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=100.0)


def _stacked() -> cb.Data:
    """Chord output whose per-window feature consumed the time dimension."""
    chord = (
        cb.feature.SlidingWindow(window_size=20, step_size=10)
        | cb.feature.LineLength()
        | cb.feature.ConcatAggregate()
    )
    return chord.apply(_data())


def test_real_time_axis_is_not_flagged() -> None:
    """Untouched time-series data has a genuine time axis."""
    assert not has_placeholder_time(_data())


def test_chord_output_carries_a_flagged_axis() -> None:
    """The time axis on a time-reducing chord's output is fabricated, and marked so."""
    stacked = _stacked()

    assert stacked.data.sizes["time"] == 1
    assert has_placeholder_time(stacked)


def test_time_feature_refuses_stacked_chord_output() -> None:
    """The silent-zeros case from issue #118 now raises."""
    stacked = _stacked()

    with pytest.raises(ValueError, match="no real time axis left"):
        cb.feature.LineLength().apply(stacked)


def test_error_message_names_the_feature_and_history() -> None:
    """The error explains which feature refused and how the data got here."""
    stacked = _stacked()

    with pytest.raises(ValueError, match="no real time axis left") as excinfo:
        cb.feature.Nonreversibility().apply(stacked)

    message = str(excinfo.value)
    assert "Nonreversibility" in message
    assert "SlidingWindow" in message  # history is included
    assert 'Mean(dim="window")' in message  # points at the fix


def test_dim_parameterised_feature_still_works_on_placeholder_data() -> None:
    """Features that take a dim argument are not time-domain features and stay allowed."""
    stacked = _stacked()

    result = cb.feature.Mean(dim="space").apply(stacked)

    assert "space" not in result.data.dims


def test_window_axis_remains_reducible_after_a_chord() -> None:
    """The legitimate follow-up to ConcatAggregate — reducing over 'window' — still works."""
    stacked = _stacked()

    result = cb.feature.Mean(dim="window").apply(stacked)

    assert "window" not in result.data.dims


def test_time_feature_still_runs_on_genuine_time_series() -> None:
    """The guard does not disturb the ordinary case."""
    result = cb.feature.LineLength().apply(_data())

    assert "time" not in result.data.dims
    assert np.all(result.to_numpy() > 0)


def test_time_preserving_feature_clears_the_flag() -> None:
    """A feature that yields a real multi-sample time axis is not flagged."""
    filtered = cb.feature.BandDecomposition(bands={"alpha": [8.0, 12.0]}).apply(_data())

    assert not has_placeholder_time(filtered)
    assert filtered.data.sizes["time"] > 1


@pytest.mark.parametrize(
    ("feature_cls", "expected"),
    [
        (cb.feature.LineLength, True),
        (cb.feature.Nonreversibility, True),
        (cb.feature.Mean, False),
        (cb.feature.Max, False),
    ],
)
def test_requires_real_time_follows_type_parameterisation(
    feature_cls: type, expected: bool
) -> None:
    """BaseFeature[SignalData] means time-domain; BaseFeature[Data] means dim-agnostic."""
    assert _requires_real_time(feature_cls) is expected
