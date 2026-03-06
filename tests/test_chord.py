"""Tests for Chord, _ChordBuilder pipe syntax, and MeanAggregate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import cobrabox as cb
from cobrabox.base_feature import BaseFeature, Pipeline, _ChordBuilder


def _make_data(n_time: int = 20, n_space: int = 2, sr: float = 100.0) -> cb.SignalData:
    arr = np.arange(n_time * n_space, dtype=float).reshape(n_time, n_space)
    return cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=sr, subjectID="sub-01"
    )


# --- pipe syntax ---


def test_splitter_pipe_feature_returns_chord_builder() -> None:
    builder = cb.feature.SlidingWindow(window_size=4, step_size=2) | cb.feature.LineLength()
    assert isinstance(builder, _ChordBuilder)


def test_chord_builder_pipe_feature_extends_pipeline() -> None:
    builder = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.Min(dim="time")
    )
    assert isinstance(builder, _ChordBuilder)
    assert isinstance(builder.pipeline, cb.Pipeline)


def test_chord_builder_pipe_aggregator_returns_chord() -> None:
    chord = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    assert isinstance(chord, cb.Chord)


def test_pipe_syntax_result_equals_explicit_chord() -> None:
    data = _make_data()
    via_pipe = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    ).apply(data)
    via_explicit = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=4, step_size=2),
        pipeline=cb.feature.LineLength(),
        aggregate=cb.feature.MeanAggregate(),
    ).apply(data)
    np.testing.assert_allclose(via_pipe.to_numpy(), via_explicit.to_numpy())


def test_incomplete_chord_builder_raises_on_apply() -> None:
    builder = cb.feature.SlidingWindow() | cb.feature.LineLength()
    with pytest.raises(TypeError, match="incomplete"):
        builder.apply(_make_data())


def test_splitter_pipe_directly_to_aggregator_raises() -> None:
    with pytest.raises(TypeError, match="at least one pipeline step"):
        cb.feature.SlidingWindow() | cb.feature.MeanAggregate()


# --- Chord behaviour ---


def test_chord_applies_pipeline_to_each_window_and_aggregates() -> None:
    data = _make_data()
    out = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    ).apply(data)
    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.subjectID == "sub-01"


def test_chord_history_contains_all_steps() -> None:
    data = _make_data()
    out = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    ).apply(data)
    assert "LineLength" in out.history
    assert "MeanAggregate" in out.history
    assert "Chord" in out.history


def test_chord_multi_step_pipeline_via_pipe() -> None:
    data = _make_data()
    out = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.Mean(dim="space")
        | cb.feature.MeanAggregate()
    ).apply(data)
    assert isinstance(out, cb.Data)


def test_chord_composes_downstream_with_pipe() -> None:
    data = _make_data(n_time=40)
    out = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
        | cb.feature.Mean(dim="time")
    ).apply(data)
    assert isinstance(out, cb.Data)


# --- MeanAggregate ---


def test_mean_aggregate_values_match_manual_mean() -> None:
    arr = np.arange(20, dtype=float).reshape(10, 2)
    data = cb.data.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)

    windows = list(cb.feature.SlidingWindow(window_size=4, step_size=2)(data))
    expected = np.mean([cb.feature.LineLength().apply(w).to_numpy() for w in windows], axis=0)

    out = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    ).apply(data)
    np.testing.assert_allclose(np.squeeze(out.to_numpy()), expected)


def test_mean_aggregate_raises_on_empty_stream() -> None:
    data = _make_data()
    with pytest.raises(ValueError, match="empty stream"):
        cb.feature.MeanAggregate()(data, iter([]))


# --- base_feature coverage ---


@dataclass
class _BadFeature(BaseFeature[cb.Data]):
    def __call__(self, data: cb.Data) -> int:  # type: ignore[override]
        return 42


def test_apply_raises_type_error_when_call_returns_wrong_type() -> None:
    data = _make_data()
    with pytest.raises(TypeError, match="must return"):
        _BadFeature().apply(data)


def test_chord_builder_pipe_extends_existing_pipeline() -> None:
    """_ChordBuilder.__or__ with an existing Pipeline (line 135) extends it."""
    builder = (
        cb.feature.SlidingWindow(window_size=4, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.Mean(dim="space")
        | cb.feature.Min(dim="time")
    )
    assert isinstance(builder, _ChordBuilder)
    assert isinstance(builder.pipeline, Pipeline)
    assert len(builder.pipeline.features) == 3


def test_pipeline_or_extends_pipeline() -> None:
    """Pipeline.__or__ returns a new Pipeline with the appended feature."""
    p = cb.feature.LineLength() | cb.feature.Mean(dim="space")
    assert isinstance(p, Pipeline)
    p2 = p | cb.feature.Min(dim="time")
    assert isinstance(p2, Pipeline)
    assert len(p2.features) == 3


def test_pipeline_call_syntax() -> None:
    """Pipeline(data) is equivalent to Pipeline.apply(data)."""
    data = _make_data()
    pipeline = cb.feature.LineLength() | cb.feature.Mean(dim="space")
    result_call = pipeline(data)
    result_apply = pipeline.apply(data)
    np.testing.assert_allclose(result_call.to_numpy(), result_apply.to_numpy())
