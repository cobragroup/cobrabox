"""Tests for Data._copy_with_new_data behavior."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb
from cobrabox.data import METADATA_FIELDS


def test_copy_with_new_data_from_dataarray_preserves_metadata_and_adds_time() -> None:
    """DataArray input preserves metadata, appends history, merges extra, and restores time."""
    base = cb.SignalData.from_numpy(
        np.arange(12, dtype=float).reshape(6, 2),
        dims=["time", "space"],
        sampling_rate=200.0,
        subjectID="sub-01",
        groupID="group-a",
        condition="rest",
        extra={"base": 1},
    )

    reduced = xr.DataArray(np.array([3.0, 7.0]), dims=["space"])
    out = base._copy_with_new_data(reduced, operation_name="line_length", extra={"new": 2})

    assert isinstance(out, cb.Data)
    assert out.data.dims == ("space", "time")
    assert out.data.shape == (2, 1)
    np.testing.assert_allclose(out.to_numpy(), np.array([[3.0], [7.0]]))
    np.testing.assert_allclose(out.data.coords["time"].values, np.array([1.0 / 200.0]))
    assert out.sampling_rate == pytest.approx(200.0)
    assert out.subjectID == "sub-01"
    assert out.groupID == "group-a"
    assert out.condition == "rest"
    assert out.history == ["line_length"]
    assert out.extra == {"base": 1, "new": 2}


def test_copy_with_new_data_from_data_merges_metadata_history_and_extra() -> None:
    """Data input merges metadata selectively, combines histories, and applies extra overrides."""
    base = cb.SignalData.from_numpy(
        np.arange(8, dtype=float).reshape(4, 2),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-a",
        condition="rest",
        extra={"keep": 1, "override": "base"},
    )
    base = base._copy_with_new_data(base.data, operation_name="seed")

    incoming = cb.SignalData.from_numpy(
        np.full((4, 2), 9.0),
        dims=["time", "space"],
        sampling_rate=100.0,
        groupID="group-b",
        extra={"override": "incoming", "incoming_only": True},
    )
    incoming = incoming._copy_with_new_data(incoming.data, operation_name="inner")

    out = base._copy_with_new_data(incoming, operation_name="outer", extra={"override": "final"})

    np.testing.assert_allclose(out.to_numpy(), np.full((2, 4), 9.0))
    assert out.subjectID == "sub-01"  # incoming subjectID is None -> keep original
    assert out.groupID == "group-b"  # incoming non-None -> override original
    assert out.condition == "rest"  # incoming condition is None -> keep original
    assert out.history == ["seed", "inner", "outer"]
    assert out.extra == {"keep": 1, "override": "final", "incoming_only": True}


def test_copy_with_new_data_without_sampling_rate_uses_fallback() -> None:
    """When time is missing and sampling rate is unknown, fallback metadata is applied."""
    base = cb.SignalData.from_numpy(np.arange(6, dtype=float).reshape(3, 2), dims=["time", "space"])
    reduced = xr.DataArray(np.array([1.0, 2.0]), dims=["space"])

    out = base._copy_with_new_data(reduced, operation_name="reduce")

    assert out.data.dims == ("space", "time")
    np.testing.assert_allclose(out.data.coords["time"].values, np.array([0.01]))
    assert out.sampling_rate == pytest.approx(100.0)
    assert out.history == ["reduce"]


def test_copy_with_new_data_history_concatenates_long_and_short_sequences() -> None:
    """Long self history and short incoming history are concatenated in order."""
    base = cb.SignalData.from_numpy(
        np.arange(10, dtype=float).reshape(5, 2), dims=["time", "space"], sampling_rate=50.0
    )
    for name in ["op_1", "op_2", "op_3", "op_4", "op_5"]:
        base = base._copy_with_new_data(base.data, operation_name=name)

    incoming = cb.SignalData.from_numpy(
        np.full((5, 2), 2.0), dims=["time", "space"], sampling_rate=50.0
    )
    incoming = incoming._copy_with_new_data(incoming.data, operation_name="incoming_only")

    out = base._copy_with_new_data(incoming, operation_name="merge")

    assert out.history == ["op_1", "op_2", "op_3", "op_4", "op_5", "incoming_only", "merge"]


def test_copy_strips_sampling_rate_when_no_time_dim() -> None:
    """_copy_with_new_data strips sampling_rate from attrs when result has no time dimension."""
    # Use plain Data class (not SignalData) to avoid auto-adding time dimension
    base = cb.Data.from_numpy(
        np.ones((10, 2), dtype=float), dims=["time", "space"], sampling_rate=100.0
    )
    # Result without time dimension but WITH sampling_rate in attrs (to test stripping)
    reduced = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=["space"],
        attrs={"sampling_rate": 100.0},  # This should be stripped
    )
    out = base._copy_with_new_data(reduced, operation_name="reduce")

    assert "time" not in out.data.dims
    assert "sampling_rate" not in out.data.attrs
    assert out.sampling_rate is None


# ----------------------------------------------------------------------
# Every METADATA_FIELDS entry must survive every rebuild site
# ----------------------------------------------------------------------
#
# These are parametrised over METADATA_FIELDS rather than over a hardcoded list, so
# adding a field (e.g. runID) automatically extends the checks. A new field dropped
# at any rebuild site fails here instead of silently going None mid-pipeline.


def _with_field(field: str, value: str) -> cb.SignalData:
    return cb.SignalData.from_numpy(
        np.arange(40, dtype=float).reshape(20, 2),
        dims=["time", "space"],
        sampling_rate=100.0,
        **{field: value},
    )


@pytest.mark.parametrize("field", METADATA_FIELDS)
def test_metadata_field_survives_copy_from_dataarray(field: str) -> None:
    base = _with_field(field, "value-x")
    out = base._copy_with_new_data(xr.DataArray(np.array([1.0, 2.0]), dims=["space"]))
    assert getattr(out, field) == "value-x"


@pytest.mark.parametrize("field", METADATA_FIELDS)
def test_metadata_field_survives_copy_from_data(field: str) -> None:
    base = _with_field(field, "value-x")
    returned = cb.Data(xr.DataArray(np.array([1.0, 2.0]), dims=["space"]))
    out = base._copy_with_new_data(returned)
    assert getattr(out, field) == "value-x"


@pytest.mark.parametrize("field", METADATA_FIELDS)
def test_metadata_field_survives_a_feature(field: str) -> None:
    base = _with_field(field, "value-x")
    assert getattr(cb.LineLength().apply(base), field) == "value-x"


@pytest.mark.parametrize("field", METADATA_FIELDS)
@pytest.mark.parametrize("aggregator", [cb.MeanAggregate, cb.ConcatAggregate])
def test_metadata_field_survives_aggregators(field: str, aggregator: type) -> None:
    base = _with_field(field, "value-x")
    chord = cb.Chord(
        split=cb.SlidingWindow(window_size=10, step_size=5),
        pipeline=cb.LineLength(),
        aggregate=aggregator(),
    )
    assert getattr(chord.apply(base), field) == "value-x"
