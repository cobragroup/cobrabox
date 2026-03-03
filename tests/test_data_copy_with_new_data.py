"""Tests for Data._copy_with_new_data behavior."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb


def test_copy_with_new_data_from_dataarray_preserves_metadata_and_adds_time() -> None:
    """DataArray input preserves metadata, appends history, merges extra, and restores time."""
    base = cb.from_numpy(
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
    base = cb.from_numpy(
        np.arange(8, dtype=float).reshape(4, 2),
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID="sub-01",
        groupID="group-a",
        condition="rest",
        extra={"keep": 1, "override": "base"},
    )
    base = base._copy_with_new_data(base.data, operation_name="seed")

    incoming = cb.from_numpy(
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
    base = cb.from_numpy(np.arange(6, dtype=float).reshape(3, 2), dims=["time", "space"])
    reduced = xr.DataArray(np.array([1.0, 2.0]), dims=["space"])

    out = base._copy_with_new_data(reduced, operation_name="reduce")

    assert out.data.dims == ("space", "time")
    np.testing.assert_allclose(out.data.coords["time"].values, np.array([0.01]))
    assert out.sampling_rate == pytest.approx(100.0)
    assert out.history == ["reduce"]


def test_copy_with_new_data_history_concatenates_long_and_short_sequences() -> None:
    """Long self history and short incoming history are concatenated in order."""
    base = cb.from_numpy(
        np.arange(10, dtype=float).reshape(5, 2), dims=["time", "space"], sampling_rate=50.0
    )
    for name in ["op_1", "op_2", "op_3", "op_4", "op_5"]:
        base = base._copy_with_new_data(base.data, operation_name=name)

    incoming = cb.from_numpy(np.full((5, 2), 2.0), dims=["time", "space"], sampling_rate=50.0)
    incoming = incoming._copy_with_new_data(incoming.data, operation_name="incoming_only")

    out = base._copy_with_new_data(incoming, operation_name="merge")

    assert out.history == ["op_1", "op_2", "op_3", "op_4", "op_5", "incoming_only", "merge"]
