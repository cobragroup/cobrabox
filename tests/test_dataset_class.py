"""Tests for Dataset[T] collection class."""

from __future__ import annotations

import numpy as np
import pytest

from cobrabox.data import Data
from cobrabox.dataset import Dataset


def _make_data(
    subjectID: str | None = None, groupID: str | None = None, condition: str | None = None
) -> Data:
    import xarray as xr

    arr = np.zeros((3, 10))
    da = xr.DataArray(arr, dims=["space", "time"])
    return Data(da, subjectID=subjectID, groupID=groupID, condition=condition)


def test_dataset_len() -> None:
    items = [_make_data(), _make_data()]
    ds = Dataset(items)
    assert len(ds) == 2


def test_dataset_getitem_int() -> None:
    d0 = _make_data(subjectID="S1")
    ds = Dataset([d0, _make_data()])
    assert ds[0] is d0


def test_dataset_getitem_slice() -> None:
    items = [_make_data() for _ in range(4)]
    ds = Dataset(items)
    sliced = ds[1:3]
    assert isinstance(sliced, Dataset)
    assert len(sliced) == 2


def test_dataset_iter() -> None:
    items = [_make_data(), _make_data()]
    ds = Dataset(items)
    assert list(ds) == items


def test_dataset_contains() -> None:
    d = _make_data()
    ds = Dataset([d])
    assert d in ds


def test_dataset_add() -> None:
    ds1 = Dataset([_make_data(subjectID="S1")])
    ds2 = Dataset([_make_data(subjectID="S2")])
    combined = ds1 + ds2
    assert isinstance(combined, Dataset)
    assert len(combined) == 2


def test_dataset_repr_nonempty() -> None:
    ds = Dataset([_make_data(), _make_data()])
    r = repr(ds)
    assert "Dataset" in r
    assert "2" in r


def test_dataset_repr_empty() -> None:
    ds = Dataset([])
    r = repr(ds)
    assert "Dataset" in r
    assert "0" in r


def test_dataset_str_shows_metadata() -> None:
    items = [
        _make_data(subjectID="S1", groupID="A", condition="rest"),
        _make_data(subjectID="S2", groupID="B", condition="task"),
    ]
    ds = Dataset(items)
    s = str(ds)
    assert "S1" in s
    assert "S2" in s
    assert "A" in s
    assert "B" in s


def test_dataset_describe_prints(capsys: pytest.CaptureFixture[str]) -> None:
    ds = Dataset([_make_data(subjectID="S1")])
    ds.describe()
    out = capsys.readouterr().out
    assert "S1" in out


def test_dataset_empty_is_valid() -> None:
    ds = Dataset([])
    assert len(ds) == 0
    assert list(ds) == []


def test_dataset_immutable_tuple_storage() -> None:
    items = [_make_data()]
    ds = Dataset(items)
    items.append(_make_data())  # mutating original list should not affect Dataset
    assert len(ds) == 1


def test_dataset_filter_by_subject() -> None:
    ds = Dataset(
        [
            _make_data(subjectID="S1", groupID="A"),
            _make_data(subjectID="S2", groupID="A"),
            _make_data(subjectID="S1", groupID="B"),
        ]
    )
    result = ds.filter(subjectID="S1")
    assert len(result) == 2
    assert all(d.subjectID == "S1" for d in result)


def test_dataset_filter_by_group() -> None:
    ds = Dataset([_make_data(groupID="A"), _make_data(groupID="B"), _make_data(groupID="A")])
    result = ds.filter(groupID="A")
    assert len(result) == 2


def test_dataset_filter_by_condition() -> None:
    ds = Dataset([_make_data(condition="rest"), _make_data(condition="task")])
    result = ds.filter(condition="rest")
    assert len(result) == 1
    assert result[0].condition == "rest"


def test_dataset_filter_combined() -> None:
    ds = Dataset(
        [
            _make_data(subjectID="S1", groupID="A", condition="rest"),
            _make_data(subjectID="S1", groupID="A", condition="task"),
            _make_data(subjectID="S2", groupID="A", condition="rest"),
        ]
    )
    result = ds.filter(subjectID="S1", condition="rest")
    assert len(result) == 1


def test_dataset_filter_no_match_returns_empty() -> None:
    ds = Dataset([_make_data(subjectID="S1")])
    result = ds.filter(subjectID="S99")
    assert isinstance(result, Dataset)
    assert len(result) == 0


def test_dataset_groupby_groupid() -> None:
    ds = Dataset([_make_data(groupID="A"), _make_data(groupID="B"), _make_data(groupID="A")])
    groups = ds.groupby("groupID")
    assert set(groups.keys()) == {"A", "B"}
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 1


def test_dataset_groupby_none_goes_to_none_key() -> None:
    ds = Dataset([_make_data(groupID="A"), _make_data(groupID=None)])
    groups = ds.groupby("groupID")
    assert "None" in groups
    assert len(groups["None"]) == 1


def test_dataset_groupby_returns_dataset_values() -> None:
    ds = Dataset([_make_data(groupID="A"), _make_data(groupID="B")])
    groups = ds.groupby("groupID")
    assert all(isinstance(v, Dataset) for v in groups.values())


def test_dataset_add_non_dataset_returns_not_implemented() -> None:
    ds = Dataset([_make_data()])
    assert ds.__add__("not a dataset") is NotImplemented


def test_dataset_groupby_invalid_attr_raises() -> None:
    ds = Dataset([_make_data()])
    with pytest.raises(ValueError, match="attr must be one of"):
        ds.groupby("history")  # type: ignore[arg-type]


def test_dataset_repr_mixed_types() -> None:
    import xarray as xr

    from cobrabox.data import SignalData

    arr = np.zeros((10,))
    da = xr.DataArray(arr, dims=["time"], coords={"time": arr})
    sd = SignalData(da)
    d = _make_data()
    ds = Dataset([sd, d])
    assert "Data" in repr(ds)  # falls back to "Data" for mixed types


def test_dataset_importable_from_cobrabox() -> None:
    import numpy as np
    import xarray as xr

    import cobrabox as cb
    from cobrabox.data import Data

    assert hasattr(cb, "Dataset")
    da = xr.DataArray(np.zeros((3, 10)), dims=["space", "time"])
    d = Data(da)
    ds = cb.Dataset([d])
    assert len(ds) == 1
