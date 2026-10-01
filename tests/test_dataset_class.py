"""Tests for Dataset[T] collection class."""

from __future__ import annotations

import numpy as np
import pytest

from cobrabox.data import METADATA_FIELDS, Data
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
    with pytest.raises(ValueError, match="Unknown field"):
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


# ----------------------------------------------------------------------
# Label-based access
# ----------------------------------------------------------------------


def _labelled(subjectID: str, condition: str | None = None, **extra: object) -> Data:
    import xarray as xr

    da = xr.DataArray(np.zeros((3, 10)), dims=["space", "time"])
    return Data(da, subjectID=subjectID, condition=condition, extra=extra or None)


def test_keys_lists_subject_labels() -> None:
    ds = Dataset([_labelled("milan"), _labelled("paris")])
    assert ds.keys() == ("milan", "paris")


def test_keys_are_composite_when_condition_present() -> None:
    ds = Dataset([_labelled("milan", "pre"), _labelled("milan", "rest")])
    assert ds.keys() == ("milan/pre", "milan/rest")


def test_keys_empty_when_items_carry_no_metadata() -> None:
    assert Dataset([_make_data(), _make_data()]).keys() == ()


def test_getitem_by_label_returns_the_item() -> None:
    milan = _labelled("milan")
    ds = Dataset([_labelled("paris"), milan])
    assert ds["milan"] is milan


def test_getitem_by_label_resolves_same_item_as_position() -> None:
    ds = Dataset([_labelled("paris"), _labelled("milan")])
    assert ds[1] is ds["milan"]


def test_getitem_by_duplicate_label_returns_all_matches() -> None:
    ds = Dataset([_labelled("ID01"), _labelled("ID01"), _labelled("ID02")])
    matches = ds["ID01"]
    assert isinstance(matches, Dataset)
    assert len(matches) == 2


def test_getitem_unknown_label_lists_available_ones() -> None:
    ds = Dataset([_labelled("milan"), _labelled("paris")])
    with pytest.raises(KeyError, match="milan, paris"):
        ds["milna"]


def test_getitem_unknown_label_explains_when_dataset_is_unlabelled() -> None:
    ds = Dataset([_make_data()])
    with pytest.raises(KeyError, match="no labels"):
        ds["milan"]


def test_getitem_int_still_works_alongside_labels() -> None:
    ds = Dataset([_labelled("milan"), _labelled("paris")])
    assert ds[0].subjectID == "milan"
    assert isinstance(ds[0:2], Dataset)


def test_integer_like_labels_do_not_collide_with_positions() -> None:
    ds = Dataset([_labelled("2"), _labelled("0")])
    assert ds[0].subjectID == "2"  # positional
    assert ds["0"].subjectID == "0"  # labelled


# ----------------------------------------------------------------------
# Discovery helpers
# ----------------------------------------------------------------------


def test_fields_includes_metadata_and_extra_keys() -> None:
    ds = Dataset([_labelled("milan", ilae=2)])
    assert ds.fields() == (*METADATA_FIELDS, "ilae")


def test_fields_accepts_every_metadata_field_data_defines() -> None:
    """Dataset's filterable fields stay in step with Data's — one list, not two."""
    assert set(METADATA_FIELDS) <= set(Dataset([_labelled("milan")]).fields())


def test_unique_returns_distinct_values_in_order() -> None:
    ds = Dataset([_labelled("a", "pre"), _labelled("b", "rest"), _labelled("c", "pre")])
    assert ds.unique("condition") == ("pre", "rest")


def test_unique_handles_unhashable_extra_values() -> None:
    ds = Dataset([_labelled("a", channels=["c1", "c2"]), _labelled("b", channels=["c1", "c2"])])
    assert ds.unique("channels") == ("['c1', 'c2']",)


def test_unique_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown field"):
        Dataset([_labelled("a")]).unique("nope")


def test_describe_rows_mention_labels_and_filterable_fields() -> None:
    ds = Dataset([_labelled("milan", "rest", ilae=2)])
    text = str(ds)
    assert "milan/rest" in text
    assert "ilae" in text


def test_describe_says_positional_only_when_unlabelled() -> None:
    text = str(Dataset([_make_data()]))
    assert "positional access only" in text
    assert "not reachable by filter on subjectID, groupID, condition" in text


def test_describe_only_names_identity_fields_that_are_actually_unset() -> None:
    # groupID is set, so it stays filterable and must not be listed as unreachable.
    ds = Dataset([_make_data(groupID="control")])
    assert "not reachable by filter on subjectID, condition" in str(ds)


def test_describe_elides_the_tail_of_a_long_label_list() -> None:
    ds = Dataset([_labelled(f"S{i:02d}") for i in range(25)])
    text = str(ds)
    assert "S19" in text  # 20th label, the last one shown
    assert "S20" not in text
    assert "(+5 more)" in text


# ----------------------------------------------------------------------
# filter / one / groupby
# ----------------------------------------------------------------------


def test_filter_accepts_a_list_of_values() -> None:
    ds = Dataset([_labelled("a"), _labelled("b"), _labelled("c")])
    assert len(ds.filter(subjectID=["a", "c"])) == 2


def test_filter_matches_against_extra() -> None:
    ds = Dataset([_labelled("a", ilae=2), _labelled("b", ilae=4)])
    assert ds.filter(ilae=2).keys() == ("a",)


def test_filter_rejects_unknown_keyword() -> None:
    ds = Dataset([_labelled("a")])
    with pytest.raises(ValueError, match="Unknown field"):
        ds.filter(nonsense="x")


def test_filter_on_empty_dataset_returns_empty_rather_than_raising() -> None:
    assert len(Dataset([]).filter(anything="x")) == 0


def test_one_returns_the_single_match() -> None:
    ds = Dataset([_labelled("milan", "pre"), _labelled("milan", "rest")])
    assert ds.one(subjectID="milan", condition="rest").condition == "rest"


def test_one_raises_when_nothing_matches() -> None:
    ds = Dataset([_labelled("milan")])
    with pytest.raises(ValueError, match="No item matches"):
        ds.one(subjectID="paris")


def test_one_raises_when_several_match() -> None:
    ds = Dataset([_labelled("milan", "pre"), _labelled("milan", "rest")])
    with pytest.raises(ValueError, match="found 2"):
        ds.one(subjectID="milan")


def test_groupby_single_attr_keeps_string_keys() -> None:
    ds = Dataset([_labelled("a", "pre"), _labelled("b", "pre")])
    groups = ds.groupby("condition")
    assert set(groups) == {"pre"}
    assert len(groups["pre"]) == 2


def test_groupby_multiple_attrs_uses_tuple_keys() -> None:
    ds = Dataset([_labelled("milan", "pre"), _labelled("milan", "rest")])
    groups = ds.groupby("subjectID", "condition")
    assert set(groups) == {("milan", "pre"), ("milan", "rest")}


def test_groupby_works_on_extra_keys() -> None:
    ds = Dataset([_labelled("a", ilae=2), _labelled("b", ilae=2), _labelled("c", ilae=4)])
    assert len(ds.groupby("ilae")["2"]) == 2


def test_groupby_requires_at_least_one_attr() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Dataset([_labelled("a")]).groupby()
