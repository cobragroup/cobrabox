# Dataset Class Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `Dataset[T]` — a generic, immutable sequence of `Data` objects — with `filter()`, `groupby()`, and `describe()` helpers, and wire it into the existing dataset loaders.

**Architecture:** `Dataset[T]` wraps a `tuple[T, ...]` internally and conforms to the sequence protocol (`__len__`, `__getitem__`, `__iter__`). All operations return new instances; no mutation. Existing `list[Data]` return types in loaders are replaced with `Dataset[...]`. The existing test suite uses `len()` and indexing, so tests mostly pass without changes.

**Tech Stack:** Python stdlib only (`collections.abc`, `typing`); xarray via existing `Data` attrs.

---

### Task 1: Create `Dataset[T]` with sequence protocol and repr

**Files:**
- Create: `src/cobrabox/dataset.py`
- Test: `tests/test_dataset_class.py`

**Step 1: Write failing tests**

Create `tests/test_dataset_class.py`:

```python
"""Tests for Dataset[T] collection class."""
from __future__ import annotations

import numpy as np
import pytest

from cobrabox.data import Data, SignalData
from cobrabox.dataset import Dataset


def _make_data(subjectID=None, groupID=None, condition=None) -> Data:
    import xarray as xr
    arr = np.zeros((3, 10))
    da = xr.DataArray(arr, dims=["space", "time"])
    return Data(da, subjectID=subjectID, groupID=groupID, condition=condition)


def test_dataset_len():
    items = [_make_data(), _make_data()]
    ds = Dataset(items)
    assert len(ds) == 2


def test_dataset_getitem_int():
    d0 = _make_data(subjectID="S1")
    ds = Dataset([d0, _make_data()])
    assert ds[0] is d0


def test_dataset_getitem_slice():
    items = [_make_data() for _ in range(4)]
    ds = Dataset(items)
    sliced = ds[1:3]
    assert isinstance(sliced, Dataset)
    assert len(sliced) == 2


def test_dataset_iter():
    items = [_make_data(), _make_data()]
    ds = Dataset(items)
    assert list(ds) == items


def test_dataset_contains():
    d = _make_data()
    ds = Dataset([d])
    assert d in ds


def test_dataset_add():
    ds1 = Dataset([_make_data(subjectID="S1")])
    ds2 = Dataset([_make_data(subjectID="S2")])
    combined = ds1 + ds2
    assert isinstance(combined, Dataset)
    assert len(combined) == 2


def test_dataset_repr_nonempty():
    ds = Dataset([_make_data(), _make_data()])
    r = repr(ds)
    assert "Dataset" in r
    assert "2" in r


def test_dataset_repr_empty():
    ds = Dataset([])
    r = repr(ds)
    assert "Dataset" in r
    assert "0" in r


def test_dataset_str_shows_metadata():
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


def test_dataset_describe_prints(capsys):
    ds = Dataset([_make_data(subjectID="S1")])
    ds.describe()
    out = capsys.readouterr().out
    assert "S1" in out


def test_dataset_empty_is_valid():
    ds = Dataset([])
    assert len(ds) == 0
    assert list(ds) == []


def test_dataset_immutable_tuple_storage():
    items = [_make_data()]
    ds = Dataset(items)
    items.append(_make_data())  # mutating original list should not affect Dataset
    assert len(ds) == 1
```

**Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_dataset_class.py -v 2>&1 | head -30
```

Expected: `ImportError` — `cobrabox.dataset` does not exist yet.

**Step 3: Implement `Dataset[T]`**

Create `src/cobrabox/dataset.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, final, overload

from .data import Data

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T", bound=Data)


@final
class Dataset(Generic[T]):
    """Immutable, typed collection of Data objects.

    Behaves like a read-only sequence: supports indexing, iteration, and len().
    All filtering and combination operations return new Dataset instances.

    Args:
        items: Sequence of Data objects (list, tuple, or another Dataset).

    Example:
        >>> ds = cb.dataset("dummy_chain")
        >>> ds[0]                             # first item
        >>> ds.filter(groupID="A")            # returns new Dataset
        >>> ds.groupby("subjectID")           # returns dict[str, Dataset]
        >>> ds.describe()                     # prints summary
    """

    __slots__ = ("_items",)

    def __init__(self, items: Sequence[T]) -> None:
        self._items: tuple[T, ...] = tuple(items)

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Dataset[T]: ...

    def __getitem__(self, index: int | slice) -> T | Dataset[T]:
        if isinstance(index, slice):
            return Dataset(self._items[index])
        return self._items[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __add__(self, other: Dataset[T]) -> Dataset[T]:
        if not isinstance(other, Dataset):
            return NotImplemented
        return Dataset(self._items + other._items)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def _item_type_name(self) -> str:
        if not self._items:
            return "Data"
        return type(self._items[0]).__name__

    def __repr__(self) -> str:
        return f"Dataset({len(self._items)} \u00d7 {self._item_type_name()})"

    def __str__(self) -> str:
        n = len(self._items)
        type_name = self._item_type_name()
        lines = [f"Dataset  {n} items  [{type_name}]"]

        def _fmt(values: list) -> str:
            return ", ".join(str(v) for v in values)

        subjects = [item.subjectID for item in self._items]
        groups = [item.groupID for item in self._items]
        conditions = [item.condition for item in self._items]

        lines.append(f"  subjectIDs : {_fmt(subjects)}")
        lines.append(f"  groupIDs   : {_fmt(groups)}")
        lines.append(f"  conditions : {_fmt(conditions)}")

        # Collapse repeated shapes
        shapes = [tuple(item.data.shape) for item in self._items]
        shape_counts: dict[tuple, int] = {}
        for s in shapes:
            shape_counts[s] = shape_counts.get(s, 0) + 1
        shape_str = ", ".join(
            f"{s} \u00d7 {c}" if c > 1 else str(s)
            for s, c in shape_counts.items()
        )
        lines.append(f"  shapes     : {shape_str}")

        return "\n".join(lines)

    def describe(self) -> None:
        """Print a human-readable summary of this Dataset."""
        print(str(self))

    # ------------------------------------------------------------------
    # Filtering and grouping
    # ------------------------------------------------------------------

    def filter(
        self,
        *,
        subjectID: str | None = None,
        groupID: str | None = None,
        condition: str | None = None,
    ) -> Dataset[T]:
        """Return a new Dataset containing only items matching all given criteria.

        Args:
            subjectID: Keep items where item.subjectID == this value.
            groupID: Keep items where item.groupID == this value.
            condition: Keep items where item.condition == this value.

        Returns:
            New Dataset with matching items. Empty Dataset if none match.

        Example:
            >>> ds.filter(groupID="control")
            >>> ds.filter(subjectID="S01", condition="rest")
        """
        result = list(self._items)
        if subjectID is not None:
            result = [d for d in result if d.subjectID == subjectID]
        if groupID is not None:
            result = [d for d in result if d.groupID == groupID]
        if condition is not None:
            result = [d for d in result if d.condition == condition]
        return Dataset(result)

    def groupby(
        self,
        attr: Literal["subjectID", "groupID", "condition"],
    ) -> dict[str, Dataset[T]]:
        """Group items by a metadata attribute.

        Args:
            attr: One of "subjectID", "groupID", or "condition".

        Returns:
            Dict mapping attribute value (as string) to a Dataset of matching items.
            Items with None for the attribute are grouped under the key "None".

        Example:
            >>> by_group = ds.groupby("groupID")
            >>> by_group["control"]
        """
        groups: dict[str, list[T]] = {}
        for item in self._items:
            key = str(getattr(item, attr))
            groups.setdefault(key, []).append(item)
        return {k: Dataset(v) for k, v in groups.items()}
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_dataset_class.py -v
```

Expected: All pass.

**Step 5: Commit**

```bash
git add src/cobrabox/dataset.py tests/test_dataset_class.py
git commit -m "feat: add Dataset[T] generic immutable collection with filter/groupby/describe"
```

---

### Task 2: Export `Dataset` from public API

**Files:**
- Modify: `src/cobrabox/__init__.py`

**Step 1: Write failing test**

Add to `tests/test_dataset_class.py` (append at bottom):

```python
def test_dataset_importable_from_cobrabox():
    import cobrabox as cb
    assert hasattr(cb, "Dataset")
    ds = cb.Dataset([_make_data()])
    assert len(ds) == 1
```

**Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_dataset_class.py::test_dataset_importable_from_cobrabox -v
```

Expected: `AttributeError: module 'cobrabox' has no attribute 'Dataset'`

**Step 3: Add to `__init__.py`**

In `src/cobrabox/__init__.py`, add the import and `__all__` entry:

```python
# existing imports ...
from .dataset import Dataset
```

And add `"Dataset"` to `__all__`.

**Step 4: Run test**

```bash
uv run pytest tests/test_dataset_class.py::test_dataset_importable_from_cobrabox -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/cobrabox/__init__.py tests/test_dataset_class.py
git commit -m "feat: export Dataset from cobrabox public API"
```

---

### Task 3: Update loaders to return `Dataset[...]`

**Files:**
- Modify: `src/cobrabox/dataset_loader.py`
- Modify: `src/cobrabox/datasets.py`

**Step 1: Check existing tests still pass (baseline)**

```bash
uv run pytest tests/test_dataset_loader.py tests/test_datasets.py -v
```

All should pass (they use `len()` and indexing, which `Dataset` supports).

**Step 2: Update `dataset_loader.py`**

At the top of `src/cobrabox/dataset_loader.py`, add the import:

```python
from .dataset import Dataset
```

Change the three loader function signatures and return statements:

**`load_structured_dummy`** — change signature:
```python
def load_structured_dummy(identifier: str, repo_root: Path | None = None) -> Dataset[SignalData]:
```
Change local variable:
```python
    datasets: list[SignalData] = []
```
→
```python
    items: list[SignalData] = []
```
And every `datasets.append(...)` → `items.append(...)`.

Return at end:
```python
    return Dataset(items)
```
(Also update the `if not datasets` guard → `if not items`.)

**`load_noise_dummy`** — same pattern:
```python
def load_noise_dummy(identifier: str = "dummy_noise", repo_root: Path | None = None) -> Dataset[SignalData]:
```
```python
    items: list[SignalData] = []
    # ... items.append(...) ...
    if not items:
        raise ValueError(...)
    return Dataset(items)
```

**`load_realistic_swiss`** — same pattern:
```python
def load_realistic_swiss(
    identifier: str = "realistic_swiss", repo_root: Path | None = None
) -> Dataset[SignalData]:
```
```python
    items: list[SignalData] = []
    # ... items.append(...) ...
    if not items:
        raise ValueError(...)
    return Dataset(items)
```

**Step 3: Update `datasets.py` return type**

```python
from .dataset import Dataset

def dataset(identifier: str) -> Dataset[Data]:
    """Load one logical dataset identifier as a Dataset of Data parts."""
    if identifier in {"dummy_chain", "dummy_random", "dummy_star"}:
        return load_structured_dummy(identifier)
    if identifier == "dummy_noise":
        return load_noise_dummy(identifier)
    if identifier == "realistic_swiss":
        return load_realistic_swiss(identifier)
    raise ValueError(f"Unknown dataset identifier: {identifier}")
```

**Step 4: Run all loader and datasets tests**

```bash
uv run pytest tests/test_dataset_loader.py tests/test_datasets.py -v
```

Expected: All pass. The existing tests use `len(out)` and `out[0]` which work on `Dataset`.

**Step 5: Run full suite**

```bash
uv run pytest
```

Expected: All pass, coverage ≥ 95%.

**Step 6: Commit**

```bash
git add src/cobrabox/dataset_loader.py src/cobrabox/datasets.py
git commit -m "feat: update dataset loaders to return Dataset[T] instead of list"
```

---

### Task 4: Final check — lint and full test run

**Step 1: Lint**

```bash
uvx ruff check src/cobrabox/dataset.py
uvx ruff format src/cobrabox/dataset.py
```

Fix any warnings. Common things to watch:
- `Literal` import must be from `typing` (Python 3.11 compat)
- `@final` and `@overload` from `typing`
- Remove unused `TYPE_CHECKING` block if nothing is inside it at runtime

**Step 2: Full test suite with coverage**

```bash
uv run pytest --cov-fail-under=95 -v
```

Expected: All pass, ≥ 95% coverage.

**Step 3: Commit lint fixes if any**

```bash
git add -u
git commit -m "style: fix ruff warnings in dataset.py"
```
