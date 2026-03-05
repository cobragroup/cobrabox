# Dataset class design

**Date:** 2026-03-05
**Status:** Approved

## Problem

`list[Data]` / `list[SignalData]` is used throughout the codebase (dataset loaders, public API) with no helpers for filtering, grouping, or inspection. A typed collection class gives users ergonomic access to metadata-based operations while staying consistent with the immutable `Data` philosophy.

## Decision

Introduce `Dataset[T]` — a generic, immutable sequence of `Data` objects — in a new file `src/cobrabox/dataset.py`.

## Design

### Class signature

```python
from __future__ import annotations

from typing import Generic, Iterator, Literal, Sequence, TypeVar, final, overload

from .data import Data

T = TypeVar("T", bound=Data)

@final
class Dataset(Generic[T]):
    """Immutable, typed collection of Data objects."""

    def __init__(self, items: Sequence[T]) -> None: ...
```

Internally stores `tuple[T, ...]`.

### Sequence protocol

| Method | Behaviour |
|--------|-----------|
| `__len__()` | number of items |
| `__getitem__(int)` | returns `T` |
| `__getitem__(slice)` | returns `Dataset[T]` |
| `__iter__()` | iterates over items |
| `__contains__(item)` | identity check |
| `__add__(other)` | concatenates two `Dataset[T]` → `Dataset[T]` |

### String representation

**`__repr__`** — compact, unambiguous:
```
Dataset(5 × SignalData)
```
Uses the concrete type name of items[0] (or `Data` if empty).

**`__str__`** — human-readable multi-line summary:
```
Dataset  5 items  [SignalData]
  subjectIDs : S1, S2, S3, None, None
  groupIDs   : A, A, B, B, B
  conditions : rest, task, rest, task, rest
  shapes     : (64, 1000) × 4, (64, 500) × 1
```
Repeated shapes are collapsed to `shape × count`.

**`describe()`** — convenience: `print(str(self))`, returns `None`.

### Filtering

```python
def filter(
    self,
    *,
    subjectID: str | None = None,
    groupID: str | None = None,
    condition: str | None = None,
) -> Dataset[T]:
```

- Keyword-only arguments; AND semantics (all specified must match).
- Returns a new `Dataset[T]`; empty `Dataset` if nothing matches (no error).
- `None` argument means "don't filter on this field".

### Grouping

```python
def groupby(
    self,
    attr: Literal["subjectID", "groupID", "condition"],
) -> dict[str, Dataset[T]]:
```

- Groups items by the given metadata attribute.
- Items where the attribute is `None` go into the `"None"` key.
- Returns `dict[str, Dataset[T]]` (order preserved, insertion order).

### Immutability

`Dataset` is immutable. All operations (`filter`, `groupby`, `__add__`) return new
instances. No `append`/`extend`/`remove` methods. Construction from a plain `list`
via the constructor is the "builder" pattern.

## Files changed

| File | Change |
|------|--------|
| `src/cobrabox/dataset.py` | **new** — `Dataset[T]` class |
| `src/cobrabox/dataset_loader.py` | return `Dataset[SignalData]` / `Dataset[Data]` instead of `list` |
| `src/cobrabox/datasets.py` | return type updated to `Dataset[Data]` |
| `src/cobrabox/__init__.py` | add `Dataset` to imports and `__all__` |
| `tests/test_dataset_class.py` | **new** — unit tests for `Dataset` |
| `tests/test_datasets.py` | update assertions (type checks) |
| `tests/test_dataset_loader.py` | update return-type assertions |
