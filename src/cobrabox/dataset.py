from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Generic, Literal, TypeVar, final, overload

from .data import Data

T = TypeVar("T", bound=Data)


@final
class Dataset(Generic[T]):
    """Immutable, typed collection of Data objects.

    Behaves like a read-only sequence: supports indexing, iteration, and len().
    All filtering and combination operations return new Dataset instances.

    Args:
        items: Sequence of Data objects (list, tuple, or another Dataset).

    Example:
        >>> ds = cb.load_dataset("dummy_chain")
        >>> ds[0]                             # first item
        >>> ds.filter(groupID="A")            # returns new Dataset
        >>> ds.groupby("subjectID")           # returns dict[str, Dataset]
        >>> ds.describe()                     # prints summary
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[T]) -> None:
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
        types = {type(item).__name__ for item in self._items}
        return types.pop() if len(types) == 1 else "Data"

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
            f"{s} \u00d7 {c}" if c > 1 else str(s) for s, c in shape_counts.items()
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

    _GROUPBY_ATTRS: frozenset[str] = frozenset({"subjectID", "groupID", "condition"})

    def groupby(self, attr: Literal["subjectID", "groupID", "condition"]) -> dict[str, Dataset[T]]:
        """Group items by a metadata attribute.

        Args:
            attr: One of "subjectID", "groupID", or "condition".

        Returns:
            Dict mapping attribute value (as string) to a Dataset of matching items.
            Items with None for the attribute are grouped under the key "None".

        Raises:
            ValueError: If attr is not one of the valid metadata attributes.

        Example:
            >>> by_group = ds.groupby("groupID")
            >>> by_group["control"]
        """
        if attr not in self._GROUPBY_ATTRS:
            raise ValueError(f"attr must be one of {sorted(self._GROUPBY_ATTRS)!r}, got {attr!r}")
        groups: dict[str, list[T]] = {}
        for item in self._items:
            key = str(getattr(item, attr))
            groups.setdefault(key, []).append(item)
        return {k: Dataset(v) for k, v in groups.items()}
