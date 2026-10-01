from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final, overload

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult

from .data import METADATA_FIELDS as _METADATA_FIELDS
from .data import Data

T = TypeVar("T", bound=Data)

#: Attributes combined (in order, skipping None) to form an item's label.
#: Ordered widest-to-narrowest so a label reads as a path — "sub-01/rest/02" — and a
#: prefix of it names a meaningful group of items. ``groupID`` is excluded: it
#: classifies items rather than identifying them.
_LABEL_FIELDS: tuple[str, ...] = ("subjectID", "condition", "runID")

#: Maximum number of values listed in an error message or summary before truncating.
_MAX_LISTED = 20


def _label_of(item: Data) -> str | None:
    """Build an item's label from its identifying metadata.

    Joins the non-None values of :data:`_LABEL_FIELDS` with ``/``, so an item with
    only a subjectID labels as ``"milan"`` and one that also has a condition labels
    as ``"milan/rest"``. Returns None when the item carries neither.
    """
    parts = [value for field in _LABEL_FIELDS if (value := getattr(item, field)) is not None]
    return "/".join(parts) if parts else None


def _value_of(item: Data, attr: str) -> Any:
    """Read *attr* from an item's metadata, falling back to its ``extra`` dict."""
    if attr in _METADATA_FIELDS:
        return getattr(item, attr)
    return item.extra.get(attr)


def _matches(value: Any, criterion: Any) -> bool:
    """Test a value against a criterion: a single value, or a collection of allowed values."""
    if isinstance(criterion, (list, tuple, set, frozenset)):
        return value in criterion
    return value == criterion


def _truncate(values: Sequence[Any]) -> str:
    """Format a sequence for display, eliding the tail when it is long."""
    shown = [str(v) for v in values[:_MAX_LISTED]]
    if len(values) > _MAX_LISTED:
        shown.append(f"... (+{len(values) - _MAX_LISTED} more)")
    return ", ".join(shown)


@final
class Dataset(Generic[T]):
    """Immutable, typed collection of Data objects.

    Behaves like a read-only sequence — indexing, iteration, and len() — and
    additionally supports label-based lookup for items that carry identifying
    metadata. All filtering and combination operations return new Dataset instances.

    Items are stored positionally; labels are a derived index over that storage, so
    duplicate labels are preserved (they resolve to every matching item) and items
    with no metadata stay reachable by position.

    Args:
        items: Sequence of Data objects (list, tuple, or another Dataset).

    Example:
        >>> ds = cb.load_dataset("dummy_chain")
        >>> ds[0]                             # first item, by position
        >>> ds.keys()                         # labels available for lookup
        >>> ds["milan/rest"]                  # by label
        >>> ds.filter(groupID="A")            # returns new Dataset
        >>> ds.groupby("subjectID")           # returns dict[str, Dataset]
        >>> ds.describe()                     # prints summary
    """

    __slots__ = ("_index", "_items")

    def __init__(self, items: Iterable[T]) -> None:
        self._items: tuple[T, ...] = tuple(items)
        index: dict[str, tuple[int, ...]] = {}
        for position, item in enumerate(self._items):
            if (label := _label_of(item)) is not None:
                index[label] = (*index.get(label, ()), position)
        self._index: dict[str, tuple[int, ...]] = index

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Dataset[T]: ...
    @overload
    def __getitem__(self, index: str) -> T | Dataset[T]: ...

    def __getitem__(self, index: int | slice | str) -> T | Dataset[T]:
        """Look up by position (int), range (slice), or label (str).

        A label matching exactly one item returns that item; a label shared by
        several items returns a Dataset of all of them. A label that names only the
        leading part of other labels matches all of them, so ``ds["sub-01"]`` reaches
        every recording of that subject even when labels run to ``"sub-01/rest/02"``.
        """
        if isinstance(index, str):
            return self._by_label(index)
        if isinstance(index, slice):
            return Dataset(self._items[index])
        return self._items[index]

    def _by_label(self, label: str) -> T | Dataset[T]:
        positions = self._index.get(label)
        if positions is None:
            # Fall back to a prefix match on segment boundaries, so a subject can be
            # named even when labels carry a condition or run below it: ds["sub-01"]
            # reaches all of sub-01's recordings. "sub-0" matches nothing.
            positions = tuple(
                position
                for key, group in self._index.items()
                if key.startswith(f"{label}/")
                for position in group
            )
        if not positions:
            if not self._index:
                raise KeyError(
                    f"{label!r}: this Dataset has no labels — its items carry no "
                    f"{' or '.join(_LABEL_FIELDS)} metadata, so only positional "
                    "access (ds[0]) works."
                )
            raise KeyError(
                f"{label!r} is not a label in this Dataset. Available: {_truncate(self.keys())}"
            )
        if len(positions) == 1:
            return self._items[positions[0]]
        return Dataset(self._items[position] for position in sorted(positions))

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __add__(self, other: Dataset[T]) -> Dataset[T]:
        if not isinstance(other, Dataset):
            return NotImplemented
        return Dataset(self._items + other._items)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def keys(self) -> tuple[str, ...]:
        """Return the labels available for lookup, in item order.

        A label joins an item's subjectID, condition and runID with ``/``, skipping
        whichever are unset — so ``"sub-01"``, ``"sub-01/rest"`` and
        ``"sub-01/rest/02"`` are all possible shapes. Items carrying none of them
        contribute no label and are reachable only by position, so this may be
        shorter than ``len(ds)``. A label shared by several items appears once.

        Lookup also accepts any leading part of a label, so a subject stays
        addressable as ``ds["sub-01"]`` even when the labels listed here are longer.

        Example:
            >>> ds.keys()
            ('milan', 'paris', 'prague')
        """
        return tuple(self._index)

    def fields(self) -> tuple[str, ...]:
        """Return every attribute name accepted by :meth:`filter` and :meth:`groupby`.

        This is the three standard metadata fields plus whatever keys the items
        carry in their ``extra`` dicts, which vary by dataset.

        Example:
            >>> ds.fields()
            ('subjectID', 'groupID', 'condition', 'ilae', 'resected_zone')
        """
        return (*_METADATA_FIELDS, *self._extra_fields())

    def _extra_fields(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for item in self._items:
            for key in item.extra:
                seen.setdefault(key, None)
        return tuple(seen)

    def unique(self, attr: str) -> tuple[Any, ...]:
        """Return the distinct values of *attr* across items, in first-seen order.

        Useful for discovering what a filter could match before writing it.

        Args:
            attr: Any name from :meth:`fields` — a metadata field, or a key from
                the items' ``extra`` dicts.

        Returns:
            Tuple of distinct values, including None if any item lacks the
            attribute. Unhashable values (e.g. a list in ``extra``) are reported
            by their repr.

        Raises:
            ValueError: If attr is not a known field.

        Example:
            >>> ds.unique("condition")
            ('pre', 'post', 'rest')
        """
        self._validate_fields((attr,))
        seen: dict[Any, None] = {}
        for item in self._items:
            value = _value_of(item, attr)
            try:
                seen.setdefault(value, None)
            except TypeError:  # unhashable, e.g. a list stored in extra
                seen.setdefault(repr(value), None)
        return tuple(seen)

    def _validate_fields(self, attrs: Iterable[str]) -> None:
        """Raise ValueError for any attr that is neither metadata nor a present extra key."""
        if not self._items:
            return
        known = set(self.fields())
        unknown = [attr for attr in attrs if attr not in known]
        if unknown:
            raise ValueError(
                f"Unknown field(s) {unknown!r}. Available: {list(self.fields())!r}. "
                "Call ds.fields() to list them."
            )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def _item_type_name(self) -> str:
        if not self._items:
            return "Data"
        types = {type(item).__name__ for item in self._items}
        return types.pop() if len(types) == 1 else "Data"

    def _summary_rows(self) -> list[tuple[str, str]]:
        """Build the (name, value) rows shared by __str__ and the rich rendering."""
        rows = [
            ("subjectIDs", _truncate([item.subjectID for item in self._items])),
            ("groupIDs", _truncate([item.groupID for item in self._items])),
            ("conditions", _truncate([item.condition for item in self._items])),
        ]

        shape_counts: dict[tuple, int] = {}
        for item in self._items:
            shape = tuple(item.data.shape)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        rows.append(
            (
                "shapes",
                ", ".join(f"{s} \u00d7 {c}" if c > 1 else str(s) for s, c in shape_counts.items()),
            )
        )

        labels = self.keys()
        if labels:
            rows.append(("labels", _truncate(labels)))
        else:
            # Name the fields that are None on every item, rather than claiming the
            # items are unfilterable outright — they may still carry usable extras.
            unset = [
                field
                for field in _METADATA_FIELDS
                if all(_value_of(item, field) is None for item in self._items)
            ]
            rows.append(
                (
                    "labels",
                    "none — positional access only, not reachable by filter on " + ", ".join(unset),
                )
            )
        rows.append(("filter on", ", ".join(self.fields())))
        return rows

    def __repr__(self) -> str:
        return f"Dataset({len(self._items)} \u00d7 {self._item_type_name()})"

    def __str__(self) -> str:
        lines = [f"Dataset  {len(self._items)} items  [{self._item_type_name()}]"]
        lines.extend(f"  {name:<11}: {value}" for name, value in self._summary_rows())
        return "\n".join(lines)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        from rich.panel import Panel
        from rich.table import Table

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        for name, value in self._summary_rows():
            table.add_row(name, value)

        yield Panel(
            table, title=f"[bold]Dataset[/bold]  {len(self._items)} \u00d7 {self._item_type_name()}"
        )

    def describe(self) -> None:
        """Print a human-readable summary of this Dataset."""
        from rich.console import Console

        Console().print(self)

    # ------------------------------------------------------------------
    # Filtering and grouping
    # ------------------------------------------------------------------

    def filter(
        self, *, subjectID: Any = None, groupID: Any = None, condition: Any = None, **extra: Any
    ) -> Dataset[T]:
        """Return a new Dataset containing only items matching all given criteria.

        Each criterion may be a single value, or a list/tuple/set of values any one
        of which is accepted. Criteria combine with AND. Additional keyword
        arguments are matched against the items' ``extra`` dicts; call
        :meth:`fields` to see what is available.

        Args:
            subjectID: Keep items whose subjectID matches.
            groupID: Keep items whose groupID matches.
            condition: Keep items whose condition matches.
            **extra: Keep items whose ``extra`` entry of that name matches.

        Returns:
            New Dataset with matching items. Empty Dataset if none match — use
            :meth:`one` instead when you expect exactly one.

        Raises:
            ValueError: If a keyword does not name a known field.

        Example:
            >>> ds.filter(groupID="control")
            >>> ds.filter(subjectID="S01", condition="rest")
            >>> ds.filter(subjectID=["S01", "S02"])     # either subject
            >>> ds.filter(ilae=2)                       # matched against extra
        """
        self._validate_fields(extra)
        criteria = {"subjectID": subjectID, "groupID": groupID, "condition": condition, **extra}
        active = {attr: value for attr, value in criteria.items() if value is not None}
        return Dataset(
            item
            for item in self._items
            if all(_matches(_value_of(item, attr), value) for attr, value in active.items())
        )

    def one(self, **criteria: Any) -> T:
        """Return the single item matching *criteria*, raising if there is not exactly one.

        Same matching rules as :meth:`filter`, but for when you want the item itself
        rather than a one-element Dataset, and want a miss to be an error rather
        than a silently empty result.

        Args:
            **criteria: Field/value pairs, as accepted by :meth:`filter`.

        Returns:
            The matching Data object.

        Raises:
            ValueError: If no item matches, or if more than one does.

        Example:
            >>> ds.one(subjectID="milan", condition="rest")
        """
        matches = self.filter(**criteria)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"No item matches {criteria!r}. Available labels: {_truncate(self.keys())}"
            )
        raise ValueError(
            f"Expected exactly one item matching {criteria!r}, found {len(matches)}. "
            "Use ds.filter(...) to get them all."
        )

    def groupby(self, *attrs: str) -> dict[Any, Dataset[T]]:
        """Group items by one or more metadata attributes.

        Args:
            *attrs: One or more names from :meth:`fields` — the standard metadata
                fields, or any key present in the items' ``extra`` dicts.

        Returns:
            Dict mapping value to a Dataset of matching items. With a single attr
            the keys are strings; with several they are tuples, one element per
            attr. Items with None for an attribute use the string "None".

        Raises:
            ValueError: If no attrs are given, or one is not a known field.

        Example:
            >>> by_group = ds.groupby("groupID")
            >>> by_group["control"]
            >>> by_both = ds.groupby("subjectID", "condition")
            >>> by_both[("milan", "rest")]
        """
        if not attrs:
            raise ValueError("groupby() requires at least one attribute name.")
        self._validate_fields(attrs)
        groups: dict[Any, list[T]] = {}
        for item in self._items:
            values = tuple(str(_value_of(item, attr)) for attr in attrs)
            key = values[0] if len(attrs) == 1 else values
            groups.setdefault(key, []).append(item)
        return {key: Dataset(items) for key, items in groups.items()}
