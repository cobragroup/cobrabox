# Feature Review: phase_locking_value

**File**: `src/cobrabox/features/phase_locking_value.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

The file contains two well-structured classes (`PhaseLockingValue` and
`PhaseLockingValueMatrix`) that are correctly migrated to the new class-based pattern.
Ruff is clean, all fields are typed, docstrings have all required sections, and
validation is appropriate for `BaseFeature[SignalData]`. Two low-severity issues prevent
a PASS: the `_compute_plv` static method is duplicated verbatim in both classes (DRY
violation), and `PhaseLockingValue`'s `Raises` section documents a `time`-dimension
error that this feature never actually raises (it is enforced upstream by `SignalData`
construction).

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

Both classes use `@dataclass` and `BaseFeature[SignalData]` correctly — they require
both `space` and `time` dimensions, so `SignalData` is the right type parameter.
`output_type: ClassVar[type[Data]] = Data` is set on both, which is correct since
neither output preserves the time dimension. `__call__` signatures match the base class
contract. No `apply()` override. Imports are clean and in standard order.
`from __future__ import annotations` is on line 1.

The file contains two classes, following the same precedent as
`partial_correlation.py`. Acceptable.

## Docstring

Both docstrings have one-line summaries, extended descriptions, `Args:`, `Returns:`,
`Raises:`, and `Example:` sections. `Returns:` correctly describes the output shape and
dims for each class. One inaccuracy: `PhaseLockingValue.Raises` (line 30) states
"If `space` or `time` dimension is not in data", but the feature only checks `space`
explicitly — `time` is guaranteed by `SignalData` at construction time and will never
raise from within this feature. `PhaseLockingValueMatrix.Raises` correctly omits the
`time` claim.

## Typing

All dataclass fields typed: `coord_x: str`, `coord_y: str`, `coords: list[str]`.
`__call__` return type is `xr.DataArray` on both classes. `_compute_plv` is fully
annotated (`np.ndarray, np.ndarray) -> float`. No bare `Any`.

## Safety & Style

No `print` statements. No mutation of input `data`. `space` dimension and coordinate
presence are validated with clear `ValueError` messages. No redundant `time` check
(correctly deferred to `SignalData`).

`_compute_plv` (lines 42–59 and 111–128) is a byte-for-byte duplicate in both classes.
The same logic is repeated in `PhaseLockingValueMatrix._compute_plv`. This creates a
maintenance hazard: a bug fix or improvement in one copy must be manually mirrored in
the other.

## Action List

1. [LOW] Remove the duplicate `_compute_plv` static method. Extract it as a
   module-level private function `_compute_plv(x, y) -> float` above both class
   definitions, and call it directly from both `__call__` implementations. This is the
   same pattern used for shared helpers in the old code, and avoids the maintenance
   hazard.

2. [LOW] Fix the `Raises` section in `PhaseLockingValue` (line 30): remove or reword
   the claim about `time` dimension. Since `SignalData` enforces the `time` requirement
   at construction, this feature cannot raise for a missing `time` dim. Change to:
   `ValueError: If ``space`` dimension is not in data or either coordinate is missing.`
