# Feature Review: spikes_calc

**File**: `src/cobrabox/features/spikes_calc.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

The `SpikesCalc` feature implements IQR-based spike/outlier counting correctly.
However, the class name uses abbreviation "Calc" and the docstring is missing
a `Raises:` section despite having validation that raises `ValueError`.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- `from __future__ import annotations` present (line 1)
- `@dataclass` decorator with `BaseFeature[Data]` inheritance (lines 13-14)
- `output_type: ClassVar[type[Data]] = Data` correctly set (line 31)
- Class name `SpikesCalc` matches filename but uses abbreviation
- `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray:` (line 33)
- Import order follows convention

## Docstring

Missing `Raises:` section. The feature raises `ValueError` at line 37 when input
is empty, but this is not documented.

The `Args:` section states "None" which is technically correct (no dataclass fields),
but could be omitted for features without parameters.

## Typing

Fully typed:

- No dataclass fields requiring types
- `__call__` return type annotation correct
- No bare `Any` types

## Safety & Style

- No `print()` statements
- Input validation: raises `ValueError` if input data is empty (lines 36-37)
- No mutation of input `data`
- Line length within 100 characters

## Action List

1. [Severity: MEDIUM] Rename class from `SpikesCalc` to `SpikeCount` (or similar
descriptive name without abbreviation) at line 14.

2. [Severity: MEDIUM] Add `Raises:` section to docstring documenting the
`ValueError` raised when input data is empty:

```python
Raises:
    ValueError: If input data is empty.
```
