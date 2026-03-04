# Feature Review: max

**File**: `src/cobrabox/features/max.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`Max` is structurally correct and ruff-clean. It has the right import order, `@dataclass` decorator, `BaseFeature` inheritance, a typed field, a proper `__call__` signature, input validation, and no mutation of input data. The sole but significant deficiency is the docstring: the one-line summary is present but `Args:`, `Returns:`, and `Example:` sections are entirely absent, making this a HIGH-severity finding. There are no other issues.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- **Line 1**: `from __future__ import annotations` is present and first. Correct.
- **Lines 3–8**: Import order is future → stdlib (`dataclasses`) → third-party (`xarray`) → internal. Correct.
- **Lines 11–12**: `@dataclass` and `BaseFeature` inheritance are present. Correct.
- **Class name**: `Max` is PascalCase and matches `max.py`. Correct.
- **Line 15**: `dim: str` is a typed dataclass field. Correct.
- **Line 17**: `__call__` signature is `(self, data: Data) -> xr.DataArray`. Correct for a `BaseFeature`.
- `data` is not a dataclass field. Correct.
- No reimplementation of `.apply()`. Correct.

## Docstring

- **Line 13**: Only a one-line summary `"""Compute maximum across a dimension."""` is present.
- Missing `Args:` section. The field `dim` is not documented (what values are valid? what does passing `"time"` vs `"space"` mean?).
- Missing `Returns:` section. The output shape (input shape minus the reduced dimension), preserved dimensions, dtype, and value semantics (element-wise maximum) are not described.
- Missing `Example:` section demonstrating `.apply()` usage.

## Typing

- `dim: str` is typed. Correct.
- `__call__` return type is `xr.DataArray`. Correct.
- No bare `Any`. Correct.

## Safety & Style

- No `print()` statements. Correct.
- **Lines 18–19**: Validates that `self.dim` is in `data.data.dims` and raises `ValueError` with a descriptive message. Correct.
- No mutation of input `data`. `data.data.max(...)` returns a new array. Correct.
- No `__post_init__` is needed (no numeric constraints on `dim`). Correct.
- All lines are within 100 characters. Correct.

## Action List

1. [HIGH] Add a complete Google-style docstring with `Args:` (documenting `dim` — its purpose, valid values, and effect), `Returns:` (output shape, remaining dimensions, value semantics), and `Example:` (minimal self-contained call via `.apply()`).
