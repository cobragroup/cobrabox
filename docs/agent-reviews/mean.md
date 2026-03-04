# Feature Review: mean

**File**: `src/cobrabox/features/mean.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`Mean` is structurally identical to `Max` and `Min` and shares the same single deficiency: the docstring is a bare one-liner with no `Args:`, `Returns:`, or `Example:` sections. Everything else — imports, decorator, base class, field typing, `__call__` signature, input validation, and immutability — is correct. The missing docstring sections are rated HIGH because they are mandatory per the review criteria.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- **Line 1**: `from __future__ import annotations` is present and first. Correct.
- **Lines 3–8**: Import order is future → stdlib (`dataclasses`) → third-party (`xarray`) → internal. Correct.
- **Lines 11–12**: `@dataclass` and `BaseFeature` inheritance are present. Correct.
- **Class name**: `Mean` is PascalCase and matches `mean.py`. Correct.
- **Line 15**: `dim: str` is a typed dataclass field. Correct.
- **Line 17**: `__call__` signature is `(self, data: Data) -> xr.DataArray`. Correct for a `BaseFeature`.
- `data` is not a dataclass field. Correct.
- No reimplementation of `.apply()`. Correct.

## Docstring

- **Line 13**: Only a one-line summary `"""Compute mean across a dimension."""` is present.
- Missing `Args:` section. The field `dim` is not documented (valid values, effect on output shape, e.g. reducing `"time"` gives per-channel mean).
- Missing `Returns:` section. The output shape (input shape minus the reduced dimension), preserved dimensions, dtype, and value semantics (arithmetic mean) are not described.
- Missing `Example:` section demonstrating `.apply()` usage.

## Typing

- `dim: str` is typed. Correct.
- `__call__` return type is `xr.DataArray`. Correct.
- No bare `Any`. Correct.

## Safety & Style

- No `print()` statements. Correct.
- **Lines 18–19**: Validates that `self.dim` is in `data.data.dims` and raises `ValueError` with a descriptive message. Correct.
- No mutation of input `data`. `data.data.mean(...)` returns a new array. Correct.
- No `__post_init__` is needed (no numeric constraints on `dim`). Correct.
- All lines are within 100 characters. Correct.

## Action List

1. [HIGH] Add a complete Google-style docstring with `Args:` (documenting `dim` — its purpose, valid values such as `"time"` or `"space"`, and effect on output shape), `Returns:` (output shape, remaining dimensions, value semantics), and `Example:` (minimal self-contained call via `.apply()`).
