# Feature Review: min

**File**: `src/cobrabox/features/min.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

Clean, well-structured feature that computes the minimum value across a specified dimension. Follows all cobrabox conventions with complete docstring, proper typing, input validation, and no style issues. Ready for production use.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Correct `@dataclass` decorator with `BaseFeature[Data]` inheritance (lines 11-12). The `Data` type parameter is appropriate since this is a generic dimension-reduction feature that does not inherently require a time dimension. Class name `Min` matches filename `min.py`. `__call__` signature correctly takes `data: Data` and returns `xr.DataArray` (line 29). No custom `apply()` implementation — correctly relies on inherited method.

Import order is correct: `__future__`, stdlib (`dataclasses`), third-party (`xarray`), then internal (`..base_feature`, `..data`).

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary**: Clear and descriptive (line 13)
- **Args**: Documents the `dim` field with type and description (lines 15-16)
- **Returns**: Describes shape change and value semantics (lines 18-21)
- **Example**: Shows correct `.apply()` usage (lines 23-24)

## Typing

All fields are typed: `dim: str` (line 27). `__call__` has explicit return type `xr.DataArray` (line 29). No bare `Any` types present.

## Safety & Style

No `print()` statements. Input validation present at lines 30-31: checks that `self.dim` exists in `data.data.dims` and raises `ValueError` with a clear message if not. Feature operates on `data.data` (the underlying xarray) and returns a new array — no mutation of the immutable `Data` container.

Line length is within 100 character limit (longest line is 86 characters).

## Action List

None.
