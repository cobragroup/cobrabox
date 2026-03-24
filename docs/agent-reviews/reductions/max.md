# Feature Review: max

**File**: `src/cobrabox/features/reductions/max.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

Clean, well-structured reduction feature. The `Max` feature correctly computes the maximum value across a specified dimension, returning a plain `Data` container (as appropriate for dimension-reducing operations). All docstring sections are present, typing is complete, ruff is clean, and input validation is implemented.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 12: `@dataclass` decorator applied.

Line 13: Correctly inherits `BaseFeature[Data]` — this is a generic reduction feature that works on any dimension, not specifically time-series.

Line 31: `output_type: ClassVar[type[Data]] = Data` is correctly set. Since this feature removes a dimension, returning plain `Data` (rather than preserving the input container type like `SignalData`) is the right choice.

Line 33: Single field `dim: str` is properly declared as a dataclass field (not a `__call__` argument).

Line 35: `__call__` signature is correct: `def __call__(self, data: Data) -> xr.DataArray:`.

No loose helper functions — the implementation is concise and self-contained.

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 14): Clear and descriptive.
- **Args:** (lines 16-17): Documents the `dim` parameter with type and purpose.
- **Returns:** (lines 19-22): Describes the output shape and value meaning.
- **Raises:** (lines 24-25): Documents `ValueError` for missing dimension.
- **Example:** (lines 27-28): Shows correct `.apply()` usage.

## Typing

- Field `dim: str` is properly typed (line 33).
- `__call__` return type `xr.DataArray` is explicit (line 35).
- The type parameter `BaseFeature[Data]` matches the `data: Data` argument.
- No bare `Any` types.

## Safety & Style

- No `print()` statements.
- Input validation at lines 36-37: Checks if `self.dim` exists in `data.data.dims` and raises a clear `ValueError` if not.
- No mutation of input `data` — operates on `data.data` and returns a new xarray result.
- Line length is within the 100-character limit.

## Action List

None.
