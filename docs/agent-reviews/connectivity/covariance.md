# Feature Review: Covariance

**File**: `src/cobrabox/features/connectivity/covariance.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

The Covariance feature is a well-implemented, clean feature that computes pairwise sample covariance matrices. It follows all cobrabox conventions: proper dataclass structure, comprehensive docstring with all required sections, appropriate input validation, and correct use of `output_type = Data` since it consumes the time dimension. The implementation correctly handles 2D input validation, dimension checking, and produces a symmetric covariance matrix with appropriately named output dimensions.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

The feature correctly uses:

- Line 1: `from __future__ import annotations` as the first import
- Line 13: `@dataclass` decorator with `BaseFeature[Data]` inheritance (appropriate for a generic connectivity feature)
- Line 49: `output_type: ClassVar[type[Data]] = Data` — correctly specified since covariance removes the time dimension
- Class name `Covariance` matches filename `covariance.py`
- Line 53: `__call__` signature is `def __call__(self, data: Data) -> xr.DataArray` — correct
- No custom `apply()` method (correctly inherited from base)
- No loose helper functions (all logic contained within the class)
- Import order follows convention: `__future__`, stdlib, third-party, internal

## Docstring

Excellent docstring with all required sections:

- **One-line summary** (line 15): "Compute pairwise sample covariance between all channel pairs."
- **Extended description** (lines 17-24): Explains the algorithm, output shape, and that the diagonal contains sample variance
- **Args:** (lines 26-30): Documents the `dim` field with type, default, and behavior
- **Raises:** (lines 32-34): Lists both `ValueError` conditions (wrong dimensions, missing dim)
- **Example:** (lines 36-40): Working snippet showing typical usage
- **Returns:** (lines 42-46): Detailed description of output dimensions and matrix properties

## Typing

- Line 51: `dim: str` — properly typed dataclass field
- Line 53: Return type `xr.DataArray` is explicit and correct for `BaseFeature`
- Line 49: `ClassVar[type[Data]]` is correctly specified
- No bare `Any` types
- Line 78: `coords: dict[str, np.ndarray]` — properly typed local variable

## Safety & Style

- No `print()` statements
- Input validation in `__call__` (lines 56-66):
  - Validates exactly 2D input with clear error message
  - Validates that `self.dim` exists in data dimensions
- No mutation of input `data` — works on `data.data` and returns new `xr.DataArray`
- Uses `np.cov(arr)` with default `ddof=1` as documented
- Properly handles coordinate preservation (lines 76-81)

## Action List

None.
