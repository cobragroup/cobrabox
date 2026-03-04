# Feature Review: max

**File**: `src/cobrabox/features/max.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

Clean, well-structured feature. The `Max` class correctly inherits from `BaseFeature[Data]`, implements proper input validation, and includes complete Google-style docstrings. All ruff checks pass. A solid example of a dimension-reducing aggregation feature.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present — correct.

Line 11: `@dataclass` decorator applied.

Line 12: Inherits `BaseFeature[Data]` — appropriate for a generic dimension-reducing feature that works on any `Data` (not just time-series).

Line 12: Class name `Max` matches filename `max.py` — correct.

Line 27: Field `dim: str` is a dataclass field (not `data`), correctly placed.

Line 29: `__call__` signature `def __call__(self, data: Data) -> xr.DataArray` is correct. The return type `xr.DataArray` is appropriate since the output removes a dimension.

No `apply()` method — correctly inherited from `BaseFeature`.

Imports are in correct order: future annotations, dataclass, third-party (xarray), internal (base_feature, data).

## Docstring

Complete Google-style docstring with all required sections:

- Line 13: One-line summary describing the feature's purpose.
- Lines 15-17: `Args:` section documents the `dim` field.
- Lines 18-21: `Returns:` section describes shape and values.
- Lines 23-25: `Example:` section shows correct usage via `.apply()`.

## Typing

Line 27: Field `dim: str` is properly typed.

Line 29: `__call__` return type `xr.DataArray` is explicit and matches the contract.

No bare `Any` types.

## Safety & Style

Line 30-31: Input validation raises `ValueError` with a clear message if `dim` is not found in the data dimensions. This is good defensive programming.

Line 32: Uses `data.data.max(dim=self.dim)` — operates on the underlying xarray without mutating the input `Data` object. Correct immutability handling.

No `print()` statements.

## Action List

None.
