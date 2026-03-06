# Feature Review: sample_entropy

**File**: `src/cobrabox/features/sample_entropy.py`
**Date**: 2026-03-05
**Verdict**: PASS (0 issues)

## Summary

The `SampleEntropy` feature is now fully compliant with all criteria. It has excellent documentation, proper type annotations, comprehensive input validation, and clean code. The redundant time-dimension check has been removed, and the line-length violation has been fixed. The feature correctly implements sample entropy calculation with configurable embedding dimension, tolerance, and logarithm base.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

✅ Correct structure:

- `@dataclass` decorator present
- Inherits `BaseFeature[SignalData]` (appropriate since it operates on time series)
- `output_type: ClassVar[type[Data] | None] = Data` correctly set (collapses time dimension)
- `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray`
- No `apply()` override (correctly inherited)
- `__post_init__` validates `m` and `log_base` fields

Imports are correctly ordered and complete.

## Docstring

✅ Excellent docstring with all required sections:

- One-line summary (line 15)
- Extended description explaining the algorithm (lines 17-25)
- Complete `Args:` section documenting all three fields (lines 30-37)
- `Returns:` section describing the collapsed time dimension (lines 39-41)
- `Example:` section showing both default and natural log usage (lines 43-45)

## Typing

✅ All fields properly typed:

- `m: int = 2`
- `r: float | None = None`
- `log_base: float = 2`

✅ `__call__` return type is `xr.DataArray` (correct for BaseFeature).

✅ `output_type` uses proper `ClassVar[type[Data] | None]` annotation.

No bare `Any` types found.

## Safety & Style

✅ No `print()` statements — clean.

✅ Input validation:

- `__post_init__` validates `m >= 1` (line 56-57)
- `__post_init__` validates `log_base > 0 and != 1` (line 58-59)
- Runtime validation in `__call__` for series length (line 71-75)

✅ No redundant time-dimension check (properly removed).

✅ No mutation of input `data` — works on `data.data` and returns new DataArray.

## Action List

None.
