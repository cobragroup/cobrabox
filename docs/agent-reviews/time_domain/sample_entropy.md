# Feature Review: sample_entropy

**File**: `src/cobrabox/features/sample_entropy.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

SampleEntropy is a well-implemented feature with excellent documentation and proper validation. The code follows all structural conventions and passes ruff checks. The only issue is a missing `Raises:` section in the docstring, which should document the `ValueError` exceptions raised in `__post_init__` and during computation.

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

Comprehensive Google-style docstring with most required sections:

- One-line summary present and descriptive
- Extended explanation of the algorithm and its interpretation (lines 17-25)
- `Args:` section documents all three fields (`m`, `r`, `log_base`) with types and constraints
- `Returns:` section describes output shape correctly
- `Example:` section includes two practical examples showing default and natural log usage

**Missing**: `Raises:` section. The feature raises `ValueError` in `__post_init__` (lines 56-59) for invalid `m` and `log_base` values, and in `_sampen_one` (lines 72-75) when time series length is insufficient. These should be documented.

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

1. [Severity: MEDIUM] Add `Raises:` section to docstring documenting `ValueError` conditions:
   - When `m < 1` or `log_base` is invalid (lines 56-59)
   - When time series length is not greater than embedding dimension `m` (lines 72-75)
