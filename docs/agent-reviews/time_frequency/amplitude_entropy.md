# Feature Review: amplitude_entropy

**File**: `src/cobrabox/features/time_frequency/amplitude_entropy.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

A well-implemented feature that computes amplitude entropy via histogram-based probability estimation. Clean code structure with comprehensive docstring, proper type annotations, and no ruff issues. The implementation correctly uses `BaseFeature[Data]` since it is dimension-agnostic (no inherent time-series dependency), despite the docstring's time-series framing.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- ✅ Line 1: `from __future__ import annotations` present
- ✅ Line 13: `@dataclass` decorator applied
- ✅ Line 14: Correctly inherits `BaseFeature[Data]` — appropriate since the feature is dimension-agnostic (processes rows generically without requiring `sampling_rate` or time-specific operations)
- ✅ Line 46: `output_type: ClassVar[type[Data] | None] = Data` — correctly set to return plain `Data` since the output is a scalar (0-dimensional, no time dimension preserved)
- ✅ Class name `AmplitudeEntropy` matches filename `amplitude_entropy.py`
- ✅ Line 54: `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray:`
- ✅ No `apply()` override (inherits from base)
- ✅ No loose helper functions — all logic contained within the class
- ✅ Import order follows convention: future, stdlib, third-party, internal

## Docstring

All required sections present (lines 15-43):

- ✅ One-line summary: "Compute amplitude entropy from time-series data..."
- ✅ Extended description: Explains histogram-based approach and Shannon entropy formula
- ✅ Args: `band_width` documented with constraints
- ✅ Returns: Correctly describes 0-dimensional DataArray with scalar value
- ✅ Raises: Documents both ValueError conditions (band_width, dimensions)
- ✅ Example: Working snippet with `.apply()` usage

The docstring mentions "time-series data" and "time points" which reflects the typical use case, though the implementation is actually dimension-agnostic (any 2D array works).

## Typing

- ✅ Line 48: Field `band_width: float` properly typed
- ✅ Line 46: `output_type` correctly typed as `ClassVar[type[Data] | None]`
- ✅ Line 50: `__post_init__` has `-> None` return type
- ✅ Line 54: `__call__` has full type annotation
- ✅ No bare `Any` types
- ✅ No `print()` statements

## Safety & Style

- ✅ Lines 51-52: Input validation in `__post_init__` — raises `ValueError` if `band_width <= 0`
- ✅ Lines 59-60: Dimension validation in `__call__` — raises `ValueError` if data has fewer than 2 dimensions
- ✅ No mutation of input `data` — works on `data.to_numpy()` copy and returns new DataArray
- ✅ Lines 99-102: Defensive guard for empty histogram (sets entropy to 0.0)
- ✅ Line 111: Correctly filters zero probabilities before computing log
- ✅ Line 119: Returns proper 0-dimensional xarray DataArray (no fake singleton dimensions)

**Style note**: The extensive MATLAB-to-Python translation comments (lines 62-114) are informative for algorithm traceability but could be considered verbose. They do not affect functionality.

## Action List

None.
