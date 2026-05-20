# Feature Review: lempel_ziv

**File**: `src/cobrabox/features/time_domain/lempel_ziv.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

Clean, well-structured feature implementing Lempel-Ziv Complexity. The code correctly inherits from `BaseFeature[SignalData]`, uses static methods for internal helpers, and includes a comprehensive docstring with algorithm description and literature reference. Ruff reports no issues. The implementation correctly handles the dimensionality reduction (time dimension removed) and returns dimensionless floats in (0, 1].

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

The feature follows the correct structure:

- ✅ `from __future__ import annotations` present (line 7)
- ✅ `@dataclass` decorator with `BaseFeature[SignalData]` inheritance (line 20-21)
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (line 48) — appropriate since the time dimension is removed
- ✅ `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray` (line 50)
- ✅ Helper methods `_lzc_1d` and `_count` are `@staticmethod` inside the class (lines 53-96)
- ✅ No `apply()` override — uses inherited implementation
- ✅ Imports follow standard order: stdlib, third-party, internal

The use of `xr.apply_ufunc` with `input_core_dims=["time"]` and `vectorize=True` is the correct pattern for channel-wise operations.

## Docstring

Complete Google-style docstring with all required sections:

- ✅ **One-line summary**: "Compute Lempel-Ziv Complexity (LZC) over the time dimension." (line 22)
- ✅ **Extended description**: Explains binarization by mean, LZC counting, and normalization (lines 24-30)
- ✅ **Args**: Empty but present — appropriate since this feature has no parameters (line 32-33)
- ✅ **Returns**: Describes shape `(space,)` and value range `(0, 1]` (lines 35-38)
- ✅ **References**: Full citation to Lempel & Ziv (1976) (lines 40-42)
- ✅ **Example**: Shows typical usage via `.apply()` (lines 44-46)

## Typing

- ✅ `__call__` return type: `xr.DataArray` (line 50)
- ✅ Helper methods have type annotations: `_lzc_1d(signal: np.ndarray) -> float` (line 54), `_count(symbolic: np.ndarray) -> tuple[int, int]` (line 61)
- ✅ `output_type` properly typed as `ClassVar[type[Data]]`
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ No mutation of input `data` — creates new arrays via `_lzc_1d`
- ✅ No validation issues — this feature requires time dimension (enforced by `SignalData`) and has no parameters to validate
- ✅ Line length within 100 character limit

The implementation correctly binarizes by mean and normalizes by the theoretical maximum. The LZ76 counting algorithm is clearly attributed to NeuroKit2 with MIT license reference.

## Action List

None.
