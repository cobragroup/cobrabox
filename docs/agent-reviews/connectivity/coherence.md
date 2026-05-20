# Feature Review: Coherence

**File**: `src/cobrabox/features/connectivity/coherence.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

The `Coherence` feature is well-implemented and follows all cobrabox conventions. It computes magnitude-squared coherence between channel pairs using Welch's method, correctly handles extra dimensions, produces a symmetric matrix with NaN diagonal, and includes comprehensive input validation. The docstring is thorough with all required sections (including the previously-missing `Raises:` section which has now been added), and the implementation is clean with no ruff issues.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present ✓

Lines 14-15: Correct `@dataclass` + `BaseFeature[SignalData]` inheritance for a time-series connectivity feature ✓

Line 52: `output_type: ClassVar[type[Data]] = Data` correctly set since the feature removes the time dimension (returns a connectivity matrix) ✓

Line 15: Class name `Coherence` matches filename `coherence.py` ✓

Line 96: `__call__` signature is `def __call__(self, data: SignalData) -> xr.DataArray` — matches base class contract ✓

Lines 60-94: Helper method `_mean_squared_coherence` is properly encapsulated as an instance method (acceptable here since it accesses no fields, but logically belongs to the feature) ✓

No `apply()` override — correctly inherited from `BaseFeature` ✓

Imports are correctly ordered: stdlib → third-party → internal ✓

## Docstring

Comprehensive Google-style docstring with all required sections:

- **One-line summary** (line 16): Clear verb phrase describing the computation ✓
- **Extended description** (lines 18-26): Explains Welch's method, symmetry, diagonal handling, and extra dimension preservation ✓
- **Args** (lines 28-30): Documents `nperseg` with type and constraints ✓
- **Returns** (lines 38-43): Detailed description of output dimensions, coordinates, value range, and symmetry ✓
- **Raises** (lines 45-49): All four `ValueError` conditions now documented ✓
- **Example** (lines 32-36): Working usage with `.apply()` ✓

**Minor suggestion**: Could optionally add a `References:` section citing Welch's method paper, though this is acceptable to omit for well-known standard methods.

## Typing

Line 54: Field `nperseg: int | None` properly typed ✓

Line 96: `__call__` return type `xr.DataArray` explicit ✓

Line 60: Helper method parameter and return types fully annotated ✓

No bare `Any` types ✓

## Safety & Style

No `print()` statements ✓

**Input validation**:

- Lines 57-58: `__post_init__` validates `nperseg >= 2` ✓
- Lines 99-100: Validates presence of 'space' dimension ✓
- Lines 106-107: Validates at least 2 spatial channels ✓
- Lines 110-115: Validates computed `nperseg` bounds ✓

**No mutation of input**: The feature works on `data.data` (line 97) and returns a new `xr.DataArray` (lines 133-137) without modifying the input ✓

Line length within 100 characters throughout ✓

## Action List

None.
