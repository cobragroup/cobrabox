# Feature Review: lempel_ziv

**File**: `src/cobrabox/features/lempel_ziv.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

A well-implemented feature following all cobrabox conventions. The LempelZiv class correctly
inherits from `BaseFeature[SignalData]`, uses `xr.apply_ufunc` for vectorized computation over
the time dimension, and properly sets `output_type = Data` since the time dimension is removed.
The docstring is complete with algorithm description, citations, and usage example.
The LZ76 counting algorithm is clearly attributed to NeuroKit2 with proper licensing note.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 20: `@dataclass` decorator present.  
Line 21: Correct inheritance `BaseFeature[SignalData]` — requires time dimension.  
Line 48: `output_type: ClassVar[type[Data]] = Data` correctly declared since the feature
removes the time dimension (returns scalar complexity values).  
Line 50: `__call__` signature matches base class: `def __call__(self, data: SignalData) -> xr.DataArray:`.  
No `apply()` override — correctly inherited from base class.  

All imports follow the standard order (future, stdlib, third-party, internal).

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 22): Clear verb phrase describing the computation.  
- **Extended description** (lines 24-30): Explains binarization, normalization, and theoretical
  basis. Includes attribution to NeuroKit2 implementation.  
- **Args** (lines 32-33): Correctly states "None" since the feature has no dataclass fields.  
- **Returns** (lines 35-38): Describes shape `(space,)` and value range `(0, 1]`.  
- **References** (lines 40-42): Proper academic citation for Lempel & Ziv (1976).  
- **Example** (lines 44-45): Shows correct `.apply()` usage pattern.  

## Typing

Line 50: `__call__` has explicit return type `xr.DataArray`.  
Line 48: `output_type` uses `ClassVar[type[Data]]` with proper typing.  
No bare `Any` types. All static methods have appropriate type annotations.

## Safety & Style

No `print()` statements found.  
No input validation required — `SignalData` enforces time dimension at construction.  
No mutation of input `data` — works on `data.data` via `xr.apply_ufunc` and returns new array.  
Algorithm correctly attributed with MIT license reference to NeuroKit2.  
Line lengths within 100 character limit.

## Action List

None.
