# Feature Review: amplitude_variation

**File**: `src/cobrabox/features/amplitude_variation.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

A clean, minimal feature that computes the standard deviation along the time dimension. Follows all conventions: proper `@dataclass` + `BaseFeature[SignalData]` inheritance, correct `output_type` declaration (returns `Data` since time dimension is removed), complete Google-style docstring with all required sections, and proper typing throughout. The implementation is simple but correct, leveraging xarray's `std()` method directly.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 12-13: Correct `@dataclass` decorator and `BaseFeature[SignalData]` inheritance. The type parameter is appropriate since this feature operates on the time axis.

Line 32: Proper `output_type: ClassVar[type[Data]] = Data` declaration. This is correctly set because the feature removes the time dimension, returning a non-time-series container.

Line 34: Correct `__call__` signature with `data: SignalData` parameter and `-> xr.DataArray` return type.

Imports are clean and follow the standard order.

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 14): Clear verb phrase describing the computation.
- **Extended description** (lines 16-17): Explains what amplitude variation measures.
- **Args** (lines 19-20): Correctly states "None" since there are no dataclass fields.
- **Returns** (lines 22-27): Detailed description of output shape and units. Correctly notes the time dimension is removed and explains how extra dimensions are handled.
- **Example** (lines 29-30): Working snippet showing `.apply()` usage.

No `Raises` or `References` sections needed — this is a straightforward statistical operation with no exceptional conditions or literature basis.

## Typing

All types are explicit:

- Class-level `output_type: ClassVar[type[Data]]`
- `__call__` parameter `data: SignalData`
- `__call__` return type `-> xr.DataArray`

No bare `Any` types present.

## Safety & Style

- **No print statements**: Clean.
- **No mutation**: Returns new xarray result without modifying input.
- **Input validation**: Not required here — `BaseFeature[SignalData]` ensures the time dimension exists at construction time, and `std()` will raise appropriate xarray errors for edge cases.

## Action List

None.
