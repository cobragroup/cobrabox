# Feature Review: hilbert

**File**: `src/cobrabox/features/hilbert.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Excellent, production-ready feature. The Hilbert transform implementation is clean, well-documented, and fully compliant with cobrabox conventions. All four feature modes (analytic, envelope, phase, frequency) are properly implemented with appropriate validation.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Perfect structure:

- Line 1: `from __future__ import annotations` present
- Line 16: `@dataclass` decorator correct
- Line 17: Inherits `BaseFeature[SignalData]` — appropriate since this operates on time-series data
- Line 62: `__call__` takes `data: SignalData` with return type `xr.DataArray`
- Lines 10-11: Correct import order (internal imports after third-party)
- No custom `apply()` — correctly uses inherited method

The class name `Hilbert` matches the filename `hilbert.py`.

## Docstring

Comprehensive Google-style docstring (lines 18-51) with all required sections:

- **One-line summary**: Clear verb phrase ("Apply the Hilbert transform...")
- **Extended description**: Explains the underlying scipy function and output shape
- **Args**: Documents the `feature` field with detailed descriptions of each option
- **Returns**: Describes dtype variations and dimension preservation
- **Raises**: Documents both ValueError cases (invalid feature, missing sampling_rate)
- **Example**: Shows all four modes with `.apply()` syntax

## Typing

Fully typed:

- Line 53: `feature: Literal["analytic", "envelope", "phase", "frequency"] = "analytic"` — excellent use of Literal for constrained string options
- Line 55: `__post_init__` has `-> None` return annotation
- Line 62: `__call__` has proper `SignalData` argument and `xr.DataArray` return type
- No bare `Any` types

## Safety & Style

Clean and safe implementation:

- **No print statements**: None found
- **Input validation**:
  - Lines 56-60: `__post_init__` validates `feature` against `_VALID_FEATURES` tuple
  - Lines 76-79: Validates `sampling_rate` is not None when `feature='frequency'`
- **No mutation**: Lines 63-66 extract values and work on a copy; returns new DataArray
- **Line length**: All lines within 100 character limit

The frequency calculation (line 81) correctly uses `np.gradient` with the proper time step derived from `sampling_rate`, and divides by 2π to convert from rad/s to Hz.

## Type Checker Notes

LSP reports false-positive type errors on lines 72, 74, and 80 related to `scipy.signal.hilbert` return type compatibility with `np.abs`, `np.angle`, and `np.unwrap`. These are upstream typing stub issues with scipy/numpy — the code runs correctly and the runtime types are compatible. No action needed.

## Action List

None.
