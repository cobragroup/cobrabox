# Feature Review: hilbert

**File**: `src/cobrabox/features/hilbert.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

The Hilbert feature is well-implemented and clean. It correctly uses `BaseFeature[SignalData]`
to indicate time-series dependency, has comprehensive docstring coverage with all required
sections, and includes proper input validation in both `__post_init__` and `__call__`. The
implementation correctly preserves the input DataArray's dimensions and coordinates, returning
new arrays without mutating input.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

The feature follows the correct structure:

- Line 1: `from __future__ import annotations` is present
- Line 16-17: `@dataclass` decorator with `BaseFeature[SignalData]` inheritance
- Line 17: Correct base class choice — Hilbert transform requires time dimension
- Line 53: Single dataclass field `feature` with proper `Literal` type annotation
- Line 62: `__call__` signature correctly typed as `(self, data: SignalData) -> xr.DataArray`
- No `output_type` classvar — correct since time dimension is preserved
- No `_is_cobrabox_feature` marker — correctly omitted for class-based features

Imports are clean and in standard order (future, stdlib, third-party, internal).

## Docstring

Comprehensive Google-style docstring with all required sections:

- **One-line summary** (line 18): Clear verb phrase describing the feature
- **Extended description** (lines 20-23): Explains what and why, not how
- **Args** (lines 25-35): Documents the `feature` field with all four valid options
- **Returns** (lines 37-39): Describes shape, dimensions, and dtype
- **Raises** (lines 41-44): Lists both ValueError conditions
- **Example** (lines 46-50): Four working examples showing all feature modes

The docstring is exemplary — detailed without being verbose.

## Typing

Fully typed:

- Line 53: Field `feature: Literal["analytic", "envelope", "phase", "frequency"] = "analytic"`
- Line 62: Return type `xr.DataArray` is explicit
- Line 55, 62: Method return types (`None`, `xr.DataArray`) are annotated
- No bare `Any` types

## Safety & Style

Clean implementation with proper safety practices:

- Line 56-60: `__post_init__` validates `feature` parameter against `_VALID_FEATURES`
- Line 76-79: `__call__` validates `sampling_rate` is set when `feature='frequency'`
- No `print()` statements
- No mutation of input `data` — works on `data.data` and returns new DataArrays
- Lines 70, 83: Returns preserve original dims and coords

## Action List

None.
