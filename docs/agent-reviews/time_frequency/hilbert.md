# Feature Review: hilbert

**File**: `src/cobrabox/features/time_frequency/hilbert.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Summary

The `Hilbert` feature is an exemplary implementation following all cobrabox conventions. It applies the Hilbert transform to extract analytic signal representations (analytic signal, envelope, phase, or instantaneous frequency). The code is clean, well-documented, properly typed, and passes all linting checks without issues.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Excellent structure throughout:

- Line 1: `from __future__ import annotations` is present
- Lines 16-17: `@dataclass` decorator with `BaseFeature[SignalData]` inheritance
- Line 17: Class name `Hilbert` matches filename
- Line 62: `__call__` signature correctly typed as `(self, data: SignalData) -> xr.DataArray`
- No `apply()` override — correctly uses inherited method
- No loose module-level helper functions
- Clean import ordering: future, stdlib, third-party, internal

The feature correctly uses `BaseFeature[SignalData]` since it operates on the time axis.

## Docstring

Comprehensive Google-style docstring with all required sections:

- Line 18: One-line summary clearly states purpose
- Lines 20-23: Extended description explains behavior and SciPy dependency
- Lines 25-35: `Args:` section documents the `feature` parameter with all four valid options
- Lines 37-39: `Returns:` section describes shape, dimensions, and dtypes
- Lines 41-44: `Raises:` section documents both ValueError conditions
- Lines 46-50: `Example:` section with 4 usage examples showing all feature modes

## Typing

Full type coverage:

- Line 53: Field `feature` properly typed as `Literal["analytic", "envelope", "phase", "frequency"]`
- Line 62: `__call__` return type `xr.DataArray` explicitly annotated
- Line 55: `__post_init__` return type `None` annotated
- No bare `Any` types
- Excellent use of `Literal` for the feature mode

## Safety & Style

Clean implementation with proper safety practices:

- Lines 56-60: `__post_init__` validates `feature` against `_VALID_FEATURES`
- Lines 76-79: `__call__` validates `sampling_rate` when `feature='frequency'`
- No `print()` statements
- No mutation of input `data` — returns new `xr.DataArray` instances
- Line length within 100 character limit

## Action List

None.
