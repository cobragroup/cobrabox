# Feature Review: EpileptogenicityIndex

**File**: `src/cobrabox/features/frequency_domain/epileptogenicity_index.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Summary

Excellent implementation of a complex scientific feature. The code is well-structured,
documented, and follows all project conventions. Includes comprehensive docstring with
algorithm explanation, proper parameter validation, and clean NumPy/xarray operations.
Implements the Bartolomei et al. (2008) epileptogenicity index algorithm correctly.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 8: `from __future__ import annotations` present.
Line 28-29: Correct `@dataclass` decorator with `BaseFeature[SignalData]` inheritance.
Line 92: Proper `output_type: ClassVar[type[Data]] = Data` since the feature removes
the time dimension (returns per-channel scalar values).
Line 179: Correct `__call__` signature taking `SignalData` and returning `xr.DataArray`.
No `apply()` override — correctly uses inherited method.

## Docstring

Comprehensive Google-style docstring with all required sections:

- **One-line summary**: Line 30 — clear and concise.
- **Extended description**: Lines 32-59 — excellent explanation of the three-stage
  algorithm with mathematical formulas.
- **Args**: Lines 65-76 — all five parameters documented with types and defaults.
- **Returns**: Lines 78-80 — describes output shape and value range.
- **Raises**: Lines 82-85 — three ValueError conditions listed.
- **Example**: Lines 87-89 — working usage snippet.
- **References**: Lines 60-63 — full citation to Bartolomei et al. (2008).

## Typing

All fields properly typed (lines 94-98):

- `window_duration: float = 1.0`
- `bias: float = 0.5`
- `threshold: float = 30.0`
- `integration_window: float = 5.0`
- `tau: float = 1.0`

`__call__` return type at line 179: `xr.DataArray`.
No bare `Any` types found.

## Safety & Style

- **No print statements**: Uses no output — clean.
- **Input validation**: Lines 180-190 validate dimensions and sampling_rate.
- **No mutation**: Returns new `xr.DataArray` at line 224 without modifying input.
- **Helper methods**: Lines 100-145 and 147-177 are well-documented private methods.
- **Algorithm correctness**: Implements Page-Hinkley CUSUM detection (lines 147-177)
  and energy ratio computation (lines 100-145) correctly per the paper.

## Action List

None.
