# Feature Review: hilbert

**File**: `src/cobrabox/features/hilbert.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Excellent feature implementation. Hilbert is a clean, well-documented `BaseFeature[SignalData]` that applies the scipy Hilbert transform to extract various signal representations. The code follows all cobrabox conventions: proper dataclass structure, complete Google-style docstring with all required sections, strong typing, and appropriate validation. No issues found.

## Ruff

### `uvx ruff check`
All checks passed!

### `uvx ruff format --check`
1 file already formatted

## Signature & Structure

Clean structure (line 16-17): `@dataclass` decorator with `BaseFeature[SignalData]` inheritance — correct for a time-series feature that operates on the time axis. Class name `Hilbert` matches filename.

`__call__` signature (line 62): `def __call__(self, data: SignalData) -> xr.DataArray:` — correct return type for `BaseFeature`. No custom `apply()` — properly inherited.

Imports (lines 1-11): Correct order — `__future__`, stdlib, third-party, internal. No unused imports.

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 18): Clear, descriptive.
- **Extended description** (lines 20-23): Explains what the feature does and notes that the time dimension is preserved.
- **Args** (lines 25-35): Documents the `feature` field with all four valid options clearly explained.
- **Returns** (lines 37-39): Specifies dtype and shape preservation.
- **Raises** (lines 41-44): Documents both validation errors.
- **Example** (lines 46-50): Four working examples showing different feature modes via `.apply()`.

## Typing

- **Field typing** (line 53): `feature: Literal["analytic", "envelope", "phase", "frequency"] = "analytic"` — precise Literal type with default.
- **Return type** (line 62): `xr.DataArray` — correct.
- **No bare `Any`**: All types explicit.

## Safety & Style

- **No print statements**: Clean.
- **Input validation**:
  - `__post_init__` (lines 55-60): Validates `feature` is in `_VALID_FEATURES` with clear error message.
  - `__call__` (lines 76-79): Validates `sampling_rate` is set when `feature='frequency'`.
- **No mutation**: Creates new `xr.DataArray` instances, never mutates input (lines 70, 83).
- **Line length**: Within 100 char limit.

## Action List

None.
