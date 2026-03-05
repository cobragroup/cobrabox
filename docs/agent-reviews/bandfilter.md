# Feature Review: bandfilter

**File**: `src/cobrabox/features/bandfilter.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Excellent, production-quality feature. BandFilter correctly implements a time-series transformation using `BaseFeature[SignalData]`, applies a Butterworth bandpass filter across standard EEG bands, and stacks results along a new `band` dimension. All criteria met with no issues.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 20: Correctly decorated with `@dataclass`.

Line 21: Inherits `BaseFeature[SignalData]` — appropriate for a feature requiring time-series data.

Line 55: `__call__` signature is `def __call__(self, data: SignalData) -> xr.DataArray`, matching the base class contract.

No custom `apply()` method — correctly inherits from `BaseFeature`.

Imports are clean: stdlib → third-party → internal, with no unused imports.

## Docstring

Comprehensive Google-style docstring with all required sections:

- **One-line summary**: "Filter a signal into frequency bands." (line 22)
- **Extended description**: Explains Butterworth filtering and stacking behavior (lines 24-25)
- **Args**: Documents all three fields — `bands`, `ord`, `keep_orig` with types and defaults (lines 27-35)
- **Raises**: Documents `ValueError` for missing `sampling_rate` (lines 37-38)
- **Returns**: Detailed description of output shape and dimensions (lines 40-44)
- **Example**: Two working examples showing typical usage via `.apply()` (lines 46-48)

## Typing

All dataclass fields are explicitly typed:

- `bands: dict[str, list[float]]` (line 51)
- `ord: int` (line 52)
- `keep_orig: bool` (line 53)

`__call__` has correct return type `xr.DataArray` (line 55).

No bare `Any` types.

## Safety & Style

- No `print()` statements
- Input validation present (lines 56-57): raises `ValueError` with clear message if `sampling_rate` is `None`
- No mutation of input `data`: works on `data.data` and returns new concatenated array
- Uses `xr.apply_ufunc` for efficient dimension-aware computation
- Lines respect 100-character limit

## Action List

None.
