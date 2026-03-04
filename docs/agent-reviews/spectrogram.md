# Feature Review: spectrogram

**File**: `src/cobrabox/features/spectrogram.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

The `Spectrogram` feature is a well-structured migration from the old `@feature` decorator pattern to the new `BaseFeature[SignalData]` dataclass pattern. The implementation correctly computes power spectrograms using scipy's Welch-style STFT, with proper support for extra dimensions, configurable parameters, and multiple scaling options. All criteria are met.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: ✅ `from __future__ import annotations` present as first import.

**Line 17**: ✅ `@dataclass` decorator properly applied.

**Line 18**: ✅ Correctly inherits `BaseFeature[SignalData]` — appropriate since this is a time-series feature that operates on the time dimension.

**Line 37**: ✅ Class name `Spectrogram` matches filename `spectrogram.py` (PascalCase → snake_case).

**Lines 59-128**: ✅ `__call__` signature correct:

- `data: SignalData` as argument (not a field)
- Return type `xr.DataArray` (implicitly `xr.DataArray | Data` via inheritance)
- No `apply()` implementation (correctly inherited from `BaseFeature`)

**Lines 1-12**: ✅ Imports in correct order:

1. `from __future__ import annotations`
2. stdlib (`dataclasses.dataclass`)
3. third-party (`numpy`, `xarray`, `scipy.signal`)
4. internal (`..base_feature`, `..data`)

No unused imports detected.

## Docstring

**Lines 18-57**: ✅ Complete Google-style docstring with all required sections:

- One-line summary (line 18)
- Extended description (lines 20-27)
- `Args:` section with all four fields documented (lines 29-46)
- `Returns:` section with dimension and coordinate details (lines 48-51)
- `Raises:` section documenting error conditions (lines 53-55)
- `Example:` section showing correct `.apply()` usage (lines 57-61)

## Typing

**Lines 37-40**: ✅ All fields properly typed:

- `nperseg: int | None = None`
- `noverlap: int | None = None`
- `window: str = "hann"`
- `scaling: str = "log"`

**Line 59**: ✅ `__call__` return type is `xr.DataArray` (acceptable shorthand for `xr.DataArray | Data`).

No bare `Any` types detected.

## Safety & Style

**Line 59**: ✅ No `print()` statements.

**Lines 63-76**: ✅ Input validation present:

- Validates `scaling` parameter against `_VALID_SCALINGS` (line 65)
- Validates `nperseg >= 2` (line 71)
- Validates `nperseg <= n_time` (line 73)
- Validates `noverlap < nperseg` (line 75)

All validation raises `ValueError` with clear messages.

**Lines 63-128**: ✅ No mutation of input `data` — operates on `data.data` and returns new `xr.DataArray`.

**Lines 80-82**: ✅ Proper dimension handling with transpose for extra dimensions support.

## Action List

None.
