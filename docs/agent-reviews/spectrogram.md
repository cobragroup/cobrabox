# Feature Review: spectrogram

**File**: `src/cobrabox/features/spectrogram.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

`Spectrogram` is a well-implemented feature that computes the power spectrogram for each spatial channel using scipy's spectrogram/STFT functions. The code is clean, properly typed, and follows all project conventions. The docstring is comprehensive with clear descriptions of all parameters, return values, and includes a working example. Input validation is thorough and raises clear error messages for invalid parameters.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

**Correct.**

- `from __future__ import annotations` present at line 1 ✅
- `@dataclass` decorator applied correctly at line 16 ✅
- Inherits `BaseFeature[SignalData]` (lines 16-17) - appropriate since it operates on time-series data ✅
- Class name `Spectrogram` matches filename `spectrogram.py` ✅
- `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray` at line 64 ✅
- No `apply()` override - correctly inherits from base class ✅
- Imports in correct order: **future**, stdlib, third-party, internal ✅
- No unused imports detected ✅

## Docstring

**Excellent.**

All required sections present and well-written:

- **One-line summary** (line 18): Clear verb phrase describing what the feature does ✅
- **Extended description** (lines 20-24): Explains the algorithm (Welch-style STFT) and behavior ✅
- **Args:** (lines 26-41): Complete documentation of all 4 dataclass fields with types and constraints ✅
- **Returns:** (lines 43-46): Describes output dimensions and coordinate meanings ✅
- **Raises:** (lines 48-50): Documents ValueError conditions ✅
- **Example:** (lines 52-56): Working snippet using `.apply()` syntax ✅

The docstring is particularly strong with detailed explanations of the four scaling options and their units.

## Typing

**Complete.**

All fields have explicit type annotations:

- `nperseg: int | None = None` ✅
- `noverlap: int | None = None` ✅
- `window: str = "hann"` ✅
- `scaling: str = "log"` ✅

`__call__` return type is explicit: `xr.DataArray` ✅

No bare `Any` types found ✅

## Safety & Style

**Clean.**

- No `print()` statements ✅
- No mutation of input data - works on `data.data` and returns new DataArray ✅
- Input validation present at lines 67-80:
  - Validates `scaling` is in `_VALID_SCALINGS` ✅
  - Validates `nperseg >= 2` ✅
  - Validates `nperseg <= n_time` ✅
  - Validates `noverlap < nperseg` ✅
- Line length compliant (max 100 chars) ✅

## Action List

None.
