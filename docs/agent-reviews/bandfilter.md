# Feature Review: bandfilter

**File**: `src/cobrabox/features/bandfilter.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

BandFilter is a well-implemented, production-quality feature. It correctly inherits from `BaseFeature[SignalData]`, has comprehensive docstring coverage with Args, Returns, Raises, and Example sections, robust input validation in `__post_init__`, and proper type annotations throughout. The code cleanly applies scipy's Butterworth filter via xarray's apply_ufunc and handles the band dimension stacking correctly.

## Ruff

### `uvx ruff check`
All checks passed!

### `uvx ruff format --check`
1 file already formatted

## Signature & Structure

All structural requirements met:

- `from __future__ import annotations` at line 1
- `@dataclass` decorator at line 20
- Correct base class `BaseFeature[SignalData]` at line 21 (time-series feature requiring sampling_rate)
- Class name `BandFilter` matches filename `bandfilter.py`
- `__call__` signature correctly typed at line 74: `def __call__(self, data: SignalData) -> xr.DataArray`
- No `apply()` override (correctly inherited from base)
- Imports follow correct order: stdlib, third-party, internal

## Docstring

Complete Google-style docstring with all required sections:

- One-line summary at line 22: "Filter a signal into frequency bands."
- Extended description at lines 24-25 explaining Butterworth filtering and stacking behavior
- Args section at lines 27-35 documenting all three fields (bands, ord, keep_orig)
- Raises section at lines 37-38 documenting ValueError for missing sampling_rate
- Returns section at lines 40-44 describing output shape and band dimension
- Example section at lines 46-48 with two usage examples

## Typing

All type annotations present and correct:

- Field `bands: dict[str, list[float]]` at line 51
- Field `ord: int` at line 52
- Field `keep_orig: bool` at line 53
- `__call__` return type `xr.DataArray` at line 74 (acceptable, though `xr.DataArray | Data` would be more explicit per base class contract)
- No bare `Any` types

## Safety & Style

Excellent safety practices:

- No `print()` statements
- Comprehensive `__post_init__` validation at lines 55-72:
  - Validates `ord > 0`
  - Validates bands dict is non-empty
  - Validates each band has exactly 2 frequencies
  - Validates frequencies are non-negative
  - Validates low < high for each band
- Additional runtime validation in `__call__` at line 75-76 for `sampling_rate`
- No mutation of input `data` - creates new arrays via `xr.apply_ufunc` and `xr.concat`
- Comment at lines 82-83 explains scipy type stub limitations

## Action List

None.
