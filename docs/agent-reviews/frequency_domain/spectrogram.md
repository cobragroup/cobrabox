# Feature Review: spectrogram

**File**: `src/cobrabox/features/spectrogram.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

This is a high-quality feature implementation. The `Spectrogram` class properly
inherits from `BaseFeature[SignalData]`, has comprehensive input validation,
well-structured docstring with all required sections, and clean code organization.
The implementation correctly handles multi-dimensional data by flattening extra
dimensions, computing spectrograms per channel, and reshaping back. All ruff checks
pass without issues.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- ✅ `from __future__ import annotations` present at line 1
- ✅ `@dataclass` decorator applied correctly at line 16
- ✅ Inherits `BaseFeature[SignalData]` appropriately (time-series feature)
- ✅ Class name `Spectrogram` matches filename
- ✅ `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray` at line 64
- ✅ No `output_type` classvar (correct — preserves input container type)
- ✅ No `apply()` override (uses inherited implementation)
- ✅ No redundant `_is_cobrabox_feature` marker (inherited from base)
- ✅ Imports in standard order: `__future__`, dataclasses, third-party, internal

## Docstring

- ✅ One-line summary at line 18: "Compute the power spectrogram for each spatial channel."
- ✅ Extended description explains algorithm (Welch-style STFT) and dimension handling
- ✅ `Args:` section documents all four fields: `nperseg`, `noverlap`, `window`, `scaling`
- ✅ `Returns:` section describes output dims `(*extra_dims, "space", "frequency", "time")`
- ✅ `Raises:` section lists all three ValueError conditions
- ✅ `Example:` section shows typical usage with `.apply()` at lines 52-56

## Typing

- ✅ All dataclass fields typed:
  - `nperseg: int | None = None`
  - `noverlap: int | None = None`
  - `window: str = "hann"`
  - `scaling: str = "log"`
- ✅ `__call__` return type: `xr.DataArray`
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation in `__call__`:
  - `scaling` validation against `_VALID_SCALINGS` at line 67-68
  - `nperseg >= 2` check at line 73-74
  - `nperseg <= n_time` check at line 75-78
  - `noverlap < nperseg` check at line 79-80
- ✅ No mutation of input `data` (works on copy via `data.data`)
- ✅ Clean separation of concerns: validation, computation, result construction

## Action List

None.
