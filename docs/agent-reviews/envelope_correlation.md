# Feature Review: envelope_correlation

**File**: `src/cobrabox/features/envelope_correlation.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

Clean, well-structured feature implementing amplitude envelope correlation (AEC). Uses `BaseFeature[SignalData]` correctly with `output_type = Data` since it produces a correlation matrix without a time dimension. Excellent docstring with all required sections (Args, Returns, Raises, Example). Good input validation for required dimensions and minimum channel count. Proper use of mne_connectivity backend. No issues found.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- ✅ `from __future__ import annotations` present (line 1)
- ✅ `@dataclass` decorator on class (line 14)
- ✅ Inherits `BaseFeature[SignalData]` (line 15) — correct since it requires time dimension
- ✅ `output_type: ClassVar[type[Data]] = Data` set (line 50) — correct since output has no time dimension
- ✅ Class name `EnvelopeCorrelation` matches filename
- ✅ `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray:` (line 55)
- ✅ No `apply()` override — correctly inherits from base
- ✅ Imports ordered correctly: **future**, stdlib, third-party, internal

## Docstring

Excellent docstring with all required sections:

- ✅ One-line summary: "Compute amplitude envelope correlation (AEC) between all channel pairs." (line 16)
- ✅ Extended description explaining AEC, Hilbert transform, orthogonalization (lines 18-25)
- ✅ Args section documenting both dataclass fields (lines 27-33)
- ✅ Returns section describing output shape and dims (lines 35-37)
- ✅ Raises section documenting validation errors (lines 39-42)
- ✅ Example section with 3 usage examples via `.apply()` (lines 44-47)

## Typing

- ✅ All fields typed: `orthogonalize: str | bool = "pairwise"` (line 52), `absolute: bool = False` (line 53)
- ✅ `__call__` return type: `xr.DataArray` (line 55)
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation present (lines 58-71):
  - Checks for extra dimensions beyond `space` and `time` (lines 58-63)
  - Validates at least 2 spatial channels (lines 65-71)
- ✅ No mutation of input `data` — works on `data.data` and returns new `DataArray`
- ✅ Uses `verbose=False` when calling mne_connectivity to suppress output

## Action List

None.
