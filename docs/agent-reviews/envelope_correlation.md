# Feature Review: envelope_correlation

**File**: `src/cobrabox/features/envelope_correlation.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

The `EnvelopeCorrelation` feature is a clean migration to the class-based pattern. It correctly computes amplitude envelope correlation (AEC) using mne-connectivity, with proper support for orthogonalization options. The implementation is concise, well-documented, and follows all criteria.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: ✅ `from __future__ import annotations` present as first import.

**Line 17**: ✅ `@dataclass` decorator properly applied.

**Line 18**: ✅ Correctly inherits `BaseFeature[SignalData]`.

**Line 18**: ✅ Class name `EnvelopeCorrelation` matches filename.

**Lines 42-81**: ✅ `__call__` signature correct:

- `data: SignalData` as argument
- Return type `xr.DataArray`
- No `apply()` implementation (inherited)

**Lines 1-8**: ✅ Imports in correct order:

1. `from __future__ import annotations`
2. stdlib (`dataclasses.dataclass`)
3. third-party (`numpy`, `xarray`, `mne_connectivity`)
4. internal (`..base_feature`, `..data`)

## Docstring

**Lines 18-40**: ✅ Complete Google-style docstring:

- One-line summary (line 18)
- Extended description (lines 20-24)
- `Args:` section for both fields (lines 26-34)
- `Returns:` section with dimension details (lines 36-40)
- `Raises:` section documenting error conditions
- `Example:` section with `.apply()` usage (lines 49-51)

## Typing

**Lines 42-43**: ✅ All fields properly typed:

- `orthogonalize: str | bool = "pairwise"`
- `absolute: bool = False`

**Line 42**: ✅ `__call__` return type is `xr.DataArray`.

No bare `Any` types detected.

## Safety & Style

**Line 42**: ✅ No `print()` statements (uses `verbose=False` in mne call).

**Lines 54-69**: ✅ Input validation present:

- Validates no extra dimensions beyond `space` and `time` (lines 56-61)
- Validates at least 2 spatial channels (lines 66-69)

**Lines 54-81**: ✅ No mutation of input `data` — operates on `data.data` and returns new `xr.DataArray`.

**Lines 74-76**: ✅ Proper coordinate handling for `space` and `space_to` dimensions.

## Action List

None.
