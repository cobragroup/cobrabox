# Feature Review: ampent

**File**: `src/cobrabox/features/ampent.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

Ampent computes amplitude entropy using histogram-based probability estimation. Most issues from the previous review have been addressed: complete docstring added, `_is_cobrabox_feature` marker present, zero-probability filtering fixed, and proper validation added. However, two significant issues remain: (1) the return value creates fake singleton dimensions instead of a proper scalar, and (2) the `data_window` parameter is still unused and should be removed.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

- ✅ `from __future__ import annotations` present (line 1)
- ✅ `@dataclass` decorator present (line 15)
- ✅ Correctly inherits `BaseFeature[Data]` (line 16)
- ✅ `output_type` classvar correctly set to `Data` (line 48)
- ✅ `__call__` signature correct (line 57)
- ✅ `_is_cobrabox_feature = True` present (line 12)
- ⚠️ `data_window` field (line 50) is **completely unused** in `__call__` — only `band_width` is used

## Docstring

- ✅ Complete Google-style docstring present with all required sections
- ✅ One-line summary clear and descriptive
- ✅ Extended description explains algorithm well
- ✅ `Args:` section documents both fields
- ❌ **Returns section is misleading**: claims output has shape `(1, 1)` with dims `("time", "space")`, but criteria specify scalar outputs should NOT use fake singleton dimensions
- ✅ `Example:` section present with working code

## Typing

- ✅ All fields typed: `data_window: Data`, `band_width: float`
- ✅ `__call__` return type annotated: `-> xr.DataArray`
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ `__post_init__` validation: checks `band_width > 0` (lines 53-55)
- ✅ Input validation in `__call__`: checks data has at least 2 dimensions (lines 62-63)
- ✅ Zero-probability filtering fixed: uses `probs[probs > 0]` (line 114)
- ✅ No mutation of input `data`
- ✅ Variable naming improved: `n_timepoints` instead of `T`

## Action List

1. **[Severity: HIGH]** Fix return value to avoid fake singleton dimensions (line 122). Change:
   ```python
   return xr.DataArray([[tabar]], dims=["time", "space"], coords={"time": [0], "space": [0]})
   ```
   to:
   ```python
   return xr.DataArray(tabar)
   ```
   Update the docstring `Returns:` section to describe the actual output (0-dimensional scalar).

2. **[Severity: HIGH]** Remove the unused `data_window` parameter. It serves no purpose and the docstring admits it's "currently unused, reserved for future windowing functionality". Remove:
   - Line 50: `data_window: Data` field
   - Lines 30-31: `data_window` from Args section of docstring
   - Lines 42, 43: `data_window` usage from Example
