# Feature Review: autocorr

**File**: `src/cobrabox/features/autocorr.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

Clean, well-structured feature that computes normalized autocorrelation at a given lag. Uses `BaseFeature[Data]` correctly since the dimension is user-configurable via the `dim` parameter. Good input validation in both `__post_init__` (mutually exclusive parameters, fs positivity) and `__call__` (dimension existence, lag bounds). The docstring is complete with MATLAB reference, all required sections, and a working example. Output type is correctly set to `Data` since the specified dimension is removed.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- `from __future__ import annotations` present at line 1 ✅
- `@dataclass` decorator applied at line 13 ✅
- Correctly inherits `BaseFeature[Data]` (line 14) — appropriate for a dimension-agnostic feature where `dim` is user-specified
- `output_type: ClassVar[type[Data]] = Data` set at line 44 — correct since the output removes the `dim` dimension
- Class name `Autocorr` matches filename `autocorr.py` ✅
- `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray:` (line 74)
- No `apply()` override — correctly inherits from `BaseFeature` ✅
- Imports are clean and ordered: **future**, dataclasses/typing, numpy/xarray, internal (lines 1-10)

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary**: "Compute normalized autocorrelation at a given lag along a required dimension." (line 15) ✅
- **Extended description**: Includes MATLAB equivalent and parameter guidance (lines 17-22) ✅
- **Args**: All three fields documented (`dim`, `fs`, `lag_steps`, `lag_ms`) with types and constraints (lines 24-28) ✅
- **Returns**: Clear description of output shape and values (lines 30-33) ✅
- **Example**: Working snippet showing `.apply()` usage (lines 35-36) ✅

## Typing

- All dataclass fields typed (lines 39-42): `dim: str`, `fs: float`, `lag_steps: int | None`, `lag_ms: float | None` ✅
- `__call__` return type: `xr.DataArray` (line 74) — matches base class contract ✅
- No bare `Any` types ✅
- Uses modern union syntax (`int | None`, `float | None`) ✅

## Safety & Style

- No `print()` statements ✅
- **Input validation**:
  - `__post_init__` validates mutually exclusive `lag_steps`/`lag_ms` (lines 47-48) ✅
  - `__post_init__` validates `fs > 0` (lines 49-50) ✅
  - `__call__` validates `dim` exists in data (lines 77-78) ✅
  - `__call__` validates lag bounds (lines 89-90) ✅
- **No mutation of input**: Works on `data.data` (line 75), uses `xr.apply_ufunc` to produce new output (line 95) ✅
- Handles NaN values gracefully in `_acf_numpy` (lines 57-63) ✅
- Line length compliant (max 100 chars) ✅

## Action List

None.
