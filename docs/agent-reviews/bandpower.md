# Feature Review: bandpower

**File**: `src/cobrabox/features/bandpower.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`Bandpower` is a well-structured `BaseFeature` subclass with solid input validation, correct
typing, and a thorough docstring. The main gap is a missing `Returns:` section — required by
the criteria — which leaves the caller without a formal description of output shape and
semantics. A secondary low-severity gap is the absence of `__post_init__` validation for
`nperseg` (a negative or zero value would be silently passed to `scipy.signal.welch` and
produce a confusing error there rather than here). Ruff is fully clean.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements are met:

- `from __future__ import annotations` is present at line 1. ✅
- Class is decorated with `@dataclass` and inherits `BaseFeature`. ✅
- Class name `Bandpower` is PascalCase and matches filename `bandpower.py`. ✅
- `__call__(self, data: Data) -> xr.DataArray` is correctly typed; `data` is a parameter, not
  a field. ✅
- `apply()` is not reimplemented — inherited from `BaseFeature`. ✅
- Imports follow the correct order (future → stdlib → third-party → internal) with no unused
  entries. ✅

## Docstring

The one-line summary and extended description (lines 22–49) are clear and accurate. The
`Args:` section documents both fields (`bands` and `nperseg`) with full detail including
sub-cases and examples. The `Example:` block (lines 47–49) shows three realistic usage
patterns via `.apply()`.

**Missing**: a `Returns:` section. The criteria require it for all features. The caller needs
to know the output dimensions, shape, and value semantics without reading the implementation.

## Typing

All fields are annotated:

- `bands: dict[str, list[float] | bool] | None` (line 55)
- `nperseg: int | None` (line 56)

`__call__` return type is `xr.DataArray`. No bare `Any`. ✅

## Safety & Style

- No `print()` statements. ✅
- Input validation in `__call__`:
  - `"time"` dimension check at line 62 raises `ValueError` with a clear message. ✅
  - `sampling_rate is None` check at line 66 raises `ValueError` with a clear message. ✅
  - Band spec validation (lines 75–88) covers the `True`/`False`/list cases correctly. ✅
- No mutation of the input `data` object. ✅
- `nperseg` is not validated in `__post_init__`: a caller passing `nperseg=0` or a negative
  integer will get an obscure error from `scipy.signal.welch` rather than a clear `ValueError`
  at construction time.

## Action List

1. [MEDIUM] Add a `Returns:` section to the class docstring. Describe output dimensions
   (``band_index``, ``space``, plus a singleton ``time`` added by `_copy_with_new_data`),
   the coordinate assigned to ``band_index``, and that values are absolute power in
   signal² / Hz.
2. [LOW] Add `__post_init__` validation for `nperseg`: raise `ValueError` if it is not
   `None` and is less than 2, consistent with the downstream constraint in
   `scipy.signal.welch`.
