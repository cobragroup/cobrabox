<!-- overwritten 2025-03-24 -->
# Feature Review: Correlation

**File**: `src/cobrabox/features/connectivity/correlation.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

Clean, well-structured feature implementing pairwise Pearson/Spearman correlation.
Docstring is comprehensive with all required sections. Ruff passes with no issues.
One minor typing improvement suggested: use `Literal` for the `method` field.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- `from __future__ import annotations` present (line 1). ✅
- `@dataclass` decorator with `BaseFeature[Data]` inheritance (line 13-14). ✅
- `output_type` correctly set to `Data` since the time dimension is consumed (line 50). ✅
- Class name `Correlation` matches filename `correlation.py`. ✅
- `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray` (line 59). ✅
- No `apply()` override — correctly inherits from base. ✅
- No loose helper functions — scipy import is inline and justified. ✅
- Imports in correct order: future, stdlib, third-party, internal. ✅

## Docstring

- One-line summary at top (line 15). ✅
- Extended description explains algorithm behavior (lines 17-23). ✅
- Args section covers both fields: `dim` and `method` (lines 25-31). ✅
- Raises section lists all 3 ValueError conditions (lines 33-36). ✅
- Example section shows working usage (lines 38-42). ✅
- Returns section describes output shape and dimensions (lines 44-47). ✅
- No References section (acceptable — basic statistical operation).

## Typing

- All fields typed: `dim: str`, `method: str` (lines 52-53). ✅
- `__call__` return type: `xr.DataArray` (line 59). ✅
- No bare `Any`. ✅
- `coords: dict[str, np.ndarray] = {}` — fully parameterised. ✅
- `method` field uses plain `str` — could be `Literal["pearson", "spearman"]` for stricter typing.

## Safety & Style

- No `print()` statements. ✅
- Input validation in `__call__`: checks ndim==2 (lines 62-66), dim presence (lines 68-72). ✅
- `__post_init__` validates method values (lines 55-57). ✅
- No mutation of input `data` — returns new `xr.DataArray`. ✅
- Line length within 100 characters. ✅

## Action List

None.
