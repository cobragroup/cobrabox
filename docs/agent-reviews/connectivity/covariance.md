<!-- updated 2026-03-06 (rev 2) -->
# Feature Review: covariance

**File**: `src/cobrabox/features/covariance.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

`Covariance` is a minimal, correct feature. All issues from the previous review have been
addressed: `coords` is now annotated as `dict[str, np.ndarray]`. The feature mirrors the
`Correlation` structure without the `method` field or `__post_init__`, which is appropriate
since `np.cov` has no variant to select. Base class, `output_type`, dim guards, docstring,
typing, and ruff are all clean.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- `from __future__ import annotations` is the first import line. ✅
- `@dataclass` + `BaseFeature[Data]` — correct; `dim` is user-configurable. ✅
- `output_type: ClassVar[type[Data]] = Data` — correct; covariance axis is consumed. ✅
- Class name `Covariance` matches filename `covariance.py`. ✅
- No `_is_cobrabox_feature` marker — correctly omitted (inherited). ✅
- `__call__(self, data: Data) -> xr.DataArray` — matches contract. ✅
- No `apply()` override. ✅
- Imports: only what is used. ✅

## Docstring

All required sections are present and complete.

- **One-line summary**: clear verb phrase. ✅
- **Extended description**: explains 2D constraint, diagonal meaning (`ddof=1`), immutability. ✅
- **Args**: `dim` documented with type and semantics. ✅
- **Raises**: two `ValueError` cases enumerated. ✅
- **Example**: working `.apply()` snippet with expected output. ✅
- **Returns**: describes dims, coordinate axes, diagonal, symmetry. ✅

## Typing

- `dim: str` typed. ✅
- `__call__` return type `-> xr.DataArray` — valid narrowing. ✅
- `coords: dict[str, np.ndarray] = {}` — fully parameterised. ✅

## Safety & Style

- No `print()` statements. ✅
- `__call__` validates ndim and dim-presence with clear messages. ✅
- No `__post_init__` needed — only field is a string validated contextually. ✅
- Input is never mutated. ✅
- Comment on `np.cov` call correctly documents `ddof=1` default. ✅
- Line length within 100 chars. ✅

## Action List

None.
