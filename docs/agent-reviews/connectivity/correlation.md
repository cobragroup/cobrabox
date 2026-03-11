<!-- overwritten 2026-03-06 (rev 2) -->
# Feature Review: correlation

**File**: `src/cobrabox/features/correlation.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

`Correlation` is a clean, production-ready feature. All issues from the previous review have
been addressed: `coords` is now annotated as `dict[str, np.ndarray]`. The feature correctly
uses `BaseFeature[Data]` (dim is user-configurable, no `sampling_rate` required), sets
`output_type = Data` because the correlation axis is consumed, validates `method` in
`__post_init__`, guards both ndim and dim-presence in `__call__`, and has a complete
Google-style docstring. Ruff is clean.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- `from __future__ import annotations` is the first import line. ✅
- `@dataclass` + `BaseFeature[Data]` — correct; `dim` is user-configurable so `SignalData` is not needed. ✅
- `output_type: ClassVar[type[Data]] = Data` — correct; the correlation axis is consumed and the output has no time dimension. ✅
- Class name `Correlation` matches filename `correlation.py` (PascalCase). ✅
- No `_is_cobrabox_feature` marker — correctly omitted (inherited from `BaseFeature`). ✅
- `__call__(self, data: Data) -> xr.DataArray` — matches `BaseFeature[Data]` contract. ✅
- No `apply()` override. ✅
- Imports: only what is used; `scipy.stats.rankdata` imported lazily inside the Spearman branch. ✅

## Docstring

All required sections are present and complete.

- **One-line summary**: clear verb phrase. ✅
- **Extended description**: explains 2D constraint, diagonal behaviour, immutability. ✅
- **Args**: both `dim` and `method` documented with types and semantics. ✅
- **Raises**: three `ValueError` cases enumerated. ✅
- **Example**: working `.apply()` snippet with expected output. ✅
- **Returns**: describes dims, coordinate axes, range, diagonal, symmetry. ✅

## Typing

- `dim: str` and `method: str` are typed. ✅
- `__post_init__(self) -> None` — annotated. ✅
- `__call__` return type `-> xr.DataArray` — valid narrowing of the base class union. ✅
- `coords: dict[str, np.ndarray] = {}` — fully parameterised. ✅

## Safety & Style

- No `print()` statements. ✅
- `__post_init__` validates `method` with a clear message including the invalid value. ✅
- `__call__` validates ndim and dim-presence before any computation. ✅
- Input is never mutated; all work is on newly created arrays from `data.data`. ✅
- Line length within 100 chars (ruff confirms). ✅

## Action List

None.
