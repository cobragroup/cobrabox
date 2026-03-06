# Feature Review: wavelet_transform

**File**: `src/cobrabox/features/wavelet_transform.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

The file contains two well-structured feature classes (`DiscreteWaveletTransform` and `ContinuousWaveletTransform`) that correctly follow the cobrabox dataclass pattern, use appropriate base classes with type parameters, validate parameters in `__post_init__`, handle extra dimensions, and preserve metadata. The implementation logic is correct and 100% branch-covered by tests. The only issues are two ruff `RUF002` linting errors — EN DASH characters (`–`) in docstrings that ruff flags as ambiguous Unicode. No structural, docstring, typing, or safety issues were found.

## Ruff

### `uvx ruff check`
```
RUF002 Docstring contains ambiguous `–` (EN DASH). Did you mean `-` (HYPHEN-MINUS)?
   --> src/cobrabox/features/wavelet_transform.py:133:25
    |
133 |     producing a 2-D time–scale representation. The output preserves the
    |                         ^

RUF002 Docstring contains ambiguous `–` (EN DASH). Did you mean `-` (HYPHEN-MINUS)?
   --> src/cobrabox/features/wavelet_transform.py:141:42
    |
141 |               components with strong time–frequency localisation.
    |                                          ^

Found 2 errors.
```

### `uvx ruff format --check`
```
RUF002 Docstring contains ambiguous `–` (EN DASH). Did you mean `-` (HYPHEN-MINUS)?
(same 2 errors as above — no formatting issues beyond the RUF002 violations)
```

## Signature & Structure

Both classes are correctly structured:

- `from __future__ import annotations` is present at line 1. ✅
- Both classes use `@dataclass` and inherit `BaseFeature[SignalData]`. ✅
- `output_type: ClassVar[type[Data] | None]` is correctly set: `Data` for DWT (time consumed), `None` for CWT (time preserved). ✅
- Class names are PascalCase and match the filename. ✅
- `__call__` takes `data: SignalData` and returns `xr.DataArray` on both classes. ✅
- `apply()` is not re-implemented. ✅
- Imports are ordered correctly and all used. ✅

## Docstring

Both docstrings are complete and well-written:

- One-line summary is present and concise. ✅
- Extended description explains the algorithm, output layout, and NaN-padding rationale for DWT. ✅
- `Args:` section covers every dataclass field with type and description. ✅
- `Returns:` section describes dims, coords, and value semantics. ✅
- `Example:` section provides working `.apply()` snippets. ✅

## Typing

- All dataclass fields are typed: `wavelet: str`, `level: int | None`, `mode: str`, `scales: list[float] | None`, `n_scales: int`, `scaling: str`. ✅
- `__call__` return type is `xr.DataArray` on both classes. ✅
- `frequencies: np.ndarray | None` local annotation in CWT `__call__` at line 242. ✅
- No bare `Any`. ✅

## Safety & Style

- No `print()` statements. ✅
- Input is never mutated; all work is done on `data.data.values` and new arrays. ✅
- `__post_init__` validates `level < 1`, wavelet name (via `pywt.Wavelet`/`pywt.ContinuousWavelet`), `scales` emptiness and positivity, `n_scales < 1`, and `scaling` against the valid set. ✅
- `__call__` validates `level > max_level` at runtime against the actual signal length. ✅
- Line length is within the 100-character limit throughout. ✅
- The CWT loop initialises `frequencies` from the first iteration and reuses it, avoiding a redundant extra `pywt.cwt` call. ✅

## Action List

1. [Severity: HIGH] Replace EN DASH (`–`) with hyphen-minus (`-`) at line 133 (`time–scale` → `time-scale`) and line 141 (`time–frequency` → `time-frequency`) in `ContinuousWaveletTransform` docstring to fix the two `RUF002` ruff errors.
