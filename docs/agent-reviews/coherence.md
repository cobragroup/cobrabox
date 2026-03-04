# Feature Review: coherence

**File**: `src/cobrabox/features/coherence.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`Coherence` is a well-implemented `BaseFeature` subclass with a clean private helper method
(`_mean_squared_coherence`), correct typing throughout, and solid pairwise-channel validation.
Three gaps need addressing: the class docstring is missing a `Returns:` section; `__call__`
does not validate that `"time"` and `"space"` are present as dimensions before accessing them
(a missing dimension yields a `KeyError` from xarray rather than a clear `ValueError`); and
there is no `__post_init__` guard on `nperseg`. Ruff is fully clean.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements are met:

- `from __future__ import annotations` is present at line 1. ✅
- Class is decorated with `@dataclass` and inherits `BaseFeature`. ✅
- Class name `Coherence` is PascalCase and matches filename `coherence.py`. ✅
- `__call__(self, data: Data) -> xr.DataArray` is correctly typed; `data` is a parameter,
  not a field. ✅
- `apply()` is not reimplemented. ✅
- The private helper `_mean_squared_coherence` is properly encapsulated as an instance method
  and has a complete docstring with `Args:` and `Returns:`. ✅
- Imports follow the correct order with no unused entries. ✅

## Docstring

The class one-line summary and extended description (lines 14–33) are accurate and informative.
The `Args:` section covers `nperseg` with constraints. The `Example:` block shows realistic
usage via `.apply()`.

**Missing**: a `Returns:` section on the class docstring. The caller needs a formal description
of output dimensions (`*extra_dims`, `space`, `space_to`, plus singleton `time` from
`_copy_with_new_data`), the NaN diagonal convention, and value range [0, 1].

## Typing

All fields and methods are annotated:

- `nperseg: int | None` (line 38)
- `_mean_squared_coherence(self, x: np.ndarray, y: np.ndarray, nperseg: int) -> np.ndarray` ✅
- `__call__(self, data: Data) -> xr.DataArray` ✅
- No bare `Any`. ✅

## Safety & Style

- No `print()` statements. ✅
- No mutation of input `data`. ✅
- Validation for `n_space < 2`, `seg < 2`, and `seg > n_time` is present (lines 84–93). ✅
- **Missing**: `"time"` dimension is not validated before `xr_data.sizes["time"]` is accessed
  at line 80. A `Data` without `"time"` yields a `KeyError` from xarray, not a `ValueError`.
- **Missing**: `"space"` dimension/coordinate is not validated before
  `xr_data.coords["space"].values` is accessed at line 81. Same issue.
- **Missing**: `__post_init__` validation for `nperseg`. A caller passing `nperseg=0` or a
  negative integer will get a confusing error inside the Welch loop rather than at
  construction time.

## Action List

1. [MEDIUM] Add a `Returns:` section to the class docstring. Describe output dimensions
   (`*extra_dims`, `space`, `space_to`, singleton `time`), the NaN diagonal, symmetry, and
   value range [0, 1].
2. [MEDIUM] Add dimension validation at the top of `__call__` (before line 80): raise
   `ValueError` if `"time"` is not in `xr_data.dims` and if `"space"` is not in
   `xr_data.dims`.
3. [LOW] Add `__post_init__` validation for `nperseg`: raise `ValueError` if it is not
   `None` and is less than 2.
