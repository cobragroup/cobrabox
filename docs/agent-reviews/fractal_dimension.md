# Feature Review: fractal_dimension

**File**: `src/cobrabox/features/fractal_dimension.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Both `FractalDimHiguchi` and `FractalDimKatz` are well-structured, correctly typed,
and follow all cobrabox conventions. The code is clean with no ruff errors,
comprehensive docstrings, proper validation, and no safety issues. All criteria
are met.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Both classes use `@dataclass`, inherit `BaseFeature[SignalData]` (correct, both
operate on the time axis), and declare `output_type: ClassVar[type[Data]] = Data`
with the required `pyright: ignore` suppression (lines 59, 130).

`from __future__ import annotations` is present at line 14. Import order is correct
(stdlib → third-party → internal). All imports are used: `field` is consumed by
`FractalDimHiguchi.k_max`; `dataclass`, `ClassVar`, `np`, `xr`, `BaseFeature`,
`Data`, `SignalData` are all referenced. Neither class implements `apply()`. `data`
is never a field in either class.

**Convention note (LOW)**: CLAUDE.md specifies one class per file. This file contains
two. This was an intentional user-directed deviation; no code change needed.

## Docstring

`FractalDimHiguchi` (lines 28–55): one-liner ✅, extended description ✅, `Args:`
with `k_max` description and constraint ✅, `Returns:` with shape and value range
✅, `References:` ✅, two `Example:` snippets ✅.

`FractalDimKatz` (lines 101–128): one-liner ✅, extended description with formula
✅, `Args: None` ✅, `Returns:` with shape and note on value range ✅, `References:`
✅, `Example:` ✅. All sections present and complete.

## Typing

`FractalDimHiguchi.k_max: int` annotated (line 57). `_higuchi_1d(signal:
np.ndarray, k_max: int) -> float` fully annotated (line 74). `_katz_1d(signal:
np.ndarray) -> float` fully annotated (line 136). Both `**call**` methods return
`xr.DataArray`, which satisfies the `xr.DataArray | Data` contract. No bare `Any`.

**LOW note**: `N` (line 76) is uppercase, deviating from PEP 8. This is an accepted
domain convention for signal length in the Higuchi literature; no change required.

## Safety & Style

No `print()` statements. `FractalDimHiguchi.__post_init__` validates `k_max >= 2`
(line 62). `_higuchi_1d` guards against `N <= k_max` with a clear `ValueError` (line
77). No mutation of input data in either class.

## Action List

None.
