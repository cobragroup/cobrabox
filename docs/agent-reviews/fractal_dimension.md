# Feature Review: fractal_dimension

**File**: `src/cobrabox/features/fractal_dimension.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Summary

Both `FractalDimHiguchi` and `FractalDimKatz` are well-structured, correctly typed,
and algorithmically verified. One HIGH-severity ruff error must be fixed before
merge: a Unicode MINUS SIGN character (`−`, U+2212) in the `FractalDimKatz`
docstring at line 105 that ruff flags as RUF002. Everything else — structure,
docstrings, typing, validation, safety — is clean. Two LOW items are noted for
awareness but require no code changes.

## Ruff

### `uvx ruff check`
```
RUF002 Docstring contains ambiguous `−` (MINUS SIGN). Did you mean `-` (HYPHEN-MINUS)?
   --> src/cobrabox/features/fractal_dimension.py:105:18
    |
103 |     Models the signal as a 2-D curve (sample index vs amplitude) and estimates
104 |     fractal dimension from the total Euclidean path length (L), the number of
105 |     steps (n = N − 1), and the maximum planar distance from the first sample to
    |                  ^
106 |     any other sample (d):

Found 1 error.
```

### `uvx ruff format --check`
```
RUF002 (same as above — format check surfaces the same error)
Found 1 error.
```

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
np.ndarray) -> float` fully annotated (line 141). Both `__call__` methods return
`xr.DataArray`, which satisfies the `xr.DataArray | Data` contract. No bare `Any`.

**LOW note**: `N` (line 76) is uppercase, deviating from PEP 8. This is an accepted
domain convention for signal length in the Higuchi literature; no change required.

## Safety & Style

No `print()` statements. `FractalDimHiguchi.__post_init__` validates `k_max >= 2`
(line 62). `_higuchi_1d` guards against `N <= k_max` with a clear `ValueError` (line
77). `_katz_1d` handles the degenerate zero-length/zero-distance case by returning
`1.0` (line 156). No mutation of input data in either class.

## Action List

1. [HIGH] Replace the Unicode MINUS SIGN `−` (U+2212) with an ASCII hyphen `-` at
   line 105 in the `FractalDimKatz` docstring: change `steps (n = N − 1)` to
   `steps (n = N - 1)`. This is a ruff RUF002 error that will fail CI.
