# Feature Review: fourier_transform_surrogates

**File**: `src/cobrabox/features/fourier_transform_surrogates.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

Well-structured SplitterFeature implementing Fourier transform surrogates for time-series
null hypothesis testing. Clean implementation with proper validation and immutability
preservation. Missing required docstring sections (Returns, Raises, References) prevent
PASS status.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Clean structure throughout:

- Line 1: `from __future__ import annotations` present
- Line 13-14: Correct `@dataclass` + `SplitterFeature[SignalData]` inheritance
- Line 75: Correct `__call__` signature with `Iterator[SignalData]` return type
- No `apply()` override (correct for SplitterFeature)
- Class name matches filename (PascalCase conversion)

The `_rng` field (line 50) is properly marked as `field(init=False, repr=False)`.

## Docstring

**Incomplete — missing required sections.**

Present sections:

- One-line summary (line 15-16)
- Args section with all 4 fields documented (lines 18-28)
- Yields section (line 30-31) — should be "Returns:" per criteria
- Example section (lines 33-44)

Missing sections:

- **Returns** — SplitterFeature docstrings should use "Returns:" per criteria,
  describing the iterator of surrogate SignalData objects
- **Raises** — Feature raises `ValueError` in `__post_init__` (lines 53-56) for
  invalid `n_surrogates` values
- **References** — FT surrogate method is published (Theiler et al.); citation required

Suggested reference:

```text
Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Doyne Farmer, J. (1992).
Testing for nonlinearity in time series: the method of surrogate data.
Physica D: Nonlinear Phenomena, 58(1-4), 77-94.
```

## Typing

Excellent typing throughout:

- Line 46-50: All 4 dataclass fields have explicit type annotations
- Line 75: `__call__` correctly returns `Iterator[SignalData]`
- Line 52, 59: Methods properly typed with `-> None` and parameter types
- No bare `Any` types

## Safety & Style

Clean implementation:

- No print statements
- Lines 53-56: Proper validation in `__post_init__` with descriptive error messages
- Line 73: Correctly uses `data._copy_with_new_data()` preserving immutability
- Line 61-62, 65-68: NumPy operations work on copies, no input mutation

## Action List

1. [Severity: MEDIUM] Add `Returns:` section to docstring describing the iterator
   of SignalData objects (line 30, change "Yields:" to "Returns:").

2. [Severity: MEDIUM] Add `Raises:` section documenting the two ValueError
   conditions in `__post_init__` (after line 28).

3. [Severity: MEDIUM] Add `References:` section citing Theiler et al. (1992)
   or other appropriate FT surrogate literature (after Yields/Returns section).
