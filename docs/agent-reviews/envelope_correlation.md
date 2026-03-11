# Feature Review: envelope_correlation

**File**: `src/cobrabox/features/envelope_correlation.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

A well-structured feature that correctly implements amplitude envelope correlation using `mne_connectivity`. The code is clean, properly typed, and follows the BaseFeature pattern. The only significant gap is the missing References section — since AEC is a published neuroimaging method with specific literature citations, this should be documented for scientific reproducibility.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 14: `@dataclass` decorator correctly applied.

Line 15: Inherits `BaseFeature[SignalData]` — appropriate since the feature operates on time-series data and uses `sampling_rate` implicitly via the Hilbert transform in the underlying library.

Line 50: `output_type: ClassVar[type[Data]] = Data` — correctly declared since the time dimension is removed and the result is a correlation matrix with dims `(space_to, space_from)`.

Line 55: `__call__` signature is correct: `def __call__(self, data: SignalData) -> xr.DataArray`. Takes `SignalData` as argument (not a field), returns `xr.DataArray`.

Line 92: No `apply()` method — correctly inherited from `BaseFeature`.

Imports (lines 1-12): Standard order maintained. No unused imports.

## Docstring

The docstring includes all required sections except References.

Lines 16-25: One-line summary and extended description are present and clear. Explains what AEC computes and mentions the orthogonalization option for reducing volume conduction effects.

Lines 27-33: Args section documents both fields (`orthogonalize`, `absolute`) with types and behavior descriptions.

Lines 35-37: Returns section describes the output dims `(space_to, space_from)` and that values are Pearson correlations.

Lines 39-42: Raises section documents both ValueError conditions (extra dimensions, insufficient spatial channels).

**Missing**: References section. Since this implements a specific published algorithm (amplitude envelope correlation), the primary literature citation should be included. This is important for scientific reproducibility and for users to understand the methodological basis.

Lines 44-47: Example section present with three usage patterns.

## Typing

Line 52: `orthogonalize: str | bool` — correctly typed union.

Line 53: `absolute: bool` — correctly typed.

Line 55: Return type `-> xr.DataArray` is explicit.

No bare `Any` types found.

## Safety & Style

No print statements found.

Lines 58-71: Input validation is present and appropriate:

- Checks for extra dimensions beyond `space` and `time` (line 58-63)
- Validates at least 2 spatial channels exist (lines 68-71)

Line 74: `values = xr_data.transpose("space", "time").values` — creates a copy, does not mutate input.

Line 87-92: Returns a new `xr.DataArray`, does not modify the input `data` object.

## Action List

1. [Severity: MEDIUM] Add a `References:` section to the docstring citing the primary literature for amplitude envelope correlation. The mne-connectivity documentation or the original AEC papers should be cited.
