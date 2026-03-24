# Feature Review: envelope_correlation

**File**: `src/cobrabox/features/time_domain/envelope_correlation.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

Excellent feature implementation. `EnvelopeCorrelation` is a well-structured, thoroughly documented connectivity feature that correctly computes amplitude envelope correlation between channel pairs. The code follows all cobrabox conventions: proper dataclass structure, comprehensive docstring with all required sections, appropriate input validation, and clean integration with `mne_connectivity`. The feature correctly declares `output_type = Data` since it removes the time dimension and returns a correlation matrix. The previously missing References section has been added with proper citation to Hipp et al. (2012).

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: `from __future__ import annotations` — correct first import.

**Lines 14-15**: `@dataclass` decorator with `BaseFeature[SignalData]` inheritance — appropriate for a time-series feature that operates on the time dimension.

**Line 56**: `output_type: ClassVar[type[Data]] = Data` — correctly declared since the feature returns a correlation matrix without time dimension.

**Line 61**: `def __call__(self, data: SignalData) -> xr.DataArray:` — correct signature matching base class contract.

**Lines 58-59**: Both dataclass fields properly typed: `orthogonalize: str | bool` and `absolute: bool`.

No loose helper functions; all logic is contained within the class. Clean import structure following the standard order.

## Docstring

Comprehensive Google-style docstring with all required sections:

- **One-line summary** (line 16): Clear verb phrase describing the computation.
- **Extended description** (lines 18-24): Explains the algorithm, Hilbert transform usage, and orthogonalization purpose.
- **Args** (lines 27-33): Both fields documented with types and behavior.
- **Returns** (lines 35-37): Describes output dimensions (`space_to`, `space_from`) and value type.
- **Raises** (lines 39-42): Two `ValueError` conditions documented (extra dimensions, insufficient channels).
- **References** (lines 44-48): Full citation to Hipp et al. (2012) with DOI — previously missing, now added.
- **Example** (lines 50-53): Three usage examples showing different parameter configurations.

## Typing

- All fields have explicit type annotations.
- `__call__` has correct return type `xr.DataArray`.
- **Minor suggestion**: The `orthogonalize` field could use `Literal["pairwise"] | bool` instead of `str | bool` for stricter typing, since only `"pairwise"` or `False` are valid values per the docstring. However, the current implementation is acceptable.

## Safety & Style

- No `print()` statements.
- Input validation (lines 64-77): Checks for extra dimensions and minimum 2 spatial channels with clear error messages.
- No mutation of input `data`: Creates new `xr.DataArray` and returns it.
- Line length within 100 characters.

## Action List

None.
