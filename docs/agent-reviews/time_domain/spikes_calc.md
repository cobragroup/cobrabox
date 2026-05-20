# Feature Review: SpikeCount

**File**: `src/cobrabox/features/time_domain/spikes_calc.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

A clean, well-structured feature implementing IQR-based outlier detection. The code follows all project conventions with proper dataclass structure, complete docstring, correct typing, and appropriate input validation. Returns a proper 0-dimensional scalar DataArray as documented. No issues found.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` — present and correct.

Line 13-14: `@dataclass` decorator with `BaseFeature[Data]` inheritance — correct. Uses `Data` (not `SignalData`) since the feature is dimension-agnostic; it flattens all values for IQR calculation regardless of dimension structure.

Line 34: `output_type: ClassVar[type[Data]] = Data` — appropriate since the output removes all dimensions (returns a scalar count).

Line 36: `def __call__(self, data: Data) -> xr.DataArray:` — correct signature matching base class contract.

Line 54: Returns `xr.DataArray(float(spike_count))` — properly returns a 0-dimensional scalar DataArray without fake singleton dimensions.

No loose helper functions — all computation is inline in `__call__`, which is appropriate for a simple feature.

## Docstring

Complete Google-style docstring with all required sections:

- **One-line summary** (line 15): Clear verb phrase describing the feature.
- **Extended description** (lines 17-18): Explains the IQR method (1.5*IQR rule) and that it returns a scalar count.
- **Args** (lines 20-21): States "None" — appropriate since the feature has no configurable parameters.
- **Returns** (lines 23-25): Correctly describes the output shape `()` with no dimensions, containing a scalar float.
- **Raises** (lines 27-28): Documents `ValueError` for empty input data.
- **Example** (lines 30-31): Shows correct usage via `.apply()`.

No `References` section — acceptable since IQR outlier detection is a standard statistical method with no specific paper reference needed.

## Typing

- Line 34: `output_type` is properly typed as `ClassVar[type[Data]]`.
- Line 36: `__call__` has explicit parameter type `Data` and return type `xr.DataArray`.
- No bare `Any` types.
- Import of `ClassVar` from `typing` is present (line 4).

## Safety & Style

- **No print statements** — clean.
- **Input validation** (lines 39-40): Raises `ValueError` with clear message if input data is empty. Appropriate for a feature that computes quantiles.
- **No mutation**: Works on `data.data.values` (numpy array) and returns a new `xr.DataArray` — does not modify input.
- **Algorithm correctness**: Correctly implements the standard IQR outlier detection method:
  - Q1 at 0.25 quantile, Q3 at 0.75 quantile (lines 43-44)
  - IQR = Q3 - Q1 (line 45)
  - Bounds at Q1 - 1.5*IQR and Q3 + 1.5*IQR (lines 47-48)
  - Counts values outside bounds (line 51)

## Action List

None.
