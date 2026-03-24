# Feature Review: reciprocal_connectivity

**File**: `src/cobrabox/features/connectivity/reciprocal_connectivity.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

`ReciprocalConnectivity` is a well-implemented connectivity feature that computes per-channel net directional roles from directed connectivity measures (primarily PDC). The code is clean, properly typed, and follows all structural conventions. It supports two usage modes (time-series input and pre-computed matrix input) with thorough validation. The only minor improvement would be adding a `References:` section for the underlying connectivity methodology.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

**Line 1**: `from __future__ import annotations` present — correct.

**Line 13-14**: `@dataclass` decorator with `BaseFeature[Data]` inheritance. The use of `BaseFeature[Data]` (not `SignalData`) is appropriate because this feature accepts both time-series data (which gets converted via internal PDC) and pre-computed connectivity matrices.

**Line 77**: `output_type: ClassVar[type[Data]] = Data` is correctly declared since the feature reduces to a per-channel scalar, removing all original dimensions.

**Line 85**: `__call__` signature is `def __call__(self, data: Data) -> xr.DataArray:` — correct.

**Lines 71-75**: All dataclass fields are properly typed with union types where appropriate (`tuple[float, float] | None`, `int | None`).

**Lines 79-83**: `__post_init__` validates `freq_band` ordering (fmin < fmax).

No loose helper functions — all logic is contained within the class. No `apply()` override — correctly uses inherited method.

## Docstring

Comprehensive Google-style docstring with:

- **One-line summary** (lines 15-16): Clear description of what the feature computes.
- **Extended description** (lines 18-29): Explains the algorithm, mathematical formulation, and two usage modes.
- **Args:** (lines 37-48): All 5 dataclass fields documented with types and behavior.
- **Returns:** (lines 49-51): Describes output dimensions and shape.
- **Raises:** (lines 53-60): Lists 5 specific ValueError conditions.
- **Example:** (lines 62-69): Two working examples covering both usage modes.

**Minor gap**: No `References:` section. Since this feature implements Reciprocal Connectivity from directed connectivity measures (PDC-based), consider adding literature references if this follows a specific published methodology.

## Typing

- All fields typed with appropriate union types.
- `__call__` return type explicitly `xr.DataArray`.
- `ClassVar` used correctly for `output_type`.
- No bare `Any` types.
- Type ignore comment on line 103 is appropriate (`# type: ignore[arg-type]`) since PDC expects `SignalData` but the type narrowing happens via runtime check.

## Safety & Style

- **No `print()` statements** — clean.
- **Input validation**: Comprehensive validation throughout `__call__`:
  - Line 95-99: Validates `connectivity` parameter for time-series input.
  - Lines 109-113: Validates required dimensions for pre-computed matrix input.
  - Lines 118-123: Symmetry check for 2-D matrices (RC requires asymmetry).
  - Lines 129-147: Frequency band validation and range checking.
- **No mutation**: Creates copies (line 150: `mat.values.copy()`) and returns new DataArray without modifying input.
- **Algorithm safety**: Lines 161-162 mask diagonal with NaN before computing means to exclude self-connections.

## Action List

1. [Severity: LOW] Consider adding a `References:` section to the docstring if this implementation follows specific published literature on Reciprocal Connectivity or PDC-based source/sink analysis.
