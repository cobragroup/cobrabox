# Feature Review: Bandpower

**File**: `src/cobrabox/features/frequency_domain/bandpower.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

A well-implemented feature following all cobrabox conventions. The Bandpower feature computes
frequency band power using Welch's method with proper validation, clear docstring, and clean
typing. The code handles default bands elegantly and supports custom band specifications.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Excellent structure throughout:

- **Line 1**: `from __future__ import annotations` present
- **Line 21-22**: Correct `@dataclass` + `BaseFeature[SignalData]` inheritance for a time-series feature
- **Line 22**: Class name `Bandpower` matches filename `bandpower.py`
- **Line 70**: `__call__` signature correctly typed as `def __call__(self, data: SignalData) -> xr.DataArray`
- **Lines 12-18**: Module-level `_DEFAULTS` dict is appropriate (band definitions are constants)
- **Lines 63-64**: All fields properly typed with `field()` from dataclasses

The feature correctly omits `output_type` since it preserves the container type behavior
(returns a DataArray that `BaseFeature.apply` will wrap appropriately).

## Docstring

Comprehensive Google-style docstring with all required sections:

- **Lines 23-61**: Full docstring present
- **Line 23**: Clear one-line summary
- **Lines 25-28**: Extended description explains algorithm (Welch's method + integration)
- **Lines 30-44**: `Args:` section documents both `bands` and `nperseg` with detailed explanations
- **Lines 51-55**: `Returns:` section describes output shape and units
- **Lines 57-60**: `Raises:` section lists all three ValueError conditions
- **Lines 46-49**: `Example:` section with three usage examples

Minor suggestion: Could add a `References:` section citing Welch's method paper, though this
is optional for well-known signal processing methods.

## Typing

Full type coverage:

- **Line 63**: `bands: dict[str, list[float] | bool] | None` — complex union type handled well
- **Line 64**: `nperseg: int | None` — optional int with None default
- **Line 70**: Return type `xr.DataArray` explicit
- **Line 66**: `__post_init__` return type `None` declared
- **Line 81**: Type annotation for `resolved` variable: `dict[str, tuple[float, float]]`

The band specification type (`dict[str, list[float] | bool] | None`) is complex but necessary
to support the flexible API (True for defaults, list for custom ranges, None for all defaults).

## Safety & Style

- **Lines 67-68**: Input validation in `__post_init__` for `nperseg < 2`
- **Lines 73-77**: Runtime validation for missing `sampling_rate` with helpful error message
- **Lines 86-97**: Validation of band names and spec values with clear error messages
- **Line 103**: Correct use of `xr_data.transpose(..., "time")` to ensure time is last axis
- **Lines 111-115**: Safe integration with mask check to handle edge cases
- **Line 121-124**: Clean coordinate assignment using dict union operator

No print statements. No mutation of input data. All operations create new arrays.

## Action List

None.
