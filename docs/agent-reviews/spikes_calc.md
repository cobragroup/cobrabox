# Feature Review: spikes_calc

**File**: `src/cobrabox/features/spikes_calc.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Clean, well-structured feature that calculates spike counts using the IQR outlier method. Follows all conventions: correct base class usage (`BaseFeature[Data]`), appropriate `output_type` for scalar output, complete docstring with all required sections, input validation for empty data, and no code quality issues.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Correct structure throughout:

- `from __future__ import annotations` present (line 1)
- `@dataclass` decorator with `BaseFeature[Data]` inheritance (lines 13-14)
- `output_type: ClassVar[type[Data]] = Data` correctly set (line 31) since the feature returns a scalar count without time dimension
- Class name `SpikesCalc` matches filename `spikes_calc.py`
- `__call__` signature: `def __call__(self, data: Data) -> xr.DataArray:` (line 33)
- No `apply()` override — correctly inherits from `BaseFeature`
- Import order follows convention: **future**, stdlib, third-party, internal

## Docstring

Complete Google-style docstring with all required sections:

- One-line summary: "Calculate spikes in the input data using the IQR method." (line 15)
- Extended description explains the algorithm (lines 17-18)
- `Args:` section present (lines 20-21) — correctly states "None" since there are no dataclass fields
- `Returns:` section describes shape and content (lines 23-25)
- `Example:` section shows typical usage via `.apply()` (lines 27-28)

## Typing

Fully typed:

- No dataclass fields requiring types (feature has no parameters)
- `__call__` return type annotation: `xr.DataArray` (line 33)
- No bare `Any` types

## Safety & Style

- No `print()` statements
- Input validation: raises `ValueError` if input data is empty (lines 36-37)
- No mutation of input `data` — operates on `data.data.values` copy (line 34)
- Line length within 100 characters

## Action List

None.
