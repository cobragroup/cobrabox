# Feature Review: spikes_calc

**File**: `src/cobrabox/features/spikes_calc.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

Clean, well-structured feature that calculates spike counts using the IQR method. Correctly inherits from `BaseFeature[Data]` (generic, no time dimension required), has proper input validation for empty data, and returns a scalar 0-dimensional DataArray. Ruff check and format both pass. Docstring includes all required sections (summary, description, Args, Returns, Example). No issues found.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- Line 1: `from __future__ import annotations` present — correct.
- Line 13: `@dataclass` decorator applied.
- Line 14: Inherits `BaseFeature[Data]` — appropriate for a dimension-agnostic feature.
- Line 31: `output_type: ClassVar[type[Data]] = Data` — correct since output is a scalar (no time dimension).
- Line 33: `__call__` signature: `def __call__(self, data: Data) -> xr.DataArray` — matches base class contract.
- No `apply()` override — inherits correctly from `BaseFeature`.
- Imports are minimal and ordered correctly (stdlib → third-party → internal).

## Docstring

Complete Google-style docstring (lines 15–29):

- One-line summary explaining the IQR spike detection method.
- Extended description clarifying outlier bounds (±1.5*IQR).
- `Args:` section present (empty, which is correct — no dataclass fields).
- `Returns:` section describes shape `()`, dims `()`, and scalar float value.
- `Example:` section shows correct `.apply()` usage.

## Typing

- No dataclass fields to type (feature has no configurable parameters).
- `__call__` return type explicitly annotated as `xr.DataArray`.
- No bare `Any` types.

## Safety & Style

- Line 36–37: Input validation for empty data raises `ValueError` with clear message.
- No `print()` statements.
- No mutation of input `data` — works on `data.data.values` (line 34) and returns new array.
- All operations use numpy/xarray correctly.

## Action List

None.
