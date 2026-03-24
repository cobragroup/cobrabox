# Feature Review: min

**File**: `src/cobrabox/features/reductions/min.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

A clean, well-structured generic feature that computes the minimum value across a specified dimension. The implementation correctly inherits from `BaseFeature[Data]` since it works on any dimension (not just time). It follows all cobrabox conventions: proper `@dataclass` decorator, complete Google-style docstring with Args, Returns, and Example sections, typed fields and return types, input validation with clear error messages, and no mutation of input data. Ruff checks pass cleanly.

The decision to NOT set `output_type` is correct — since `Min` can reduce any dimension (not inherently time), forcing `output_type = Data` would be inappropriate. The base class correctly preserves the input container type.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 11: Correct `@dataclass` decorator applied.

Line 12: Properly inherits `BaseFeature[Data]` — appropriate choice since this is a dimension-agnostic reduction feature (works on any dimension, not just time). The generic `BaseFeature[Data]` is correct; `SignalData` would be inappropriate since the feature does not require a time dimension.

Line 12: Class name `Min` matches filename `min.py` in PascalCase.

Line 29: `__call__` signature correctly takes `data: Data` and returns `xr.DataArray`.

Lines 1-8: Import order follows convention (`__future__`, stdlib, third-party, internal).

No `apply()` override — correctly inherits from `BaseFeature`.

No `output_type` set — this is correct! Since `Min` is a generic reducer that works on any dimension, setting `output_type = Data` would incorrectly force all outputs to plain `Data` even when reducing non-time dimensions (e.g., reducing "space" should keep `SignalData`).

## Docstring

Lines 13-25: Google-style docstring with most required sections:

- One-line summary (line 13) clearly states purpose
- `Args:` section (lines 15-16) documents the `dim` field with type and description
- `Returns:` section (lines 18-21) describes shape change and value semantics
- `Example:` section (lines 23-24) shows correct `.apply()` usage pattern

**Minor gap**: Missing `Raises:` section. The feature raises `ValueError` on line 31 when the specified dimension is not found, but this is not documented in the docstring.

## Typing

Line 27: Field `dim: str` has explicit type annotation.

Line 29: `__call__` has full type signature `def __call__(self, data: Data) -> xr.DataArray`.

No bare `Any` types present.

## Safety & Style

Line 30-31: Input validation checks that `dim` exists in data dimensions and raises `ValueError` with a clear message if not.

Line 32: Returns result of `data.data.min()` without mutating input — follows immutability convention.

No `print()` statements present.

Line 31: Line length is 100 characters, within the limit.

## Action List

1. [Severity: LOW] Add `Raises:` section to docstring documenting: `ValueError: If ``dim`` is not found in data dimensions.`
