# Feature Review: wavelet_transform

**File**: `src/cobrabox/features/wavelet_transform.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

Both `DiscreteWaveletTransform` and `ContinuousWaveletTransform` are well-implemented features with excellent documentation, comprehensive validation, and proper typing. The implementation follows all cobrabox conventions including correct use of `output_type` (DWT removes time dimension, CWT preserves it), thorough input validation in `__post_init__`, and clear docstrings with Args/Returns/Raises/Example sections.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 65-181: `DiscreteWaveletTransform` is a `@dataclass` inheriting `BaseFeature[SignalData]` — correct for a time-series feature.

Line 183-324: `ContinuousWaveletTransform` is a `@dataclass` inheriting `BaseFeature[SignalData]` — correct for a time-series feature.

Line 123: `output_type: ClassVar[type[Data] | None] = Data` — correctly set to `Data` since DWT removes the time dimension.

Line 243: `output_type: ClassVar[type[Data] | None] = None` — correctly set to `None` since CWT preserves the time dimension.

Both classes use descriptive PascalCase names that clearly indicate functionality. No `apply()` method — correctly inherited from `BaseFeature`.

Imports are well-organized: stdlib, third-party (pywt, numpy, xarray), then internal modules.

## Docstring

Both features have comprehensive Google-style docstrings:

- **One-line summary**: Present and descriptive (lines 67, 185)
- **Extended description**: Detailed algorithm explanations with behavior notes
- **Args**: All dataclass fields documented with types and constraints
- **Returns**: Shape, dimensions, and coordinate details clearly specified
- **Raises**: Lists all ValueError conditions (lines 95-98, 223-227)
- **Example**: Working code snippets using `.apply()` syntax (lines 100-106, 229-235)

Excellent use of reStructuredText-style inline code markers (`` `code` ``) for parameter names and code references.

## Typing

All fields have explicit type annotations:

- `_DwtWavelet` and `_CwtWavelet` type aliases for wavelet names (lines 15-62)
- Literal types for constrained string parameters (`mode`, `scaling`)
- `ClassVar` properly used for `output_type`

`__call__` signatures are correctly typed:

- `def __call__(self, data: SignalData) -> xr.DataArray:` (lines 136, 265)

Return types match the base class contract. No bare `Any` usage.

## Safety & Style

No `print()` statements found.

Input validation is comprehensive:

- `DiscreteWaveletTransform.__post_init__` (lines 125-134): Validates level >= 1, checks wavelet exists
- `ContinuousWaveletTransform.__post_init__` (lines 245-263): Validates scales not empty/positive, n_scales >= 1, scaling valid, wavelet exists
- Runtime validation in `DiscreteWaveletTransform.__call__` (lines 147-151): Checks level doesn't exceed max possible

No mutation of input `data` — both features work on `data.data` and return new DataArrays.

Code style is clean with appropriate line lengths.

## Action List

None.
