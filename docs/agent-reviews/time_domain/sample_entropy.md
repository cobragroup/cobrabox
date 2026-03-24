# Feature Review: sample_entropy

**File**: `src/cobrabox/features/time_domain/sample_entropy.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

A well-implemented feature following the dataclass pattern. The SampleEntropy feature correctly computes sample entropy using a nested helper function approach that captures local variables efficiently. All structural requirements are met, docstring sections are complete (with one minor omission), and the implementation properly handles validation and dimensionality reduction.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: Correctly has `from __future__ import annotations` as the first import.

**Line 14**: Proper `@dataclass` decorator with `BaseFeature[SignalData]` inheritance. The `SignalData` type parameter is appropriate since this feature operates on the time axis.

**Line 54**: Correctly declares `output_type: ClassVar[type[Data] | None] = Data` since the feature collapses the time dimension.

**Line 66**: `__call__` signature correctly typed as `def __call__(self, data: SignalData) -> xr.DataArray`. The return type could be `xr.DataArray | Data` per the base class contract, but `xr.DataArray` is acceptable.

**Lines 74-99**: The helper functions `_sampen_one` and `_count` are defined inside `__call__` rather than at module level. This is acceptable here because they are closures that capture `self` (for `m`, `r`, `log_base`) and `ln_base`, avoiding the need to pass these as repeated arguments.

## Docstring

All required sections are present:

- **One-line summary** (line 15): Brief but clear.
- **Extended description** (lines 17-25): Explains the algorithm well, including the default base-2 logarithm divergence from the original definition.
- **Args** (lines 30-37): Documents all three dataclass fields (`m`, `r`, `log_base`) with types and constraints.
- **Returns** (lines 39-41): Correctly describes the collapsed time dimension.
- **Raises** (lines 43-46): Lists all three `ValueError` conditions raised in `__post_init__` and `_sampen_one`.
- **Example** (lines 48-50): Shows both default base-2 and natural logarithm usage.

**Missing**: `References:` section. Sample Entropy is a published algorithm (Richman & Moorman, 2000), so a citation should be added for completeness.

## Typing

All fields properly typed:

- `m: int = 2` (line 56)
- `r: float | None = None` (line 57)
- `log_base: float = 2` (line 58)

No bare `Any` types. The `ClassVar` type annotation on line 54 is correctly used for `output_type`.

## Safety & Style

- **No print statements**: Clean.
- **Input validation**: `__post_init__` (lines 60-64) validates `m >= 1` and valid `log_base`. Additional validation in `_sampen_one` (lines 76-80) checks time series length against embedding dimension.
- **No mutation**: The implementation works on `data.data` and returns a new `xr.DataArray` without modifying the input.
- **Line length**: All lines within 100 character limit.

The use of `xr.apply_ufunc` with `vectorize=True` and `dask="parallelized"` (lines 103-111) is idiomatic for xarray operations that reduce over a dimension.

## Action List

1. [Severity: LOW] Add a `References:` section citing the original Sample Entropy paper (Richman & Moorman, 2000) to align with the criteria for published algorithms.
