# Feature Review: SlidingWindowReduce

**File**: `src/cobrabox/features/windowing/sliding_window_reduce.py`
**Date**: 2026-03-24
**Verdict**: NEEDS WORK

## Summary

A well-structured feature that combines windowing and aggregation into a single step. The implementation is clean, properly typed, and uses xarray's rolling operations efficiently. However, the docstring is missing a `Raises:` section despite raising `ValueError` in multiple places.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

The feature correctly inherits from `BaseFeature[SignalData]` (line 13) since it operates on time-series data. The `output_type: ClassVar[type[Data]] = Data` (line 48) is properly declared because the feature removes the time dimension and adds a "window" dimension — the container type changes from `SignalData` to plain `Data`.

All dataclass fields are properly typed:

- `window_size: int` (line 43)
- `step_size: int` (line 44)
- `dim: str` (line 45)
- `agg: Literal["mean", "std", "sum", "min", "max"]` (line 46)

The `__call__` signature (line 59) correctly takes `SignalData` and returns `xr.DataArray`. The implementation uses xarray's rolling window with the aggregation method accessed dynamically via `getattr(rolling, self.agg)` (line 73), which is clean and Pythonic.

## Docstring

The docstring follows Google style with most required sections present:

- One-line summary (line 14)
- Extended description explaining the feature's purpose (lines 16-20)
- `Args:` section documenting all four fields (lines 22-27)
- `Returns:` section describing the dimension transformation (lines 29-32)
- `Example:` section showing typical usage (lines 34-40)

**Missing**: The `Raises:` section is absent despite the feature raising `ValueError` in both `__post_init__` (lines 51-57) and `__call__` (lines 62-69). All raised exceptions must be documented.

## Typing

Excellent typing throughout:

- All fields have explicit type annotations
- `Literal` is used correctly for the `agg` parameter with five valid options
- `__call__` return type is `xr.DataArray` (line 59)
- No bare `Any` types
- `ClassVar` is properly used for `output_type` (line 48)

## Safety & Style

- **No print statements** — clean implementation
- **Input validation** — thorough validation in `__post_init__` (window_size >= 1, step_size >= 1, agg in valid set) and `__call__` (dim exists, window_size <= data length)
- **No mutation** — works on `data.data` and returns a new renamed DataArray (line 81)
- **Logic is clear** — the window selection logic using `range(self.window_size - 1, n_dim, self.step_size)` (line 77) correctly aligns the window centers

## Action List

1. [Severity: MEDIUM] Add a `Raises:` section to the docstring documenting the three `ValueError` conditions:
   - When `window_size` or `step_size` is less than 1 (in `__post_init__`, lines 51-54)
   - When `agg` is not one of the valid strings (in `__post_init__`, line 56)
   - When the specified `dim` is not found in data dimensions (in `__call__`, line 63)
   - When `window_size` exceeds the data length along the specified dimension (in `__call__`, lines 66-69)
