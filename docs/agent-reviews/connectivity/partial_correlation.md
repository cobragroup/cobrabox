# Feature Review: partial_correlation

**File**: `src/cobrabox/features/partial_correlation.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

Well-structured feature with two related classes for partial correlation computation. Clean implementation with good validation and clear docstrings. However, `PartialCorrelation` returns fake singleton dimensions instead of a proper 0-dimensional scalar, violating the no-fake-dimensions guideline. `PartialCorrelationMatrix` correctly returns a 2D matrix with appropriate dimensions.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Both classes properly inherit from `BaseFeature[SignalData]` and use `@dataclass`. They correctly set `output_type: ClassVar[type[Data]] = Data` since they remove the time dimension.

- Line 54: `PartialCorrelation` — correct base class
- Line 133: `PartialCorrelationMatrix` — correct base class
- Line 78, 157: Correct `output_type` declarations

Helper function `_compute_partial_correlation` is well-structured with proper docstring.

## Docstring

Both classes have complete Google-style docstrings:

- One-line summaries present (lines 55, 134)
- Extended descriptions explain algorithm context
- Args sections document all dataclass fields
- Returns sections describe output shape
- Raises sections list expected exceptions
- Example sections show `.apply()` usage (lines 73-76, 152-155)

## Typing

All fields properly typed:

- Line 80-82: `coord_x: str | int`, `coord_y: str | int`, `control_vars: list[str] | list[int]`
- Line 159-160: `coords: list[str] | list[int]`, `control_vars: list[str] | list[int]`

`__call__` return types correctly annotated as `xr.DataArray` (lines 84, 162).

## Safety & Style

No `print()` statements. Input validation is thorough:

- Lines 88-93: Checks for required dimensions
- Lines 97-113: Validates coordinates exist in space dimension
- Lines 106-107, 173-177: Ensures non-empty coordinate lists
- Lines 39-46: Handles singular matrix error with clear message

No mutation of input data — works on `data.data` and returns new arrays.

## Action List

1. [Severity: MEDIUM] `PartialCorrelation.__call__` creates fake singleton dimensions (lines 123-129). The result is expanded to have `time` and `space` dimensions with single values, which violates the guideline that scalar outputs should be 0-dimensional. Return a proper 0-d DataArray instead:

```python
# Current (incorrect)
return (
    xr.DataArray(result, dims=[])
    .expand_dims(time_dim, axis=0)
    .assign_coords({time_dim: [time_coord[0]]})
    .expand_dims(space_dim, axis=0)
    .assign_coords({space_dim: [self.coord_x]})
)

# Should be
return xr.DataArray(result)
```
