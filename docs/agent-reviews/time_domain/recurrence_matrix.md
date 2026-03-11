# Feature Review: recurrence_matrix

**File**: `src/cobrabox/features/recurrence_matrix.py`
**Date**: 2026-03-11
**Verdict**: PASS

## Summary

The implementation is well-structured and follows project conventions. The feature correctly uses `BaseFeature[SignalData]`, has thorough `__post_init__` validation, and includes proper type annotations. The docstring is comprehensive with all required sections. Ruff check and format are clean.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

`@dataclass class RecurrenceMatrix(BaseFeature[SignalData])` is the correct declaration. `output_type: ClassVar[type[Data]] = Data` is correctly set since the `(T, T)` output eliminates the time dimension. The feature has three operating modes based on input shape (2-D state-vector, 2-D window/FC, 3-D pre-computed FC) and supports five FC metrics (`pearson`, `spearman`, `MI`, `PLV`, `AEC`) and three recurrence metrics (`cosine`, `correlation`, `euclidean`). No issues found.

## Docstring

The class docstring includes a one-line summary, extended description covering all input modes, `Args:`, `Returns:`, `Raises:`, `Example:`, and `References:` sections. All sections are well-populated.

## Typing

`rec_metric: str`, `fc_options: list` are typed. Private derived fields are typed. Return annotation `-> xr.DataArray` is present and correct.

## Safety & Style

- No `print()` statements.
- No input mutation.
- `__post_init__` validation is thorough.
- Helper functions `_similarity_matrix` and `_fc_matrix` are well-structured.

## Action List

None.