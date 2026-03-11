# Feature Review: granger_causality

**File**: `src/cobrabox/features/granger_causality.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

Both `GrangerCausality` and `GrangerCausalityMatrix` are well-implemented features with excellent documentation. The code correctly implements the log-ratio method for Granger causality testing, includes comprehensive parameter validation, and produces properly shaped outputs. The helper function `_granger_log_ratio` is thoroughly documented with mathematical formulas and interpretation guidelines. Minor improvements suggested around docstring completeness.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Both classes correctly use:

- `from __future__ import annotations` as first import (line 1)
- `@dataclass` decorator with `BaseFeature[SignalData]` inheritance (lines 69, 145)
- `output_type: ClassVar[type[Data]] = Data` appropriately set since these features remove the time dimension (lines 109, 178)
- Descriptive PascalCase class names matching the filename
- Correct `__call__` signatures taking `data: SignalData` and returning `xr.DataArray` (lines 123, 193)
- No `apply()` override — correctly inherits from base class
- Clean import structure following the standard order

Note: `_is_cobrabox_feature` marker is not needed — class-based features automatically inherit this from `BaseFeature`.

## Docstring

Both classes have comprehensive Google-style docstrings with:

- Clear one-line summaries
- Extended descriptions explaining the algorithm
- Complete `Args:` sections for all dataclass fields
- Detailed `Returns:` sections describing output dimensions
- Working `Example:` sections showing `.apply()` usage

**Missing sections** (not blocking but recommended):

- No `Raises:` section documenting the `ValueError` exceptions raised in `__post_init__` (lines 117-121, 184-191)
- No `References:` section for this published algorithm — Granger causality is a well-established method (Granger, 1969) that should be cited

The helper function `_granger_log_ratio` (lines 13-66) has exceptional documentation with mathematical formulas, interpretation guidelines, and detailed argument descriptions.

## Typing

All fields have proper type annotations:

- `coord_x: str | int | None = None` (line 111)
- `coord_y: str | int | None = None` (line 112)
- `lag: int | None = None` (line 113)
- `maxlag: int = 1` (line 114)
- `coords: list[str] | list[int] | None = None` (line 180)

`__call__` return types are correctly annotated as `xr.DataArray`. No bare `Any` types found.

## Safety & Style

- No `print()` statements
- Input validation in `__post_init__` for both classes (lines 117-121, 184-191), including validation that `coords` is not an empty list (line 190-191)
- No mutation of input `data` — works on `data.data` and returns new arrays
- Line length within 100 characters
- Clean separation of concerns: helper function handles the core computation, classes handle the feature interface

## Action List

1. [Severity: LOW] Add `Raises:` section to `GrangerCausality` docstring documenting `ValueError` conditions (line 117-121).
2. [Severity: LOW] Add `Raises:` section to `GrangerCausalityMatrix` docstring documenting `ValueError` conditions (line 184-191).
3. [Severity: LOW] Add `References:` section to both class docstrings citing Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods". Econometrica, 37(3), 424-438.
