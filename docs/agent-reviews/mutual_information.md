# Feature Review: mutual_information

**File**: `src/cobrabox/features/mutual_information.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

The `MutualInformation` feature implements a solid algorithm for computing pairwise mutual information between series. However, it has several documentation issues that need to be fixed: missing required docstring sections (Args, Returns, Example), typos in the existing docstring, and the required `from __future__ import annotations` import is absent. The implementation itself is clean and well-structured, but the documentation does not meet the project standards.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: Missing `from __future__ import annotations` — this is a required import that must be the first line.

**Line 12**: The class correctly inherits from `BaseFeature[SignalData]` and is decorated with `@dataclass`.

**Line 53**: The `output_type: ClassVar[type[Data] | None] = Data` is correctly set since MI removes the time dimension.

**Line 66**: The `__call__` signature has `data: Data` but should be `data: SignalData` to match the type parameter of the base class.

## Docstring

**Line 17**: Typo — "Mututal" should be "Mutual".

**Line 26**: Typo — "parmeters" should be "Parameters", and the section should be named "Args:" not "Optional parmeters".

**Line 29-42**: The Args section is present but uses the wrong header format. It should be:

```python
Args:
    dim: ...
    other_dim: ...
```

instead of the current indented "Optional parmeters" section.

**Missing Returns section**: The docstring lacks a `Returns:` section describing the output DataArray shape, dimensions (space_from, space_to), and what the values represent.

**Missing Example section**: There is no `Example:` section showing typical usage via `.apply()`.

## Typing

All dataclass fields are properly typed (lines 46-51). The `__call__` return type is `xr.DataArray` (line 66), which is acceptable though the base class contract allows `xr.DataArray | Data`. No bare `Any` types are present.

## Safety & Style

No `print()` statements found. Input validation is present in `__post_init__` (lines 55-64) and `__call__` (lines 67-80). No mutation of input `data` — the feature works on `data.data` and returns new arrays. Line length is within the 100 character limit.

## Action List

1. [Severity: HIGH] Add `from __future__ import annotations` as the first import line.
2. [Severity: HIGH] Fix docstring section header from "Optional parmeters" to "Args:" and fix the typo "parmeters" → "parameters".
3. [Severity: HIGH] Fix typo "Mututal" → "Mutual" in the docstring (line 17).
4. [Severity: MEDIUM] Add missing `Returns:` section describing the output DataArray with dimensions "space_from" and "space_to".
5. [Severity: MEDIUM] Add missing `Example:` section showing usage: `>>> result = cb.feature.MutualInformation().apply(data)`.
6. [Severity: LOW] Change `__call__` parameter type from `data: Data` to `data: SignalData` to match the base class type parameter.
