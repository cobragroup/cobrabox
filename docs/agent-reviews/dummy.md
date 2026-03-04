# Feature Review: dummy

**File**: `src/cobrabox/features/dummy.py`
**Date**: 2026-03-04
**Verdict**: NEEDS WORK

## Summary

`Dummy` is explicitly called out in CLAUDE.md as a negative reference ("do not model new features after it"), and the review confirms why. It is missing the mandatory `from __future__ import annotations` first import, has a one-line docstring with no `Args:`, `Returns:`, or `Example:` sections, and its `__call__` implementation reconstructs a `Data` object manually instead of returning an `xr.DataArray` (bypassing the `BaseFeature.apply` wrapping contract cleanly). There is also no input validation on the required `time` and `space` dimensions. The file exists as a skeleton test fixture rather than a production-quality feature.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

- **Line 1**: Missing `from __future__ import annotations`. This is mandatory as the first import in every feature file.
- **Lines 7–8**: `@dataclass` and `BaseFeature` inheritance are correct.
- **Class name**: `Dummy` is PascalCase and matches the filename `dummy.py`. Correct.
- **Line 14**: `__call__` signature is `(self, data: Data) -> Data`. The return type `Data` is acceptable (the union allows `xr.DataArray | Data`), so this is not a violation by itself, but see Docstring and Safety sections for related issues.
- **Fields**: `mandatory_arg: int` (line 11) and `optional_arg: int = 0` (line 12) are both typed. Correct.
- `data` is not a dataclass field. Correct.
- No reimplementation of `.apply()`. Correct.
- Import order: `dataclasses` (stdlib) before internal imports. Correct order, but `from __future__ import annotations` is absent.

## Docstring

- **Line 9**: Only a one-line summary `"""Dummy feature that returns the input data."""` is present.
- Missing `Args:` section. The fields `mandatory_arg` and `optional_arg` are not documented at all.
- Missing `Returns:` section describing shape, dimensions, and values of the output.
- Missing `Example:` section demonstrating `.apply()` usage.
- The one-line summary itself is technically inaccurate: `Dummy` does not return its input unchanged — it reconstructs a new `Data` from the numpy representation, losing all dimension coordinates (labels) and metadata beyond `sampling_rate`.

## Typing

- All fields are explicitly typed (`int`). Correct.
- `__call__` return type is annotated as `-> Data`. Correct.
- No bare `Any` usage. Correct.

## Safety & Style

- No `print()` statements. Correct.
- **No input validation**: `__call__` does not check that `data` has the required `time` and `space` dimensions before calling `data.to_numpy()`.
- **Lossy reconstruction**: Line 17 reconstructs `Data` via `Data.from_numpy(a, dims=list(data.data.dims), sampling_rate=sampling_rate)`. This drops all dimension coordinate labels (e.g. channel names, timestamps), all metadata fields (`subjectID`, `groupID`, `condition`, `history`, `extra`), and any non-`sampling_rate` attributes. For a feature meant only as a test fixture this may be acceptable, but it silently discards information in a surprising way.
- `__post_init__` is absent. No validation is performed on `mandatory_arg` or `optional_arg` constraints (e.g. whether they must be positive). If there are no constraints that is fine, but the field name `mandatory_arg` suggests some semantic constraint that is left undocumented and unenforced.

## Action List

1. [HIGH] Add `from __future__ import annotations` as the first line of the file.
2. [HIGH] Expand the docstring to include `Args:` (documenting `mandatory_arg` and `optional_arg`), `Returns:` (shape, dims, values), and `Example:` (via `.apply()`).
3. [MEDIUM] Add input validation in `__call__` to raise `ValueError` when `time` or `space` dimensions are absent.
4. [MEDIUM] Clarify or fix the lossy reconstruction on line 17: either use `_copy_with_new_data` to preserve metadata, or document explicitly that this feature strips all coordinates and metadata.
5. [LOW] Add a `__post_init__` method if `mandatory_arg` or `optional_arg` have numeric constraints (e.g. must be non-negative).
