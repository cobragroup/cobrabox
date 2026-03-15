# Feature Review: sliding_window

**File**: `src/cobrabox/features/sliding_window.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

Excellent implementation of a `SplitterFeature`. The code is clean, well-documented, and follows all project conventions. The docstring is comprehensive with all required sections (Args, Returns, Example). Input validation is thorough with `__post_init__` checking parameter constraints and `__call__` verifying data dimensions. The lazy generator pattern is correctly implemented using `yield`. No issues found.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 12: Correct `@dataclass` decorator applied.
Line 13: Correctly inherits `SplitterFeature[SignalData]` — appropriate for a time-series windowing feature.
Lines 36-37: Properly typed dataclass fields with default values using `field()`.
Lines 45-57: Correct `__call__` signature matching base class contract: `Iterator[Data]`.
No `output_type` classvar — correct omission since `SplitterFeature` yields `Data` objects.
Class name `SlidingWindow` matches filename `sliding_window.py` in PascalCase.

## Docstring

Comprehensive Google-style docstring with all required sections:

- Lines 14-16: Clear one-line summary + extended description about lazy generation
- Lines 18-20: Args section with type and constraint documentation for both fields
- Lines 22-27: Returns section describing generator behavior and metadata preservation
- Lines 29-33: Working example showing typical usage

The docstring correctly notes that `history` is appended on each yielded window and metadata is preserved.

## Typing

Line 36-37: Both fields properly typed as `int`.
Line 45: `__call__` has correct parameter type `SignalData` matching the base class type parameter.
Line 45: `__call__` has correct return type `Iterator[Data]`.
All imports properly typed with `from __future__ import annotations` (line 1).
No bare `Any` types present.

## Safety & Style

Line 39-43: Excellent `__post_init__` validation for both `window_size` and `step_size` ensuring they are >= 1.
Line 49-50: Additional validation in `__call__` ensuring window_size does not exceed data length.
Line 57: Uses `_copy_with_new_data` for immutability — correct pattern.
No `print()` statements found.
Proper handling of input data without mutation.

## Action List

None.
