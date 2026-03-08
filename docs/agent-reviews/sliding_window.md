# Feature Review: sliding_window

**File**: `src/cobrabox/features/sliding_window.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Excellent implementation of a `SplitterFeature`. The code is clean, well-documented, and follows all conventions. Proper use of lazy generation with `yield`, correct validation in `__post_init__`, and comprehensive docstring with all required sections. A model feature implementation.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 12: Correct `@dataclass` decorator.
Line 13: Correct inheritance `SplitterFeature[SignalData]` — appropriate for a time-dimension window splitter.
Line 13: Class name `SlidingWindow` matches filename `sliding_window.py`.
Lines 36-37: Fields properly declared as dataclass fields with type annotations.
Line 45: Correct `__call__` signature `def __call__(self, data: SignalData) -> Iterator[Data]:` — matches `SplitterFeature` contract.
No `apply()` override — correctly relies on base class behavior for `SplitterFeature`.

## Docstring

Lines 14-34: Complete Google-style docstring with all required sections:

- One-line summary (line 14): Clear verb phrase describing the purpose.
- Extended description (line 16): Explains lazy generation behavior.
- Args section (lines 18-20): Documents both `window_size` and `step_size` with constraints.
- Returns section (lines 22-27): Describes generator behavior, output dimensions, and history handling.
- Example section (lines 29-33): Shows typical usage via constructor call.

## Typing

Line 36-37: Both fields have explicit type annotations (`int`).
Line 45: `__call__` return type explicitly annotated as `Iterator[Data]`.
No bare `Any` types found.

## Safety & Style

No `print()` statements.

Lines 39-43: `__post_init__` validates both `window_size >= 1` and `step_size >= 1` with clear `ValueError` messages.

Lines 49-50: Additional validation in `__call__` ensures `window_size <= n_time` with informative error.

Line 57: Correctly uses `data._copy_with_new_data()` to create new instances without mutating input — preserves immutability contract.

All lines under 100 characters.

## Action List

None.
