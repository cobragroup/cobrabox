# Feature Review: coherence

**File**: `src/cobrabox/features/coherence.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

The Coherence feature is well-structured and implements magnitude-squared coherence computation correctly. It follows the dataclass pattern, uses appropriate base class typing (`BaseFeature[SignalData]`), and sets `output_type` correctly since it removes the time dimension. Code is clean, typed, and follows conventions. The only significant gap is the missing `Raises:` section in the class docstring.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 14: Correct use of `@dataclass` decorator.
Line 15: Properly inherits `BaseFeature[SignalData]` since this feature operates on time-series data.
Line 46: Correctly sets `output_type: ClassVar[type[Data]] = Data` because the output removes the time dimension (returns a coherence matrix instead of time-series).
Line 90: `__call__` signature correctly typed as `(self, data: SignalData) -> xr.DataArray`.

The class correctly does NOT implement `apply()` — this is inherited from `BaseFeature`.

Imports are clean and ordered correctly (stdlib → third-party → internal).

## Docstring

Comprehensive Google-style docstring with most required sections present:

- ✅ One-line summary (line 16)
- ✅ Extended description explaining algorithm (lines 17-26)
- ✅ Args section documenting `nperseg` field (lines 28-30)
- ✅ Returns section describing output shape and dimensions (lines 38-43)
- ✅ Example section showing typical usage (lines 32-36)
- ❌ **Missing Raises section** — the feature raises `ValueError` in both `__post_init__` (line 52) and `__call__` (lines 94, 101, 105, 108) but these are not documented

## Typing

All fields are properly typed:

- Line 48: `nperseg: int | None = field(default=None)` — correct union type with default

`__call__` return type is explicit: `-> xr.DataArray` (line 90).

No bare `Any` types present.

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation present: checks for 'space' dimension (line 93-94), minimum 2 spatial channels (lines 100-101), and validates `nperseg` constraints (lines 104-109)
- ✅ `__post_init__` validates `nperseg` parameter (lines 50-52)
- ✅ No mutation of input `data` — works on copy and returns new DataArray
- ✅ Line length within 100 characters

## Action List

1. [Severity: MEDIUM] Add `Raises:` section to class docstring documenting the ValueError conditions:
   - `ValueError`: If `nperseg` is provided and less than 2.
   - `ValueError`: If input data lacks a 'space' dimension.
   - `ValueError`: If fewer than 2 spatial channels are present.
   - `ValueError`: If computed `nperseg` is less than 2 or exceeds time samples.
