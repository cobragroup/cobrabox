# Feature Review: amplitude_variation

**File**: `src/cobrabox/features/amplitude_variation.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

A clean, minimal feature that computes standard deviation over the time dimension. Well-structured with proper dataclass inheritance, correct docstring sections, and clean ruff output. The feature correctly sets `output_type = Data` since it removes the time dimension, and sampling_rate becomes None as expected.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

- ✅ `from __future__ import annotations` present (line 1)
- ✅ `@dataclass` decorator with `BaseFeature[SignalData]` inheritance (line 13)
- ✅ Class name `AmpVar` matches filename `amplitude_variation.py` (PascalCase convention)
- ✅ Correct `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray` (line 34)
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly set (line 32) — feature removes time dimension
- ✅ No `apply()` override — correctly inherited from BaseFeature

## Docstring

- ✅ One-line summary present: "Compute amplitude variation over the time dimension."
- ✅ Extended description explains what amplitude variation is (std of EEG signal)
- ✅ `Args:` section present — correctly notes "None" since feature has no parameters
- ✅ `Returns:` section present — describes shape `(space,)` and mentions extra_dims handling
- ✅ `Example:` section present with correct `.apply()` usage

## Typing

- ✅ All imports typed correctly
- ✅ `__call__` return type `xr.DataArray` is correct (feature returns underlying array, not Data)
- ✅ ClassVar type annotation correct: `ClassVar[type[Data]]`
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ No input mutation — works on `data.data` and returns new array
- ✅ No validation needed — SignalData enforces 'time' dimension at construction
- ✅ Line length compliant (100 chars)

## Action List

None.
