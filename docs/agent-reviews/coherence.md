# Feature Review: coherence

**File**: `src/cobrabox/features/coherence.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Clean, well-structured feature implementing Welch's method for magnitude-squared coherence computation. Follows all project conventions with excellent docstring coverage, proper typing, and comprehensive input validation. The feature correctly uses `BaseFeature[SignalData]` since it operates on the time axis, and sets `output_type = Data` appropriately since the output is a correlation matrix (no time dimension).

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 14-15: Correctly decorated with `@dataclass` and inherits `BaseFeature[SignalData]` — appropriate for a time-series feature.

Line 46: `output_type: ClassVar[type[Data]] = Data` correctly set since coherence produces a spatial correlation matrix without time dimension.

Line 90: `__call__` signature is correct: `def __call__(self, data: SignalData) -> xr.DataArray:`.

No `apply()` override — correctly inherits from `BaseFeature`.

Imports are in correct order: stdlib → third-party → internal.

## Docstring

Comprehensive Google-style docstring with all required sections:

- One-line summary (line 16)
- Extended description explaining the algorithm and symmetry (lines 17-26)
- Args section documenting `nperseg` field (lines 28-30)
- Example section showing typical usage (lines 32-36)
- Returns section describing output dimensions and properties (lines 38-43)

Note: The Returns section correctly documents the extra singleton `time` dimension added by `BaseFeature.apply`.

## Typing

All fields are typed:

- Line 48: `nperseg: int | None = field(default=None)`

Line 90: `__call__` has proper return type annotation `-> xr.DataArray`.

No bare `Any` types present.

## Safety & Style

No `print()` statements — clean.

Input validation is thorough:

- `__post_init__` (lines 50-52): Validates `nperseg >= 2` at construction time
- `__call__` (lines 91-109): Validates presence of 'space' dimension, minimum 2 channels, and `nperseg` constraints against actual data length

No mutation of input `data` — operates on `data.data` and returns new `xr.DataArray`.

Line length within 100 character limit.

## Action List

None.
