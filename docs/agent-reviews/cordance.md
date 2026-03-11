# Feature Review: cordance

**File**: `src/cobrabox/features/cordance.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

Excellent, publication-quality feature implementing the Leuchter Cordance algorithm.
The code is well-structured, thoroughly documented, and follows all cobrabox conventions.
The docstring is exemplary with detailed algorithm description, complete parameter
documentation, proper citations, and clear examples. Input validation is robust
with checks in both `__post_init__` and `__call__`.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present.

Line 14: Correct `@dataclass` decorator on class.

Line 15: Correct inheritance `BaseFeature[SignalData]` — appropriate since this
is a spectral analysis feature requiring time-series data.

Line 92: Correct `output_type: ClassVar[type[Data]] = Data` — feature removes
the time dimension (returns band-wise spatial maps), so container downgrade is appropriate.

Line 110: Correct `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray`.

Lines 14, 94-98: All dataclass fields properly typed with no bare `Any`.

Lines 10-11: Internal imports use correct relative paths from `..base_feature` and `..data`.

Lines 100-108: `__post_init__` validation present for `nperseg`, `threshold`, and `output`.

## Docstring

Lines 16-90: Comprehensive Google-style docstring with all required sections:

- **One-line summary** (line 16): Clear verb phrase describing the feature.

- **Extended description** (lines 18-38): Excellent algorithm walkthrough with
detailed explanation of the 5-step Leuchter method, including the classification
logic and scoring formulas.

- **Args** (lines 40-68): Complete documentation for all 5 parameters with types,
defaults, and behavior descriptions.

- **Returns** (lines 75-78): Describes output dimensions `(band_index, space)`
and the meaning of positive/negative values.

- **References** (lines 80-89): Full academic citation plus patent reference — exemplary.

- **Example** (lines 70-73): Three usage examples showing different parameter combinations.

The only minor gap is the absence of an explicit `Raises:` section documenting
the `ValueError` exceptions raised in `__post_init__` (lines 101-108) and
`__call__` (lines 111-121). This is a LOW severity omission since the errors
have clear messages and the validation logic is self-documenting.

## Typing

All fields fully typed:

- `bands: dict[str, list[float] | bool] | None`
- `nperseg: int | None`
- `threshold: float`
- `output: Literal["cordance", "concordance", "discordance"]`
- `nan_on_zero: bool`

Return type `xr.DataArray` is explicit and correct.

No bare `Any` types present.

## Safety & Style

No `print()` statements — clean.

Lines 100-108: `__post_init__` validates numeric constraints:

- `nperseg >= 2`
- `0 < threshold < 1`
- `output` is valid literal

Lines 111-121: `__call__` validates dimension requirements:

- Checks for "space" dimension presence
- Checks for at least 2 spatial channels

Lines 129-135: Graceful handling of zero-power channels with informative error
message and `nan_on_zero` escape hatch for batch processing.

Lines 124-179: No mutation of input data — works on derived arrays and returns
new results.

## Action List

1. [Severity: LOW] Consider adding a `Raises:` section to the docstring documenting
   the `ValueError` conditions in `__post_init__` and `__call__` (lines 101-121).
   This is optional since the validation is well-implemented with clear messages.
