# Feature Review: coherence

**File**: `src/cobrabox/features/coherence.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

The `Coherence` feature is a high-quality implementation that computes magnitude-squared coherence between all pairwise channel combinations using Welch's method. It correctly inherits from `BaseFeature[SignalData]`, sets `output_type = Data` (since coherence removes the time dimension), and produces a symmetric matrix output with proper handling of extra dimensions. The code is well-structured, thoroughly documented, and passes all linting checks.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` — correct.

Line 14-15: `@dataclass` decorator with `BaseFeature[SignalData]` inheritance — correct. Uses `SignalData` because the algorithm inherently operates on time-series data.

Line 46: `output_type: ClassVar[type[Data]] = Data` — correctly declared since the output is a correlation matrix without time dimension.

Line 48: `nperseg: int | None = field(default=None)` — properly typed dataclass field.

Line 90: `def __call__(self, data: SignalData) -> xr.DataArray:` — correct signature, return type matches base class contract.

Imports are clean and in correct order: future annotations, stdlib (itertools, dataclasses, typing), third-party (numpy, xarray), internal (base_feature, data).

## Docstring

Excellent Google-style docstring with all required sections:

- **One-line summary** (line 16): "Compute magnitude-squared coherence for all pairwise channel combinations."
- **Extended description** (lines 18-26): Explains Welch's method, symmetric matrix output, handling of extra dimensions.
- **Args** (lines 28-30): Documents `nperseg` with type, default behavior, and constraints.
- **Example** (lines 32-36): Working snippet using `.apply()` with typical usage pattern.
- **Returns** (lines 38-43): Describes output dimensions, coordinates, value range, and symmetry property.

## Typing

All fields are typed:

- Line 46: `output_type: ClassVar[type[Data]]`
- Line 48: `nperseg: int | None`

Line 50: `def __post_init__(self) -> None:` — proper return type.

Line 54: `def _mean_squared_coherence(self, x: np.ndarray, y: np.ndarray, nperseg: int) -> np.ndarray:` — fully typed private helper.

Line 90: `def __call__(self, data: SignalData) -> xr.DataArray:` — correct.

No bare `Any` types found.

## Safety & Style

No `print()` statements — clean.

Input validation is comprehensive:

- Line 51-52: `__post_init__` validates `nperseg >= 2`
- Line 93-94: Validates presence of 'space' dimension
- Line 100-101: Validates at least 2 spatial channels
- Lines 103-109: Validates `nperseg` against data length

No mutation of input data — the feature works on `data.data` and returns a new `xr.DataArray` (lines 127-131).

Line length is within 100 characters throughout.

## Action List

None.
