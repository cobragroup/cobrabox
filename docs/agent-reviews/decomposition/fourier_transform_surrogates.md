# Feature Review: FourierTransformSurrogates

**File**: `src/cobrabox/features/decomposition/fourier_transform_surrogates.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Summary

This feature is well-implemented and passes all quality criteria. It correctly implements the `SplitterFeature[SignalData]` pattern for generating Fourier transform surrogates, with complete docstring documentation, proper type annotations, input validation in `__post_init__`, and clean ruff formatting. The feature follows the cobrabox conventions precisely.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- **Line 1**: `from __future__ import annotations` is correctly placed as the first import.
- **Line 14**: Class correctly decorated with `@dataclass` and inherits `SplitterFeature[SignalData]` — appropriate for a feature that yields multiple `SignalData` windows.
- **Lines 60-63**: All dataclass fields are properly typed (`n_surrogates: int`, `multivariate: bool`, `return_data: bool`, `random_state: np.random.Generator | int | None`).
- **Line 64**: The `_rng` field uses `field(init=False, repr=False)` correctly for runtime-generated state.
- **Lines 66-71**: `__post_init__` validates `n_surrogates` is a non-negative integer as required.
- **Lines 89-94**: `__call__` correctly yields `Iterator[SignalData]` — first optionally returns original data, then surrogates.

No issues found.

## Docstring

The docstring is comprehensive and follows Google style:

- **Line 15**: One-line summary clearly describes the feature's purpose.
- **Lines 16-26**: Extended description explains the algorithm (phase randomization preserving power spectrum) and its use for null-hypothesis testing.
- **Lines 27-35**: `Args:` section documents all four dataclass fields with types and descriptions.
- **Lines 37-40**: `Raises:` section correctly documents the validation errors.
- **Lines 41-46**: `References:` includes the full citation to Theiler et al. (1992).
- **Lines 47-58**: `Example:` shows practical usage via `.apply()` with the feature name `cb.feature.FourierTransformSurrogates`.

All required sections are present and complete.

## Typing

- All dataclass fields have explicit type annotations.
- `__call__` return type `Iterator[SignalData]` matches the `SplitterFeature[SignalData]` contract.
- No bare `Any` types are used.
- `random_state` correctly uses union type `np.random.Generator | int | None`.

No issues found.

## Safety & Style

- No `print()` statements.
- Input validation for `n_surrogates` is properly implemented in `__post_init__`.
- Input `data` is not mutated — `_surrogate()` creates new `SignalData` via `_copy_with_new_data`.
- Line lengths comply with the 100-character limit.

No issues found.

## Action List

None.
