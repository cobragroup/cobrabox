# Feature Review: lempel_ziv

**File**: `src/cobrabox/features/lempel_ziv.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

This is a clean, well-structured feature implementing Lempel-Ziv Complexity (LZC) for time-series data. The code correctly inherits from `BaseFeature[SignalData]`, uses `output_type = Data` since the feature removes the time dimension, and implements an efficient LZ76 counting algorithm adapted from NeuroKit2. The docstring is complete with all required sections including citations. The feature uses `xr.apply_ufunc` for vectorized computation across channels.

## Ruff

### `uvx ruff check`
Clean — no issues found.

### `uvx ruff format --check`
Clean — no formatting issues.

## Signature & Structure

Line 21: Correct `@dataclass` decorator with `BaseFeature[SignalData]` base class.

Line 48: Proper `output_type: ClassVar[type[Data]] = Data` declaration indicating the feature removes the time dimension.

Line 50: Correct `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray`.

No `apply()` override — correctly relies on inherited behavior.

Class name `LempelZiv` matches filename `lempel_ziv.py`.

Imports are ordered correctly:
1. `from __future__ import annotations`
2. stdlib (`math`, `dataclasses`, `typing`)
3. third-party (`numpy`, `xarray`)
4. internal (`..base_feature`, `..data`)

## Docstring

Complete Google-style docstring with all required sections:

- Line 22: One-line summary explaining what the feature computes.
- Lines 24-30: Extended description explaining the binarization process, LZC meaning, and algorithm attribution.
- Lines 32-33: `Args:` section (empty, appropriate since no dataclass fields).
- Lines 35-38: `Returns:` section describing shape `(space,)` and value range `(0, 1]`.
- Lines 40-42: `References:` section with proper IEEE citation.
- Lines 44-45: `Example:` section with correct `.apply()` usage.

## Typing

- Line 48: `output_type: ClassVar[type[Data]] = Data` — correctly typed with pyright ignore comment for the override.
- Line 50: `__call__` returns `xr.DataArray` as expected for `BaseFeature`.
- Line 54: `_lzc_1d` is a static method returning `float`.
- Line 61: `_count` is a static method returning `tuple[int, int]`.
- No bare `Any` types.

## Safety & Style

- No `print()` statements.
- Line 56: Correctly works on `signal` (numpy array), not mutating input `data`.
- Uses `SignalData` type parameter so the time dimension is enforced at construction — no redundant validation needed.
- The LZ76 algorithm implementation (lines 60-96) is a faithful adaptation from NeuroKit2 with proper attribution.
- Line 51: Uses `xr.apply_ufunc` with `vectorize=True` for efficient computation across channels.

## Action List

None.
