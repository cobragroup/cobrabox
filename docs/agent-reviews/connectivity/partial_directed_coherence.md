# Feature Review: partial_directed_coherence

**File**: `src/cobrabox/features/connectivity/partial_directed_coherence.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

Excellent, production-ready feature implementing Partial Directed Coherence (PDC) via VAR modeling. The code is clean, well-documented, follows all conventions, and ruff passes without issues. Strong mathematical implementation with proper validation and clear dimension semantics.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- Line 1: `from __future__ import annotations` present ✓
- Line 14-15: `@dataclass` decorator with `BaseFeature[SignalData]` inheritance ✓
- Line 58: `output_type: ClassVar[type[Data]] = Data` correctly set since PDC removes time dimension ✓
- Line 15: Class name `PartialDirectedCoherence` matches filename ✓
- Line 66: `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray` ✓
- No loose helper functions — all computation lives in `__call__` ✓
- Imports are clean and ordered correctly ✓

## Docstring

Comprehensive Google-style docstring with all required sections:

- Line 16: Clear one-line summary
- Lines 18-26: Extended description explaining PDC mathematics and normalization
- Lines 28-31: `Args:` section documenting both dataclass fields
- Lines 33-36: `Returns:` section with explicit shape `(n_channels, n_channels, n_freqs)` and dimensions
- Lines 38-42: `Raises:` section listing all validation errors
- Lines 44-47: `References:` section with full citation to Baccalá & Sameshima (2001)
- Lines 49-52: `Example:` section with usage via `.apply()`

## Typing

- Lines 55-56: All fields typed: `var_order: int | None = None`, `n_freqs: int = 128` ✓
- Line 66: `__call__` return type `xr.DataArray` explicit ✓
- No bare `Any` types ✓
- Line 58: `ClassVar` properly imported and used for `output_type` ✓

## Safety & Style

- No `print()` statements ✓
- Lines 61-64: `__post_init__` validates `var_order` and `n_freqs` are positive ✓
- Lines 67-83: `__call__` validates sampling_rate, dimensions, and channel count ✓
- Lines 86-131: No mutation of input `data` — works on `values` and returns new `xr.DataArray` ✓
- Line 116: Handles edge case of zero division in normalization ✓

## Action List

None.
