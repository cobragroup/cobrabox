# Feature Review: emd

**File**: `src/cobrabox/features/emd.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Summary

High-quality implementation of Empirical Mode Decomposition (EMD) with excellent documentation and thoughtful handling of multi-channel data with varying IMF counts. The feature correctly uses `BaseFeature[SignalData]`, validates parameters in `__post_init__`, and includes comprehensive docstrings. Only minor issue: missing `References` section for this published algorithm.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Correct structure throughout:

- `from __future__ import annotations` present at line 1
- `@dataclass` decorator with `BaseFeature[SignalData]` inheritance (line 16-17)
- No unnecessary `output_type` declaration — correctly preserves container type
- Class name `EMD` matches filename `emd.py`
- No redundant `_is_cobrabox_feature` marker
- `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray:` (line 80)
- Does not implement `apply()` — uses inherited implementation
- Imports in correct order: stdlib, third-party, internal

## Docstring

Excellent Google-style docstring with one minor omission:

- **One-line summary**: Clear and descriptive (line 18)
- **Extended description**: Comprehensive explanation of IMF handling and NaN padding rationale (lines 20-29)
- **Args**: All three fields documented with types and defaults (lines 31-42)
- **Returns**: Detailed description including shape, dimensions, and the `n_imfs` metadata dict (lines 48-58)
- **Raises**: Two `ValueError` conditions documented (lines 44-46)
- **References**: **MISSING** — EMD is a published algorithm (Huang et al., 1998) and should include a citation
- **Example**: Three usage examples showing typical patterns (lines 60-63)

## Typing

Fully typed:

- All fields have type annotations:
  - `max_imfs: int | None = None` (line 66)
  - `method: Literal["sift", "mask_sift", "iterated_mask_sift"] = "sift"` (line 67)
  - `keep_orig: bool = False` (line 68)
- `__call__` return type: `-> xr.DataArray` (line 80)
- `__post_init__` return type: `-> None` (line 70)
- Internal helper `_apply_emd` return type: `-> tuple[xr.DataArray, int]` (line 86)
- No bare `Any` types

## Safety & Style

Clean implementation:

- No `print()` statements
- Input validation in `__post_init__`:
  - Validates `method` is in `_SIFT_METHODS` (lines 72-76)
  - Validates `max_imfs` is positive if not `None` (lines 77-78)
- No mutation of input `data` — works on `data.data` and returns new arrays
- Line length compliant (100 chars max)
- Handles multi-channel data correctly with NaN padding for varying IMF counts (well-commented at lines 149-151)

## Action List

1. [Severity: LOW] Add `References` section citing Huang et al. (1998) "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis" and any other relevant EMD literature.
