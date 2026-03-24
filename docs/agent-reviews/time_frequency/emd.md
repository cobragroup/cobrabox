# Feature Review: EMD

**File**: `src/cobrabox/features/time_frequency/emd.py`
**Date**: 2025-03-24
**Verdict**: NEEDS WORK

## Summary

The EMD feature is well-structured overall with good typing, comprehensive docstring sections, and clean implementation. However, it is missing a required `References:` section for this published algorithm. The ruff checks pass cleanly and the code follows most conventions, but the literature citation is a significant omission for a feature implementing a well-established method like Empirical Mode Decomposition.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: `from __future__ import annotations` present — correct.

**Lines 16-17**: `@dataclass` decorator with `BaseFeature[SignalData]` inheritance — correct for a time-series feature that requires the `time` dimension.

**Lines 66-68**: All dataclass fields properly typed:

- `max_imfs: int | None = None`
- `method: Literal["sift", "mask_sift", "iterated_mask_sift"] = "sift"`
- `keep_orig: bool = False`

**Line 80**: `__call__` signature is correct: `def __call__(self, data: SignalData) -> xr.DataArray:`

**No `output_type` declared**: Correct — the feature adds an `imf` dimension but preserves the time dimension, so the default behavior (preserve input container type) is appropriate.

**Helper organization**: The nested `_apply_emd` function inside `__call__` (lines 86-114) is acceptable since it is only used within that method and keeps the module-level namespace clean.

**No `_is_cobrabox_feature` marker**: Correctly omitted (inherited from `BaseFeature`).

## Docstring

The docstring is comprehensive with most required sections present:

- One-line summary (line 18)
- Extended description (lines 20-29)
- `Args:` section with all fields documented (lines 31-42)
- `Raises:` section documenting ValueError conditions (lines 44-46)
- `Returns:` section with detailed output description (lines 48-58)
- `Example:` section with working snippets (lines 60-63)

**Missing**: `References:` section. EMD is a published algorithm (Huang et al., 1998) and should include a proper citation. This is required per the criteria for features implementing published algorithms.

## Typing

All typing is excellent:

- All fields have explicit type annotations
- `__call__` has correct return type `xr.DataArray`
- Uses `Literal` for the `method` parameter with proper validation via `_SIFT_METHODS` tuple
- Uses `int | None` union syntax (Python 3.10+)

## Safety & Style

- No `print()` statements
- Input validation in `__post_init__` (lines 70-78) validates `method` and `max_imfs`
- No mutation of input `data` — works on `data.data` and returns new arrays
- Line length within 100 characters (ruff verified)

**Minor note**: The validation uses a separate `_SIFT_METHODS` tuple (line 13) rather than `typing.get_args()` on the `Literal` type. While this works, using `get_args()` would make the `Literal` the single source of truth.

## Action List

1. [Severity: HIGH] Add a `References:` section to the docstring citing the original EMD paper:

   ```python
   References:
       Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis.
       *Proceedings of the Royal Society of London. Series A: Mathematical,
       Physical and Engineering Sciences*, 454(1971), 903-995.
   ```

2. [Severity: LOW] Consider using `typing.get_args()` to extract allowed values from the `Literal` type for validation instead of maintaining a separate `_SIFT_METHODS` tuple, making the type annotation the single source of truth.
