# Feature Review: bandpower

**File**: `src/cobrabox/features/bandpower.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

Bandpower is a well-implemented feature that computes frequency band power using Welch's method. The code is clean, properly typed, has comprehensive docstrings, and includes thorough input validation. All ruff checks pass. The implementation correctly handles default bands, custom band specifications, and edge cases like empty frequency masks.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Line 21-22: Correct `@dataclass` decorator and `BaseFeature[SignalData]` inheritance — appropriate since this is a time-series feature requiring the `time` dimension.

Line 65: `__call__` signature is correct: `def __call__(self, data: SignalData) -> xr.DataArray:`. Uses `SignalData` as the type parameter, which enforces the time dimension at the container level.

Line 119: Feature correctly returns `xr.DataArray` (not wrapped `Data`) — the `BaseFeature.apply()` method will handle wrapping and history.

Class name `Bandpower` matches filename `bandpower.py`.

No `apply()` override — correctly inherits implementation from `BaseFeature`.

## Docstring

Excellent docstring with all required sections:

- Line 23: Concise one-line summary
- Lines 25-28: Extended description explaining the algorithm
- Lines 30-44: `Args:` section documenting both `bands` and `nperseg` fields with detailed explanations of default bands
- Lines 46-49: `Example:` section with 3 working examples showing different usage patterns
- Lines 51-55: `Returns:` section describing output shape, dimensions, and units

The docstring is particularly strong in explaining the `bands` parameter flexibility (None, True for defaults, or explicit ranges).

## Typing

All fields properly typed:

- Line 58: `bands: dict[str, list[float] | bool] | None`
- Line 59: `nperseg: int | None`

Line 65: `__call__` return type `xr.DataArray` is correct for `BaseFeature`.

No bare `Any` types used.

## Safety & Style

**Validation:**

- Lines 61-63: `__post_init__` validates `nperseg >= 2`
- Lines 68-72: Validates `sampling_rate` is set on input data
- Lines 78-94: Validates band specifications — handles `True` (use default), `False` (error), unknown band names, and custom ranges

**Edge case handling:**

- Lines 107-110: Gracefully handles case where no frequencies fall within a band (returns zeros)
- Line 97-98: Transposes data to ensure time is last axis for welch

**No mutation:**

- Line 66: Works on `data.data` (the underlying xarray), not mutating input
- Line 99: Gets values via `.values` property
- Returns new `xr.DataArray` (line 115) with proper coordinates (lines 116-118)

**No print statements:** Clean — uses no debug output.

## Action List

None.
