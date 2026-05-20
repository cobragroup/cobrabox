# Feature Review: spectrogram

**File**: `src/cobrabox/features/frequency_domain/spectrogram.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Summary

The `Spectrogram` feature is well-implemented and follows all major conventions.
It computes time-frequency power spectrograms using scipy's Welch-style STFT,
supporting multiple scaling modes (log, density, spectrum, magnitude). The code
is clean, properly typed, thoroughly documented, and passes all ruff checks.
The only minor suggestion is to consider using `Literal` typing for the scaling
parameter to improve type safety.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 1: `from __future__ import annotations` present and correct.

Line 16-17: Proper `@dataclass` decorator with `BaseFeature[SignalData]` inheritance.
Correct choice of base class since the feature operates on time-series data.

Line 17: Class name `Spectrogram` matches filename `spectrogram.py`.

Lines 59-62: All dataclass fields properly typed:

- `nperseg: int | None = None`
- `noverlap: int | None = None`
- `window: str = "hann"`
- `scaling: str = "log"`

Line 64: `__call__` signature correctly typed:
`def __call__(self, data: SignalData) -> xr.DataArray`

Lines 85-86: Good handling of extra dimensions — transposes to uniform indexing
before processing, preserving arbitrary extra dimensions.

Lines 93-136: Clean loop-based implementation that handles both `spectrogram`
and `stft` paths depending on scaling mode. Output shape determination via
test call (lines 97-111) is a pragmatic approach.

No loose helper functions — all logic contained within the class.

## Docstring

Lines 18-57: Comprehensive Google-style docstring with all required sections.

- **One-line summary** (line 18): Clear and concise.
- **Extended description** (lines 20-24): Explains the algorithm (Welch-style STFT)
  and dimension handling.
- **Args** (lines 26-41): All four fields documented with types and constraints.
  The scaling options are well-explained with inline bullets.
- **Returns** (lines 43-46): Describes output dimensions clearly:
  `(*extra_dims, "space", "frequency", "time")`.
- **Raises** (lines 48-50): Documents ValueError conditions.
- **Example** (lines 52-56): Working doctest-style example showing `.apply()` usage.

## Typing

All fields have explicit type annotations. The `__call__` return type is correct.

**Minor suggestion**: The `scaling` field uses plain `str` with a separate
`_VALID_SCALINGS` tuple for runtime validation (line 13). Consider using
`Literal` typing for better static analysis:

```python
from typing import Literal, get_args

Scaling = Literal["log", "density", "spectrum", "magnitude"]

@dataclass
class Spectrogram(BaseFeature[SignalData]):
    scaling: Scaling = "log"
    ...
```

This would allow IDEs to autocomplete valid values and catch typos at type-check time.

## Safety & Style

- No `print()` statements.
- Proper input validation in `__call__` (lines 67-80):
  - Validates `scaling` against allowed values.
  - Validates `nperseg >= 2` and `nperseg <= n_time`.
  - Validates `noverlap < nperseg` when provided.
- Handles `sampling_rate is None` gracefully (falls back to 1.0, line 82).
- No mutation of input `data` — works on `data.data` and returns new `xr.DataArray`.
- Line 134: Good safety practice using `np.maximum()` with `tiny` to avoid `-inf`
  in log scaling.

## Action List

1. [Severity: LOW] Consider using `Literal` typing for `scaling` parameter
   (lines 13, 62) to improve type safety and enable IDE autocomplete.
   This is optional — the current implementation is functional.
