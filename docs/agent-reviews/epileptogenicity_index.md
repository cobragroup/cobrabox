# Feature Review: epileptogenicity_index

**File**: `src/cobrabox/features/epileptogenicity_index.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

The `EpileptogenicityIndex` feature is a comprehensive migration implementing the Bartolomei et al. (2008) algorithm for quantifying epileptogenicity per channel. The implementation includes two helper methods (`_energy_ratio` and `_page_hinkley`) as class methods, proper handling of frequency bands, and complete documentation. All criteria are met.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 9**: ✅ `from __future__ import annotations` present as first import (after module docstring).

**Line 33**: ✅ `@dataclass` decorator properly applied.

**Line 34**: ✅ Correctly inherits `BaseFeature[SignalData]`.

**Line 34**: ✅ Class name `EpileptogenicityIndex` matches filename.

**Lines 115-225**: ✅ `__call__` signature correct:

- `data: SignalData` as argument
- Return type `xr.DataArray`
- No `apply()` implementation (inherited)

**Lines 9-16**: ✅ Imports in correct order:

1. `from __future__ import annotations`
2. stdlib (`dataclasses.dataclass`)
3. third-party (`numpy`, `xarray`)
4. internal (`..base_feature`, `..data`)

Module docstring present (lines 1-7) with reference citation.

## Docstring

**Lines 34-113**: ✅ Complete and comprehensive Google-style docstring:

- One-line summary (line 34)
- Extended description (lines 36-43)
- Detailed algorithm explanation (lines 45-60)
- Reference citation (lines 62-65)
- `Args:` section for all five fields (lines 67-81)
- `Returns:` section (lines 83-86)
- `Raises:` section (lines 88-92)
- `Example:` section with `.apply()` usage (lines 94-96)

**Lines 115-168** (`_energy_ratio`): ✅ Complete docstring for helper method.

**Lines 170-203** (`_page_hinkley`): ✅ Complete docstring for helper method.

## Typing

**Lines 106-110**: ✅ All fields properly typed:

- `window_duration: float = 1.0`
- `bias: float = 0.5`
- `threshold: float = 30.0`
- `integration_window: float = 5.0`
- `tau: float = 1.0`

**Line 115**: ✅ `__call__` return type is `xr.DataArray`.

**Lines 115, 170**: ✅ Helper methods properly typed with return types `np.ndarray` and `int | None`.

No bare `Any` types detected.

## Safety & Style

**Lines 115-225**: ✅ No `print()` statements.

**Lines 182-192**: ✅ Input validation present:

- Validates exactly `time` and `space` dimensions (lines 182-186)
- Validates `sampling_rate` is set (lines 187-192)
- Validates signal length against window (in `_energy_ratio`, lines 149-153)

**Lines 115-225**: ✅ No mutation of input `data` — operates on `data.data` and returns new `xr.DataArray`.

**Lines 115, 170**: ✅ Helper methods `_energy_ratio` and `_page_hinkley` properly implemented as class methods with `self` parameter.

**Lines 17-23**: ✅ Constants defined at module level for frequency bands (`_THETA`, `_ALPHA`, `_BETA`, `_GAMMA_MIN`, `_EPS`).

## Action List

None.
