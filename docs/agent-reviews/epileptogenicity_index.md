# Feature Review: epileptogenicity_index

**File**: `src/cobrabox/features/epileptogenicity_index.py`
**Date**: 2026-03-04
**Verdict**: PASS

## Summary

Excellent feature implementation. The `EpileptogenicityIndex` class is a sophisticated, publication-grade feature implementing the Bartolomei et al. (2008) algorithm for quantifying epileptogenicity from intracranial EEG. The code is well-structured, thoroughly documented, and follows all project conventions. The docstring is exemplary—comprehensive, includes mathematical formulas, frequency band specifications, and proper academic citations. Ruff checks pass cleanly.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

Clean and correct throughout:

- ✅ `from __future__ import annotations` present at line 8
- ✅ `@dataclass` decorator + `BaseFeature[SignalData]` inheritance (lines 28-29)
- ✅ `output_type: ClassVar[type[Data]] = Data` correctly declared at line 92—feature removes time dimension as expected
- ✅ Class name `EpileptogenicityIndex` matches filename
- ✅ `__call__` signature: `def __call__(self, data: SignalData) -> xr.DataArray` at line 179
- ✅ No `apply()` override—correctly inherits from base
- ✅ Imports follow standard order: stdlib → third-party → internal

## Docstring

Outstanding. Sets the standard for scientific features:

- ✅ One-line summary clearly states purpose
- ✅ Extended description (lines 30-58) includes:
  - Algorithm overview with three numbered stages
  - Mathematical formula for Energy Ratio: `ER[n] = (E_beta + E_gamma) / (E_theta + E_alpha)`
  - Frequency band table matching the paper (θ, α, β, γ ranges)
  - EI formula with normalisation explanation
  - Complete academic reference with DOI
- ✅ `Args:` section (lines 65-76) documents all 5 dataclass fields with clear descriptions
- ✅ `Returns:` section (lines 78-80) specifies output dimensions and normalisation range
- ✅ `Raises:` section (lines 82-85) documents 3 specific validation cases
- ✅ `Example:` section (lines 87-89) shows correct `.apply()` usage

## Typing

Fully typed:

- ✅ All 5 dataclass fields have type annotations (lines 94-98)
- ✅ `__call__` return type: `xr.DataArray` (line 179)
- ✅ Helper method `_energy_ratio` return type: `np.ndarray` (line 117)
- ✅ Helper method `_page_hinkley` return type: `int | None` (line 161)
- ✅ No bare `Any` types

## Safety & Style

- ✅ No `print()` statements
- ✅ Input validation in `__call__` (lines 180-190):
  - Validates exactly 2 dimensions (`time` and `space`)
  - Validates `sampling_rate` is set
- ✅ No mutation of input `data`—works on `.data` and returns new array
- ✅ `__post_init__` not needed—field validation handled by numpy/xarray downstream
- ✅ Uses `np.finfo(float).eps` constant for numerical stability (line 25)

## Action List

None.
