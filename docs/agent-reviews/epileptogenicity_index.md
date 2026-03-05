# Feature Review: epileptogenicity_index

**File**: `src/cobrabox/features/epileptogenicity_index.py`
**Date**: 2025-03-04
**Verdict**: PASS

## Summary

Excellent feature implementation. The `EpileptogenicityIndex` feature is a sophisticated, well-documented implementation of the Bartolomei et al. (2008) algorithm for quantifying epileptogenicity from intracranial EEG. The code is clean, ruff-compliant, thoroughly documented with Google-style docstrings including theoretical background and mathematical formulas, and properly typed throughout. Input validation is comprehensive, and the implementation avoids mutation of input data.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

All structural requirements met:

- `from __future__ import annotations` present at line 8
- `@dataclass` decorator applied at line 28
- Correct base class `BaseFeature[SignalData]` at line 29 (this is a time-series feature)
- Appropriate `output_type: ClassVar[type[Data]] = Data` at line 92 — correct since the output removes the time dimension and returns per-channel scalar values
- Class name `EpileptogenicityIndex` matches filename (PascalCase conversion)
- `__call__` signature correct at line 179: `def __call__(self, data: SignalData) -> xr.DataArray:`
- No custom `apply()` implementation (correctly uses inherited method)
- Clean import ordering: stdlib → third-party → internal

## Docstring

Outstanding documentation quality:

- One-line summary at line 30: "Compute the Epileptogenicity Index (EI) per channel (Bartolomei et al., 2008)."
- Extended description (lines 32-90) provides comprehensive theoretical background including:
  - Explanation of what EI quantifies (spectral + temporal properties)
  - Three-stage algorithm breakdown with mathematical formulas
  - Frequency band definitions referencing Table 2 of the paper
  - Page-Hinkley detection explanation
- `Args:` section at lines 65-76 documents all 5 dataclass fields with clear descriptions and units
- `Returns:` section at lines 78-80 describes output shape and normalization
- `Raises:` section at lines 82-85 documents 3 specific validation errors
- `Example:` section at lines 87-89 shows correct `.apply()` usage
- Citation and DOI included both in module docstring and class docstring

## Typing

All type annotations present and correct:

- All dataclass fields typed (lines 94-98): `window_duration: float`, `bias: float`, etc.
- `__call__` return type `xr.DataArray` at line 179 (acceptable — returns a bare DataArray which will be wrapped by `apply()`)
- Private method `_energy_ratio` fully typed at line 100
- Private method `_page_hinkley` fully typed at line 147
- No bare `Any` types
- Uses modern union syntax (`int | None` at line 147)

## Safety & Style

Excellent safety practices:

- No `print()` statements
- Input validation in `__call__`:
  - Dimension check at lines 180-184: validates exactly `{"time", "space"}` dimensions
  - Sampling rate check at lines 185-190: validates `sampling_rate` is set
  - Signal length check inside `_energy_ratio` at lines 124-128
- No mutation of input `data` — works on `data.data.values` copy at line 197
- Epsilon constant `_EPS` used at line 145 to prevent division by zero
- Strided window creation uses `writeable=False` view implicitly via `as_strided`
- Line length within 100 characters throughout

## Action List

None.
