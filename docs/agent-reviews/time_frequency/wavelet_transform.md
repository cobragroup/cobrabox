# Feature Review: wavelet_transform

**File**: `src/cobrabox/features/time_frequency/wavelet_transform.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

Both `DiscreteWaveletTransform` and `ContinuousWaveletTransform` are well-implemented, production-ready features. The code follows all cobrabox conventions: proper dataclass structure, comprehensive docstrings with all required sections, thorough input validation, and clean separation of concerns. The use of Literal types for wavelet names and modes is exemplary.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

**Line 1**: `from __future__ import annotations` — present.

**Lines 65-66, 183-184**: Both classes correctly decorated with `@dataclass` and inherit `BaseFeature[SignalData]`. The type parameter is correct since both features operate on the time axis.

**Line 123, 243**: `output_type` correctly declared:

- `DiscreteWaveletTransform`: `output_type = Data` (removes time dimension, produces wavelet coefficients)
- `ContinuousWaveletTransform`: `output_type = None` (preserves SignalData since time dimension is retained)

**Class names**: Both `DiscreteWaveletTransform` and `ContinuousWaveletTransform` are descriptive and follow PascalCase convention.

**Lines 136, 265**: `__call__` signatures are correct:

```python
def __call__(self, data: SignalData) -> xr.DataArray:
```

No loose helper functions — all computation logic lives inside class methods.

Import order is correct: future annotations → stdlib → third-party → internal.

## Docstring

Both classes have excellent Google-style docstrings with all required sections:

### DiscreteWaveletTransform (lines 67-107)

- **One-line summary**: Clear verb phrase describing the feature
- **Extended description**: Explains the padding strategy (NaN-padding for rectangular output)
- **Args**: All three fields documented (`wavelet`, `level`, `mode`)
- **Returns**: Detailed dimension and coordinate description
- **Raises**: Three ValueError conditions documented
- **Example**: Working doctest-style example showing `.apply()` usage

### ContinuousWaveletTransform (lines 185-236)

- **One-line summary**: Clear description
- **Extended description**: Explains time-scale representation and composition
- **Args**: All five fields documented with detailed explanations for wavelet choices and scaling modes
- **Returns**: Comprehensive dimension/coordinate description including the `frequency` non-index coordinate
- **Raises**: Four validation conditions documented
- **Example**: Working example with attribute checks

## Typing

All fields are fully typed:

- **Lines 109-121**: `DiscreteWaveletTransform` fields use `Literal` types for wavelet names and mode options
- **Lines 238-241**: `ContinuousWaveletTransform` fields similarly well-typed

**Type aliases** (lines 15-62): Excellent use of `TypeAlias` for wavelet name Literals — `_DwtWavelet` and `_CwtWavelet` are comprehensive and make the code self-documenting.

**Return types**: Both `__call__` methods have explicit `xr.DataArray` return type annotations.

No bare `Any` types used.

## Safety & Style

**No print statements**: Clean — uses no output statements.

**Input validation**: Excellent validation in `__post_init__`:

- **Lines 126-134**: `DiscreteWaveletTransform` validates level >= 1 and wavelet name via pywt.Wavelet()
- **Lines 246-263**: `ContinuousWaveletTransform` validates scales, n_scales, scaling mode, and wavelet name

Additional validation in `__call__`:

- **Lines 147-151**: Checks level doesn't exceed max possible for signal length

**No mutation**: Both features correctly work on `data.data` and return new DataArray instances without modifying input.

**Line length**: All lines within 100 character limit.

## Action List

None.
