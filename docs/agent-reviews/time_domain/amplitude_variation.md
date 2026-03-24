# Feature Review: AmplitudeVariation

**File**: `src/cobrabox/features/time_domain/amplitude_variation.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

The `AmplitudeVariation` feature is a clean, well-structured implementation that correctly computes the standard deviation of the signal over the time dimension. It follows all cobrabox conventions including proper base class inheritance, complete docstring coverage, correct typing, and appropriate use of `output_type` to indicate the time dimension is removed. No issues found.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

The feature correctly inherits from `BaseFeature[SignalData]` (line 13), which is appropriate since it operates on the time dimension. The `@dataclass` decorator is present.

The `output_type: ClassVar[type[Data]] = Data` declaration (line 32) is correctly set since this feature removes the `time` dimension, returning a result that should be wrapped as a plain `Data` object rather than preserving the input `SignalData` type.

The `__call__` signature (line 34) accepts `data: SignalData` and returns `xr.DataArray`, matching the base class contract.

No `apply()` method is implemented (correctly inherited from `BaseFeature`).

No loose helper functions — the implementation is a single line using xarray's built-in `std()` method.

Imports follow the correct order: `__future__`, stdlib, third-party, internal.

## Docstring

All required sections are present:

- **One-line summary**: Clear verb phrase describing the computation (line 14)
- **Extended description**: Explains what amplitude variation represents (lines 16-17)
- **Args**: Correctly documents `None` since there are no dataclass fields (lines 19-20)
- **Returns**: Detailed description of output shape and units (lines 22-26)
- **Example**: Working snippet using `.apply()` syntax (lines 28-29)

## Typing

All type annotations are correct:

- Class properly parameterized as `BaseFeature[SignalData]`
- `output_type` declared as `ClassVar[type[Data]]`
- `__call__` parameter `data: SignalData` matches the base class type parameter
- Return type `xr.DataArray` is explicit

No bare `Any` types. No `Literal` types needed (no string option fields).

## Safety & Style

No `print()` statements found.

No input validation required in `__call__` — since this is a `BaseFeature[SignalData]`, the `SignalData` type already enforces the presence of a `time` dimension at construction time. No additional feature-specific constraints need validation.

No mutation of input `data` — the implementation returns a new xarray DataArray computed via `data.data.std(dim="time")` (line 35).

Line length is within the 100 character limit.

## Action List

None.
