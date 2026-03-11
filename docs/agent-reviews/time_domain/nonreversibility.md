# Feature Review: nonreversibility

**File**: `src/cobrabox/features/nonreversibility.py`
**Date**: 2026-03-11
**Verdict**: PASS

## Summary

The implementation is algorithmically sound and well-structured. The feature correctly uses `BaseFeature[SignalData]`, has thorough validation, and includes comprehensive docstrings. Helper functions are cleanly separated. Ruff check and format are clean.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

`@dataclass class Nonreversibility(BaseFeature[SignalData])` is the correct declaration. `output_type: ClassVar[type[Data]] = Data` is correctly set since the output is a scalar with no time dimension. The feature is parameterless (no dataclass fields). Imports are ordered correctly (stdlib, third-party, internal). No issues found.

## Docstring

The class docstring includes a one-line summary, extended description explaining the forward/reverse VAR(1) model and dc_norm formula, `Args:`, `Returns:`, `Raises:`, `Example:`, and `References:` sections. All sections are well-populated with appropriate content.

## Typing

The feature has no dataclass fields. `__call__` return type is `xr.DataArray`, which satisfies the contract. Helper functions have full parameter and return type annotations. No bare `Any` usage.

## Safety & Style

- No `print()` statements.
- No input mutation — the feature works on `data.data` and returns a new array.
- `__call__` has validation for single channel, too few timepoints, and missing space dimension.
- Helper functions `_fit_var1`, `_rescale_to_unit_spectral_radius`, and `_compute_dc_norm` are well-structured.

## Action List

None.