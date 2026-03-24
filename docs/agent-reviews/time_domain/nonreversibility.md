# Feature Review: Nonreversibility

**File**: `src/cobrabox/features/time_domain/nonreversibility.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

A well-implemented feature computing the dc_norm time-irreversibility measure via VAR(1) models in forward and reverse time directions. The code is clean, properly typed, and follows all structural conventions. The only issue is a placeholder TODO in the References section that should be replaced with an actual citation.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

Line 15-16: Correct `@dataclass` decorator with `BaseFeature[SignalData]` inheritance. The type parameter is appropriate since the feature operates on time-series data and uses the time dimension for VAR(1) fitting.

Line 50: Proper `output_type: ClassVar[type[Data]] = Data` declaration — required because the feature removes the time dimension and returns a scalar-like result.

Line 52-66, 68-74: Helper methods `_fit_var1` and `_rescale_to_unit_spectral_radius` are correctly implemented as `@staticmethod` inside the class, keeping the module's public surface clean.

Line 107: `__call__` signature correctly typed as `(self, data: SignalData) -> xr.DataArray`.

No `apply()` method — correctly inherited from `BaseFeature`.

## Docstring

Line 17-48: Comprehensive Google-style docstring with all required sections.

Line 17: Good one-line summary.

Lines 19-30: Extended description clearly explains the VAR(1) fitting in both directions and the dc_norm formula. Mathematical notation is clear.

Lines 32-34: Args section correctly states "None" since the feature has no dataclass fields.

Lines 35-37: Returns section describes the output shape and dimension removal.

Lines 39-41: Raises section documents the two ValueError conditions.

Lines 46-47: **Issue** — References section contains a TODO placeholder instead of an actual citation. This should be replaced with the proper literature reference for the dc_norm measure.

Lines 43-44: Example section shows correct usage via `.apply()`.

## Typing

Line 50: `output_type` properly typed as `ClassVar[type[Data]]`.

Line 52, 68: Static methods have complete type annotations for arguments and return values.

Line 76: `_compute_dc_norm` is fully typed with `np.ndarray` inputs and `float` return.

Line 107: `__call__` return type correctly declared as `xr.DataArray`.

No bare `Any` types found.

## Safety & Style

No `print()` statements found.

Lines 110-116: Proper input validation in `__call__` — checks for required 'space' dimension and validates at least 2 channels are present. Error messages are clear and actionable.

Line 94-95: Additional validation inside `_compute_dc_norm` ensures at least 2 timepoints are available for VAR(1) fitting.

No mutation of input `data` — the feature works on `data.data.values` and returns new arrays.

All lines are within the 100-character limit.

## Action List

1. [Severity: MEDIUM] Replace TODO placeholder in References section (line 47) with actual citation for the dc_norm / time-irreversibility VAR(1) measure.
