# Feature Review: partial_directed_coherence

**File**: `src/cobrabox/features/partial_directed_coherence.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

A well-structured feature implementing Partial Directed Coherence analysis. The code is clean, properly typed, and follows most conventions. However, it's missing a `References:` section for a published algorithm, which is required per the review criteria.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements are met. The class correctly inherits from `BaseFeature[SignalData]` (line 15), sets `output_type: ClassVar[type[Data]] = Data` (line 53) since the time dimension is removed, and uses `@dataclass` decorator (line 14). The `__call__` signature is properly typed as `def __call__(self, data: SignalData) -> xr.DataArray:` (line 61). No `apply()` override — correctly uses inherited method. Imports follow standard order and only import what is used.

## Docstring

Comprehensive Google-style docstring with most required sections. One-line summary is present and clear. Extended description explains the algorithm well, including the mathematical normalization property. `Args:` section documents both dataclass fields (`var_order`, `n_freqs`) with types and descriptions. `Returns:` section specifies output dimensions `("space_to", "space_from", "frequency")` and shape. `Raises:` section lists all four validation conditions. `Example:` section shows correct `.apply()` usage.

**Issue**: Missing `References:` section. PDC is a published algorithm (Baccalá & Sameshima, 2001) and requires citation per criteria.

## Typing

Excellent typing throughout. Both dataclass fields are annotated: `var_order: int | None = None` and `n_freqs: int = 128`. The `output_type` class variable is correctly typed as `ClassVar[type[Data]]`. Return type annotation on `__call__` is `xr.DataArray`. No bare `Any` types found. All local variables are properly inferred.

## Safety & Style

No `print()` statements. Comprehensive input validation: `__post_init__` validates `var_order` and `n_freqs` are positive (lines 56-59), and `__call__` validates `sampling_rate` is not None (line 62), input is 2-D (lines 68-72), and at least 2 channels exist (lines 77-78). No mutation of input `data` — works on `data.data` and returns new `xr.DataArray`. Line length is within 100 characters.

## Action List

1. [Severity: MEDIUM] Add a `References:` section to the docstring citing the original PDC paper:
   - Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence: a new concept in neural structure determination. *Biological Cybernetics*, 84(6), 463-474.
