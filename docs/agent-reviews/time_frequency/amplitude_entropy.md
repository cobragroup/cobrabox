# Feature Review: amplitude_entropy

**File**: `src/cobrabox/features/amplitude_entropy.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Summary

AmplitudeEntropy is a well-structured feature that computes histogram-based Shannon entropy. The implementation is correct and follows most conventions. The feature properly returns a 0-dimensional scalar (not fake singleton dimensions) and validates inputs appropriately. The only issue is a missing `Raises:` section in the docstring to document the ValueError exceptions raised in `__post_init__` and `__call__`.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

The feature correctly uses `@dataclass` decorator and inherits `BaseFeature[Data]`. The choice of `BaseFeature[Data]` is appropriate since the feature does not inherently require time-series structure — it operates on any 2D+ array by treating the first dimension as "time points". The `output_type = Data` is correctly set since the output is a scalar (0-dimensional) with no time dimension.

The class name `AmplitudeEntropy` matches the filename and is descriptive. The `__call__` signature is correct with proper type annotations. No `_is_cobrabox_feature` marker is needed (correctly omitted for class-based features).

## Docstring

The Google-style docstring is comprehensive with one-line summary, extended description, Args, Returns, and Example sections. However, it is missing a `Raises:` section to document the two ValueError conditions:

1. `__post_init__` (line 47-48): raises when `band_width <= 0`
2. `__call__` (line 55-56): raises when input has fewer than 2 dimensions

## Typing

All fields are properly typed (`band_width: float`). The `__call__` return type is explicitly annotated as `xr.DataArray`. No bare `Any` types are present.

## Safety & Style

The feature properly validates its `band_width` parameter in `__post_init__` and validates input dimensionality in `__call__`. No print statements are present. The implementation does not mutate input data — it works on a copy via `data.to_numpy()` and returns a new DataArray.

The code includes helpful comments explaining the MATLAB-to-Python translation, which aids maintainability for researchers familiar with the original MATLAB implementation.

## Action List

1. [Severity: MEDIUM] Add a `Raises:` section to the docstring documenting:
   - `ValueError`: If `band_width` is not positive (raised in `__post_init__`, line 47-48).
   - `ValueError`: If input data has fewer than 2 dimensions (raised in `__call__`, line 55-56).
