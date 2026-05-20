# Feature Review: svd

**File**: `src/cobrabox/features/reductions/svd.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

The SVD feature is well-implemented with excellent documentation and clean code structure. It correctly inherits from `BaseFeature[Data]`, has comprehensive docstring coverage with all required sections (Args, Returns, Raises, References, Example), and implements proper input validation. The ruff checks pass cleanly. There is one minor formatting issue in the docstring examples where line 69 has incorrect indentation.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

- **Line 2**: `from __future__ import annotations` present ✓
- **Line 16-17**: `@dataclass` decorator with `BaseFeature[Data]` inheritance ✓
- **Class name**: `SVD` is PascalCase matching filename `svd.py` ✓
- **Naming**: `SVD` is a widely recognized acronym (Singular Value Decomposition), acceptable per criteria ✓
- **Line 91**: `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray | Data` ✓
- No loose helper functions - all logic contained within the class ✓
- No `apply()` override - correctly uses inherited method ✓
- Imports properly ordered: future, stdlib, third-party, internal ✓
- `output_type` not declared - correct since feature does not remove time dimension (it operates over any specified dimension) ✓

## Docstring

Excellent documentation coverage:

- **One-line summary** (line 18): "Compute truncated SVD over one dimension." ✓
- **Extended description** (lines 20-21): Clear explanation of N-D input handling ✓
- **Args** (lines 23-48): All 8 dataclass fields documented with types and intuitive explanations:
  - `dim`, `n_components`, `center`, `zscore`, `mask`, `return_unstacked_V`, `output` ✓
- **Returns** (lines 49-56): Describes both output modes (V and U) with shapes ✓
- **Raises** (lines 72-75): Lists all 3 ValueError conditions ✓
- **References** (lines 77-80): Citation to Golub & Kahan (1965) ✓
- **Example** (lines 57-71): Multiple usage examples showing different scenarios

## Typing

- All 8 dataclass fields have type annotations (lines 83-89) ✓
- `__call__` return type annotation present: `xr.DataArray | Data` ✓
- No bare `Any` types ✓
- `Literal` used correctly for `_SVDOutputMode` (line 13) ✓
- `get_args()` used for runtime validation (line 98) ✓

## Safety & Style

- No `print()` statements ✓
- Input validation in `__call__` (lines 94-101):
  - Validates `dim` exists in data dimensions
  - Validates `n_components` is positive
  - Validates `output` is one of allowed values ✓
- No mutation of input `data` - works on `data.data` and returns new objects ✓
- Line length within 100 characters ✓

## Action List

1. [Severity: LOW] Fix indentation in docstring example at line 69. The line `>>>         U = svd["U"]` has extra leading spaces that do not match the pattern of other example lines. Should be `>>> U = svd["U"]` aligned with other `>>>` lines.
