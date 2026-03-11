# Feature Review: reciprocal_connectivity

**File**: `src/cobrabox/features/reciprocal_connectivity.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Summary

A well-implemented feature that computes Reciprocal Connectivity from directed connectivity measures. The code is clean, well-documented, and follows all cobrabox conventions. It supports two usage modes (time-series input and pre-computed matrix input) with comprehensive validation. The docstring is excellent, explaining both the mathematical concept and practical usage.

## Ruff

### `uvx ruff check`

All checks passed!

### `uvx ruff format --check`

1 file already formatted

## Signature & Structure

All structural requirements met:

- `from __future__ import annotations` at line 1
- `@dataclass` decorator at line 13 with `BaseFeature[Data]` inheritance
- `output_type: ClassVar[type[Data]] = Data` correctly declared at line 77 (output removes time dimension)
- Class name `ReciprocalConnectivity` matches filename
- `__call__` signature correct: `def __call__(self, data: Data) -> xr.DataArray:` (line 85)
- No custom `apply()` — correctly inherited
- Imports in proper order (stdlib, third-party, internal)

## Docstring

Excellent, comprehensive docstring with all required sections:

- Clear one-line summary explaining the feature's purpose
- Extended description explaining the mathematical formulation (RC[i] = in_strength - out_strength)
- Mathematical notation using reStructuredText-style formatting
- **Two usage modes** clearly documented (time-series vs pre-computed matrix)
- `Args:` section covering all 5 dataclass fields with type info and constraints
- `Returns:` section with shape `(n_channels,)` and dims `("space",)`
- `Raises:` section documenting 5 distinct ValueError conditions
- `Example:` section with two code snippets showing both usage modes

## Typing

All typing requirements satisfied:

- All 5 fields have explicit type annotations (lines 71-75)
- `__call__` return type correctly annotated as `xr.DataArray` (line 85)
- `output_type` class variable properly typed with `ClassVar` (line 77)
- One `type: ignore[arg-type]` at line 103 is justified (PDC feature expects SignalData, but this is called conditionally when time dim exists)

## Safety & Style

No safety concerns:

- No `print()` statements
- Comprehensive input validation throughout `__call__`:
  - Validates connectivity measure for time-series mode (lines 95-99)
  - Checks for required dimensions in pre-computed input (lines 109-113)
  - Symmetry check for directional matrices (lines 117-123)
  - Frequency band validation against available range (lines 136-140)
  - Cross-validation between freq_band setting and frequency dimension presence (lines 129-147)
- `__post_init__` validation ensures `fmin < fmax` for freq_band (lines 79-83)
- No mutation of input — creates copies (line 150: `mat_vals = mat.values.copy()`)
- Line length within 100 characters (ruff enforced)

## Action List

None.
