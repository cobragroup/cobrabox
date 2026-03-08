# Feature Review: PartialDirectedCoherence

**File**: `src/cobrabox/features/partial_directed_coherence.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Summary

Excellent, production-ready feature implementing Partial Directed Coherence (PDC) analysis via VAR modeling. The implementation is mathematically sound, well-documented, and follows all cobrabox conventions. The feature correctly handles the transformation from time-series `SignalData` to a frequency-domain connectivity matrix returned as plain `Data` (no time dimension). The code includes robust input validation, clear docstrings with proper Raises sections, and uses statsmodels appropriately for VAR fitting.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

All structural requirements met:

- Line 1: `from __future__ import annotations` present
- Line 14: `@dataclass` decorator applied
- Line 15: Inherits `BaseFeature[SignalData]` (correct for time-series analysis)
- Line 53: `output_type: ClassVar[type[Data]] = Data` set (returns plain Data, no time dim)
- Line 61: `__call__` signature is `def __call__(self, data: SignalData) -> xr.DataArray`
- No custom `apply()` — correctly inherited from base
- Class name `PartialDirectedCoherence` matches filename `partial_directed_coherence.py`
- Imports follow standard order (stdlib → third-party → internal)

## Docstring

Comprehensive Google-style docstring with all required sections:

- Line 16: Concise one-line summary
- Lines 18-27: Extended description explaining PDC mathematics and normalization
- Lines 28-31: `Args:` section covering both dataclass fields
- Lines 33-36: `Returns:` section with dimension and shape details
- Lines 38-42: `Raises:` section listing all four validation conditions
- Lines 44-47: `Example:` section showing `.apply()` usage

The docstring clearly explains the directional interpretation (`PDC[i, j, f]` = influence from j to i) and the normalization constraint (sum of squares = 1).

## Typing

All typing requirements satisfied:

- Line 50: `var_order: int | None = None` — typed with union
- Line 51: `n_freqs: int = 128` — typed with default
- Line 53: `output_type: ClassVar[type[Data]] = Data` — proper classvar
- Line 61: Return type `xr.DataArray` declared
- Line 55: `__post_init__` has `-> None` return type
- `# type: ignore[override]` on line 61 is acceptable — narrows return type from base class union

No bare `Any` types present.

## Safety & Style

Excellent safety practices:

- **No print statements** — clean mathematical code
- **Input validation in `__post_init__`** (lines 55-59):
  - Validates `var_order >= 1` if not None
  - Validates `n_freqs >= 1`
- **Input validation in `__call__`** (lines 62-78):
  - Checks `data.sampling_rate is not None` (line 62-63)
  - Validates 2-D input shape (lines 68-72)
  - Validates at least 2 channels (lines 77-78)
- **No mutation of input** — works on `xr_data.values` copy and returns new DataArray
- **Numerical safety** — handles division by zero on line 111 with `np.where`

The VAR model fitting (lines 85-89) correctly handles automatic order selection via AIC when `var_order` is None. The PDC computation (lines 94-115) properly implements the frequency-domain transfer matrix and column-wise normalization.

## Action List

None.
