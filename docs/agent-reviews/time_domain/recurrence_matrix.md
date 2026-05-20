# Feature Review: RecurrenceMatrix

**File**: `src/cobrabox/features/time_domain/recurrence_matrix.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Summary

RecurrenceMatrix is a high-quality, well-documented feature that computes pairwise recurrence (self-similarity) matrices. It supports multiple input modes (2-D state vectors, 2-D with windowed FC computation, and 3-D pre-computed FC matrices) and offers extensive configuration through Literal-typed parameters. The implementation follows all cobrabox conventions, includes comprehensive academic references, and handles edge cases with appropriate warnings and validation.

## Ruff

### `uvx ruff check`

Clean — no issues found.

### `uvx ruff format --check`

Clean — no formatting issues.

## Signature & Structure

The feature follows the dataclass pattern correctly:

- ✅ `@dataclass` decorator on line 24
- ✅ Inherits `BaseFeature[SignalData]` (line 25) — appropriate since it operates on time-series
- ✅ `output_type: ClassVar[type[Data]] = Data` (line 97) — correctly returns plain `Data` since time dimension is removed
- ✅ Class name `RecurrenceMatrix` matches filename
- ✅ `from __future__ import annotations` present (line 3)
- ✅ No `_is_cobrabox_feature` marker (not needed for class-based features)
- ✅ `__call__` signature correct: `def __call__(self, data: SignalData) -> xr.DataArray:` (line 186)
- ✅ No `apply()` override (correctly inherited from `BaseFeature`)
- ✅ Helper methods `_similarity_matrix` (line 127) and `_fc_matrix` (line 141) are `@staticmethod` inside the class
- ✅ Clean imports: stdlib → third-party → internal (lines 3–18)

## Docstring

Comprehensive Google-style docstring with all required sections:

- ✅ **One-line summary** (line 26): "Compute a pairwise recurrence (self-similarity) matrix from a time-series."
- ✅ **Extended description** (lines 28–47): Detailed explanation of behavior for different input shapes (2-D vs 3-D) and `fc_options` configurations
- ✅ **Args:** (lines 49–57): Both fields documented (`rec_metric`, `fc_options`) with allowed values and defaults
- ✅ **Returns:** (lines 59–60): Shape `(n, n)` with dims `('t1', 't2')` specified
- ✅ **Raises:** (lines 62–65): Lists `ValueError` conditions for missing dimensions, window size, and invalid metrics
- ✅ **Example:** (lines 67–78): Four working examples covering all usage modes (state-vector, FC with defaults, FC with full control, 3-D input)
- ✅ **References:** (lines 80–86): Two academic citations with DOIs — Eckmann et al. (1987) and Marwan et al. (2007)

## Typing

Excellent type coverage throughout:

- ✅ All dataclass fields typed:
  - `rec_metric: RecMetric = field(default="cosine")` (line 89)
  - `fc_options: list[str | int | float] = field(default_factory=list)` (line 90)
  - Derived fields `_fc_metric`, `_window_size`, `_overlap` all typed (lines 93–95)
- ✅ `RecMetric` and `FcMetric` are `Literal` type aliases (lines 20–21)
- ✅ `__call__` return type: `xr.DataArray` (line 186)
- ✅ `__post_init__` return type: `None` (line 99)
- ✅ Static methods have full type annotations for parameters and returns
- ✅ `cast(FcMetric, fc_metric_str)` used appropriately (line 118)
- ✅ `output_type` uses `ClassVar[type[Data]]` (line 97)

## Safety & Style

- ✅ No `print()` statements — uses `warnings.warn()` appropriately (lines 225–229)
- ✅ **Input validation** in `__post_init__`:
  - Validates `rec_metric` against `get_args(RecMetric)` (lines 100–103)
  - Validates `fc_options` length ≤ 3 (lines 106–109)
  - Validates `fc_metric` against `get_args(FcMetric)` (lines 112–117)
  - Validates `window_size >= 1` (lines 122–123)
  - Validates `overlap` in range `[0, 1)` (lines 124–125)
- ✅ **Input validation** in `__call__`:
  - Validates 1 or 2 spatial dimensions (lines 192–193)
  - Validates 3-D input has equal spatial dims (lines 201–205)
  - Validates `window_size < n_time` (lines 230–231)
- ✅ No mutation of input `data` — works on `data.data` and returns new `xr.DataArray`
- ✅ Uses modern Python `match-case` statements (lines 130–139, 152–184)
- ✅ Guards against zero-norm vectors in cosine similarity (line 133)

## Action List

None.
