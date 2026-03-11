# Test Review: granger_causality

**Feature**: `src/cobrabox/features/granger_causality.py`
**Test file**: `tests/test_feature_granger_causality.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
granger_causality.py: 99% (88 statements, 1 missing)
Missing: line 191 (empty coords validation in GrangerCausalityMatrix)
```

Coverage is excellent at 99%. Only one line is uncovered: the `ValueError` for empty coords list in `GrangerCausalityMatrix.__post_init__`.

## Summary

The test file provides good coverage for both `GrangerCausality` and `GrangerCausalityMatrix` classes. It tests happy paths, error cases, history tracking, metadata preservation, and immutability. The main gap is the missing test for empty coords validation. Additionally, several tests lack docstrings and some naming conventions could be improved for consistency.

## Keep

Tests that are correct and complete — no changes needed:

- `test_granger_causality_detects_causality` — correctly tests that positive causality is detected
- `test_granger_causality_directionality` — tests directional causality correctly
- `test_granger_causality_single_lag` — tests scalar output shape for single lag
- `test_granger_causality_multiple_lags` — tests array output for maxlag range
- `test_granger_causality_lag_precedence` — tests that lag takes precedence over maxlag
- `test_granger_causality_matrix_shape` — tests correct 2D output shape
- `test_granger_causality_matrix_diagonal_nan` — tests NaN diagonal correctly
- `test_granger_causality_matrix_directional` — tests matrix directionality
- `test_granger_causality_matrix_default_coords` — tests auto-detection of coords
- `test_granger_causality_matrix_multiple_lags` — tests 3D output with multiple lags
- `test_granger_causality_history_tracking` — tests history correctly
- `test_granger_causality_preserves_metadata` — tests all metadata fields
- `test_granger_causality_no_mutation` — tests immutability comprehensively
- `test_granger_causality_matrix_history_updated` — tests matrix history
- `test_granger_causality_matrix_preserves_metadata` — tests matrix metadata
- `test_granger_causality_matrix_no_mutation` — tests matrix immutability
- `test_granger_causality_invalid_lag` — tests lag < 1 raises ValueError
- `test_granger_causality_invalid_maxlag` — tests maxlag < 1 raises ValueError
- `test_granger_causality_matrix_invalid_lag` — tests matrix lag validation
- `test_granger_causality_matrix_invalid_maxlag` — tests matrix maxlag validation

## Fix

Tests that exist but need changes:

### Missing docstrings

Several tests lack docstrings. Add one-line descriptions:

- `test_granger_causality_correct_value` (lines 21-27)
- `test_granger_causality_detects_causality` (lines 30-34)
- `test_granger_causality_directionality` (lines 37-52)
- `test_granger_causality_single_lag` (lines 55-60)
- `test_granger_causality_multiple_lags` (lines 63-68)
- `test_granger_causality_lag_precedence` (lines 71-75)
- `test_granger_causality_matrix_shape` (lines 78-82)
- `test_granger_causality_matrix_diagonal_nan` (lines 85-90)
- `test_granger_causality_matrix_directional` (lines 93-105)
- `test_granger_causality_matrix_default_coords` (lines 108-112)
- `test_granger_causality_returns_positive_for_causality` (lines 115-121)
- `test_granger_causality_no_causality_small` (lines 124-132)
- `test_granger_causality_requires_time_dimension` (lines 135-139)
- `test_granger_causality_requires_space_dimension` (lines 142-146)
- `test_granger_causality_invalid_coord` (lines 149-152)
- `test_granger_causality_small_samples` (lines 165-171)
- `test_granger_causality_small_dataset` (lines 174-184)
- `test_granger_causality_detects_known_causality` (lines 187-199)

Example fix:

```python
def test_granger_causality_correct_value() -> None:
    """GrangerCausality returns positive value for causal relationship."""
    ...
```

## Add

Missing scenarios — new tests to add:

### `test_granger_causality_matrix_empty_coords`

The only uncovered line (191) is the empty coords validation. Add this test:

```python
def test_granger_causality_matrix_empty_coords() -> None:
    """GrangerCausalityMatrix raises ValueError for empty coords list."""
    with pytest.raises(ValueError, match="coords cannot be an empty list"):
        cb.feature.GrangerCausalityMatrix(coords=[], lag=2)
```

## Action List

1. [Severity: MEDIUM] Add docstrings to 18 test functions that currently lack them
2. [Severity: HIGH] Add `test_granger_causality_matrix_empty_coords` to cover line 191 and achieve 100% coverage
