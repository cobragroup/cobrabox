# Test Review: granger_causality

**Feature**: `src/cobrabox/features/granger_causality.py`
**Test file**: `tests/test_feature_granger_causality.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```text
granger_causality.py: 99% (90 statements, 1 missing)
Missing line: 194 (GrangerCausalityMatrix.__post_init__ coords empty list check)
```

## Summary

Comprehensive test suite with 31 tests covering GrangerCausality and GrangerCausalityMatrix classes. Tests validate causality detection, directionality, multiple lags, matrix operations, and error handling. Coverage is at 99% with only one uncovered line (empty coords list validation).

## Keep

- `test_granger_causality_correct_value` — validates positive causality detection
- `test_granger_causality_detects_causality` — basic causality test
- `test_granger_causality_directionality` — directional causality verification
- `test_granger_causality_multiple_lags` — maxlag parameter testing
- `test_granger_causality_matrix_shape` — matrix output structure
- `test_granger_causality_matrix_diagonal_nan` — self-causality handling
- `test_granger_causality_history_tracking` — history preservation
- `test_granger_causality_preserves_metadata` — metadata propagation
- `test_granger_causality_no_mutation` — input immutability
- `test_granger_causality_invalid_lag` — parameter validation
- `test_granger_causality_matrix_multiple_lags` — multi-lag matrix testing
- `test_granger_causality_matrix_invalid_lag` — matrix parameter validation
- `test_granger_causality_matrix_invalid_maxlag` — matrix maxlag validation

## Fix

None. All existing tests are correct.

## Add

Optional: Add test for empty coords validation (line 194):

### `test_granger_causality_matrix_empty_coords_raises`

```python
def test_granger_causality_matrix_empty_coords_raises() -> None:
    """GrangerCausalityMatrix raises ValueError for empty coords list."""
    with pytest.raises(ValueError, match="coords cannot be an empty list"):
        cb.feature.GrangerCausalityMatrix(coords=[], lag=2)
```

## Action List

1. [Severity: LOW] Add test for empty coords validation in GrangerCausalityMatrix (optional - coverage already meets 95% threshold)
