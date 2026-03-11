# Test Review: recurrence_matrix

**Feature**: `src/cobrabox/features/recurrence_matrix.py`
**Test file**: `tests/test_feature_recurrence_matrix.py`
**Date**: 2026-03-11
**Verdict**: PASS

## Coverage

```text
src/cobrabox/features/recurrence_matrix.py   113      0   100%
```

Excellent coverage — all lines are exercised by the test suite.

## Summary

The test file has been comprehensively updated and now meets all project requirements. All test functions follow the `test_recurrence_matrix_<scenario>` naming convention, carry `-> None` return annotations, and have appropriate one-line docstrings. Coverage is at 100%, including the previously uncovered lines: `window_size < 1` validation, `MI` fc_metric branch, `n_spatial not in (1, 2)` guard, and the non-square 3‑D input guard. The metadata test verifies all four metadata fields (`subjectID`, `groupID`, `condition`, `sampling_rate`), and immutability is checked by a dedicated no‑mutation test. All five FC metrics (`pearson`, `spearman`, `MI`, `PLV`, `AEC`) are exercised, and output finiteness is validated.

## Keep

All tests are correct and complete — no changes needed.

- `test_recurrence_matrix_statevector_shape` — parametrizes all three `rec_metric` values, checks dims, shape, and history.
- `test_recurrence_matrix_statevector_cosine_diagonal_is_one` — structural invariant for cosine similarity.
- `test_recurrence_matrix_statevector_euclidean_diagonal_is_zero` — structural invariant for euclidean distance.
- `test_recurrence_matrix_window_fc_options_metric_only` — covers all five FC metrics with default window/overlap.
- `test_recurrence_matrix_window_fc_options_with_window_size` — two‑element `fc_options` variant.
- `test_recurrence_matrix_window_fc_options_full` — three‑element `fc_options` variant.
- `test_recurrence_matrix_window_small_warns` — catches `UserWarning` for `window_size < 5`.
- `test_recurrence_matrix_3d_shape` — parametrizes all three `rec_metric` values for 3‑D input.
- `test_recurrence_matrix_3d_cosine_diagonal_is_one` — diagonal invariant for 3‑D cosine path.
- `test_recurrence_matrix_statevector_symmetric` / `test_recurrence_matrix_3d_symmetric` — symmetry invariants.
- `test_recurrence_matrix_metadata_preserved` — verifies `subjectID`, `groupID`, `condition`, `sampling_rate`.
- `test_recurrence_matrix_invalid_rec_metric_raises` — construction‑time validation.
- `test_recurrence_matrix_invalid_fc_metric_in_options_raises` — construction‑time validation.
- `test_recurrence_matrix_too_many_fc_options_raises` — construction‑time validation.
- `test_recurrence_matrix_invalid_overlap_raises` — construction‑time validation.
- `test_recurrence_matrix_window_size_too_large_raises` — runtime guard.
- `test_recurrence_matrix_missing_time_dim_raises` — time‑dimension guard.
- `test_recurrence_matrix_window_size_zero_raises` — covers `window_size < 1` validation.
- `test_recurrence_matrix_fc_mi` — exercises the `MI` fc_metric branch.
- `test_recurrence_matrix_invalid_spatial_dims_raises` — covers `n_spatial not in (1, 2)` guard.
- `test_recurrence_matrix_3d_nonsquare_raises` — covers non‑square 3‑D input guard.
- `test_recurrence_matrix_no_mutation` — verifies input immutability.
- `test_recurrence_matrix_output_finite` — ensures output values are finite.

## Fix

None.

## Add

None — all required scenarios are covered.

## Action List

None.