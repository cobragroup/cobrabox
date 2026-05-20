# Test Review: Correlation

**Feature**: `src/cobrabox/features/connectivity/correlation.py`
**Test file**: `tests/features/connectivity/test_feature_correlation.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Coverage

Per-file coverage report:

```text
src/cobrabox/features/connectivity/correlation.py                        35      0   100%
```

All 35 statements in `correlation.py` are covered. **No action needed** — coverage meets the 95% threshold.

## Summary

26 tests, 100% coverage, ruff clean. The test suite is comprehensive and well-structured, covering:
- Output structure (dims, shape, coordinates)
- Pearson correlation correctness (matches `np.corrcoef`)
- Spearman correlation correctness (matches `scipy.stats.spearmanr`)
- Metadata preservation and history tracking
- Custom dimension names
- All error handling cases (wrong dim count, missing dim, invalid method)
- Immutability of input data

All required scenarios from the review criteria are covered. No gaps or issues detected.

## Keep

- `test_correlation_output_dims_and_shape` — shape, type, and dim checks on default call.
- `test_correlation_output_is_square` — matrix squareness.
- `test_correlation_channel_coords_preserved` — coord labels on both axes.
- `test_correlation_pearson_matches_numpy_corrcoef` — exact correspondence with `np.corrcoef`.
- `test_correlation_pearson_diagonal_is_one` — diagonal exactly 1.0.
- `test_correlation_pearson_matrix_is_symmetric` — allclose symmetry check.
- `test_correlation_pearson_values_in_minus_one_to_one` — range bounds with tolerance.
- `test_correlation_pearson_identical_channels_give_one` — known-signal ground truth.
- `test_correlation_pearson_anti_correlated_gives_minus_one` — known-signal ground truth.
- `test_correlation_spearman_matches_scipy` — exact correspondence with `spearmanr`.
- `test_correlation_spearman_diagonal_is_one` — Spearman diagonal.
- `test_correlation_spearman_matrix_is_symmetric` — Spearman symmetry.
- `test_pearson_and_spearman_differ_on_nonlinear_data` — method distinguishability.
- `test_correlation_preserves_metadata_and_history` — all metadata fields + `extra`.
- `test_correlation_history_appends_to_existing` — PascalCase history entry.
- `test_correlation_custom_dim_name` — non-default dim name.
- `test_correlation_correlates_along_non_default_dim` — first-axis dim case.
- `test_correlation_raises_on_3d_input` — ndim guard.
- `test_correlation_raises_on_1d_input` — ndim guard.
- `test_correlation_raises_when_dim_missing` — dim-presence guard with match.
- `test_correlation_raises_when_dim_missing_includes_hint` — error message hint.
- `test_correlation_raises_on_invalid_method` — `__post_init__` guard.
- `test_correlation_raises_on_invalid_method_pearson_typo` — case-sensitivity guard.
- `test_correlation_does_not_mutate_input` — checks history, shape, and values.
- `test_correlation_accessible_via_feature_module` — autodiscovery smoke test.
- `test_correlation_is_data_instance` — return type check.

## Fix

No tests need changes.

## Add

No missing scenarios identified.

## Action List

None.
