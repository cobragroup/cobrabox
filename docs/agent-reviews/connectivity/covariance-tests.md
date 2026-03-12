<!-- updated 2026-03-06 (rev 2) -->
# Test Review: covariance

**Feature**: `src/cobrabox/features/covariance.py`
**Test file**: `tests/test_feature_covariance.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
covariance.py: 100% (27 statements, 0 missing)
```

## Summary

20 tests, 100% coverage, ruff clean. All issues from the previous review have been addressed:
the shared module-level `rng` has been replaced with per-test `rng = np.random.default_rng(N)`
calls (seeds 0–18), ensuring full test isolation. The full scenario set — output shape/dims,
coordinate preservation, numerical correctness (vs `np.cov`), diagonal equality to
`np.var(..., ddof=1)`, symmetry, positive diagonal, metadata, custom dims, all error paths,
no mutation, API accessibility — is present and correct. No further changes needed.

## Keep

- `test_covariance_output_dims_and_shape` — shape, type, and dim checks on default call.
- `test_covariance_output_is_square` — matrix squareness.
- `test_covariance_channel_coords_preserved` — coord labels on both axes.
- `test_covariance_matches_numpy_cov` — exact correspondence with `np.cov`.
- `test_covariance_diagonal_equals_sample_variance` — diagonal vs `np.var(ddof=1)`.
- `test_covariance_matrix_is_symmetric` — allclose symmetry check.
- `test_covariance_diagonal_positive_for_nonzero_signals` — positivity of variance.
- `test_covariance_identical_channels_diagonal_matches_off_diagonal` — var == cov ground truth.
- `test_covariance_preserves_metadata_and_history` — all metadata fields + `extra`.
- `test_covariance_history_appends_correctly` — PascalCase history entry.
- `test_covariance_custom_dim_name` — non-default dim name.
- `test_covariance_correlates_along_non_default_dim` — first-axis dim case.
- `test_covariance_custom_dim_matches_numpy_cov` — numerical correctness for custom dim.
- `test_covariance_raises_on_3d_input` — ndim guard.
- `test_covariance_raises_on_1d_input` — ndim guard.
- `test_covariance_raises_when_dim_missing` — dim-presence guard with match.
- `test_covariance_raises_when_dim_missing_includes_hint` — error message hint.
- `test_covariance_does_not_mutate_input` — checks history, shape, and values.
- `test_covariance_accessible_via_feature_module` — autodiscovery smoke test.
- `test_covariance_output_is_data_instance` — return type check.

## Fix

No tests need changes.

## Add

No missing scenarios identified.

## Action List

None.
