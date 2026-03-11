# Test Review: reciprocal_connectivity

**Feature**: `src/cobrabox/features/reciprocal_connectivity.py`
**Test file**: `tests/test_feature_reciprocal_connectivity.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
ReciprocalConnectivity: 100% (62 statements, 0 missing)
```

Full coverage achieved. All lines in the feature file are exercised by the test suite.

## Summary

Excellent test coverage for a complex dual-path feature. The test file correctly handles both usage modes (time-series input via VAR/PDC and pre-computed connectivity matrix input) with thorough validation of numerical correctness, edge cases, and error paths. All required scenarios per the test criteria are covered.

## Keep

Tests that are correct and complete — no changes needed:

- `test_reciprocal_connectivity_is_registered` — verifies feature auto-discovery
- `test_reciprocal_connectivity_from_timeseries_output_dims` — output dims for time-series path
- `test_reciprocal_connectivity_from_timeseries_output_shape` — output shape validation
- `test_reciprocal_connectivity_from_timeseries_space_coords_preserved` — space coords propagation
- `test_reciprocal_connectivity_history_updated` — history append verification
- `test_reciprocal_connectivity_driver_is_negative_sink_is_positive` — numerical correctness with synthetic driver/sink data
- `test_reciprocal_connectivity_normalize_changes_values_not_shape` — normalize param effect
- `test_reciprocal_connectivity_from_precomputed_2d_matrix` — 2-D matrix path with manual RC calculation verification
- `test_reciprocal_connectivity_from_precomputed_3d_matrix_with_freq_band` — 3-D matrix with frequency averaging
- `test_reciprocal_connectivity_precomputed_space_coords_preserved` — space coords on matrix input
- `test_reciprocal_connectivity_unsupported_connectivity_raises` — validation for unsupported connectivity measure
- `test_reciprocal_connectivity_symmetric_matrix_raises` — symmetry check for pre-computed matrices
- `test_reciprocal_connectivity_freq_band_set_but_no_freq_dim_raises` — freq_band validation
- `test_reciprocal_connectivity_freq_band_outside_range_raises` — frequency range bounds check
- `test_reciprocal_connectivity_freq_band_none_but_freq_dim_present_raises` — freq_band=None validation
- `test_reciprocal_connectivity_invalid_freq_band_fmin_ge_fmax_raises` — constructor validation via `__post_init__`
- `test_reciprocal_connectivity_precomputed_missing_space_dims_raises` — dimension validation for matrix input
- `test_reciprocal_connectivity_metadata_preserved` — subjectID, groupID, condition preservation; sampling_rate=None
- `test_reciprocal_connectivity_does_not_mutate_input` — immutability check
- `test_reciprocal_connectivity_no_space_coords_fallback` — fallback to integer indices when coords missing

## Fix

None. All existing tests are correct and complete.

## Add

No missing scenarios. The test suite comprehensively covers:

1. **Both input paths**: time-series (via PDC) and pre-computed matrix (2-D and 3-D)
2. **All parameters**: connectivity, freq_band, var_order, n_freqs, normalize
3. **All validation paths**: 9 distinct error cases covered
4. **Numerical correctness**: Manual RC calculation verification and driver/sink detection
5. **Metadata preservation**: All fields verified
6. **Edge cases**: Missing coords fallback, normalize effect

## Action List

None.
