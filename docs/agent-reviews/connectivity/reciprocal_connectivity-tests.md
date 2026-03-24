# Test Review: reciprocal_connectivity

**Feature**: `src/cobrabox/features/connectivity/reciprocal_connectivity.py`
**Test file**: `tests/features/connectivity/test_feature_reciprocal_connectivity.py`
**Date**: 2025-03-24
**Verdict**: PASS

## Coverage

```text
ReciprocalConnectivity: 100% (62 statements, 0 missing)
```

All statements in the feature file are covered by the test suite.

## Summary

The test file for `ReciprocalConnectivity` is comprehensive and well-structured. It covers both usage paths (time-series input and pre-computed matrix input) with thorough validation of output structure, numerical correctness, error handling, and metadata preservation. The 20 test cases achieve 100% code coverage and exercise all parameter validation paths.

The feature is a `BaseFeature[Data]` that supports two distinct input modes:

1. **Time-series path**: Uses `PartialDirectedCoherence` internally to compute a directed connectivity matrix
2. **Pre-computed matrix path**: Accepts an asymmetric `(space_to, space_from)` matrix directly

Tests are organized into logical sections with clear helper functions for data construction.

## Keep

Tests that are correct and complete — no changes needed:

- `test_reciprocal_connectivity_is_registered` — verifies auto-discovery
- `test_reciprocal_connectivity_from_timeseries_output_dims` — verifies output has only `('space',)` dim
- `test_reciprocal_connectivity_from_timeseries_output_shape` — verifies output shape matches n_channels
- `test_reciprocal_connectivity_from_timeseries_space_coords_preserved` — verifies coordinate propagation from time-series input
- `test_reciprocal_connectivity_history_updated` — verifies history appends class name
- `test_reciprocal_connectivity_driver_is_negative_sink_is_positive` — numerical correctness with known driver/sink
- `test_reciprocal_connectivity_normalize_changes_values_not_shape` — verifies normalize parameter effect
- `test_reciprocal_connectivity_from_precomputed_2d_matrix` — 2-D matrix path with exact RC value verification
- `test_reciprocal_connectivity_from_precomputed_3d_matrix_with_freq_band` — 3-D matrix with frequency averaging
- `test_reciprocal_connectivity_precomputed_space_coords_preserved` — verifies coordinate propagation from matrix input
- `test_reciprocal_connectivity_unsupported_connectivity_raises` — validates unsupported connectivity measure
- `test_reciprocal_connectivity_symmetric_matrix_raises` — validates symmetry check for pre-computed matrices
- `test_reciprocal_connectivity_freq_band_set_but_no_freq_dim_raises` — validates freq_band/frequency dim mismatch
- `test_reciprocal_connectivity_freq_band_outside_range_raises` — validates freq_band bounds checking
- `test_reciprocal_connectivity_freq_band_none_but_freq_dim_present_raises` — validates freq_band=None with freq dim
- `test_reciprocal_connectivity_invalid_freq_band_fmin_ge_fmax_raises` — validates `__post_init__` constraint
- `test_reciprocal_connectivity_precomputed_missing_space_dims_raises` — validates required dims for matrix input
- `test_reciprocal_connectivity_metadata_preserved` — verifies subjectID, groupID, condition preserved; sampling_rate=None
- `test_reciprocal_connectivity_does_not_mutate_input` — verifies input Data immutability
- `test_reciprocal_connectivity_no_space_coords_fallback` — verifies integer fallback when no coords present

## Fix

None. All tests are correct and well-structured.

## Add

None. All required scenarios are covered.

## Action List

None.
