# Test Review: hilbert

**Feature**: `src/cobrabox/features/hilbert.py`
**Test file**: `tests/test_feature_hilbert.py`
**Date**: 2025-03-05
**Verdict**: PASS

## Coverage

```
hilbert.py: 100% (83 statements, 0 missing)
```

## Summary

Excellent test coverage with 20 tests. Tests validate all four Hilbert transform modes (analytic, envelope, phase, frequency), dimension/shape preservation, dtype correctness (complex128 for analytic), numerical correctness (envelope of sine, instantaneous frequency), metadata preservation, error handling (invalid feature, missing sampling_rate for frequency mode, missing time dim), and pipeline compatibility.

## Keep

- `test_hilbert_output_is_signal_data` — return type
- `test_hilbert_output_dims_preserved` — dimension preservation
- `test_hilbert_analytic_dtype_is_complex` — dtype correctness
- `test_hilbert_analytic_real_part_equals_input` — analytic signal property
- `test_hilbert_envelope_pure_sine` — envelope computation
- `test_hilbert_envelope_nonnegative` — envelope constraint
- `test_hilbert_phase_range` — phase bounds
- `test_hilbert_frequency_pure_sine` — frequency computation
- `test_hilbert_history_appended` — history tracking
- `test_hilbert_subject_id_preserved` — metadata preservation
- `test_hilbert_sampling_rate_preserved` — metadata preservation
- `test_hilbert_coords_preserved` — coordinate preservation
- `test_hilbert_group_id_and_condition_preserved` — metadata preservation
- `test_hilbert_does_not_mutate_input` — immutability check
- `test_hilbert_invalid_feature_raises` — feature parameter validation
- `test_hilbert_frequency_without_sampling_rate_raises` — required field check
- `test_hilbert_missing_time_raises` — dimension validation
- `test_hilbert_pipe_with_line_length` — pipeline compatibility

## Fix

None. All tests are correct and comprehensive.

## Add

None. All required scenarios covered.

## Action List

None.
