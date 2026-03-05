# Test Review: hilbert

**Feature**: `src/cobrabox/features/hilbert.py`  
**Test file**: `tests/test_feature_hilbert.py`  
**Date**: 2025-03-05  
**Verdict**: PASS

## Coverage

```
hilbert.py: 100% (31 statements, 0 missing)
```

All lines covered. No missing statements.

## Summary

Comprehensive test suite covering all 4 Hilbert feature modes (analytic, envelope, phase, frequency),
error handling, metadata preservation, and pipeline compatibility. Tests include numerical correctness
validation against known ground truth (pure sine waves). All required scenarios from criteria.md are met.

## Keep

All existing tests are correct and complete — no changes needed:

- `test_hilbert_output_is_signal_data` — verifies SignalData return type
- `test_hilbert_output_dims_preserved` — all 4 modes preserve dims
- `test_hilbert_output_shape_preserved` — all 4 modes preserve shape
- `test_hilbert_analytic_dtype_is_complex` — correct dtype verification
- `test_hilbert_analytic_real_part_equals_input` — mathematical property check
- `test_hilbert_default_feature_is_analytic` — default parameter verification
- `test_hilbert_envelope_pure_sine` — numerical correctness (envelope ≈ 1 for unit sine)
- `test_hilbert_envelope_nonnegative` — invariant check
- `test_hilbert_phase_range` — invariant check (phase ∈ [-π, π])
- `test_hilbert_frequency_pure_sine` — numerical correctness (frequency ≈ input frequency)
- `test_hilbert_history_appended` — history tracking
- `test_hilbert_subject_id_preserved` — metadata preservation
- `test_hilbert_sampling_rate_preserved` — metadata preservation
- `test_hilbert_coords_preserved` — coordinate preservation
- `test_hilbert_group_id_and_condition_preserved` — metadata preservation
- `test_hilbert_does_not_mutate_input` — immutability check
- `test_hilbert_invalid_feature_raises` — __post_init__ validation
- `test_hilbert_frequency_without_sampling_rate_raises` — runtime guard
- `test_hilbert_missing_time_raises` — dimension validation (with noqa: PT011)
- `test_hilbert_pipe_with_line_length` — pipeline integration

## Fix

None. All tests pass criteria.

## Add

None required. All criteria are covered.

## Action List

None.
