# Test Review: partial_directed_coherence

**Feature**: `src/cobrabox/features/connectivity/partial_directed_coherence.py`
**Test file**: `tests/features/connectivity/test_feature_partial_directed_coherence.py`
**Date**: 2026-03-24
**Verdict**: PASS

## Coverage

```text
PartialDirectedCoherence: 100% (51 statements, 0 missing)
```

## Summary

Excellent test coverage with 18 comprehensive tests. All required scenarios are covered plus additional numerical correctness tests that validate the PDC algorithm's mathematical properties. The test file uses good helper functions, proper naming conventions, and all tests have docstrings.

## Keep

All existing tests are correct and well-written:

- `test_pdc_is_registered` — verifies feature auto-discovery
- `test_pdc_output_dims` — validates output dimension names
- `test_pdc_output_shape` — validates output shape (n_ch, n_ch, n_freqs)
- `test_pdc_space_coords_preserved` — input space labels propagate to space_to/space_from
- `test_pdc_frequency_coord_range` — frequency axis spans [0, Nyquist]
- `test_pdc_returns_data_not_signal_data` — output_type=Data correctly returns Data type
- `test_pdc_history_updated` — history appends class name
- `test_pdc_values_in_unit_interval` — PDC values bounded in [0, 1]
- `test_pdc_column_sums_to_one` — mathematical property: squared PDC sums to 1
- `test_pdc_directed_coupling_detected` — directional detection with synthetic data
- `test_pdc_fixed_var_order` — var_order parameter respected
- `test_pdc_requires_sampling_rate` — raises ValueError when sampling_rate is None
- `test_pdc_requires_at_least_2_channels` — raises ValueError for single-channel input
- `test_pdc_invalid_n_freqs_raises` — `__post_init__` validates n_freqs >= 1
- `test_pdc_invalid_var_order_raises` — `__post_init__` validates var_order >= 1
- `test_pdc_metadata_preserved` — subjectID, groupID, condition preserved; sampling_rate=None
- `test_pdc_does_not_mutate_input` — input Data unchanged after apply()
- `test_pdc_invalid_ndim_raises` — raises ValueError for non-2-D input

## Action List

None.
