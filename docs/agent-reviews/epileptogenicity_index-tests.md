# Test Review: epileptogenicity_index

**Feature**: `src/cobrabox/features/epileptogenicity_index.py`
**Test file**: `tests/test_feature_epileptogenicity_index.py`
**Date**: 2025-03-04
**Verdict**: NEEDS WORK

## Summary

Strong test coverage for the EpileptogenicityIndex feature with excellent validation of algorithm correctness (temporal ordering, value ranges, edge cases). Tests include sophisticated synthetic signal generation for gamma burst detection and comprehensive error handling. Two required scenarios are missing: explicit type checking and input immutability verification.

## Keep

Tests that are correct and complete — no changes needed:

- `test_ei_dims` — Correctly verifies output has exactly `(space,)` dimensions
- `test_ei_output_shape` — Confirms output shape matches number of channels
- `test_ei_space_coords_preserved` — Validates space coordinates carry through unchanged
- `test_ei_values_in_unit_interval` — Asserts all EI values are in [0, 1]
- `test_ei_max_is_one` — Confirms normalisation produces max value of 1.0
- `test_ei_early_channel_scores_higher` — Tests core algorithm behavior: earlier detection = higher EI
- `test_ei_no_discharge_channel_near_zero` — Validates channels without rapid discharge receive near-zero EI
- `test_ei_flat_signal_all_zero` — Edge case: constant signal produces all zeros
- `test_ei_three_channels_ordering` — Multi-channel temporal ordering validation
- `test_ei_history_appended` — Correctly checks history ends with "EpileptogenicityIndex"
- `test_ei_metadata_preserved` — Validates subjectID, groupID, condition preserved; correctly notes sampling_rate becomes None
- `test_ei_window_duration_accepted_and_shape_unchanged` — Parameter sensitivity testing
- `test_ei_very_high_threshold_suppresses_detection` — Extreme parameter value testing
- `test_ei_raises_without_time_dim` — Validates ValueError when 'time' dimension missing
- `test_ei_raises_without_sampling_rate` — Validates ValueError when sampling_rate is None
- `test_ei_raises_with_extra_dims` — Validates ValueError when extra dimensions present
- `test_ei_raises_when_signal_shorter_than_window` — Runtime guard: signal shorter than window duration

## Fix

None — all existing tests are correct.

## Add

Missing scenarios — new tests to add:

### `test_ei_returns_data_instance`

```python
def test_ei_returns_data_instance() -> None:
    """EpileptogenicityIndex.apply() returns a Data instance."""
    out = cb.feature.EpileptogenicityIndex().apply(_two_channel_data())
    assert isinstance(out, cb.Data)
```

### `test_ei_does_not_mutate_input`

```python
def test_ei_does_not_mutate_input() -> None:
    """EpileptogenicityIndex.apply() does not modify the input Data object."""
    data = _two_channel_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.EpileptogenicityIndex().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: MEDIUM] Add `test_ei_returns_data_instance` to verify `isinstance(result, cb.Data)` (tests/test_feature_epileptogenicity_index.py)

2. [Severity: MEDIUM] Add `test_ei_does_not_mutate_input` to verify input Data is not modified (tests/test_feature_epileptogenicity_index.py)
