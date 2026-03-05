# Test Review: epileptogenicity_index

**Feature**: `src/cobrabox/features/epileptogenicity_index.py`
**Test file**: `tests/test_feature_epileptogenicity_index.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
EpileptogenicityIndex: 100% (79 statements, 0 missing)
```

Full coverage achieved — no uncovered lines.

## Summary

The test file is comprehensive with 17 tests covering value correctness, metadata preservation, error handling, and parameter sensitivity. Coverage is 100%. However, there are naming convention violations (test functions should use full feature name prefix) and missing tests for parameter validation. The file uses `test_ei_*` instead of the required `test_epileptogenicity_index_*` pattern.

## Keep

Tests that are correct and complete:

- `test_ei_dims` — Verifies output has exactly (space,) dimensions
- `test_ei_output_shape` — Verifies output shape is (n_channels,)
- `test_ei_space_coords_preserved` — Verifies space coordinates are carried through
- `test_ei_values_in_unit_interval` — Verifies all EI values in [0, 1]
- `test_ei_max_is_one` — Verifies maximum EI equals 1 after normalization
- `test_ei_early_channel_scores_higher` — Verifies temporal ordering of EI values
- `test_ei_no_discharge_channel_near_zero` — Verifies channels without discharge get near-zero EI
- `test_ei_flat_signal_all_zero` — Verifies constant signals yield all-zero EI
- `test_ei_three_channels_ordering` — Verifies correct ordering with three channels
- `test_ei_history_appended` — Verifies 'EpileptogenicityIndex' appended to history
- `test_ei_metadata_preserved` — Verifies metadata preservation and sampling_rate=None
- `test_ei_window_duration_accepted_and_shape_unchanged` — Verifies parameter sensitivity
- `test_ei_very_high_threshold_suppresses_detection` — Verifies threshold behavior
- `test_ei_raises_without_time_dim` — Verifies ValueError for missing time dimension
- `test_ei_raises_without_sampling_rate` — Verifies ValueError for missing sampling_rate
- `test_ei_raises_with_extra_dims` — Verifies ValueError for extra dimensions
- `test_ei_raises_when_signal_shorter_than_window` — Verifies ValueError for short signals

## Fix

Tests that need changes:

### All test function names
Issue: Uses `test_ei_*` prefix instead of required `test_epileptogenicity_index_*` pattern per naming conventions.

Rename all 17 functions:
- `test_ei_dims` → `test_epileptogenicity_index_dims`
- `test_ei_output_shape` → `test_epileptogenicity_index_output_shape`
- `test_ei_space_coords_preserved` → `test_epileptogenicity_index_space_coords_preserved`
- `test_ei_values_in_unit_interval` → `test_epileptogenicity_index_values_in_unit_interval`
- `test_ei_max_is_one` → `test_epileptogenicity_index_max_is_one`
- `test_ei_early_channel_scores_higher` → `test_epileptogenicity_index_early_channel_scores_higher`
- `test_ei_no_discharge_channel_near_zero` → `test_epileptogenicity_index_no_discharge_channel_near_zero`
- `test_ei_flat_signal_all_zero` → `test_epileptogenicity_index_flat_signal_all_zero`
- `test_ei_three_channels_ordering` → `test_epileptogenicity_index_three_channels_ordering`
- `test_ei_history_appended` → `test_epileptogenicity_index_history_appended`
- `test_ei_metadata_preserved` → `test_epileptogenicity_index_metadata_preserved`
- `test_ei_window_duration_accepted_and_shape_unchanged` → `test_epileptogenicity_index_window_duration_accepted`
- `test_ei_very_high_threshold_suppresses_detection` → `test_epileptogenicity_index_high_threshold_suppresses_detection`
- `test_ei_raises_without_time_dim` → `test_epileptogenicity_index_raises_without_time_dim`
- `test_ei_raises_without_sampling_rate` → `test_epileptogenicity_index_raises_without_sampling_rate`
- `test_ei_raises_with_extra_dims` → `test_epileptogenicity_index_raises_with_extra_dims`
- `test_ei_raises_when_signal_shorter_than_window` → `test_epileptogenicity_index_raises_signal_shorter_than_window`

## Add

Missing scenarios to add:

### `test_epileptogenicity_index_returns_data_instance`

```python
def test_epileptogenicity_index_returns_data_instance() -> None:
    """EpileptogenicityIndex.apply() returns a Data instance."""
    data = _two_channel_data()
    result = cb.feature.EpileptogenicityIndex().apply(data)
    assert isinstance(result, cb.Data)
```

### `test_epileptogenicity_index_does_not_mutate_input`

```python
def test_epileptogenicity_index_does_not_mutate_input() -> None:
    """EpileptogenicityIndex.apply() leaves input Data unchanged."""
    data = _two_channel_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.EpileptogenicityIndex().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_epileptogenicity_index_negative_window_duration`

```python
def test_epileptogenicity_index_negative_window_duration() -> None:
    """Negative window_duration should raise ValueError."""
    data = _two_channel_data()
    with pytest.raises(ValueError):
        cb.feature.EpileptogenicityIndex(window_duration=-1.0).apply(data)
```

### `test_epileptogenicity_index_zero_integration_window`

```python
def test_epileptogenicity_index_zero_integration_window() -> None:
    """Zero integration_window should raise ValueError."""
    data = _two_channel_data()
    with pytest.raises(ValueError):
        cb.feature.EpileptogenicityIndex(integration_window=0.0).apply(data)
```

## Action List

1. [Severity: MEDIUM] Rename all test functions from `test_ei_*` to `test_epileptogenicity_index_*` pattern (17 renames required)
2. [Severity: MEDIUM] Add `test_epileptogenicity_index_returns_data_instance` test
3. [Severity: MEDIUM] Add `test_epileptogenicity_index_does_not_mutate_input` test
4. [Severity: LOW] Add parameter validation tests for negative/zero window_duration and integration_window if __post_init__ validation exists; otherwise skip
