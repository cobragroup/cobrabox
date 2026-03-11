# Test Review: cordance

**Feature**: `src/cobrabox/features/cordance.py`
**Test file**: `tests/test_feature_cordance.py`
**Date**: 2025-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
Cordance: 100% (58 statements, 0 missing)
```

Coverage meets the 95% threshold.

## Summary

The test suite is comprehensive with 30 tests covering shape/dims, algorithm correctness, output modes, threshold behavior, parameter validation, error handling, and edge cases. The helpers `_sine_data`, `_varied_data`, and `_varied_amplitude_data` provide good test data variety. However, a few required scenarios from the test criteria are missing, and one existing test uses an inappropriate testing approach.

## Keep

Tests that are correct and complete:

- `test_cordance_default_dims_and_shape` — verifies output dims and shape
- `test_cordance_default_band_coords` — verifies band_index coordinate
- `test_cordance_custom_bands_shape` — verifies custom bands work
- `test_cordance_mixed_spec_shape` — verifies mixed True + custom range
- `test_cordance_values_are_finite` — basic sanity check
- `test_cordance_values_bounded` — verifies mathematical bounds [-1, 1]
- `test_cordance_concordance_positive_discordance_negative` — verifies sign semantics
- `test_cordance_channel_with_dominant_band_highest_relative_power` — semantic test
- `test_cordance_agrees_with_manual_calculation` — gold-standard comparison
- `test_cordance_output_concordance_only` — output mode test
- `test_cordance_output_discordance_only` — output mode test
- `test_cordance_combined_equals_concordance_minus_discordance` — mathematical identity
- `test_cordance_threshold_changes_classification` — threshold behavior
- `test_cordance_threshold_validation` — parameter validation
- `test_cordance_nperseg_changes_nothing_in_shape` — nperseg parameter test
- `test_cordance_history_appended` — history test
- `test_cordance_metadata_preserved` — metadata test (partial, see Fix below)
- `test_cordance_in_pipeline` — pipeline composition
- `test_cordance_identical_channels_all_concordant` — edge case
- `test_cordance_empty_bands_equals_none` — edge case
- `test_cordance_raises_when_no_space_dim` — error handling
- `test_cordance_raises_when_single_channel` — error handling
- `test_cordance_raises_for_true_with_unknown_band` — error handling (delegated to Bandpower)
- `test_cordance_raises_when_nperseg_less_than_2` — parameter validation
- `test_cordance_raises_on_zero_signal` — error handling
- `test_cordance_nan_on_zero_outputs_nan_for_silent_channels` — nan_on_zero parameter
- `test_cordance_raises_for_false_band_spec` — error handling
- `test_cordance_true_alias_matches_explicit_range` — semantic equivalence
- `test_cordance_invalid_output_parameter` — parameter validation

## Fix

### `test_cordance_raises_when_sampling_rate_missing`

Issue: Uses a `_FakeData` class hack to bypass SignalData construction. This is not appropriate for `BaseFeature[SignalData]` features because SignalData enforces the time dimension (and sampling_rate) at construction time. You cannot construct a SignalData without sampling_rate through normal means, so this error scenario cannot occur in practice. The test should be removed.

```python
# Remove this test entirely - not applicable for BaseFeature[SignalData]
```

### `test_cordance_metadata_preserved`

Issue: Only checks `subjectID` and `sampling_rate`. Should also verify `groupID` and `condition` are preserved.

```python
def test_cordance_metadata_preserved() -> None:
    """subjectID, groupID, condition preserved; sampling_rate is None since time dim is removed."""
    data = _sine_data(freq_hz=10.0, subjectID="sub-42")
    # Add groupID and condition to the data
    data = cb.SignalData.from_numpy(
        data.to_numpy(),
        dims=["time", "space"],
        sampling_rate=data.sampling_rate,
        subjectID="sub-42",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.Cordance().apply(data)

    assert out.subjectID == "sub-42"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    # output_type=Data and no time dim → sampling_rate is stripped
    assert out.sampling_rate is None
```

## Add

### `test_cordance_does_not_mutate_input`

Missing required scenario: verify input Data is unchanged after `.apply()`.

```python
def test_cordance_does_not_mutate_input() -> None:
    """Cordance.apply() leaves the input Data object unchanged."""
    data = _varied_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = cb.feature.Cordance().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

### `test_cordance_returns_data_instance`

Missing explicit check that return type is Data (currently only checked implicitly via `isinstance(out, cb.Data)` in other tests).

```python
def test_cordance_returns_data_instance() -> None:
    """Cordance.apply() always returns a Data instance."""
    data = _sine_data(freq_hz=10.0)
    result = cb.feature.Cordance().apply(data)
    assert isinstance(result, cb.Data)
```

## Action List

1. [Severity: MEDIUM] Remove `test_cordance_raises_when_sampling_rate_missing` — uses inappropriate `_FakeData` hack; SignalData enforces sampling_rate at construction (`tests/test_feature_cordance.py:356-372`)
2. [Severity: MEDIUM] Fix `test_cordance_metadata_preserved` — add assertions for `groupID` and `condition` (`tests/test_feature_cordance.py:296-303`)
3. [Severity: MEDIUM] Add `test_cordance_does_not_mutate_input` — missing required "no mutation" scenario
4. [Severity: LOW] Add `test_cordance_returns_data_instance` — explicit return type check (already covered implicitly but good for completeness)
