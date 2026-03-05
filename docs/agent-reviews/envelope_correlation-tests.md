# Test Review: envelope_correlation

**Feature**: `src/cobrabox/features/envelope_correlation.py`
**Test file**: `tests/test_feature_envelope_correlation.py`
**Date**: 2025-03-04
**Verdict**: NEEDS WORK

## Summary

Good coverage of numerical properties and parameter variations. Tests correctly verify symmetry, valid correlation ranges, and coordinate preservation. However, missing the critical "no mutation of input" scenario and incomplete metadata preservation checks. No explicit guard for missing `time` dimension exists in the feature (inherits from `BaseFeature[SignalData]`), so that test is not applicable.

## Keep

Tests that are correct and complete — no changes needed:

- `test_envelope_correlation_output_dims_and_shape` — verifies output shape and dims
- `test_envelope_correlation_space_coords_preserved` — verifies coordinate preservation
- `test_envelope_correlation_matrix_is_symmetric` — good numerical property check
- `test_envelope_correlation_diagonal_is_finite` — verifies self-correlation values
- `test_envelope_correlation_absolute_true_gives_nonnegative` — parameter effect verified
- `test_envelope_correlation_values_in_valid_range` — boundary check
- `test_envelope_correlation_orthogonalize_false_same_shape` — parameter variation
- `test_envelope_correlation_orthogonalize_changes_values` — verifies different outputs
- `test_envelope_correlation_history_appended` — history tracking
- `test_envelope_correlation_raises_for_single_channel` — runtime validation
- `test_envelope_correlation_raises_for_extra_dims` — dimension validation

## Fix

Tests that exist but need changes:

### `test_envelope_correlation_metadata_preserved`

Issue: Incomplete coverage — only checks `subjectID` and `sampling_rate`, missing `groupID` and `condition`.

```python
def test_envelope_correlation_metadata_preserved() -> None:
    """subjectID, groupID, condition preserved; sampling_rate None for Data without time."""
    data = _make_data(
        subjectID="sub-99",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.EnvelopeCorrelation().apply(data)

    assert out.subjectID == "sub-99"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
    # sampling_rate is not preserved for Data without time dimension
    assert out.sampling_rate is None
```

## Add

Missing scenarios — new tests to add:

### `test_envelope_correlation_no_mutation`

```python
def test_envelope_correlation_no_mutation() -> None:
    """EnvelopeCorrelation does not modify the input Data object."""
    data = _make_data()
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.data.values.copy()

    _ = cb.feature.EnvelopeCorrelation().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.data.values, original_values)
```

## Action List

1. [Severity: HIGH] Add `test_envelope_correlation_no_mutation` to verify input is not mutated (`tests/test_feature_envelope_correlation.py`, after line 143)

2. [Severity: MEDIUM] Update `test_envelope_correlation_metadata_preserved` to also assert `groupID` and `condition` are preserved (`tests/test_feature_envelope_correlation.py`, line 136)
