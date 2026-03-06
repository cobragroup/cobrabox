# Test Review: mutual_information

**Feature**: `src/cobrabox/features/mutual_information.py`
**Test file**: `tests/test_feature_mutual_information.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```
mutual_information: 100% (72 statements, 0 missing)
```

## Summary

The test file has excellent coverage (100%) and thoroughly tests the algorithmic correctness of mutual information computation with both equiprobable and equidistant binning strategies. Tests include validation of the mathematical implementation against known ground truth values.

However, several standard feature test scenarios are missing: history tracking, metadata preservation, and input immutability. The file docstring also incorrectly references "line_length" instead of "mutual_information".

## Keep

Tests that are correct and complete:

- `test_entropy` — Correctly validates internal `_vector_entropy` helper against expected value
- `test_binning` — Validates `_get_binned` binning logic produces correct indices
- `test_bad_inits` — Comprehensive parameter validation tests (negative bins, non-integer bins, wrong dim types, missing dims, missing other_dim for high-D data)
- `test_low_dim_equidistant_bins` / `test_low_dim_equiprobable_bins` — Happy path tests with ground truth validation
- `test_high_dim_equiprobable_bins` / `test_high_dim_equidistant_bins` — Multi-dimensional data tests with correctness checks

## Fix

### File docstring (line 1)
Issue: Docstring says "line_length" instead of "mutual_information"
```python
# Corrected:
"""Tests for the mutual_information feature behavior."""
```

## Add

Missing standard feature test scenarios:

### `test_mutual_information_history_updated`

```python
def test_mutual_information_history_updated() -> None:
    """MutualInformation appends 'MutualInformation' to history."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=100.0)
    result = MutualInformation().apply(data)
    assert result.history[-1] == "MutualInformation"
```

### `test_mutual_information_metadata_preserved`

```python
def test_mutual_information_metadata_preserved() -> None:
    """MutualInformation preserves subjectID, groupID, condition."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(
        arr,
        dims=["space", "time"],
        sampling_rate=100.0,
        subjectID="s42",
        groupID="control",
        condition="rest",
    )
    result = MutualInformation().apply(data)
    assert result.subjectID == "s42"
    assert result.groupID == "control"
    assert result.condition == "rest"
```

### `test_mutual_information_sampling_rate_none`

```python
def test_mutual_information_sampling_rate_none() -> None:
    """MutualInformation sets sampling_rate to None since time dimension is removed."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=100.0)
    result = MutualInformation().apply(data)
    assert result.sampling_rate is None
```

### `test_mutual_information_does_not_mutate_input`

```python
def test_mutual_information_does_not_mutate_input() -> None:
    """MutualInformation does not modify the input Data object."""
    arr = np.zeros([6, 100])
    data = cb.SignalData.from_numpy(arr, dims=["space", "time"], sampling_rate=100.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    original_values = data.to_numpy().copy()

    _ = MutualInformation().apply(data)

    assert data.history == original_history
    assert data.data.shape == original_shape
    np.testing.assert_array_equal(data.to_numpy(), original_values)
```

## Action List

1. [Severity: MEDIUM] Fix test file docstring: change "line_length" to "mutual_information" (`tests/test_feature_mutual_information.py`, line 1)
2. [Severity: MEDIUM] Add `test_mutual_information_history_updated` to verify history tracking
3. [Severity: MEDIUM] Add `test_mutual_information_metadata_preserved` to verify metadata preservation
4. [Severity: MEDIUM] Add `test_mutual_information_sampling_rate_none` to verify output_type = Data behavior
5. [Severity: MEDIUM] Add `test_mutual_information_does_not_mutate_input` to verify input immutability
