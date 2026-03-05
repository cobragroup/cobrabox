# Test Review: amplitude_variation

**Feature**: `src/cobrabox/features/amplitude_variation.py`
**Test file**: `tests/test_feature_amplitude_variation.py`
**Date**: 2025-03-05
**Verdict**: NEEDS WORK

## Coverage

```
src/cobrabox/features/amplitude_variation.py: 100% (11 statements, 0 missing)
```

✅ 100% coverage — meets the ≥95% threshold.

## Summary

The test file has good coverage and tests core functionality including expected values, edge cases (constant signal), single-channel handling, Chord integration, and input immutability. However, it is missing a critical error-case test for missing `time` dimension. Additionally, metadata preservation testing is incomplete (only `subjectID` checked, missing `groupID` and `condition`).

## File & Function Conventions

- ✅ File naming: `test_feature_amplitude_variation.py` matches feature file
- ⚠️ Function naming: Uses `test_feature_amp_var_*` prefix (extra `feature_` not ideal, but acceptable)
- ✅ Return annotations: All functions have `-> None`
- ✅ Docstrings: Every test has a one-line docstring

## Required Scenarios

| Scenario | Status | Notes |
|----------|--------|-------|
| Happy path | ✅ | `test_feature_amp_var_expected_values_and_history` — checks shape, dims, values |
| History updated | ✅ | Line 29: `assert out.history == ["AmpVar"]` |
| Metadata preserved | ⚠️ | Only `subjectID` checked. Missing `groupID`, `condition` assertions |
| Output type handling | ✅ | Line 28: `assert out.sampling_rate is None` (time dim removed) |
| Invalid dims — missing `time` | ❌ | **NOT TESTED** — HIGH severity gap |
| Invalid params | ⚪ | **SKIPPED** — feature has no parameters |
| Output type is `Data` | ✅ | Line 23: `assert isinstance(out, cb.Data)` |
| No mutation of input | ✅ | `test_feature_amp_var_no_mutation` added |
| Edge case: constant signal | ✅ | `test_feature_amp_var_constant_signal_is_zero` |
| Edge case: single channel | ✅ | `test_feature_amp_var_single_channel` |
| Integration: via Chord | ✅ | `test_feature_amp_var_via_chord` — tests pipeline usage |

## Keep

Tests that are correct and complete — no changes needed:

- `test_feature_amp_var_expected_values_and_history` — Tests correct values, shape, dims, history, output type, and sampling_rate=None. Comprehensive happy path test.
- `test_feature_amp_var_constant_signal_is_zero` — Good edge case: constant signal produces zero std.
- `test_feature_amp_var_single_channel` — Tests single-channel input correctly.
- `test_feature_amp_var_via_chord` — Verifies Chord composition works correctly.
- `test_feature_amp_var_no_mutation` — Correctly verifies input data is not modified.

## Fix

Tests that exist but need changes:

### `test_feature_amp_var_expected_values_and_history`
**Issue**: Only checks `subjectID` for metadata preservation. Missing `groupID` and `condition`.

**Fix**: Either add assertions for all metadata fields, or create a dedicated test:

```python
# Option 1: Extend existing test (add after line 27):
assert out.groupID == data.groupID
assert out.condition == data.condition

# Option 2: Create dedicated test:
def test_amp_var_metadata_preserved() -> None:
    """AmpVar preserves subjectID, groupID, and condition."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    data = cb.SignalData.from_numpy(
        arr,
        dims=["time", "space"],
        sampling_rate=200.0,
        subjectID="sub-01",
        groupID="group-A",
        condition="rest",
    )
    out = cb.feature.AmpVar().apply(data)

    assert out.subjectID == "sub-01"
    assert out.groupID == "group-A"
    assert out.condition == "rest"
```

## Add

Missing scenarios — new tests to add:

### `test_amp_var_missing_time_raises`

AmpVar requires `SignalData` which has a `time` dimension. Must test that missing `time` raises ValueError.

```python
def test_amp_var_missing_time_raises() -> None:
    """AmpVar raises ValueError when 'time' dimension is missing."""
    import xarray as xr
    # Build a Data-like object whose underlying array lacks 'time'
    arr = np.random.randn(10)
    xr_data = xr.DataArray(arr, dims=["space"])
    # Bypass Data.__init__ validation to isolate the feature guard
    raw = cb.Data.__new__(cb.Data)
    object.__setattr__(raw, "_data", xr_data)
    with pytest.raises(ValueError, match="time"):
        cb.feature.AmpVar().apply(raw)
```

## Action List

1. **[Severity: HIGH]** Add missing test `test_amp_var_missing_time_raises` to verify ValueError is raised when time dimension is absent.

2. **[Severity: MEDIUM]** Extend `test_feature_amp_var_expected_values_and_history` to assert `groupID` and `condition` are preserved, or create a dedicated `test_amp_var_metadata_preserved` test with all metadata fields set explicitly.

3. **[Severity: LOW]** Consider renaming tests from `test_feature_amp_var_*` to `test_amp_var_*` to align with naming convention (feature name prefix only, not `feature_` prefix).
