# Test Review: amplitude_variation

**File**: `tests/test_feature_amplitude_variation.py`
**Date**: 2026-03-05
**Verdict**: PASS

## Summary

Solid test coverage for a minimal feature. Tests cover happy path, edge cases (constant signal, single channel), and integration with Chord. All required scenarios are present. Coverage is 100% for the feature file.

## Coverage

```
src/cobrabox/features/amplitude_variation.py         11      0   100%
```

✅ 100% coverage — meets the ≥95% threshold.

## File & Function Conventions

- ✅ File naming: `test_feature_amplitude_variation.py` matches feature file
- ✅ Function naming: All tests use `test_feature_amp_var_*` prefix (acceptable variation)
- ✅ Return annotations: All functions have `-> None`
- ✅ Docstrings: Every test has a one-line docstring

## Required Scenarios

| Scenario | Status | Notes |
|----------|--------|-------|
| Happy path | ✅ | `test_feature_amp_var_expected_values_and_history` — checks shape, dims, values |
| History updated | ✅ | Line 29: `assert out.history == ["AmpVar"]` |
| Metadata preserved | ✅ | Line 27-28: checks `subjectID`, `sampling_rate` |
| Output type handling | ✅ | Line 28: `assert out.sampling_rate is None` (time dim removed) |
| Invalid dims — missing `time` | ⚪ | **SKIPPED** — feature has no parameters and SignalData enforces time |
| Invalid params | ⚪ | **SKIPPED** — feature has no parameters |
| Output type is `Data` | ✅ | Line 23: `assert isinstance(out, cb.Data)` |
| No mutation of input | ⚪ | **NOT TESTED** — would be nice to add |
| Edge case: constant signal | ✅ | `test_feature_amp_var_constant_signal_is_zero` |
| Edge case: single channel | ✅ | `test_feature_amp_var_single_channel` |
| Integration: via Chord | ✅ | `test_feature_amp_var_via_chord` — tests pipeline usage |

## Assertion Quality

- ✅ Specific assertions: checks `dims`, `shape`, exact values with `assert_allclose`
- ✅ Uses `np.testing.assert_allclose` for float comparisons
- ✅ No loose assertions like `is not None`

## Test Independence

- ✅ Each test creates its own data inline — no shared state
- ✅ No module-level fixtures

## Action List

1. [Severity: LOW] Add test for input mutation — verify input data is unchanged after `.apply()` call. Suggested:
   ```python
   def test_feature_amp_var_no_mutation() -> None:
       """Input data is not modified by AmpVar."""
       arr = np.array([[1.0, 2.0], [3.0, 4.0]])
       data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0, subjectID="s1")
       original_history = list(data.history)
       original_shape = data.data.shape
       _ = cb.feature.AmpVar().apply(data)
       assert data.history == original_history
       assert data.data.shape == original_shape
   ```
