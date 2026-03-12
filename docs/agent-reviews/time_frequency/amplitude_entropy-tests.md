# Test Review: amplitude_entropy

**Feature**: `src/cobrabox/features/amplitude_entropy.py`
**Test file**: `tests/test_feature_amplitude_entropy.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

Per-file coverage for amplitude_entropy.py:

```text
src/cobrabox/features/amplitude_entropy.py: 95% (37 statements, 2 missing)
Missing lines: 101-102
```

Lines 101-102 contain a defensive guard for the edge case where histogram counts sum to zero (empty row). This scenario is not currently covered by tests.

## Summary

The test file has 15 tests covering all required scenarios from the criteria. The tests are well-structured with clear docstrings, proper naming conventions, and good assertion quality. However, coverage is at 95% (just under the 95% threshold) due to two uncovered lines that handle an edge case in the histogram calculation.

## Keep

Tests that are correct and complete — no changes needed:

- `test_amplitude_entropy_basic` — checks output is Data with scalar shape, finite values
- `test_amplitude_entropy_deterministic` — verifies reproducibility
- `test_amplitude_entropy_constant_data` — tests zero entropy case
- `test_amplitude_entropy_uniform_data` — tests expected entropy for uniform distribution
- `test_amplitude_entropy_different_band_widths` — verifies parameter sensitivity
- `test_amplitude_entropy_history_updated` — verifies history append
- `test_amplitude_entropy_metadata_preserved` — verifies all metadata fields preserved; correctly asserts `sampling_rate is None` for `output_type = Data`
- `test_amplitude_entropy_returns_data_instance` — verifies return type
- `test_amplitude_entropy_does_not_mutate_input` — verifies input immutability
- `test_amplitude_entropy_negative_band_width` — `__post_init__` validation for negative param
- `test_amplitude_entropy_zero_band_width` — `__post_init__` validation for zero param
- `test_amplitude_entropy_single_row` — edge case: single time point
- `test_amplitude_entropy_single_column` — edge case: single spatial point
- `test_amplitude_entropy_1d_input_raises` — validates ndim >= 2 requirement

## Fix

None — all existing tests are correct.

## Add

Missing scenario — the defensive guard for `total == 0` is not covered:

### `test_amplitude_entropy_zero_total_counts`

Lines 101-102 handle the case where `counts.sum() == 0`. While this is a defensive guard that is hard to trigger with normal data (non-empty rows with finite values always produce some counts), it should be tested for completeness.

```python
def test_amplitude_entropy_zero_total_counts() -> None:
    """AmplitudeEntropy handles edge case where histogram has zero total counts."""
    # This tests the defensive guard at lines 101-102
    # Create data with NaN values which result in zero counts after histogram
    arr = np.full((5, 5), np.nan)
    data = cb.Data.from_numpy(arr, dims=["time", "space"])

    feature = cb.feature.AmplitudeEntropy(band_width=1.0)
    result = feature.apply(data)

    # Should not raise, should return finite value (0.0 from the guard)
    assert np.isfinite(float(result.to_numpy()))
```

Alternatively, mock the histogram to force `total == 0`:

```python
def test_amplitude_entropy_zero_total_counts_mocked() -> None:
    """AmplitudeEntropy handles zero total counts via defensive guard."""
    from unittest.mock import patch

    data = _make_data(n_time=5, n_space=5)
    feature = cb.feature.AmplitudeEntropy(band_width=1.0)

    # Mock np.histogram to return zero counts
    with patch('numpy.histogram', return_value=(np.array([0, 0]), np.array([0, 1, 2]))):
        result = feature.apply(data)

    # Should return 0.0 from the guard
    assert float(result.to_numpy()) == 0.0
```

## Action List

1. [Severity: HIGH] Add test to cover lines 101-102 (zero total counts defensive guard) to reach 100% coverage

Severity rationale: Coverage is at 95% with 2 lines uncovered. Per the criteria, <95% coverage is a HIGH severity issue. While the current coverage is borderline, the uncovered lines are defensive guards that should be tested.
